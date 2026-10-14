"""
Test suite for answer-position metrics (spec PR 3, tests D09-D10).

Covers:
  * answer_predictions: shift convention, masking
  * D09: hand-made logits give exact accuracy / per-slice / per-class /
    leakage / other values
  * D10: cross-entropy with the repo convention equals a manual mean over
    answer positions only

Run with: pytest src/research_lib/data/synthetic/tests/test_metrics.py -v
"""

import pytest
import torch

from ..metrics import CLASS_NAMES, RecallMetrics, answer_predictions
from ..mqar import MQARConfig, generate_mqar
from ..overwrite import OverwriteConfig, generate_overwrite

IGNORE = -100
NO_CLASS = -1


# Hand-built overwrite-style batch: T=6, 1 example, 2 answers.
#   positions:        0      1      2      3      4      5
#   input_ids:        k0     v0     k1     v1     k0?    ...
# We only need labels/key_class/old_labels + logits; content is arbitrary
# as long as it matches the shift convention.
def _hand_batch():
    labels = torch.tensor([[IGNORE, IGNORE, IGNORE, 11, IGNORE, 7]], dtype=torch.int64)
    old_labels = torch.tensor(
        [[IGNORE, IGNORE, IGNORE, 10, IGNORE, IGNORE]], dtype=torch.int64
    )
    key_class = torch.tensor(
        [[NO_CLASS, NO_CLASS, NO_CLASS, 1, NO_CLASS, 3]], dtype=torch.int64
    )
    write_distance = torch.tensor([[-1, -1, -1, 4, -1, -1]], dtype=torch.int64)
    slice_id = torch.tensor([8], dtype=torch.int64)
    return {
        "labels": labels,
        "old_labels": old_labels,
        "key_class": key_class,
        "write_distance": write_distance,
        "slice_id": slice_id,
    }


def _hand_logits():
    """(1, 6, 12) logits with hand-set argmaxes at the answer positions.

    Answers: position 3 (target 11, old 10, class ADJACENT) is predicted
    from logits[:, 2]; position 5 (target 7, class SINGLE) from logits[:, 4].
    Preds: position-3 answer -> 11 (correct); position-5 answer -> 10
    (a *wrong* value, equal to the old value of the other key — pure noise
    example, but exact for the metric math).
    """
    logits = torch.zeros(1, 6, 12, dtype=torch.float64)
    logits[0, 2, 11] = 5.0  # predicts 11 at position 3 -> correct
    logits[0, 4, 10] = 5.0  # predicts 10 at position 5 -> wrong
    return logits


# =============================================================================
# Tests: answer_predictions
# =============================================================================


def test_answer_predictions_shift_and_mask():
    B, T, V = 2, 5, 7
    logits = torch.arange(B * T * V, dtype=torch.float64).reshape(B, T, V)
    labels = torch.full((B, T), IGNORE, dtype=torch.int64)
    labels[0, 3] = 4
    labels[1, 5 - 1] = 6  # predicted from logits[:, 3]
    preds, targets, mask = answer_predictions(logits, labels)
    assert preds.shape == targets.shape == mask.shape == (B, T - 1)
    assert torch.equal(preds, logits[:, :-1, :].argmax(-1))
    assert torch.equal(targets, labels[:, 1:])
    assert torch.equal(mask, labels[:, 1:] != IGNORE)
    assert int(mask.sum()) == 2
    # Batch shape mismatch raises.
    with pytest.raises(ValueError):
        answer_predictions(logits, torch.zeros(3, T, dtype=torch.int64))


# =============================================================================
# Tests: D09 metrics
# =============================================================================


def test_D09_exact_accuracy_slices_classes_leak_other():
    batch = _hand_batch()
    logits = _hand_logits()
    metrics = RecallMetrics()
    metrics.update(logits, batch)
    out = metrics.compute()

    # Two answers: one correct (11), one wrong (10, not the old value of
    # key with class SINGLE; old_labels at position 5 is IGNORE).
    assert out["acc"] == pytest.approx(0.5)
    assert out["acc/slice=8"] == pytest.approx(0.5)
    # Class totals: ADJACENT (code 1) has the correct answer; SINGLE (code 3)
    # the wrong one.
    assert out["acc/class=adjacent"] == pytest.approx(1.0)
    assert out["acc/class=single"] == pytest.approx(0.0)
    assert out["leak/class=adjacent"] == pytest.approx(0.0)
    assert out["other/class=single"] == pytest.approx(1.0)
    # Unobserved classes are OMITTED (not emitted as 0.0).
    for name in ("same", "distant"):
        assert f"acc/class={name}" not in out
        assert f"leak/class={name}" not in out
        assert f"other/class={name}" not in out
    # Cross slice x class.
    assert out["acc/slice=8/class=adjacent"] == pytest.approx(1.0)
    assert out["leak/slice=8/class=single"] == pytest.approx(0.0)
    assert out["other/slice=8/class=single"] == pytest.approx(1.0)
    # (slice, class) cells of unobserved classes are absent too.
    assert "acc/slice=8/class=same" not in out
    assert "leak/slice=8/class=distant" not in out


def test_D09_leakage_counts_old_value_predictions():
    batch = _hand_batch()
    logits = torch.zeros(1, 6, 12, dtype=torch.float64)
    # Answer at position 3 (target 11, old 10): predict the OLD value -> leak.
    logits[0, 2, 10] = 5.0
    # Answer at position 5 (target 7, class SINGLE): predict correctly.
    logits[0, 4, 7] = 5.0
    metrics = RecallMetrics()
    metrics.update(logits, batch)
    out = metrics.compute()
    assert out["acc"] == pytest.approx(0.5)
    assert out["leak/class=adjacent"] == pytest.approx(1.0)
    assert out["acc/class=adjacent"] == pytest.approx(0.0)
    assert out["other/class=adjacent"] == pytest.approx(0.0)
    # SINGLE answers never leak (old_labels is IGNORE there).
    assert out["leak/class=single"] == pytest.approx(0.0)
    # Unobserved classes are omitted.
    assert "acc/class=same" not in out
    assert "leak/class=distant" not in out


def test_D09_accumulation_across_updates_and_slices():
    metrics = RecallMetrics()
    # Slice 4: 1 correct answer; slice 8: 2 answers, 1 correct.
    labels = torch.full((2, 4), IGNORE, dtype=torch.int64)
    labels[0, 3] = 5
    labels[1, 3] = 6
    logits = torch.zeros(2, 4, 8, dtype=torch.float64)
    logits[0, 2, 5] = 1.0  # correct
    logits[1, 2, 0] = 1.0  # wrong
    metrics.update(
        logits,
        {
            "labels": labels,
            "slice_id": torch.tensor([4, 8], dtype=torch.int64),
            # Position 3 answers belong to "adjacent" class; old values
            # differ from the targets (as in real examples).
            "old_labels": torch.tensor(
                [[IGNORE, IGNORE, IGNORE, 4], [IGNORE, IGNORE, IGNORE, 7]],
                dtype=torch.int64,
            ),
            "key_class": torch.tensor(
                [[-1, -1, -1, 1], [-1, -1, -1, 1]], dtype=torch.int64
            ),
            "write_distance": torch.tensor(
                [[-1, -1, -1, 2], [-1, -1, -1, 6]], dtype=torch.int64
            ),
        },
    )
    out = metrics.compute()
    # Two answers (one per example): the slice-4 answer is correct, the
    # slice-8 answer is wrong.
    assert out["acc"] == pytest.approx(0.5)
    assert out["acc/slice=4"] == pytest.approx(1.0)
    assert out["acc/slice=8"] == pytest.approx(0.0)
    assert out["acc/slice=4/class=adjacent"] == pytest.approx(1.0)
    assert out["leak/class=adjacent"] == pytest.approx(0.0)
    # One wrong answer in the adjacent class that did not leak.
    assert out["other/class=adjacent"] == pytest.approx(0.5)
    # Only the adjacent class was observed; the others are omitted.
    for name in set(CLASS_NAMES) - {"adjacent"}:
        assert f"acc/class={name}" not in out
        assert f"acc/slice=4/class={name}" not in out
        assert f"acc/slice=8/class={name}" not in out


def test_D09_mqar_mode_without_class_keys():
    config = MQARConfig(vocab_size=128, seq_len=32, num_kv_pairs=(2,))
    batch = generate_mqar(config, num_examples=4, seed=0)
    logits = torch.zeros(4, config.seq_len, config.vocab_size, dtype=torch.float64)
    metrics = RecallMetrics()
    metrics.update(logits, batch)
    out = metrics.compute()
    # All predictions are 0; never equal to a value (>= V/2 = 64) -> acc 0.
    assert out["acc"] == pytest.approx(0.0)
    assert set(out) == {"acc"} | {f"acc/slice={n}" for n in config.slices}
    # Perfect logits give acc 1.
    metrics.reset()
    perfect = torch.full((4, config.seq_len, config.vocab_size), -30.0)
    labels = batch["labels"]
    ans_b, ans_p = (labels != IGNORE).nonzero(as_tuple=True)
    perfect[ans_b, ans_p - 1, labels[ans_b, ans_p]] = 30.0
    metrics.update(perfect, batch)
    assert metrics.compute()["acc"] == pytest.approx(1.0)


def test_D09_perfect_and_random_overwrite_batches():
    config = OverwriteConfig(
        vocab_size=512,
        seq_len=128,
        num_keys=(8,),
        block_size=16,
        distant_min_blocks=4,
    )
    batch = generate_overwrite(config, num_examples=4, seed=0, num_keys=8)
    logits = torch.full((4, config.seq_len, config.vocab_size), -30.0)
    labels = batch["labels"]
    ans_b, ans_p = (labels != IGNORE).nonzero(as_tuple=True)
    logits[ans_b, ans_p - 1, labels[ans_b, ans_p]] = 30.0
    metrics = RecallMetrics()
    metrics.update(logits, batch)
    out = metrics.compute()
    assert out["acc"] == pytest.approx(1.0)
    observed = {
        CLASS_NAMES[int(c)]
        for c in torch.unique(batch["key_class"][batch["labels"] != IGNORE]).tolist()
    }
    assert all(out[f"acc/class={name}"] == pytest.approx(1.0) for name in observed)
    assert all(out[f"leak/class={name}"] == 0.0 for name in observed)
    assert all(out[f"other/class={name}"] == pytest.approx(0.0) for name in observed)
    for name in set(CLASS_NAMES) - observed:
        assert f"acc/class={name}" not in out
    # Old-value leakage: point every double-write answer at its old value.
    old_labels = batch["old_labels"]
    metrics.reset()
    leak_logits = torch.full((4, config.seq_len, config.vocab_size), -30.0)
    double = (old_labels != IGNORE).nonzero(as_tuple=True)
    leak_logits[double[0], double[1] - 1, old_labels[double]] = 30.0
    # Single answers still correct.
    single_answers = (labels != IGNORE).nonzero(as_tuple=True)
    single_mask = old_labels[single_answers] == IGNORE
    sa = (
        single_answers[0][single_mask],
        single_answers[1][single_mask],
    )
    leak_logits[sa[0], sa[1] - 1, labels[sa]] = 30.0
    metrics.update(leak_logits, batch)
    out = metrics.compute()
    # Every double-write answer leaked; every single answer correct.
    num_double = int((old_labels != IGNORE).sum())
    num_single = int(single_mask.sum())
    total = num_double + num_single
    assert out["acc"] == pytest.approx(num_single / total)
    answer_mask = batch["labels"] != IGNORE
    double_mask = answer_mask & (old_labels != IGNORE)
    single_cls = {
        CLASS_NAMES[int(c)]
        for c in torch.unique(
            batch["key_class"][answer_mask & (old_labels == IGNORE)]
        ).tolist()
    }
    double_cls = {
        CLASS_NAMES[int(c)]
        for c in torch.unique(batch["key_class"][double_mask]).tolist()
    }
    assert all(out[f"leak/class={name}"] == pytest.approx(1.0) for name in double_cls)
    assert all(out[f"acc/class={name}"] == pytest.approx(0.0) for name in double_cls)
    assert all(out[f"acc/class={name}"] == pytest.approx(1.0) for name in single_cls)
    assert all(out[f"leak/class={name}"] == pytest.approx(0.0) for name in single_cls)
    # Empty cells stay omitted.
    for name in set(CLASS_NAMES) - double_cls - single_cls:
        assert f"leak/class={name}" not in out


# =============================================================================
# Tests: D10 loss restricted to answers
# =============================================================================


def test_D10_loss_equals_manual_answer_mean():
    import torch.nn.functional as F

    config = OverwriteConfig(
        vocab_size=512,
        seq_len=128,
        num_keys=(8,),
        block_size=16,
        distant_min_blocks=4,
    )
    batch = generate_overwrite(config, num_examples=4, seed=3, num_keys=8)
    torch.manual_seed(0)
    logits = torch.randn(4, config.seq_len, config.vocab_size, dtype=torch.float64)

    # Repo convention (DualOptimizerModule.compute_loss): shift by 1 and
    # ignore -100.
    shift_logits = logits[:, :-1, :]
    shift_labels = batch["labels"][:, 1:]
    repo_loss = F.cross_entropy(
        shift_logits.reshape(-1, config.vocab_size),
        shift_labels.reshape(-1),
        ignore_index=IGNORE,
    )
    # Manual mean over answer positions only.
    labels = batch["labels"]
    ans_b, ans_p = (labels != IGNORE).nonzero(as_tuple=True)
    manual = F.cross_entropy(logits[ans_b, ans_p - 1], labels[ans_b, ans_p])
    assert torch.allclose(repo_loss, manual, atol=1e-10, rtol=1e-10)

    # Perturbing any logit not used by the loss leaves the loss unchanged.
    # The loss uses logits at positions (answer position - 1).
    perturbed = logits.clone()
    used = torch.zeros_like(labels, dtype=torch.bool)
    used[ans_b, ans_p - 1] = True
    for b in range(perturbed.shape[0]):
        for p in range(perturbed.shape[1]):
            if not used[b, p]:
                perturbed[b, p, 0] += 1.0
    perturbed_loss = F.cross_entropy(
        perturbed[:, :-1, :].reshape(-1, config.vocab_size),
        shift_labels.reshape(-1),
        ignore_index=IGNORE,
    )
    assert torch.allclose(
        perturbed_loss,
        repo_loss,
        atol=1e-10,
        rtol=1e-10,
    )


def test_D10_loss_equals_answer_mean_mqar():
    import torch.nn.functional as F

    config = MQARConfig(vocab_size=128, seq_len=32, num_kv_pairs=(2, 4))
    batch = generate_mqar(config, num_examples=8, seed=5)
    torch.manual_seed(1)
    logits = torch.randn(8, config.seq_len, config.vocab_size, dtype=torch.float64)
    shift_labels = batch["labels"][:, 1:]
    repo_loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, config.vocab_size),
        shift_labels.reshape(-1),
        ignore_index=IGNORE,
    )
    labels = batch["labels"]
    ans_b, ans_p = (labels != IGNORE).nonzero(as_tuple=True)
    manual = F.cross_entropy(logits[ans_b, ans_p - 1], labels[ans_b, ans_p])
    assert torch.allclose(repo_loss, manual, atol=1e-10, rtol=1e-10)
