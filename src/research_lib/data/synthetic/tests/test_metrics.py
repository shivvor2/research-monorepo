"""
Test suite for answer-position metrics.

Covers:
  * answer_predictions: shift convention, masking
  * hand-made logits give exact accuracy / per-slice / per-class /
    leakage / other values
  * cross-entropy with the repo convention equals a manual mean over
    answer positions only
  * update_answers: the answer-row path accumulates identically to update

Run with: pytest src/research_lib/data/synthetic/tests/test_metrics.py -v
"""

import pytest
import torch

from research_lib.training.losses import answer_position_logits

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
# Tests: metrics
# =============================================================================


def test_exact_accuracy_slices_classes_leak_other():
    batch = _hand_batch()
    logits = _hand_logits()
    metrics = RecallMetrics(slice_values=[8])
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


def test_leakage_counts_old_value_predictions():
    batch = _hand_batch()
    logits = torch.zeros(1, 6, 12, dtype=torch.float64)
    # Answer at position 3 (target 11, old 10): predict the OLD value -> leak.
    logits[0, 2, 10] = 5.0
    # Answer at position 5 (target 7, class SINGLE): predict correctly.
    logits[0, 4, 7] = 5.0
    metrics = RecallMetrics(slice_values=[8])
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


def test_accumulation_across_updates_and_slices():
    metrics = RecallMetrics(slice_values=[4, 8])
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


def test_mqar_mode_without_class_keys():
    config = MQARConfig(vocab_size=128, seq_len=32, num_kv_pairs=(2,))
    batch = generate_mqar(config, num_examples=4, seed=0)
    logits = torch.zeros(4, config.seq_len, config.vocab_size, dtype=torch.float64)
    metrics = RecallMetrics(slice_values=config.slices)
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


def test_perfect_and_random_overwrite_batches():
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
    metrics = RecallMetrics(slice_values=config.slices)
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


def test_constructor_validation():
    with pytest.raises(ValueError):
        RecallMetrics(slice_values=())
    with pytest.raises(ValueError):
        RecallMetrics(slice_values=(8, -1))
    with pytest.raises(ValueError, match="8"):
        RecallMetrics(slice_values=(4, 8, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_update_is_sync_free():
    """update() performs no host synchronization: under CUDA sync debug mode
    'error' (which raises on any synchronizing call, including internal
    .item()/D2H transfers) the update runs clean. torch.bincount was
    measured to synchronize (it computes input.max() on the host), which is
    why the accumulation uses index_add_."""
    config = OverwriteConfig(
        vocab_size=512,
        seq_len=128,
        num_keys=(8,),
        block_size=16,
        distant_min_blocks=4,
    )
    batch = generate_overwrite(config, num_examples=4, seed=0, num_keys=8)
    batch = {key: value.cuda() for key, value in batch.items()}
    labels = batch["labels"]
    logits = torch.full((4, config.seq_len, config.vocab_size), -30.0, device="cuda")
    ans_b, ans_p = (labels != -100).nonzero(as_tuple=True)
    logits[ans_b, ans_p - 1, labels[ans_b, ans_p]] = 30.0

    metrics = RecallMetrics(slice_values=config.slices)
    metrics.update(logits, batch)  # warmup: allocates the counts on device
    metrics.reset()
    torch.cuda.set_sync_debug_mode("error")
    try:
        metrics.update(logits, batch)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert metrics.compute()["acc"] == pytest.approx(1.0)


def test_two_rank_reduction_matches_combined():
    """Summing two accumulators' count tensors (what all_reduce(SUM) does in
    compute() under DDP) and deriving through the same compute() path
    matches one accumulator fed both halves."""
    config = OverwriteConfig(
        vocab_size=512,
        seq_len=128,
        num_keys=(8,),
        block_size=16,
        distant_min_blocks=4,
    )
    batch = generate_overwrite(config, num_examples=8, seed=3, num_keys=8)
    half1 = {key: value[:4] for key, value in batch.items()}
    half2 = {key: value[4:] for key, value in batch.items()}
    torch.manual_seed(0)
    logits1 = torch.randn(4, config.seq_len, config.vocab_size, dtype=torch.float64)
    torch.manual_seed(1)
    logits2 = torch.randn(4, config.seq_len, config.vocab_size, dtype=torch.float64)

    rank0 = RecallMetrics(slice_values=config.slices)
    rank1 = RecallMetrics(slice_values=config.slices)
    combined = RecallMetrics(slice_values=config.slices)
    rank0.update(logits1, half1)
    rank1.update(logits2, half2)
    combined.update(logits1, half1)
    combined.update(logits2, half2)

    # The all_reduce(SUM) of the per-rank counts, applied to rank 0.
    # (The distance counts are reduced the same way in _global_counts.)
    rank0._counts += rank1._counts
    rank0._unknown += rank1._unknown
    assert rank0._distance_counts is not None
    rank0._distance_counts += rank1._distance_counts
    assert rank0.compute() == combined.compute()


def test_unknown_slice_values_raise_in_compute():
    """Answers whose slice_id is not among slice_values are counted in a
    separate on-device counter and reported (ValueError) by compute()."""
    config = MQARConfig(vocab_size=128, seq_len=32, num_kv_pairs=(2,))
    batch = generate_mqar(config, num_examples=4, seed=0)
    batch["slice_id"] = batch["slice_id"] + 100  # outside slice_values
    logits = torch.zeros(4, config.seq_len, config.vocab_size, dtype=torch.float64)
    metrics = RecallMetrics(slice_values=config.slices)
    metrics.update(logits, batch)
    with pytest.raises(ValueError, match="outside slice_values"):
        metrics.compute()


# =============================================================================
# Tests: loss restricted to answers
# =============================================================================


def test_loss_equals_manual_answer_mean():
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


def test_loss_equals_answer_mean_mqar():
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


class TestWriteDistanceMetrics:
    """Accuracy and old-value leakage binned by the write distance."""

    def _hand_batch(self):
        """Two hand-built examples with known answers, classes, distances.

        Example 0 (slice 8): answers at label positions 3, 6, 9 (predicted
        from 2, 5, 8): correct, leaked, other; classes same/distant/single;
        distances 8, 40, -1 (the single key has no distance).
        Example 1 (slice 16): answers at 3, 6: leaked, correct; classes
        adjacent/same; distances 20, 100.
        """
        T, V = 10, 8
        labels = torch.full((2, T), -100, dtype=torch.long)
        old_labels = torch.full((2, T), -100, dtype=torch.long)
        key_class = torch.full((2, T), -1, dtype=torch.long)
        write_distance = torch.full((2, T), -1, dtype=torch.long)
        logits = torch.zeros(2, T, V, dtype=torch.float64)

        def answer(ex, pos, target, old, cls, dist, pred):
            labels[ex, pos] = target
            old_labels[ex, pos] = old
            key_class[ex, pos] = cls
            write_distance[ex, pos] = dist
            logits[ex, pos - 1, pred] = 10.0  # predicted from pos - 1

        # Example 0: correct (dist 8 -> bin 0-16), leaked (dist 40 -> bin
        # 32-64), other (single class, no distance).
        answer(0, 3, target=4, old=5, cls=0, dist=8, pred=4)
        answer(0, 6, target=4, old=5, cls=2, dist=40, pred=5)
        answer(0, 9, target=4, old=5, cls=3, dist=-1, pred=6)
        # Example 1: leaked (dist 20 -> bin 16-32... no: 20 < 32 -> bin
        # 16-32? no: bins are [0,16), [16,32), ... so 20 -> bin 16-32),
        # correct (dist 100 -> bin 64-128).
        answer(1, 3, target=4, old=6, cls=1, dist=20, pred=6)
        answer(1, 6, target=5, old=4, cls=0, dist=100, pred=5)

        return {
            "input_ids": torch.zeros(2, T, dtype=torch.long),
            "labels": labels,
            "old_labels": old_labels,
            "key_class": key_class,
            "write_distance": write_distance,
            "slice_id": torch.tensor([8, 16], dtype=torch.long),
        }, logits

    def test_exact_binned_values(self):
        batch, logits = self._hand_batch()
        metrics = RecallMetrics(slice_values=(8, 16))
        metrics.update(logits, batch)
        result = metrics.compute()
        # Distances: 8 -> bin 0-16 (example 0, correct), 40 -> bin 32-64
        # (example 0, leaked), 20 -> bin 16-32 (example 1, leaked),
        # 100 -> bin 64-128 (example 1, correct). The single-class answer
        # (distance -1) never enters the distance metrics.
        assert result["acc/dist=0-16"] == pytest.approx(1.0)
        assert result["leak/dist=0-16"] == pytest.approx(0.0)
        assert result["acc/dist=16-32"] == pytest.approx(0.0)
        assert result["leak/dist=16-32"] == pytest.approx(1.0)
        assert result["acc/dist=32-64"] == pytest.approx(0.0)
        assert result["leak/dist=32-64"] == pytest.approx(1.0)
        assert result["acc/dist=64-128"] == pytest.approx(1.0)
        assert result["leak/dist=64-128"] == pytest.approx(0.0)
        # Empty bins are omitted, not emitted as 0.0.
        assert "acc/dist=128-256" not in result
        assert "acc/dist=256-inf" not in result
        # Per (slice, bin): slice 8 owns distances 8 and 40; slice 16 owns
        # 20 and 100.
        assert result["acc/slice=8/dist=0-16"] == pytest.approx(1.0)
        assert result["leak/slice=8/dist=32-64"] == pytest.approx(1.0)
        assert result["leak/slice=16/dist=16-32"] == pytest.approx(1.0)
        assert result["acc/slice=16/dist=64-128"] == pytest.approx(1.0)
        # Cross cells are absent (each slice has one answer per bin here).
        assert "acc/slice=8/dist=16-32" not in result
        assert "acc/slice=16/dist=0-16" not in result

    def test_distances_below_the_first_edge_are_excluded(self):
        """The bins are half-open ``[lo, hi)`` with the last running to
        infinity: a distance below ``bins[0]`` is in no bin (the default
        bins start at 0, so the pilot is unaffected)."""
        T, V = 6, 8
        labels = torch.full((1, T), -100, dtype=torch.long)
        old_labels = torch.full((1, T), -100, dtype=torch.long)
        key_class = torch.full((1, T), -1, dtype=torch.long)
        write_distance = torch.full((1, T), -1, dtype=torch.long)
        logits = torch.zeros(1, T, V, dtype=torch.float64)
        # A leaked answer at distance 3: below the first edge (8).
        labels[0, 3], old_labels[0, 3], key_class[0, 3] = 4, 5, 0
        write_distance[0, 3] = 3
        logits[0, 2, 5] = 10.0  # predicts the old label -> leaked
        # A correct answer at distance 10: inside [8, 16).
        labels[0, 5], old_labels[0, 5], key_class[0, 5] = 6, 7, 1
        write_distance[0, 5] = 10
        logits[0, 4, 6] = 10.0
        batch = {
            "input_ids": torch.zeros(1, T, dtype=torch.long),
            "labels": labels,
            "old_labels": old_labels,
            "key_class": key_class,
            "write_distance": write_distance,
            "slice_id": torch.tensor([8], dtype=torch.long),
        }
        metrics = RecallMetrics(slice_values=(8,), distance_bins=(8, 16))
        metrics.update(logits, batch)
        result = metrics.compute()
        # Only the distance-10 answer is in the bin; the distance-3
        # answer is not counted in bin 0 (it would drag acc to 0.5).
        assert result["acc/dist=8-16"] == pytest.approx(1.0)
        assert result["leak/dist=8-16"] == pytest.approx(0.0)
        assert result["acc/slice=8/dist=8-16"] == pytest.approx(1.0)
        assert result["leak/slice=8/dist=8-16"] == pytest.approx(0.0)

    def test_overall_accuracy_unchanged_by_distance_bins(self):
        batch, logits = self._hand_batch()
        with_distance = RecallMetrics(slice_values=(8, 16))
        without_distance = RecallMetrics(slice_values=(8, 16), distance_bins=None)
        with_distance.update(logits, batch)
        without_distance.update(logits, batch)
        assert with_distance.compute()["acc"] == without_distance.compute()["acc"]
        assert "acc/dist=0-16" not in without_distance.compute()

    def test_mqar_batches_have_no_distance_keys(self):
        config = MQARConfig(vocab_size=128, seq_len=32, num_kv_pairs=(2,))
        batch = generate_mqar(config, num_examples=4, seed=0)
        logits = torch.zeros(4, config.seq_len, config.vocab_size, dtype=torch.float64)
        metrics = RecallMetrics(slice_values=config.slices)
        metrics.update(logits, batch)
        result = metrics.compute()
        assert not any(key.startswith("acc/dist=") for key in result)

    def test_invalid_distance_bins_rejected(self):
        with pytest.raises(ValueError):
            RecallMetrics(slice_values=(8,), distance_bins=(16,))
        with pytest.raises(ValueError):
            RecallMetrics(slice_values=(8,), distance_bins=(32, 16, 0))
        with pytest.raises(ValueError):
            RecallMetrics(slice_values=(8,), distance_bins=(-1, 16))


# ---------------------------------------------------------------------------
# update_answers: the answer-row metrics path
# ---------------------------------------------------------------------------


class TestPerSequenceAccuracy:
    """Per-sequence mean accuracy (not pooled over answers)."""

    def test_mqar_like_with_unequal_answers_per_sequence(self):
        """Equal sequences per slice, but answers per sequence equal the
        slice value. The per-sequence mean equals the unweighted mean of
        the per-slice accuracies and differs from the pooled figure."""
        labels = torch.full((4, 8), -100, dtype=torch.long)
        # Sequence 0 (slice 2): 2 answers, both correct.
        labels[0, 1], labels[0, 2] = 5, 6
        # Sequence 1 (slice 2): 2 answers, both correct.
        labels[1, 1], labels[1, 2] = 7, 8
        # Sequence 2 (slice 4): 4 answers, 2 correct, 2 wrong.
        labels[2, 1], labels[2, 2], labels[2, 3], labels[2, 4] = 5, 6, 7, 8
        # Sequence 3 (slice 4): 4 answers, 2 correct, 2 wrong.
        labels[3, 1], labels[3, 2], labels[3, 3], labels[3, 4] = 5, 6, 7, 8
        logits = torch.zeros(4, 8, 10, dtype=torch.float64)
        # Seq 0, 1: all correct.
        logits[0, 0, 5], logits[0, 1, 6] = 10.0, 10.0
        logits[1, 0, 7], logits[1, 1, 8] = 10.0, 10.0
        # Seq 2, 3: half correct.
        logits[2, 0, 5], logits[2, 1, 6] = 10.0, 10.0
        logits[2, 2, 0], logits[2, 3, 0] = 10.0, 10.0
        logits[3, 0, 5], logits[3, 1, 6] = 10.0, 10.0
        logits[3, 2, 0], logits[3, 3, 0] = 10.0, 10.0
        metrics = RecallMetrics(slice_values=(2, 4))
        metrics.update(
            logits,
            {
                "labels": labels,
                "slice_id": torch.tensor([2, 2, 4, 4], dtype=torch.long),
            },
        )
        out = metrics.compute()
        # Per-slice: slice 2 = 1.0, slice 4 = 0.5.
        assert out["acc/slice=2"] == pytest.approx(1.0)
        assert out["acc/slice=4"] == pytest.approx(0.5)
        # Per-sequence mean: (1.0 + 1.0 + 0.5 + 0.5) / 4 = 0.75.
        assert out["acc"] == pytest.approx(0.75)
        # Pooled would be 8 / 12 = 0.667, which differs.

    def test_unequal_slice_sizes_differs_from_pooled(self):
        """When answers per sequence differ, the per-sequence mean
        differs from the pooled-over-answers figure."""
        labels = torch.full((2, 6), -100, dtype=torch.long)
        labels[0, 2] = 5  # slice 4, 1 answer, correct
        labels[1, 2] = 6  # slice 8
        labels[1, 3] = 7  # slice 8
        labels[1, 4] = 8  # slice 8
        logits = torch.zeros(2, 6, 10, dtype=torch.float64)
        logits[0, 1, 5] = 10.0  # correct
        logits[1, 1, 0] = 10.0  # wrong
        logits[1, 2, 0] = 10.0  # wrong
        logits[1, 3, 0] = 10.0  # wrong
        metrics = RecallMetrics(slice_values=(4, 8))
        metrics.update(
            logits,
            {
                "labels": labels,
                "slice_id": torch.tensor([4, 8], dtype=torch.long),
            },
        )
        out = metrics.compute()
        # Per-sequence mean: (1.0 + 0.0) / 2 = 0.5.
        assert out["acc"] == pytest.approx(0.5)
        # Pooled would be 1 / 4 = 0.25.
        assert out["acc/slice=4"] == pytest.approx(1.0)
        assert out["acc/slice=8"] == pytest.approx(0.0)

    def test_accumulation_over_batches(self):
        """Accumulating over several batches equals one batch with all rows."""
        labels = torch.full((4, 4), -100, dtype=torch.long)
        labels[0, 2] = 5  # correct
        labels[1, 2] = 6  # wrong
        labels[2, 2] = 7  # correct
        labels[3, 2] = 8  # wrong
        logits = torch.zeros(4, 4, 10, dtype=torch.float64)
        logits[0, 1, 5] = 10.0
        logits[1, 1, 0] = 10.0
        logits[2, 1, 7] = 10.0
        logits[3, 1, 0] = 10.0
        batch = {
            "labels": labels,
            "slice_id": torch.tensor([4, 4, 8, 8], dtype=torch.long),
        }
        metrics_single = RecallMetrics(slice_values=(4, 8))
        metrics_single.update(logits, batch)
        out_single = metrics_single.compute()

        metrics_batch = RecallMetrics(slice_values=(4, 8))
        metrics_batch.update(logits[:2], {k: v[:2] for k, v in batch.items()})
        metrics_batch.update(logits[2:], {k: v[2:] for k, v in batch.items()})
        out_batch = metrics_batch.compute()
        assert out_batch["acc"] == out_single["acc"]
        assert out_batch["acc/slice=4"] == out_single["acc/slice=4"]
        assert out_batch["acc/slice=8"] == out_single["acc/slice=8"]

    def test_update_answers_gives_same_per_sequence_accuracy(self):
        """The answer-row path and the full-logits path yield the same
        per-sequence accuracy."""
        config = MQARConfig(vocab_size=128, seq_len=32, num_kv_pairs=(2,))
        batch = generate_mqar(config, num_examples=4, seed=0)
        torch.manual_seed(0)
        embed = torch.nn.Embedding(config.vocab_size, 16)
        head = torch.nn.Linear(16, config.vocab_size)
        hidden = embed(batch["input_ids"])
        full_logits = head(hidden)
        from research_lib.training.losses import answer_position_logits

        answers = answer_position_logits(hidden, head, batch["labels"])

        metrics_full = RecallMetrics(slice_values=config.slices)
        metrics_full.update(full_logits, batch)
        metrics_ans = RecallMetrics(slice_values=config.slices)
        metrics_ans.update_answers(answers.logits, batch)
        assert metrics_full.compute()["acc"] == metrics_ans.compute()["acc"]

    def test_distributed_all_reduce_covers_new_counters(self, monkeypatch):
        """compute() all-reduces the per-sequence counters and leaves the
        accumulator unchanged."""
        labels = torch.full((4, 4), -100, dtype=torch.long)
        labels[0, 2] = 5  # correct
        labels[1, 2] = 6  # wrong
        labels[2, 2] = 7  # correct
        labels[3, 2] = 8  # wrong
        logits = torch.zeros(4, 4, 10, dtype=torch.float64)
        logits[0, 1, 5] = 10.0
        logits[1, 1, 0] = 10.0
        logits[2, 1, 7] = 10.0
        logits[3, 1, 0] = 10.0
        batch = {
            "labels": labels,
            "slice_id": torch.tensor([4, 4, 8, 8], dtype=torch.long),
        }
        rank0 = RecallMetrics(slice_values=(4, 8))
        rank1 = RecallMetrics(slice_values=(4, 8))
        rank0.update(logits[:2], {k: v[:2] for k, v in batch.items()})
        rank1.update(logits[2:], {k: v[2:] for k, v in batch.items()})

        # Save pre-compute state of rank0.
        pre_seq_acc_sum = rank0._seq_acc_sum.clone()
        pre_seq_count = rank0._seq_count.clone()

        # Monkeypatch torch.distributed to simulate SUM all-reduce.
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(
            torch.distributed, "all_reduce", lambda t, op=None: t.mul_(2)
        )

        result = rank0.compute()
        # The result matches the combined accumulator.
        combined = RecallMetrics(slice_values=(4, 8))
        combined.update(logits, batch)
        assert result["acc"] == combined.compute()["acc"]
        # The accumulator itself is unchanged (_global_counts clones).
        assert torch.equal(rank0._seq_acc_sum, pre_seq_acc_sum)
        assert torch.equal(rank0._seq_count, pre_seq_count)

    def test_zero_answer_sequences_raise(self):
        """Sequences with no answer positions are counted and raise."""
        labels = torch.full((2, 4), -100, dtype=torch.long)
        labels[0, 2] = 5  # one answer in seq 0
        # seq 1 has no answers at all
        logits = torch.zeros(2, 4, 10, dtype=torch.float64)
        logits[0, 1, 5] = 10.0
        metrics = RecallMetrics(slice_values=(8,))
        metrics.update(
            logits,
            {"labels": labels, "slice_id": torch.tensor([8, 8], dtype=torch.long)},
        )
        with pytest.raises(ValueError, match="zero answer positions"):
            metrics.compute()


class TestUpdateAnswers:
    """update_answers accumulates the same counters as update.

    The answer-head path evaluates the LM head only at the answer
    positions and hands the metric its ``(N, V)`` answer rows; given
    those, every accumulated metric is unchanged, on real batches from
    both synthetic tasks.
    """

    @pytest.mark.parametrize("task", ["mqar", "overwrite"])
    def test_update_answers_equals_update(self, task):
        if task == "mqar":
            config = MQARConfig(vocab_size=128, seq_len=32, num_kv_pairs=(2,))
            batch = generate_mqar(config, num_examples=4, seed=0)
        else:
            config = OverwriteConfig(
                vocab_size=512,
                seq_len=128,
                num_keys=(8,),
                block_size=16,
                distant_min_blocks=4,
            )
            batch = generate_overwrite(config, num_examples=4, seed=3, num_keys=8)

        # A tiny untied LM: hidden states from an embedding, a linear head.
        torch.manual_seed(0)
        embed = torch.nn.Embedding(config.vocab_size, 16)
        head = torch.nn.Linear(16, config.vocab_size)
        hidden = embed(batch["input_ids"])
        full_logits = head(hidden)
        answers = answer_position_logits(hidden, head, batch["labels"])

        metrics_full = RecallMetrics(slice_values=config.slices)
        metrics_full.update(full_logits, batch)
        metrics_answers = RecallMetrics(slice_values=config.slices)
        metrics_answers.update_answers(answers.logits, batch)

        result_full = metrics_full.compute()
        result_answers = metrics_answers.compute()
        assert "acc" in result_full
        assert result_answers == result_full
        # The class and write-distance fields are carried by the overwrite
        # batch only, so those metric keys appear for overwrite only.
        assert any("/class=" in key for key in result_full) == (task == "overwrite")
        assert any(key.startswith("acc/dist=") for key in result_full) == (
            task == "overwrite"
        )
