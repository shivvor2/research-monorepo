"""
Test suite for the Zoology-format MQAR generator (spec PR 3).

Covers:
  * MQARConfig construction validation
  * D01: determinism per seed
  * D02: example structure (context layout, key/value ranges, query placement)
  * D03: query-gap power-law distribution
  * D04: loss convention (loss lands exactly on the logits at query positions)

Run with: pytest src/research_lib/data/synthetic/tests/test_mqar.py -v
"""

import numpy as np
import pytest
import torch

from ..mqar import MQARConfig, generate_mqar

IGNORE = -100

# Small but validated config: vocab_size > seq_len and 4 * D <= T for every D.
SMALL = MQARConfig(vocab_size=256, seq_len=64, num_kv_pairs=(2, 4, 8))


def check_mqar_example(config, input_ids, labels, expected_d):
    """Structural checks for one MQAR example (shared by several tests).

    Args:
        config: MQARConfig the example was generated with.
        input_ids: (T,) int64 sequence.
        labels: (T,) int64 unshifted labels.
        expected_d: the number of KV pairs D used for this example.
    """
    seq_len, vocab = config.seq_len, config.vocab_size
    d = expected_d
    x = input_ids.tolist()
    lab = labels.tolist()

    # Context: positions 0..2D-1 hold k_1 v_1 k_2 v_2 ... k_D v_D.
    keys = x[0 : 2 * d : 2]
    values = x[1 : 2 * d : 2]
    assert len(keys) == d and len(values) == d
    assert all(1 <= k < vocab // 2 for k in keys), "keys outside [1, V/2)"
    assert all(vocab // 2 <= v < vocab for v in values), "values outside [V/2, V)"
    assert len(set(keys)) == d, "keys not distinct"
    assert len(set(values)) == d, "values not distinct"

    # Answers: labels[q_i + 1] = value_i at exactly D positions; each query
    # q_i sits at an even offset >= 0 after the context.
    answers = [p for p, t in enumerate(lab) if t != IGNORE]
    assert len(answers) == d, f"expected {d} answer positions"
    assert len(set(answers)) == d, "answer positions not distinct"
    queried = []
    for a in answers:
        q = a - 1
        assert a < seq_len, "answer position must satisfy q + 1 < T"
        assert q >= 2 * d, "query before the context"
        assert (q - 2 * d) % 2 == 0, "query not at an even offset after context"
        key = x[q]
        assert key in keys, f"query token {key} at position {q} is not a key"
        assert (
            lab[a] == values[keys.index(key)]
        ), "labels[q + 1] must be the paired value"
        queried.append(key)
    # Every key is queried exactly once.
    assert sorted(queried) == sorted(keys), "every key must be queried exactly once"


# =============================================================================
# Tests: config validation
# =============================================================================


def test_mqar_config_validation():
    with pytest.raises(ValueError):
        MQARConfig(vocab_size=64, seq_len=31)  # odd seq_len
    with pytest.raises(ValueError):
        MQARConfig(vocab_size=32, seq_len=32)  # vocab_size not > seq_len
    with pytest.raises(ValueError):
        MQARConfig(vocab_size=256, seq_len=64, num_kv_pairs=())  # empty slices
    with pytest.raises(ValueError):
        MQARConfig(vocab_size=512, seq_len=64, num_kv_pairs=(32,))  # 4D > T
    with pytest.raises(ValueError):
        generate_mqar(SMALL, num_examples=2, seed=0, num_kv_pairs=32)


# =============================================================================
# Tests: D01 determinism
# =============================================================================


def test_D01_same_seed_reproduces_batch():
    first = generate_mqar(SMALL, num_examples=16, seed=0)
    second = generate_mqar(SMALL, num_examples=16, seed=0)
    assert set(first) == set(second)
    for key in first:
        assert torch.equal(first[key], second[key]), f"{key} differs"


def test_D01_different_seed_differs():
    first = generate_mqar(SMALL, num_examples=16, seed=0)
    second = generate_mqar(SMALL, num_examples=16, seed=1)
    assert not torch.equal(first["input_ids"], second["input_ids"])
    assert not torch.equal(first["labels"], second["labels"])


def test_D01_fixed_num_kv_pairs_uses_slice():
    batch = generate_mqar(SMALL, num_examples=16, seed=0, num_kv_pairs=8)
    assert batch["slice_id"].tolist() == [8] * 16


# =============================================================================
# Tests: D02 structure
# =============================================================================


@pytest.mark.parametrize("d", [2, 4, 8])
def test_D02_fixed_num_kv_pairs_structure(d):
    batch = generate_mqar(SMALL, num_examples=32, seed=42, num_kv_pairs=d)
    assert batch["slice_id"].tolist() == [d] * 32
    for e in range(32):
        check_mqar_example(SMALL, batch["input_ids"][e], batch["labels"][e], d)


def test_D02_drawn_num_kv_pairs_structure():
    batch = generate_mqar(SMALL, num_examples=64, seed=7)
    for e in range(64):
        d = int(batch["slice_id"][e])
        assert d in SMALL.num_kv_pairs
        check_mqar_example(SMALL, batch["input_ids"][e], batch["labels"][e], d)


def test_D02_dtypes_shapes_and_filler():
    batch = generate_mqar(SMALL, num_examples=4, seed=0)
    assert set(batch) == {"input_ids", "labels", "slice_id"}
    assert all(t.dtype == torch.int64 for t in batch.values())
    assert batch["input_ids"].shape == (4, SMALL.seq_len)
    assert batch["slice_id"].shape == (4,)
    # Filler is uniform over the full vocabulary (random_non_queries=True).
    x = batch["input_ids"]
    assert torch.all(x >= 0) and torch.all(x < SMALL.vocab_size)
    # Labels are -100 everywhere except answers; answer count sums to D.
    answers = int((batch["labels"] != IGNORE).sum())
    assert answers == sum(int(d) for d in batch["slice_id"])


def test_D02_non_random_filler_is_zero():
    config = MQARConfig(
        vocab_size=256, seq_len=64, num_kv_pairs=(4,), random_non_queries=False
    )
    batch = generate_mqar(config, num_examples=16, seed=0, num_kv_pairs=4)
    for e in range(16):
        x, lab = batch["input_ids"][e], batch["labels"][e]
        answers = [p for p in range(config.seq_len) if int(lab[p]) != IGNORE]
        occupied = set(range(8)) | {a - 1 for a in answers}
        for p in range(config.seq_len):
            if p not in occupied:
                assert int(x[p]) == 0, f"filler at position {p} must be 0"


# =============================================================================
# Tests: D03 query-gap distribution
# =============================================================================


def _gap_pooled_tv(config, d, num_examples, seed):
    """Pooled empirical gap distribution vs. the power-law weights, as TV."""
    batch = generate_mqar(config, num_examples=num_examples, seed=seed, num_kv_pairs=d)
    space = (config.seq_len - 2 * d) // 2
    weights = config.power_a * np.arange(1, space + 1, dtype=np.float64) ** (
        config.power_a - 1
    )
    p = weights / weights.sum()
    gaps = []
    for e in range(num_examples):
        lab = batch["labels"][e]
        for a in (lab != IGNORE).nonzero().flatten().tolist():
            gaps.append((a - 1 - 2 * d) // 2)
    assert all(0 <= g < space for g in gaps)
    counts = np.bincount(gaps, minlength=space).astype(np.float64)
    empirical = counts / counts.sum()
    return 0.5 * float(np.abs(empirical - p).sum())


@pytest.mark.slow
def test_D03_gap_distribution_matches_power_law():
    """With D=1 every gap is an unconditioned draw from the power law, so
    the pooled empirical distribution converges to the weights (20k draws,
    TV < 0.02; the noise floor at space=15, 20k draws is ~0.008)."""
    config = MQARConfig(vocab_size=64, seq_len=32, num_kv_pairs=(1,))
    # 20k draws: 20000 examples x 1 gap.
    tv = _gap_pooled_tv(config, d=1, num_examples=20_000, seed=123)
    assert tv < 0.02, f"total-variation distance {tv:.4f} >= 0.02"


@pytest.mark.slow
def test_D03_multi_gap_distribution_close_to_power_law():
    """With D > 1 the gaps are drawn without replacement, so the *pooled*
    marginal is biased away from the raw weights (a heavy slot can be used
    at most once per example): the TV of Zoology's own procedure at D=4,
    T=256 is ~0.05-0.06. Guard against gross deviations only (uniform gaps
    would give TV ~0.85)."""
    config = MQARConfig(vocab_size=512, seq_len=256, num_kv_pairs=(4,))
    # 20k draws: 5000 examples x 4 gaps.
    tv = _gap_pooled_tv(config, d=4, num_examples=5_000, seed=123)
    assert tv < 0.10, f"total-variation distance {tv:.4f} >= 0.10"


# =============================================================================
# Tests: D04 loss convention
# =============================================================================


def _dummy_module():
    from torch import nn

    from ....training.modules.dual_optimizer import DualOptimizerModule
    from ....training.presets import default_adamw_config

    matrix_opt, matrix_sched = default_adamw_config()
    vector_opt, vector_sched = default_adamw_config()
    return DualOptimizerModule(
        model=nn.Identity(),
        matrix_optimizer_config=matrix_opt,
        vector_optimizer_config=vector_opt,
        matrix_schedule_config=matrix_sched,
        vector_schedule_config=vector_sched,
    )


def test_D04_loss_lands_on_query_logits_only():
    """DualOptimizerModule.compute_loss-style shifting puts the loss exactly
    on the logits at the query positions (hand-built example).

    Input (V=12, T=16, D=2; keys {2, 4}, values {8, 7}, filler 0):

        key val key val               query                query
        2   8   4   7   0 0 0 0        2     0 0 0 0 0      4     0 0 0 0
        labels: -100 except labels[5] = 8 and labels[11] = 7
    """
    import torch.nn.functional as F

    seq_len, vocab = 16, 12
    input_ids = torch.tensor(
        [[2, 8, 4, 7, 2, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0]],
        dtype=torch.int64,
    )
    labels = torch.full((1, seq_len), IGNORE, dtype=torch.int64)
    labels[0, 5] = 8  # query k=2 at position 4
    labels[0, 11] = 7  # query k=4 at position 10
    batch = {"input_ids": input_ids, "labels": labels}

    module = _dummy_module()
    torch.manual_seed(0)
    logits = torch.randn(1, seq_len, vocab, dtype=torch.float64)

    loss = module.compute_loss(logits, batch)
    answer_logits = torch.stack([logits[0, 4], logits[0, 10]])
    answer_targets = torch.tensor([8, 7], dtype=torch.int64)
    manual = F.cross_entropy(answer_logits, answer_targets)
    assert torch.allclose(loss, manual, atol=1e-10, rtol=1e-10)

    # Perturbing any non-answer logit leaves the loss unchanged.
    perturbed = logits.clone()
    mask = torch.ones(1, seq_len, dtype=torch.bool)
    mask[0, 4] = False
    mask[0, 10] = False
    perturbed[mask] += 1.0
    assert torch.allclose(
        module.compute_loss(perturbed, batch), loss, atol=1e-10, rtol=1e-10
    )

    # Perturbing an answer logit changes the loss (a single vocab entry;
    # shifting a whole position slice is softmax-invariant).
    answer_perturbed = logits.clone()
    answer_perturbed[0, 4, 0] += 1.0
    assert not torch.allclose(
        module.compute_loss(answer_perturbed, batch), loss, atol=1e-8
    )


def test_D04_generated_batch_loss_equals_answer_mean():
    """The repo shift + ignore_index loss equals a manual mean over the
    answer positions of a generated MQAR batch."""
    import torch.nn.functional as F

    batch = generate_mqar(SMALL, num_examples=8, seed=5)
    logits = torch.randn(8, SMALL.seq_len, SMALL.vocab_size, dtype=torch.float64)
    shift_logits = logits[:, :-1, :]
    shift_labels = batch["labels"][:, 1:]
    repo_loss = F.cross_entropy(
        shift_logits.reshape(-1, SMALL.vocab_size),
        shift_labels.reshape(-1),
        ignore_index=IGNORE,
    )
    labels = batch["labels"]
    ans_b, ans_p = (labels != IGNORE).nonzero(as_tuple=True)
    manual = F.cross_entropy(logits[ans_b, ans_p - 1], labels[ans_b, ans_p])
    assert torch.allclose(repo_loss, manual, atol=1e-10, rtol=1e-10)
