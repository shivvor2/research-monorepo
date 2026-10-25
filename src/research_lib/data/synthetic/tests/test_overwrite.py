"""
Test suite for the repeated-key overwrite generator.

Covers:
  * OverwriteConfig construction / feasibility validation
  * Determinism per seed
  * example structure (placement classes, latest-value labels, filler)
  * batch-vectorized implementation vs the literal sequential
    reference (_reference.py), compared distributionally
  * class proportions match class_probs
  * infeasible configs raise ValueError

Run with: pytest src/research_lib/data/synthetic/tests/test_overwrite.py -v
"""

import hashlib

import numpy as np
import pytest
import torch

from ..overwrite import OverwriteClass, OverwriteConfig, generate_overwrite
from ._reference import generate_overwrite_reference

IGNORE = -100
NO_CLASS = -1
NO_DISTANCE = -1

# Small but feasible config: 6 * K <= T for the largest K and the write
# region spans >= distant_min_blocks + 1 blocks.
SMALL = OverwriteConfig(
    vocab_size=512,
    seq_len=128,
    num_keys=(4, 8, 16),
    class_probs=(0.25, 0.25, 0.25, 0.25),
    block_size=16,
    distant_min_blocks=4,
)


def check_overwrite_example(
    config, input_ids, labels, old_labels, key_class, write_distance, expected_k
):
    """Structural checks for one overwrite example (shared by several tests).

    Args:
        config: OverwriteConfig the example was generated with.
        input_ids: (T,) int64 sequence.
        labels: (T,) int64 unshifted labels (-100 except answers).
        old_labels: (T,) int64 old values at double-write answers, else -100.
        key_class: (T,) int64 class codes at answers, else -1.
        write_distance: (T,) int64 token distance at double-write answers,
            else -1.
        expected_k: the number of keys K used for this example.
    """
    seq_len, vocab = config.seq_len, config.vocab_size
    block, distant_min = config.block_size, config.distant_min_blocks
    k = expected_k
    x = input_ids.tolist()
    lab = labels.tolist()
    old = old_labels.tolist()
    cls = key_class.tolist()
    dist = write_distance.tolist()
    write_region = seq_len - 2 * k

    # Answer slots: the odd offsets of the last 2K tokens.
    answers = [p for p, t in enumerate(lab) if t != IGNORE]
    assert answers == list(
        range(write_region + 1, seq_len, 2)
    ), "answers must be the answer slots of the query block"
    assert [p for p, c in enumerate(cls) if c != NO_CLASS] == answers
    assert [p for p, d in enumerate(dist) if d != NO_DISTANCE] == [
        a for a in answers if cls[a] != int(OverwriteClass.SINGLE)
    ]

    # Query keys: every key exactly once, in [1, V/2).
    query_keys = [x[a - 1] for a in answers]
    assert len(set(query_keys)) == k, "each key must be queried exactly once"
    assert all(1 <= key < vocab // 2 for key in query_keys), "keys outside [1, V/2)"

    written_values = []
    occupied = set()
    for a, key in zip(answers, query_keys, strict=True):
        cls_code = cls[a]
        assert cls_code in {int(c) for c in OverwriteClass}
        occupied.add(a - 1)
        # Every occurrence of the key token in the write region is one of its
        # writes (filler excludes this example's keys).
        writes = [p for p in range(write_region) if x[p] == key]
        if cls_code == int(OverwriteClass.SINGLE):
            assert len(writes) == 1, "a single key must be written exactly once"
            (s,) = writes
            assert s % 2 == 0, "writes land on pair-slot key positions"
            value = x[s + 1]
            assert vocab // 2 <= value < vocab, "value outside [V/2, V)"
            assert lab[a] == value, "labels[a] must be the (only) value"
            assert old[a] == IGNORE, "single keys have no old value"
            assert dist[a] == NO_DISTANCE
            written_values.append(value)
            occupied.update({s, s + 1})
        else:
            assert len(writes) == 2, "a double-write key must be written twice"
            s1, s2 = writes
            assert s1 % 2 == 0 and s2 % 2 == 0
            assert s1 < s2, "writes must be ordered s1 < s2"
            b1, b2 = s1 // block, s2 // block
            if cls_code == int(OverwriteClass.SAME):
                assert b1 == b2, f"same-class writes in blocks {b1} != {b2}"
            elif cls_code == int(OverwriteClass.ADJACENT):
                assert b2 == b1 + 1, f"adjacent-class blocks {b1} -> {b2}"
            else:  # DISTANT
                assert b2 - b1 >= distant_min, f"distant-class blocks {b1} -> {b2}"
            v_old, v_new = x[s1 + 1], x[s2 + 1]
            assert v_old != v_new, "v_old must differ from v_new"
            assert all(
                vocab // 2 <= v < vocab for v in (v_old, v_new)
            ), "values outside [V/2, V)"
            assert lab[a] == v_new, "labels[a] must be the latest value"
            assert old[a] == v_old, "old_labels[a] must be v_old"
            # Token distance between the two writes: positions are tokens, so
            # dist = s2 - s1 (== 2 * (slot_2 - slot_1)).
            assert dist[a] == s2 - s1, "token distance between writes"
            written_values.extend([v_old, v_new])
            occupied.update({s1, s1 + 1, s2, s2 + 1})

    # Old labels only at double-write answers.
    assert all(
        old[p] == IGNORE
        for p in range(seq_len)
        if p not in answers or cls[p] == int(OverwriteClass.SINGLE)
    )

    # Filler avoids this example's keys and (written) values.
    bad_tokens = set(query_keys) | set(written_values)
    for p in range(seq_len):
        if p not in occupied:
            assert x[p] not in bad_tokens, f"filler at position {p} collides"


# =============================================================================
# Tests: config validation
# =============================================================================


def test_overwrite_config_validation():
    with pytest.raises(ValueError):
        OverwriteConfig(vocab_size=512, seq_len=127)  # odd seq_len
    with pytest.raises(ValueError):
        OverwriteConfig(vocab_size=128, seq_len=128)  # vocab not > seq_len
    with pytest.raises(ValueError):
        OverwriteConfig(num_keys=())  # empty slices
    # Every num_keys entry is validated, not only the maximum.
    with pytest.raises(ValueError, match=">= 1"):
        OverwriteConfig(vocab_size=1024, seq_len=256, num_keys=(0, 8))
    with pytest.raises(ValueError, match=">= 1"):
        OverwriteConfig(vocab_size=1024, seq_len=256, num_keys=(-1, 8))
    with pytest.raises(ValueError, match="8"):
        # Duplicate slice values are rejected, naming the duplicates.
        OverwriteConfig(vocab_size=1024, seq_len=256, num_keys=(8, 16, 8))
    with pytest.raises(ValueError):
        OverwriteConfig(class_probs=(0.25, 0.25, 0.25, 0.3))  # sums > 1
    with pytest.raises(ValueError):
        # Intentionally invalid length (validation test).
        OverwriteConfig(class_probs=(0.5, 0.5))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OverwriteConfig(block_size=1)  # < 2
    with pytest.raises(ValueError):
        OverwriteConfig(distant_min_blocks=0)  # < 1
    with pytest.raises(ValueError):
        OverwriteConfig(max_placement_attempts=0)  # < 1


# =============================================================================
# Tests: structure
# =============================================================================


@pytest.mark.parametrize("k", [4, 8, 16])
def test_fixed_num_keys_structure(k):
    batch = generate_overwrite(SMALL, num_examples=24, seed=3, num_keys=k)
    assert batch["slice_id"].tolist() == [k] * 24
    for e in range(24):
        check_overwrite_example(
            SMALL,
            batch["input_ids"][e],
            batch["labels"][e],
            batch["old_labels"][e],
            batch["key_class"][e],
            batch["write_distance"][e],
            k,
        )


def test_drawn_num_keys_structure():
    batch = generate_overwrite(SMALL, num_examples=48, seed=17)
    for e in range(48):
        k = int(batch["slice_id"][e])
        assert k in SMALL.num_keys
        check_overwrite_example(
            SMALL,
            batch["input_ids"][e],
            batch["labels"][e],
            batch["old_labels"][e],
            batch["key_class"][e],
            batch["write_distance"][e],
            k,
        )


def test_dtypes_shapes_and_distant_extreme():
    batch = generate_overwrite(SMALL, num_examples=4, seed=0, num_keys=8)
    assert set(batch) == {
        "input_ids",
        "labels",
        "slice_id",
        "old_labels",
        "key_class",
        "write_distance",
    }
    assert all(t.dtype == torch.int64 for t in batch.values())
    assert batch["input_ids"].shape == (4, SMALL.seq_len)
    assert batch["slice_id"].shape == (4,)
    # Non-answer positions carry -100 labels and -1 class / distance.
    mask = batch["labels"] != IGNORE
    assert torch.all(batch["key_class"][~mask] == NO_CLASS)
    assert torch.all(batch["write_distance"][~mask] == NO_DISTANCE)


def test_all_distant_class_extreme():
    distant_only = OverwriteConfig(
        vocab_size=512,
        seq_len=128,
        num_keys=(8,),
        class_probs=(0.0, 0.0, 1.0, 0.0),
        block_size=16,
        distant_min_blocks=4,
    )
    batch = generate_overwrite(distant_only, num_examples=16, seed=1, num_keys=8)
    mask = batch["labels"] != IGNORE
    assert torch.all(batch["key_class"][mask] == int(OverwriteClass.DISTANT))
    assert torch.all(batch["old_labels"][mask] != IGNORE)
    assert torch.all(batch["write_distance"][mask] != NO_DISTANCE)
    for e in range(16):
        check_overwrite_example(
            distant_only,
            batch["input_ids"][e],
            batch["labels"][e],
            batch["old_labels"][e],
            batch["key_class"][e],
            batch["write_distance"][e],
            8,
        )


def test_all_single_class_extreme():
    single_only = OverwriteConfig(
        vocab_size=512,
        seq_len=128,
        num_keys=(8,),
        class_probs=(0.0, 0.0, 0.0, 1.0),
        block_size=16,
        distant_min_blocks=4,
    )
    batch = generate_overwrite(single_only, num_examples=16, seed=2, num_keys=8)
    mask = batch["labels"] != IGNORE
    assert torch.all(batch["key_class"][mask] == int(OverwriteClass.SINGLE))
    assert torch.all(batch["old_labels"] == IGNORE)
    assert torch.all(batch["write_distance"] == NO_DISTANCE)


# =============================================================================
# Tests: determinism
# =============================================================================


def test_overwrite_determinism():
    """Same seed reproduces the batch bit-exactly; a different seed differs."""
    batch = generate_overwrite(SMALL, num_examples=16, seed=42)
    again = generate_overwrite(SMALL, num_examples=16, seed=42)
    for key in batch:
        assert torch.equal(batch[key], again[key]), f"{key} differs"
    other = generate_overwrite(SMALL, num_examples=16, seed=43)
    assert not torch.equal(batch["input_ids"], other["input_ids"])


# =============================================================================
# Tests: reserved filler (reserved_filler option)
# =============================================================================

# Digest of the fixed-seed batch ``(config=SMALL, num_examples=64, seed=3)``
# from the generator before ``reserved_filler`` existed.
_SNAPSHOT_DEFAULT_MODE = (
    "399cf21407ab24c6d244ec798946509720a5ce464b285c89e75e87e2877d6696"
)


def test_default_filler_mode_matches_snapshot():
    """The default mode (``reserved_filler=0``) reproduces the fixed-seed
    batch byte for byte, untouched by the reserved-filler option.

    The digest pins the generator's default-mode output at the state
    before the option existed; it guards the promise that ``F = 0``
    changes nothing (existing seeds reproduce existing data).
    ``numpy.random.default_rng`` streams are stable across numpy
    releases, which keeps the fixed seed well-defined.
    """
    config = OverwriteConfig(vocab_size=512, seq_len=128, num_keys=(4, 8, 16))
    batch = generate_overwrite(config, num_examples=64, seed=3)
    digest = hashlib.sha256()
    for name in sorted(batch):
        digest.update(name.encode())
        digest.update(batch[name].numpy().tobytes())
    assert digest.hexdigest() == _SNAPSHOT_DEFAULT_MODE


def test_explicit_zero_reserved_filler_matches_default():
    """``reserved_filler=0`` is the default: the explicit config is equal
    to the bare one and generates an identical batch from the same seed.
    """
    config = OverwriteConfig(vocab_size=512, seq_len=128, num_keys=(8,))
    explicit = OverwriteConfig(
        vocab_size=512, seq_len=128, num_keys=(8,), reserved_filler=0
    )
    assert explicit == config
    batch = generate_overwrite(config, num_examples=16, seed=3)
    batch_explicit = generate_overwrite(explicit, num_examples=16, seed=3)
    for name in batch:
        assert torch.equal(batch[name], batch_explicit[name]), name


def _check_reserved_filler_ranges(config, input_ids, labels, expected_k):
    """Range guarantees of the reserved mode for one example.

    Keys come from ``[1, V/2 - F)``, values from ``[V/2, V)``, and every
    filler token — including the tokens at the label answer slots — lies
    in the reserved range ``[V/2 - F, V/2)``, which is disjoint from both
    key and value ranges.
    """
    seq_len, vocab = config.seq_len, config.vocab_size
    f_lo = vocab // 2 - config.reserved_filler
    x = input_ids.tolist()
    lab = labels.tolist()
    write_region = seq_len - 2 * expected_k
    answers = [p for p, t in enumerate(lab) if t != IGNORE]
    keys = {x[a - 1] for a in answers}
    assert all(1 <= key < f_lo for key in keys), "keys must come from [1, V/2-F)"
    # Key occurrences in the write region are exactly the writes (filler
    # can never equal a key in this mode).
    write_key_positions = [p for p in range(write_region) if x[p] in keys]
    occupied = (
        set(write_key_positions)
        | {p + 1 for p in write_key_positions}
        | {a - 1 for a in answers}
    )
    values = [x[p + 1] for p in write_key_positions]
    assert all(vocab // 2 <= v < vocab for v in values), "values outside [V/2, V)"
    filler_tokens = [x[p] for p in range(seq_len) if p not in occupied]
    assert all(
        f_lo <= t < vocab // 2 for t in filler_tokens
    ), f"filler outside the reserved range [{f_lo}, {vocab // 2})"


def test_reserved_filler_structure():
    """With ``reserved_filler=F`` examples keep the task's geometry
    (placement classes, latest-value labels, query block) and restrict
    keys, values and filler to their three disjoint ranges.
    """
    config = OverwriteConfig(
        vocab_size=1024, seq_len=256, num_keys=(16, 32), reserved_filler=256
    )
    batch = generate_overwrite(config, num_examples=12, seed=5)
    for e in range(12):
        k = int(batch["slice_id"][e])
        check_overwrite_example(
            config,
            batch["input_ids"][e],
            batch["labels"][e],
            batch["old_labels"][e],
            batch["key_class"][e],
            batch["write_distance"][e],
            k,
        )
        _check_reserved_filler_ranges(
            config, batch["input_ids"][e], batch["labels"][e], k
        )


def test_reserved_filler_validation():
    """Construction-time validation: F must be >= 0, must leave at least
    one key id, and must leave enough key ids for the largest K.
    """
    with pytest.raises(ValueError, match="reserved_filler must be >= 0"):
        OverwriteConfig(vocab_size=512, seq_len=128, num_keys=(8,), reserved_filler=-4)
    with pytest.raises(ValueError, match="at least one key id"):
        OverwriteConfig(vocab_size=512, seq_len=128, num_keys=(8,), reserved_filler=255)
    with pytest.raises(ValueError, match="key ids"):
        OverwriteConfig(
            vocab_size=512, seq_len=128, num_keys=(8, 16), reserved_filler=240
        )
    # Exact-boundary config: V/2 - 1 - F == max(num_keys) is accepted.
    config = OverwriteConfig(
        vocab_size=512, seq_len=128, num_keys=(16,), reserved_filler=239
    )
    batch = generate_overwrite(config, num_examples=4, seed=3)
    assert batch["input_ids"].shape == (4, 128)


# =============================================================================
# Tests: vectorized vs literal reference (distributional)
# =============================================================================


def _first_write_slots(config, batch):
    """Per-answer (class, first-write slot) and per-class write distances.

    Returns ``{class_code: {"s1": list, "wd": list}}`` extracted from the
    generated arrays (the write positions of each queried key are exactly
    the occurrences of its token in the write region, since filler
    excludes keys).
    """
    seq_len = config.seq_len
    x = batch["input_ids"]
    cls_arr = batch["key_class"]
    wd_arr = batch["write_distance"]
    stats = {c: {"s1": [], "wd": []} for c in range(4)}
    for e in range(int(batch["slice_id"].shape[0])):
        k = int(batch["slice_id"][e])
        write_region = seq_len - 2 * k
        slot_keys = x[e, 0:write_region:2]  # key token of each pair slot
        for a in range(write_region + 1, seq_len, 2):
            cls_code = int(cls_arr[e, a])
            key = int(x[e, a - 1])
            s1 = int(np.flatnonzero(slot_keys == key)[0])
            stats[cls_code]["s1"].append(s1)
            if cls_code != int(OverwriteClass.SINGLE):
                stats[cls_code]["wd"].append(int(wd_arr[e, a]))
    return stats


@pytest.mark.slow
def test_vectorized_matches_reference_distribution():
    """The batch-vectorized generator reproduces the literal sequential
    reference (kept verbatim in ``_reference.py``) distributionally.

    Two-sample Kolmogorov-Smirnov tests (alpha = 1e-3) on the per-class
    write-distance and first-write-slot distributions and on the class
    frequencies: any placement-law difference (wrong slice bounds, biased
    s_1 / s_2 / retry handling, query-order bias) shows up here, while the
    structural tests only check per-example invariants.
    """
    from scipy.stats import ks_2samp

    config = OverwriteConfig(
        vocab_size=1024,
        seq_len=256,
        num_keys=(16,),
        block_size=16,
        distant_min_blocks=4,
    )
    n = 6000
    vec = generate_overwrite(config, num_examples=n, seed=101, num_keys=16)
    ref = generate_overwrite_reference(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        num_keys=config.num_keys,
        class_probs=config.class_probs,
        block_size=config.block_size,
        distant_min_blocks=config.distant_min_blocks,
        max_placement_attempts=config.max_placement_attempts,
        num_examples=n,
        seed=202,
        fixed_num_keys=16,
    )
    alpha = 1e-3
    names = {0: "same", 1: "adjacent", 2: "distant", 3: "single"}

    vec_stats = _first_write_slots(config, vec)
    ref_stats = _first_write_slots(config, ref)
    for cls_code in (0, 1, 2):  # write distance per double-write class
        result = ks_2samp(vec_stats[cls_code]["wd"], ref_stats[cls_code]["wd"])
        assert result.pvalue > alpha, (names[cls_code], "wd", result)
    for cls_code in range(4):  # first-write slot per class
        result = ks_2samp(vec_stats[cls_code]["s1"], ref_stats[cls_code]["s1"])
        assert result.pvalue > alpha, (names[cls_code], "s1", result)

    # Class frequencies at answer positions.
    for batch in (vec, ref):
        assert int((batch["labels"] != IGNORE).sum()) == n * 16
    mask_vec = (vec["labels"] != IGNORE).flatten()
    mask_ref = (ref["labels"] != IGNORE).flatten()
    result = ks_2samp(
        vec["key_class"].flatten()[mask_vec].tolist(),
        ref["key_class"].flatten()[mask_ref].tolist(),
    )
    assert result.pvalue > alpha, ("class", result)


@pytest.mark.slow
def test_restarts_keep_k_per_example():
    """K is drawn once per example and kept across restarts: with a tight
    config whose restart rate depends on K (all-distant K=42 at T=256
    restarts ~79% of examples), the slice mix stays at the configured 50/50
    instead of collapsing toward the easy K. (When restarts redrew K for the
    pending rows, the K=42 share collapsed to ~0.19.)"""
    config = OverwriteConfig(num_keys=(8, 42), class_probs=(0.0, 0.0, 1.0, 0.0))
    num_examples = 2000
    batch = generate_overwrite(config, num_examples=num_examples, seed=7)
    share_42 = float((batch["slice_id"] == 42).sum()) / num_examples
    assert 0.45 < share_42 < 0.55, f"K=42 share {share_42:.3f}"
    # Restart-heavy generation still produces valid structure, for both K.
    for e in range(0, num_examples, 200):
        k = int(batch["slice_id"][e])
        check_overwrite_example(
            config,
            batch["input_ids"][e],
            batch["labels"][e],
            batch["old_labels"][e],
            batch["key_class"][e],
            batch["write_distance"][e],
            k,
        )


# =============================================================================
# Tests: class proportions
# =============================================================================


@pytest.mark.slow
def test_class_frequencies_match_probs():
    probs = (0.2, 0.3, 0.4, 0.1)
    config = OverwriteConfig(
        vocab_size=512, seq_len=128, num_keys=(8,), class_probs=probs
    )
    num_examples = 2000
    batch = generate_overwrite(config, num_examples=num_examples, seed=11, num_keys=8)
    total = num_examples * 8
    for code, prob in enumerate(probs):
        freq = float((batch["key_class"] == code).sum()) / total
        assert abs(freq - prob) < 0.01, f"class {code}: freq {freq} vs {prob}"


# =============================================================================
# Tests: feasibility
# =============================================================================


def test_too_many_keys_raises():
    # All-distant worst case: 4K write tokens + 2K query tokens = 6K > T.
    with pytest.raises(ValueError):
        OverwriteConfig(vocab_size=1024, seq_len=128, num_keys=(32,))


def test_write_region_too_short_raises():
    # 6K <= T but the write region spans < distant_min_blocks + 1 blocks.
    with pytest.raises(ValueError):
        OverwriteConfig(
            vocab_size=2048,
            seq_len=128,
            num_keys=(16,),
            block_size=32,
            distant_min_blocks=4,
        )


def test_generate_override_validates():
    with pytest.raises(ValueError):
        generate_overwrite(SMALL, num_examples=2, seed=0, num_keys=32)
