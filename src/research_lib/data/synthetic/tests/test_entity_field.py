"""
Test suite for the entity-field generator, its datamodule registration and
the class-conditioned metrics.

Covers:
  * the overwrite generator staying byte-for-byte unchanged (a frozen
    fixture is regenerated and compared)
  * EntityFieldConfig construction / feasibility validation
  * Determinism per seed
  * Token ranges (entities, fields, filler, values disjoint)
  * Example structure (write slots, placement classes, latest-value labels)
  * Update count and per-entity update choice
  * sibling / sibling_late / untouched class assignment
  * updated-pair class proportions vs placement_probs
  * infeasible configs raise ValueError
  * the datamodule (slice counts, fixed val/test, worker independence)
  * RecallMetrics on entity-field batches (exact per-class acc / leak / other)
  * the answer-position loss on an entity-field batch

Run with: pytest src/research_lib/data/synthetic/tests/test_entity_field.py -v
"""

import itertools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datamodule import SyntheticRecallDataModule, _SyntheticTrainDataset
from ..entity_field import (
    ENTITY_FIELD_CLASSES,
    EntityFieldClass,
    EntityFieldConfig,
    generate_entity_field,
)
from ..metrics import RecallMetrics
from ..overwrite import OverwriteConfig, generate_overwrite

IGNORE = -100
NO_CLASS = -1
NO_DISTANCE = -1

SAME = int(EntityFieldClass.SAME)
ADJACENT = int(EntityFieldClass.ADJACENT)
DISTANT = int(EntityFieldClass.DISTANT)
SIBLING = int(EntityFieldClass.SIBLING)
SIBLING_LATE = int(EntityFieldClass.SIBLING_LATE)
UNTOUCHED = int(EntityFieldClass.UNTOUCHED)
UPDATED_CLASSES = {SAME, ADJACENT, DISTANT}

# Feasible grid close to the pilot: N in {8, 16, 32}, reserved filler 512.
# For E = 8: N = 32, U = 6, R = 256 - 96 = 160 -> 53 slots, N + U = 38 fits;
# the region spans 10 blocks (>= distant_min_blocks + 1 = 5); entity ids
# [1, 508) hold 8; values N + U = 38 fit in [1024, 2048).
SMALL = EntityFieldConfig(
    vocab_size=2048,
    seq_len=256,
    num_fields=4,
    num_entities=(2, 4, 8),
    block_size=16,
    distant_min_blocks=4,
    reserved_filler=512,
)

# Tiny config used for the metrics and loss tests (fast; N in {4, 8}).
TINY = EntityFieldConfig(
    vocab_size=256,
    seq_len=96,
    num_fields=2,
    num_entities=(2, 4),
    block_size=16,
    distant_min_blocks=4,
    reserved_filler=2,
)

SMALL_N = tuple(sorted(e * SMALL.num_fields for e in SMALL.num_entities))
TINY_N = tuple(sorted(e * TINY.num_fields for e in TINY.num_entities))


def _check_example(
    config,
    input_ids,
    labels,
    old_labels,
    key_class,
    write_distance,
    n_entities,
):
    """Structural checks for one entity-field example (shared by several tests).

    Args:
        config: EntityFieldConfig the example was generated with.
        input_ids: (T,) int64 sequence.
        labels: (T,) int64 unshifted labels (-100 except answers).
        old_labels: (T,) int64 old values at updated answers, else -100.
        key_class: (T,) int64 class codes at answers, else -1.
        write_distance: (T,) int64 token distance at updated answers, else -1.
        n_entities: the number of entities E used for this example.
    """
    V = config.vocab_size
    M = config.num_fields
    F = config.reserved_filler
    B = config.block_size
    dmin = config.distant_min_blocks
    T = config.seq_len
    half = V // 2
    field_offset = half - F - M
    N = n_entities * M
    R = T - 3 * N
    num_slots = R // 3

    x = input_ids.tolist()
    lab = labels.tolist()
    old = old_labels.tolist()
    cls = key_class.tolist()
    dist = write_distance.tolist()

    # Answers sit at the +2 of each query triplet, one per pair in order.
    answers = [p for p in range(T) if lab[p] != IGNORE]
    assert answers == list(range(R + 2, T, 3)), "answers at the query +2 slots"

    queries = [(x[R + 3 * q], x[R + 3 * q + 1], R + 3 * q + 2) for q in range(N)]
    # Every pair queried exactly once -> N distinct (e, f) pairs.
    assert len({(e, f) for e, f, _ in queries}) == N, "each pair queried once"

    used_slots = set()
    n_updated = 0
    for e_tok, f_tok, a in queries:
        assert 1 <= e_tok < field_offset, "entity token outside its range"
        assert field_offset <= f_tok < half - F, "field token outside its range"
        assert 0 <= f_tok - field_offset < M, "field id out of range"
        c = int(cls[a])
        write_slots = [
            i for i in range(num_slots) if x[3 * i] == e_tok and x[3 * i + 1] == f_tok
        ]
        if c in UPDATED_CLASSES:
            assert len(write_slots) == 2, "an updated pair is written twice"
            s1, s2 = write_slots
            assert s1 < s2, "writes ordered s1 < s2"
            v_old = x[3 * s1 + 2]
            v_new = x[3 * s2 + 2]
            assert half <= v_old < V and half <= v_new < V, "values in [V/2, V)"
            assert v_old != v_new, "v_old must differ from v_new"
            b1, b2 = (3 * s1 + 2) // B, (3 * s2 + 2) // B
            if c == SAME:
                assert b1 == b2, "same-class writes in the same block"
            elif c == ADJACENT:
                assert b2 == b1 + 1, "adjacent-class blocks differ by 1"
            else:  # DISTANT
                assert b2 - b1 >= dmin, "distant-class block separation"
            assert lab[a] == v_new, "labels[a] the latest value"
            assert old[a] == v_old, "old_labels[a] is v_old"
            assert dist[a] == 3 * (s2 - s1), "token distance between writes"
            n_updated += 1
            used_slots.update((s1, s2))
        else:
            assert c in (SIBLING, SIBLING_LATE, UNTOUCHED)
            assert len(write_slots) == 1, "a non-updated pair is written once"
            (s,) = write_slots
            v = x[3 * s + 2]
            assert half <= v < V, "value in [V/2, V)"
            assert lab[a] == v, "labels[a] the (single) value"
            assert old[a] == IGNORE, "non-updated pair has no old value"
            assert dist[a] == NO_DISTANCE
            used_slots.add(s)
    # Every write slot is claimed by exactly one pair: the set of slots the
    # pairs use has exactly the N + U entries counted above (a slot shared by
    # two pairs would shrink the set below that count).
    assert len(used_slots) == n_updated * 2 + (N - n_updated)

    # Filler: unused region positions and every answer placeholder are in the
    # reserved range.
    used_positions = set()
    for s in used_slots:
        used_positions.update((3 * s, 3 * s + 1, 3 * s + 2))
    for p in range(R):
        if p not in used_positions:
            assert half - F <= x[p] < half, "unused region position is filler"
    for _, _, a in queries:
        assert half - F <= x[a] < half, "answer placeholder is filler"


def _entity_idx(e_tok, batch, ex):
    """Index of the entity that owns ``e_tok`` in example ``ex``.

    Entity tokens are the distinct tokens below the field range; sorting
    them gives a stable token -> index map for the test.
    """
    V = SMALL.vocab_size
    half = V // 2
    F = SMALL.reserved_filler
    M = SMALL.num_fields
    entity_range_end = half - F - M
    toks = sorted(
        {int(t) for t in batch["input_ids"][ex] if 1 <= int(t) < entity_range_end}
    )
    return toks.index(e_tok)


# =============================================================================
# Overwrite unchanged (a frozen fixture)
# =============================================================================


def test_overwrite_generator_still_reproduces_frozen_batches():
    """The overwrite generator is untouched: batches regenerated here match
    the frozen fixture byte for byte (both filler modes, two seeds, all
    slices)."""
    here = Path(__file__).resolve().parent
    fixture_path = here / "overwrite_unchanged_fixture.npz"
    with np.load(fixture_path) as data:
        slices = (8, 16, 32)
        for F in (0, 512):
            cfg = OverwriteConfig(
                vocab_size=8192, seq_len=256, num_keys=slices, reserved_filler=F
            )
            for seed in (0, 1):
                for k in slices:
                    batch = generate_overwrite(
                        cfg, num_examples=8, seed=seed, num_keys=k
                    )
                    for name in (
                        "input_ids",
                        "labels",
                        "old_labels",
                        "key_class",
                        "write_distance",
                        "slice_id",
                    ):
                        assert np.array_equal(
                            batch[name].numpy(), data[f"F{F}_s{seed}_K{k}_{name}"]
                        ), f"F={F} seed={seed} K={k} {name} differs from fixture"


# =============================================================================
# Determinism
# =============================================================================


def test_determinism_same_seed_identical_different_seed_differs():
    a = generate_entity_field(SMALL, num_examples=24, seed=7)
    b = generate_entity_field(SMALL, num_examples=24, seed=7)
    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} not identical for seed 7"
    c = generate_entity_field(SMALL, num_examples=24, seed=8)
    assert not torch.equal(a["input_ids"], c["input_ids"]), "different seed same data"


# =============================================================================
# Token ranges
# =============================================================================


def test_token_ranges_and_disjointness():
    batch = generate_entity_field(SMALL, num_examples=32, seed=11)
    V = SMALL.vocab_size
    half = V // 2
    F = SMALL.reserved_filler
    M = SMALL.num_fields
    field_offset = half - F - M
    for e in range(batch["input_ids"].shape[0]):
        x = batch["input_ids"][e]
        entity_toks = {int(t) for t in x if 1 <= t < field_offset}
        field_toks = {int(t) for t in x if field_offset <= t < half - F}
        filler_toks = {int(t) for t in x if half - F <= t < half}
        value_toks = {int(t) for t in x if half <= t < V}
        assert entity_toks.isdisjoint(field_toks)
        assert entity_toks.isdisjoint(filler_toks)
        assert entity_toks.isdisjoint(value_toks)
        assert field_toks.isdisjoint(filler_toks)
        assert field_toks.isdisjoint(value_toks)
        assert filler_toks.isdisjoint(value_toks)
        n_entities = int(batch["slice_id"][e]) // M
        assert len(entity_toks) == n_entities, "entity tokens distinct and = E"
        # Every value written in the example is distinct and there are
        # N + U of them (an update adds one extra value per updated pair).
        n = int(batch["slice_id"][e])
        u = (
            (batch["key_class"][e] == SAME).sum()
            + (batch["key_class"][e] == ADJACENT).sum()
            + (batch["key_class"][e] == DISTANT).sum()
        )
        assert len(value_toks) == n + int(u), "all written values distinct (N+U)"
        assert 0 not in entity_toks | field_toks | filler_toks | value_toks


# =============================================================================
# Structure
# =============================================================================


@pytest.mark.parametrize("e", [2, 4, 8])
def test_fixed_entity_structure(e):
    batch = generate_entity_field(SMALL, num_examples=12, seed=5, num_entities=e)
    assert batch["slice_id"].tolist() == [e * SMALL.num_fields] * 12
    for ex in range(12):
        _check_example(
            SMALL,
            batch["input_ids"][ex],
            batch["labels"][ex],
            batch["old_labels"][ex],
            batch["key_class"][ex],
            batch["write_distance"][ex],
            e,
        )


def test_drawn_entity_structure_and_length():
    batch = generate_entity_field(SMALL, num_examples=24, seed=19)
    for e in range(24):
        n = int(batch["slice_id"][e])
        assert n in SMALL_N
        n_entities = n // SMALL.num_fields
        assert batch["input_ids"].shape[1] == SMALL.seq_len
        _check_example(
            SMALL,
            batch["input_ids"][e],
            batch["labels"][e],
            batch["old_labels"][e],
            batch["key_class"][e],
            batch["write_distance"][e],
            n_entities,
        )


# =============================================================================
# Update count
# =============================================================================


def test_update_count_and_one_field_per_entity():
    batch = generate_entity_field(SMALL, num_examples=20, seed=13, num_entities=8)
    expected_u = int(np.floor(SMALL.update_fraction * 8 + 0.5))
    for ex in range(20):
        kc = batch["key_class"][ex]
        updated_pos = [p for p in range(SMALL.seq_len) if int(kc[p]) in UPDATED_CLASSES]
        assert len(updated_pos) == expected_u, "round_half_up updated pairs"
        # Each updated pair is on a distinct entity (one field per entity).
        entities = {
            _entity_idx(int(batch["input_ids"][ex][p - 2]), batch, ex)
            for p in updated_pos
        }
        assert len(entities) == expected_u, "one updated field per entity"


# =============================================================================
# Classes
# =============================================================================


def test_sibling_vs_late_and_untouched():
    """Sibling precedes its entity's update, sibling_late follows it, and
    untouched entities have no update at all."""
    batch = generate_entity_field(SMALL, num_examples=60, seed=29)
    M = SMALL.num_fields
    for ex in range(batch["input_ids"].shape[0]):
        x = batch["input_ids"][ex]
        kc = batch["key_class"][ex]
        n = int(batch["slice_id"][ex])
        n_entities = n // M
        R = SMALL.seq_len - 3 * n
        num_slots = R // 3
        # Per entity: the second-write slot of its update (and whether it has
        # an update at all).
        entity_s2 = {}
        for p, c in enumerate(kc.tolist()):
            if c not in UPDATED_CLASSES:
                continue
            e_tok = int(x[p - 2])
            f_tok = int(x[p - 1])
            s2 = max(
                i
                for i in range(num_slots)
                if int(x[3 * i]) == e_tok and int(x[3 * i + 1]) == f_tok
            )
            entity_s2[_entity_idx(e_tok, batch, ex)] = s2
        for a, c in enumerate(kc.tolist()):
            if c == NO_CLASS:
                continue
            e_tok = int(x[a - 2])
            f_tok = int(x[a - 1])
            e_idx = _entity_idx(e_tok, batch, ex)
            if c in UPDATED_CLASSES:
                continue
            slot = next(
                i
                for i in range(num_slots)
                if int(x[3 * i]) == e_tok and int(x[3 * i + 1]) == f_tok
            )
            if c == UNTOUCHED:
                assert e_idx not in entity_s2, "untouched entity has no update"
            else:
                assert e_idx in entity_s2, "sibling family needs an update"
                s2 = entity_s2[e_idx]
                assert c == SIBLING and slot < s2 or c == SIBLING_LATE and slot > s2


def test_shortest_slice_has_no_untouched_pairs():
    """At the default config the shortest slice (N = 8) updates every
    entity, so it has no untouched pairs; the longer slices do."""
    config = EntityFieldConfig()
    for e, expected in ((2, 0), (4, 4), (8, 8)):
        batch = generate_entity_field(config, num_examples=8, seed=61, num_entities=e)
        for ex in range(8):
            n_untouched = int((batch["key_class"][ex] == UNTOUCHED).sum())
            assert n_untouched == expected


# =============================================================================
# Class proportions
# =============================================================================


def test_updated_class_proportions_match_placement_probs():
    batch = generate_entity_field(SMALL, num_examples=1500, seed=41, num_entities=8)
    kc = batch["key_class"].reshape(-1).tolist()
    counts = {c: kc.count(c) for c in (SAME, ADJACENT, DISTANT)}
    total = sum(counts.values())
    assert total > 500, "enough updated pairs to measure proportions"
    index = {name: i for i, name in enumerate(("same", "adjacent", "distant"))}
    for c, name in ((SAME, "same"), (ADJACENT, "adjacent"), (DISTANT, "distant")):
        expected = SMALL.placement_probs[index[name]]
        ratio = counts[c] / total
        assert (
            abs(ratio - expected) < 0.02
        ), f"{name} proportion {ratio:.3f} deviates from {expected:.3f}"


# =============================================================================
# Feasibility
# =============================================================================


def test_infeasible_config_raises():
    with pytest.raises(ValueError):
        # Too many entities: writes cannot fit the write-region slots.
        EntityFieldConfig(
            vocab_size=2048, seq_len=256, num_fields=4, num_entities=(16,)
        )
    with pytest.raises(ValueError):
        # Region too short for distant_min_blocks + 1 blocks (the 160-token
        # write region spans 10 blocks, too few for 11).
        EntityFieldConfig(
            vocab_size=2048,
            seq_len=256,
            num_fields=4,
            num_entities=(8,),
            distant_min_blocks=10,
        )
    with pytest.raises(ValueError):
        # No reserved filler allowed (the layout needs it).
        EntityFieldConfig(vocab_size=2048, seq_len=256, reserved_filler=0)
    with pytest.raises(ValueError):
        EntityFieldConfig(num_entities=())
    with pytest.raises(ValueError):
        EntityFieldConfig(num_entities=(2, 4, 4))
    with pytest.raises(ValueError):
        EntityFieldConfig(placement_probs=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        EntityFieldConfig(block_size=1)
    with pytest.raises(ValueError):
        # Too few entity ids for the requested count.
        EntityFieldConfig(vocab_size=32, seq_len=64, num_fields=2, num_entities=(24,))


# =============================================================================
# Sampler safeguards
# =============================================================================


def test_full_update_fraction_generates_without_stalling():
    """Every entity updated (the worst case for the distinct-entity draw)
    generates quickly and only emits valid class codes."""
    config = EntityFieldConfig(
        update_fraction=1.0, num_fields=1, num_entities=(16,), seq_len=256
    )
    batch = generate_entity_field(config, num_examples=64, seed=71)
    assert batch["input_ids"].shape == (64, 256)
    codes = set(batch["key_class"].reshape(-1).tolist())
    assert codes <= {NO_CLASS, SAME, ADJACENT, DISTANT, SIBLING, SIBLING_LATE}
    assert UNTOUCHED not in codes, "every entity is updated, so no untouched"


def test_value_sampler_floor_raises():
    """A value draw below the 1% per-round success floor is rejected at
    construction instead of stalling the redraw loop."""
    with pytest.raises(ValueError, match="value sampler"):
        EntityFieldConfig(
            vocab_size=128,
            seq_len=256,
            num_fields=1,
            num_entities=(16,),
            update_fraction=1.0,
            reserved_filler=2,
        )


def test_zero_probability_sampler_floor_raises():
    """A draw whose success probability underflows to zero is rejected with
    a finite message (no division by zero)."""
    with pytest.raises(ValueError, match="effectively unbounded"):
        EntityFieldConfig(
            vocab_size=5028,
            seq_len=13000,
            num_fields=1,
            num_entities=(2000,),
            update_fraction=0.0,
        )


# =============================================================================
# Datamodule
# =============================================================================


def _dm_ef(**kwargs):
    task_config = kwargs.pop("task_config", SMALL)
    return SyntheticRecallDataModule(
        task="entity_field",
        task_config=task_config,
        batch_size=kwargs.pop("batch_size", 8),
        num_val_per_slice=kwargs.pop("num_val_per_slice", 4),
        num_test_per_slice=kwargs.pop("num_test_per_slice", 4),
        **kwargs,
    )


def test_datamodule_slices_fixed_and_deterministic():
    dm = _dm_ef(num_val_per_slice=5, num_test_per_slice=6)
    dm.setup()
    for value in SMALL_N:
        assert int((dm.val_set["slice_id"] == value).sum()) == 5
        assert int((dm.test_set["slice_id"] == value).sum()) == 6
    assert set(dm.val_set) == {
        "input_ids",
        "labels",
        "slice_id",
        "old_labels",
        "key_class",
        "write_distance",
    }
    dm2 = _dm_ef(num_val_per_slice=5, num_test_per_slice=6)
    dm2.setup()
    assert torch.equal(dm.val_set["input_ids"], dm2.val_set["input_ids"])
    assert torch.equal(dm.test_set["input_ids"], dm2.test_set["input_ids"])


def test_datamodule_train_stream_worker_sharding():
    """Entity-field train batches are per-(seed, global batch index) and
    independent of num_workers: the same global index yields identical
    tensors for 1 and 2 workers."""
    expected_count = 3
    by_workers = {}
    for num_workers in (1, 2):
        dataset = _SyntheticTrainDataset(
            task="entity_field",
            config=SMALL,
            batch_size=8,
            train_seed=1,
        )
        batches = {}
        for worker_id in range(num_workers):
            with patch(
                "research_lib.data.synthetic.datamodule.torch.utils.data.get_worker_info",
                return_value=SimpleNamespace(id=worker_id, num_workers=num_workers),
            ):
                for batch in itertools.islice(iter(dataset), expected_count):
                    batches[int(batch["batch_index"])] = batch
        by_workers[num_workers] = batches
    assert sorted(by_workers[1]) == list(range(expected_count))
    assert sorted(by_workers[2]) == list(range(expected_count * 2))
    # The batch with a given global index is identical for 1 and 2 workers.
    for index in by_workers[1]:
        one = by_workers[1][index]
        two = by_workers[2][index]
        assert set(one) == set(two)
        for key in one:
            assert torch.equal(one[key], two[key]), f"batch {index} {key} differs"


# =============================================================================
# Metrics
# =============================================================================


def _metric_preds(batch, mode):
    """Crafted next-token predictions over an entity-field batch.

    ``mode="all_correct"`` predicts the target at every answer;
    ``mode="updated_leak"`` predicts v_old at updated answers and the
    target elsewhere (leak=1 for the updated classes only).
    """
    labels = batch["labels"]
    old = batch["old_labels"]
    kc = batch["key_class"]
    mask = labels[:, 1:] != IGNORE
    targets = labels[:, 1:]
    preds = torch.zeros_like(targets)
    if mode == "all_correct":
        preds[mask] = targets[mask]
    else:
        preds[mask] = targets[mask]
        updated_mask = mask & (kc[:, 1:] >= 0) & (kc[:, 1:] <= DISTANT)
        preds[updated_mask] = old[:, 1:][updated_mask]
    return preds


def _handmade_metric_batch():
    """A hand-made batch with four answers per class: two correct, one
    ``v_old`` (updated classes only) and one wrong non-old value."""
    classes = [SAME, ADJACENT, DISTANT, SIBLING, SIBLING_LATE, UNTOUCHED]
    per_class = 4
    n = len(classes) * per_class
    T = n + 1
    labels = torch.full((1, T), IGNORE, dtype=torch.long)
    old_labels = torch.full((1, T), IGNORE, dtype=torch.long)
    key_class = torch.full((1, T), NO_CLASS, dtype=torch.long)
    preds = torch.zeros((1, T - 1), dtype=torch.long)
    idx = 0
    for c in classes:
        for k in range(per_class):
            pos = idx + 1
            target = 100 + idx
            labels[0, pos] = target
            key_class[0, pos] = c
            if c in UPDATED_CLASSES:
                old_labels[0, pos] = 200 + idx
            if k < 2:
                preds[0, pos - 1] = target
            elif k == 2 and c in UPDATED_CLASSES:
                preds[0, pos - 1] = 200 + idx
            else:
                preds[0, pos - 1] = 300 + idx
            idx += 1
    batch = {
        "labels": labels,
        "old_labels": old_labels,
        "key_class": key_class,
        "slice_id": torch.tensor([16]),
    }
    return preds, batch


def test_metrics_exact_fractions():
    """Hand-made predictions give exact non-trivial per-class fractions:
    updated classes 0.5 correct / 0.25 leaked / 0.25 other, the rest 0.5
    correct / 0.5 other."""
    preds, batch = _handmade_metric_batch()
    metrics = RecallMetrics(slice_values=(16,), class_names=ENTITY_FIELD_CLASSES)
    metrics.update_predictions(preds, batch)
    out = metrics.compute()
    for c in ("same", "adjacent", "distant"):
        assert out[f"acc/class={c}"] == pytest.approx(0.5)
        assert out[f"leak/class={c}"] == pytest.approx(0.25)
        assert out[f"other/class={c}"] == pytest.approx(0.25)
    for c in ("sibling", "sibling_late", "untouched"):
        assert out[f"acc/class={c}"] == pytest.approx(0.5)
        assert out[f"leak/class={c}"] == pytest.approx(0.0)
        assert out[f"other/class={c}"] == pytest.approx(0.5)
    for c in ENTITY_FIELD_CLASSES:
        assert out[f"acc/slice=16/class={c}"] == pytest.approx(0.5)
    assert out["acc"] == pytest.approx(0.5)


def test_metrics_per_class_exact():
    batch = generate_entity_field(SMALL, num_examples=60, seed=47, num_entities=8)
    m_all = RecallMetrics(slice_values=SMALL_N, class_names=ENTITY_FIELD_CLASSES)
    m_all.update_predictions(_metric_preds(batch, "all_correct"), batch)
    out = m_all.compute()
    for c in ENTITY_FIELD_CLASSES:
        assert out[f"acc/class={c}"] == 1.0, f"{c} accuracy not 1"
        assert out[f"leak/class={c}"] == 0.0, f"{c} leak not 0"
        assert out[f"other/class={c}"] == 0.0, f"{c} other not 0"
    assert out["acc"] == 1.0

    m_leak = RecallMetrics(slice_values=SMALL_N, class_names=ENTITY_FIELD_CLASSES)
    m_leak.update_predictions(_metric_preds(batch, "updated_leak"), batch)
    out2 = m_leak.compute()
    for c in ("same", "adjacent", "distant"):
        assert out2[f"acc/class={c}"] == 0.0
        assert out2[f"leak/class={c}"] == 1.0
        assert out2[f"other/class={c}"] == 0.0
    for c in ("sibling", "sibling_late", "untouched"):
        assert out2[f"acc/class={c}"] == 1.0
        assert out2[f"leak/class={c}"] == 0.0
        assert out2[f"other/class={c}"] == 0.0


# =============================================================================
# Loss on answers
# =============================================================================


class _TinyAnswerModel(nn.Module):
    """A minimal token mixer with the answer-head interface the training
    module requires (``hidden_states`` and ``head``)."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden_states(input_ids))


def _loss_module():
    """A tiny training module with the answer head over a random model."""
    from research_lib.training.configs import ScheduleConfig
    from research_lib.training.modules.mgdn2_synthetic_recall import (
        MGDN2SyntheticRecallModule,
    )
    from research_lib.training.presets import build_weight_decay_configs
    from research_lib.training.scheduling import WarmupStableDecaySchedule

    matrix_config, vector_config = build_weight_decay_configs(
        "grouped", weight_decay=0.1, betas=(0.9, 0.95)
    )
    schedule = ScheduleConfig(
        global_schedules=[
            WarmupStableDecaySchedule(
                param_name="lr",
                warmup_frac=0.0,
                cooldown_frac=1.0,
                min_value=0.0,
                max_value=1e-3,
                decay_type="cosine",
            )
        ]
    )
    return MGDN2SyntheticRecallModule(
        model=_TinyAnswerModel(TINY.vocab_size, 16),
        slice_values=TINY_N,
        answer_head=True,
        loss_on="answers",
        matrix_optimizer_config=matrix_config,
        vector_optimizer_config=vector_config,
        matrix_schedule_config=schedule,
        vector_schedule_config=schedule,
        vector_target_modules=("embedding", ".*norm.*", "head"),
    )


def test_answer_position_loss_equals_manual_mean():
    """The module's answer-position loss equals a manual mean cross-entropy
    over the answer positions of an entity-field batch."""
    torch.manual_seed(0)
    module = _loss_module()
    batch = generate_entity_field(TINY, num_examples=6, seed=2, num_entities=2)
    out = module(batch["input_ids"], batch["labels"])  # AnswerLogits
    module_loss = float(module.compute_loss(out, batch).detach())

    labels = batch["labels"]
    mask = labels[:, 1:] != IGNORE
    targets = labels[:, 1:][mask]
    full_logits = module.model(batch["input_ids"])
    answer_rows = full_logits[:, :-1][mask]
    manual = float(F.cross_entropy(answer_rows, targets))
    assert module_loss == pytest.approx(manual)


# =============================================================================
# Export sanity
# =============================================================================


def test_entity_field_classes_export():
    from research_lib.data.synthetic import ENTITY_FIELD_CLASSES as C

    assert C == ("same", "adjacent", "distant", "sibling", "sibling_late", "untouched")
