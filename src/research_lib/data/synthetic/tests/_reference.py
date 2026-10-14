"""Spec-literal sequential reference generator for the overwrite task.

Test-only oracle: the original per-example (sequential) implementation of
``generate_overwrite``, kept verbatim as the reference for the
distributional comparison test
(``test_overwrite.py::test_D05_vectorized_matches_reference_distribution``)
of the batch-vectorized production implementation.

The reference follows the spec (``impl/03_spec_synthetic_tasks.md`` §4)
literally: keys processed in random order; ``s_1`` uniform over the free
pair slots; ``s_2`` uniform over the free slots satisfying the class block
rule given ``s_1``; ``s_1`` redrawn up to ``max_placement_attempts`` times;
the example restarted (classes, keys, values redrawn; the example's
``K`` is kept across restarts, matching the production generator) on
exhaustion;
query block in uniform random order; filler drawn from ``[1, V)`` excluding
the example's keys and values.
"""

from __future__ import annotations

import numpy as np
import torch

IGNORE_INDEX = -100
NO_CLASS = -1
NO_DISTANCE = -1

_MAX_EXAMPLE_RESTARTS = 100

_SAME, _ADJACENT, _DISTANT, _SINGLE = 0, 1, 2, 3


def _placement_ok(
    cls_code: int, block_a: int, block_b: int, distant_min_blocks: int
) -> bool:
    """Class rule for placing the second write in block ``block_b`` when the
    first write sits in block ``block_a``."""
    if cls_code == _SAME:
        return block_b == block_a
    if cls_code == _ADJACENT:
        return block_b == block_a + 1
    # DISTANT
    return block_b - block_a >= distant_min_blocks


def _generate_reference_example(
    vocab_size: int,
    seq_len: int,
    block_size: int,
    distant_min_blocks: int,
    max_placement_attempts: int,
    class_probs: tuple[float, float, float, float],
    rng: np.random.Generator,
    k: int,
) -> dict[str, np.ndarray] | None:
    """Generate one overwrite example with ``k`` keys (spec-literal loop).

    Returns:
        Dict of int64 arrays of shape ``(T,)``, or ``None`` to signal that
        placement failed and the example should be restarted.
    """
    write_region = seq_len - 2 * k
    num_slots = write_region // 2

    # Classes (multinomial per key), keys and values (distinct).
    class_p = np.asarray(class_probs, dtype=np.float64)
    classes = rng.choice(4, size=k, p=class_p / class_p.sum())
    keys = rng.choice(np.arange(1, vocab_size // 2), size=k, replace=False)
    values = rng.choice(
        np.arange(vocab_size // 2, vocab_size), size=2 * k, replace=False
    )

    # Placement: process keys in random order; s_2 > s_1 per the class rule.
    free = list(range(num_slots))
    placements: dict[int, tuple[int, ...]] = {}
    for i in rng.permutation(k):
        cls_code = int(classes[i])
        if cls_code == _SINGLE:
            s = free[int(rng.integers(0, len(free)))]
            free.remove(s)
            placements[i] = (s,)
            continue
        placed = False
        for _ in range(max_placement_attempts):
            s1 = free[int(rng.integers(0, len(free)))]
            b1 = (2 * s1) // block_size
            candidates = [
                s2
                for s2 in free
                if s2 > s1
                and _placement_ok(
                    cls_code, b1, (2 * s2) // block_size, distant_min_blocks
                )
            ]
            if candidates:
                s2 = candidates[int(rng.integers(0, len(candidates)))]
                free.remove(s2)
                free.remove(s1)
                placements[i] = (s1, s2)
                placed = True
                break
        if not placed:
            return None  # restart the example

    input_ids = np.zeros(seq_len, dtype=np.int64)
    labels = np.full(seq_len, IGNORE_INDEX, dtype=np.int64)
    old_labels = np.full(seq_len, IGNORE_INDEX, dtype=np.int64)
    key_class = np.full(seq_len, NO_CLASS, dtype=np.int64)
    write_distance = np.full(seq_len, NO_DISTANCE, dtype=np.int64)
    occupied = np.zeros(seq_len, dtype=bool)

    # Write tokens: slot s holds (key, value) at positions (2s, 2s + 1).
    for i in range(k):
        key = keys[i]
        slots = placements[i]
        if len(slots) == 1:
            s = slots[0]
            input_ids[2 * s] = key
            input_ids[2 * s + 1] = values[2 * i]
        else:
            s1, s2 = slots
            input_ids[2 * s1] = key
            input_ids[2 * s1 + 1] = values[2 * i]
            input_ids[2 * s2] = key
            input_ids[2 * s2 + 1] = values[2 * i + 1]
        for s in slots:
            occupied[2 * s] = True
            occupied[2 * s + 1] = True

    # Query block: each key once in random order, [key, answer_slot].
    latest = {
        i: (values[2 * i] if len(placements[i]) == 1 else values[2 * i + 1])
        for i in range(k)
    }
    for j, i in enumerate(rng.permutation(k)):
        query_pos = write_region + 2 * j
        answer_pos = query_pos + 1
        input_ids[query_pos] = keys[i]
        occupied[query_pos] = True
        labels[answer_pos] = latest[i]
        key_class[answer_pos] = int(classes[i])
        if len(placements[i]) == 2:
            s1, s2 = placements[i]
            old_labels[answer_pos] = values[2 * i]
            write_distance[answer_pos] = 2 * (s2 - s1)

    # Filler: uniform over [1, V) excluding this example's keys and values
    # (all 2K drawn values, used or not).
    forbidden = np.concatenate([keys, values])
    fill_positions = np.flatnonzero(~occupied)
    filled = np.empty(fill_positions.size, dtype=np.int64)
    index = 0
    while index < filled.size:
        draws = rng.integers(1, vocab_size, size=filled.size - index)
        accepted = draws[~np.isin(draws, forbidden)]
        filled[index : index + accepted.size] = accepted
        index += accepted.size
    input_ids[fill_positions] = filled

    return {
        "input_ids": input_ids,
        "labels": labels,
        "old_labels": old_labels,
        "key_class": key_class,
        "write_distance": write_distance,
    }


def generate_overwrite_reference(
    vocab_size: int,
    seq_len: int,
    num_keys: tuple[int, ...],
    class_probs: tuple[float, float, float, float],
    block_size: int,
    distant_min_blocks: int,
    max_placement_attempts: int,
    num_examples: int,
    seed: int,
    fixed_num_keys: int | None = None,
) -> dict[str, torch.Tensor]:
    """Sequential spec-literal reference for ``generate_overwrite``.

    Same batch format as the production generator; same validation-free
    calling convention as the test configs (which are validated by
    ``OverwriteConfig`` in the tests).
    """
    rng = np.random.default_rng(seed)
    keys = ("input_ids", "labels", "old_labels", "key_class", "write_distance")
    arrays: dict[str, np.ndarray] = {
        "input_ids": np.zeros((num_examples, seq_len), dtype=np.int64),
        "labels": np.full((num_examples, seq_len), IGNORE_INDEX, dtype=np.int64),
        "old_labels": np.full((num_examples, seq_len), IGNORE_INDEX, dtype=np.int64),
        "key_class": np.full((num_examples, seq_len), NO_CLASS, dtype=np.int64),
        "write_distance": np.full((num_examples, seq_len), NO_DISTANCE, dtype=np.int64),
    }
    slice_id = np.zeros(num_examples, dtype=np.int64)

    for n in range(num_examples):
        # K drawn once per example and kept across restarts, matching the
        # production generator (hard-to-place K values are not replaced by
        # easier ones).
        k = (
            fixed_num_keys
            if fixed_num_keys is not None
            else num_keys[rng.integers(0, len(num_keys))]
        )
        example = None
        for _ in range(_MAX_EXAMPLE_RESTARTS):
            example = _generate_reference_example(
                vocab_size,
                seq_len,
                block_size,
                distant_min_blocks,
                max_placement_attempts,
                class_probs,
                rng,
                int(k),
            )
            if example is not None:
                break
        if example is None:
            raise RuntimeError(
                f"reference: placement failed for example {n} after "
                f"{_MAX_EXAMPLE_RESTARTS} restarts"
            )
        slice_id[n] = k
        for name in keys:
            arrays[name][n] = example[name]

    out = {name: torch.from_numpy(arrays[name]) for name in keys}
    out["slice_id"] = torch.from_numpy(slice_id)
    return out
