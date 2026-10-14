"""Repeated-key overwrite generator with placement classes.

Procedural generator for the second pilot task (protocol ``06`` §4): ``K``
distinct keys, each belonging to one **placement class** controlling the
geometry of its writes:

- ``same``: ``(k, v_old)`` then ``(k, v_new)`` in the same ``block_size``
  token block;
- ``adjacent``: the two writes sit in blocks differing by exactly 1;
- ``distant``: the two writes sit at least ``distant_min_blocks`` blocks
  apart;
- ``single``: one write ``(k, v)``, never overwritten.

Per example: the write region occupies positions ``[0, R)`` with
``R = T - 2K``, divided into ``R / 2`` two-token pair slots (slot ``i`` =
positions ``2i``, ``2i + 1``); the query block occupies the last ``2K``
positions as ``[key, answer_slot]`` pairs, each key once in random order.
Values are drawn distinct from ``[V/2, V)`` (two per double-write key, one
per single key); keys are drawn distinct from ``[1, V/2)``.

The label at the answer slot of key ``k`` is the **latest** value (the
value of the second write for double-write keys). ``old_labels`` carries
``v_old`` at the answer slots of double-write keys, ``key_class`` the class
code and ``write_distance`` the token distance between the two writes
(``-1`` for single keys); both ``-100`` / ``-1`` elsewhere. Labels follow
the repo's unshifted convention (``labels[p]`` is predicted from ``p - 1``).

Unlike MQAR, filler is drawn from ``[1, V)`` **excluding this example's
keys and values** (rejection sampling), so a filler token can never act as
an extra write to a queried key.

Pure and deterministic in ``(config, seed)`` via ``np.random.default_rng``.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
"""Label reserved for non-answer positions (repo convention)."""

NO_CLASS = -1
"""key_class value at non-answer positions."""
NO_DISTANCE = -1
"""write_distance value at non-answer positions."""

_MAX_EXAMPLE_RESTARTS = 100
"""Hard cap on example restarts (safety net; unreachable for feasible
configs, which are validated at construction)."""


class OverwriteClass(enum.IntEnum):
    """Placement classes, encoded in ``key_class`` at answer positions."""

    SAME = 0
    ADJACENT = 1
    DISTANT = 2
    SINGLE = 3


@dataclass(frozen=True)
class OverwriteConfig:
    """Configuration for the repeated-key overwrite generator.

    Args:
        vocab_size: vocabulary size ``V``. Keys come from ``[1, V/2)``,
            values from ``[V/2, V)``.
        seq_len: sequence length ``T`` (must be even).
        num_keys: the slice values: number of keys ``K`` drawn uniformly per
            example. Protocol: ``K in {d_k / 2, d_k, 2 d_k}``.
        class_probs: class assignment probabilities per key, in
            ``(same, adjacent, distant, single)`` order. Must sum to 1.
        block_size: placement block size ``B`` in tokens. The protocol uses
            ``B = C = 16``.
        distant_min_blocks: minimum block distance for the ``distant``
            class.
        max_placement_attempts: per-key retry budget when drawing the first
            write slot ``s_1``; an example is restarted when a key exhausts
            it.
    """

    vocab_size: int = 8192
    seq_len: int = 256
    num_keys: tuple[int, ...] = (8, 16, 32)
    class_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    block_size: int = 16
    distant_min_blocks: int = 4
    max_placement_attempts: int = 100

    def __post_init__(self) -> None:
        if self.seq_len % 2 != 0:
            raise ValueError(f"seq_len must be even, got {self.seq_len}")
        if self.vocab_size <= self.seq_len:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must exceed seq_len "
                f"({self.seq_len}) so keys/values cannot collide with filler"
            )
        if not self.num_keys:
            raise ValueError("num_keys must be non-empty")
        # Feasibility check for the largest K: an all-distant assignment must
        # fit (4K write tokens + 2K query tokens <= T), and the write region
        # must span at least distant_min_blocks + 1 blocks.
        k_max = max(self.num_keys)
        _validate_key_count(self, k_max)
        if self.block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {self.block_size}")
        if self.distant_min_blocks < 1:
            raise ValueError(
                f"distant_min_blocks must be >= 1, got {self.distant_min_blocks}"
            )
        if self.max_placement_attempts < 1:
            raise ValueError(
                f"max_placement_attempts must be >= 1, "
                f"got {self.max_placement_attempts}"
            )
        if len(self.class_probs) != 4:
            raise ValueError(
                f"class_probs must have 4 entries "
                f"(same, adjacent, distant, single), got {self.class_probs}"
            )
        if any(p < 0 for p in self.class_probs):
            raise ValueError(
                f"class_probs entries must be >= 0, got {self.class_probs}"
            )
        if abs(sum(self.class_probs) - 1.0) > 1e-9:
            raise ValueError(f"class_probs must sum to 1, got {sum(self.class_probs)}")

    @property
    def slices(self) -> tuple[int, ...]:
        """The per-example slice values ``K`` (sorted)."""
        return tuple(sorted(self.num_keys))


def _validate_key_count(config: OverwriteConfig, k: int) -> None:
    """Feasibility checks for a fixed key count ``k``."""
    if k < 1:
        raise ValueError(f"num_keys entries must be >= 1, got {k}")
    # All-distant worst case: 2K pair slots = 4K write tokens, plus the 2K
    # query tokens.
    if 6 * k > config.seq_len:
        raise ValueError(
            f"infeasible num_keys entry {k}: need 4 * K write tokens + 2 * K "
            f"query tokens <= seq_len = {config.seq_len}"
        )
    write_region = config.seq_len - 2 * k
    # Slot i occupies positions 2i, 2i + 1; the last slot spans blocks up to
    # (R - 2) // block_size, so the region spans ((R - 2) // B + 1) blocks.
    if write_region < config.block_size * config.distant_min_blocks + 2:
        raise ValueError(
            f"infeasible num_keys entry {k}: the {write_region}-token write "
            f"region spans fewer than distant_min_blocks + 1 = "
            f"{config.distant_min_blocks + 1} blocks of "
            f"{config.block_size} tokens"
        )


def _placement_ok(
    cls_code: int, block_a: int, block_b: int, config: OverwriteConfig
) -> bool:
    """Class rule for placing the second write in block ``block_b`` when the
    first write sits in block ``block_a``."""
    cls_code = OverwriteClass(cls_code)
    if cls_code == OverwriteClass.SAME:
        return block_b == block_a
    if cls_code == OverwriteClass.ADJACENT:
        return block_b == block_a + 1
    # DISTANT
    return block_b - block_a >= config.distant_min_blocks


def _generate_example(
    config: OverwriteConfig, rng: np.random.Generator, k: int
) -> dict[str, np.ndarray] | None:
    """Generate one overwrite example with ``k`` keys.

    Returns:
        Dict of int64 arrays of shape ``(T,)`` (input_ids, labels,
        old_labels, key_class, write_distance), or ``None`` to signal that
        placement failed and the example should be restarted.
    """
    seq_len, vocab = config.seq_len, config.vocab_size
    block = config.block_size
    write_region = seq_len - 2 * k
    num_slots = write_region // 2

    # Classes (multinomial per key), keys and values (distinct).
    class_p = np.asarray(config.class_probs, dtype=np.float64)
    classes = rng.choice(4, size=k, p=class_p / class_p.sum())
    keys = rng.choice(np.arange(1, vocab // 2), size=k, replace=False)
    values = rng.choice(np.arange(vocab // 2, vocab), size=2 * k, replace=False)

    # Placement: process keys in random order; s_2 > s_1 per the class rule.
    free = list(range(num_slots))
    placements: dict[int, tuple[int, ...]] = {}
    for i in rng.permutation(k):
        cls_code = int(classes[i])
        if cls_code == int(OverwriteClass.SINGLE):
            s = free[int(rng.integers(0, len(free)))]
            free.remove(s)
            placements[i] = (s,)
            continue
        placed = False
        for _ in range(config.max_placement_attempts):
            s1 = free[int(rng.integers(0, len(free)))]
            b1 = (2 * s1) // block
            candidates = [
                s2
                for s2 in free
                if s2 > s1 and _placement_ok(cls_code, b1, (2 * s2) // block, config)
            ]
            if candidates:
                s2 = candidates[int(rng.integers(0, len(candidates)))]
                free.remove(s2)  # s1 is removed below; keep order simple
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
    used_values = []
    for i in range(k):
        key = keys[i]
        slots = placements[i]
        if len(slots) == 1:
            s = slots[0]
            value = values[2 * i]
            input_ids[2 * s] = key
            input_ids[2 * s + 1] = value
            used_values.append(value)
        else:
            s1, s2 = slots
            v_old, v_new = values[2 * i], values[2 * i + 1]
            input_ids[2 * s1] = key
            input_ids[2 * s1 + 1] = v_old
            input_ids[2 * s2] = key
            input_ids[2 * s2 + 1] = v_new
            used_values.extend([v_old, v_new])
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
        draws = rng.integers(1, vocab, size=filled.size - index)
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


def generate_overwrite(
    config: OverwriteConfig,
    num_examples: int,
    seed: int,
    num_keys: int | None = None,
) -> dict[str, torch.Tensor]:
    """Generate a batch of repeated-key overwrite examples.

    Args:
        config: task configuration.
        num_examples: number of examples ``N`` to generate.
        seed: seed for ``np.random.default_rng``. Pure and deterministic.
        num_keys: if given, every example uses exactly this many keys (a
            fixed slice) instead of drawing one from ``config.num_keys``.

    Returns:
        Dict of stacked int64 tensors, all of shape ``(N, T)`` except
        ``slice_id`` of shape ``(N,)``:

        - ``input_ids``: token sequences.
        - ``labels``: unshifted; latest value at answer positions, else
          ``IGNORE_INDEX``.
        - ``slice_id``: the number of keys ``K``.
        - ``old_labels``: ``v_old`` at answer positions of double-write
          keys, else ``IGNORE_INDEX``.
        - ``key_class``: class code at answer positions (see
          :class:`OverwriteClass`), else ``-1``.
        - ``write_distance``: token distance between the two writes at
          answer positions of double-write keys, else ``-1``.

    Raises:
        ValueError: if ``num_keys`` is infeasible for ``config.seq_len``.
        RuntimeError: if placement fails for an example after
            ``max_placement_attempts`` restarts (unreachable for configs that
            pass the construction-time feasibility check).

    Shape:
        input_ids/labels/old_labels/key_class/write_distance:
        ``(num_examples, config.seq_len)``; slice_id: ``(num_examples,)``.
    """
    if num_examples < 1:
        raise ValueError(f"num_examples must be >= 1, got {num_examples}")
    if num_keys is not None:
        _validate_key_count(config, num_keys)

    rng = np.random.default_rng(seed)
    seq_len = config.seq_len

    input_ids = np.zeros((num_examples, seq_len), dtype=np.int64)
    labels = np.full((num_examples, seq_len), IGNORE_INDEX, dtype=np.int64)
    slice_id = np.zeros(num_examples, dtype=np.int64)
    old_labels = np.full((num_examples, seq_len), IGNORE_INDEX, dtype=np.int64)
    key_class = np.full((num_examples, seq_len), NO_CLASS, dtype=np.int64)
    write_distance = np.full((num_examples, seq_len), NO_DISTANCE, dtype=np.int64)

    for n in range(num_examples):
        example = None
        k = num_keys if num_keys is not None else 0
        for _ in range(_MAX_EXAMPLE_RESTARTS):
            k = (
                num_keys
                if num_keys is not None
                else int(config.num_keys[rng.integers(0, len(config.num_keys))])
            )
            example = _generate_example(config, rng, k)
            if example is not None:
                break
        if example is None:
            raise RuntimeError(
                f"overwrite: placement failed for example {n} after "
                f"{_MAX_EXAMPLE_RESTARTS} restarts; the config is likely "
                f"infeasible: {config}"
            )
        slice_id[n] = k
        input_ids[n] = example["input_ids"]
        labels[n] = example["labels"]
        old_labels[n] = example["old_labels"]
        key_class[n] = example["key_class"]
        write_distance[n] = example["write_distance"]

    return {
        "input_ids": torch.from_numpy(input_ids),
        "labels": torch.from_numpy(labels),
        "slice_id": torch.from_numpy(slice_id),
        "old_labels": torch.from_numpy(old_labels),
        "key_class": torch.from_numpy(key_class),
        "write_distance": torch.from_numpy(write_distance),
    }
