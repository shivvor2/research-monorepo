"""Entity-field recall generator with per-entity update classes.

Procedural generator for the entity-field task: every example stores
``E`` entities, each holding ``M`` fields, so there are ``N = E * M``
(entity, field, value) bindings -- ``N`` pairs of distinct codes. A
fraction of the entities are **updated**: one of their fields is written
twice, first with ``v_old`` then with ``v_new != v_old``. The other
bindings of an updated entity keep their single write.

An update tests whether an erase can drop the stale ``(e, f) -> v_old``
binding while sparing the entity's *other* fields, whose keys share the
entity's code. Placement classes therefore separate the entities that
share a code with an update from the rest:

- ``same`` / ``adjacent`` / ``distant``: an updated pair's two writes,
  with the geometry of its value tokens (same block, blocks differing by
  one, blocks at least ``distant_min_blocks`` apart) -- the classes an
  erase gate has to handle;
- ``sibling``: a never-rewritten pair of an updated entity, written
  before the entity's update (exposed to its erase);
- ``sibling_late``: a never-rewritten pair of an updated entity, written
  after the update;
- ``untouched``: a pair of an entity with no updated field at all.

``sibling`` is the screening class: only a sibling written before the
update can be damaged by it, so the gap between ``untouched`` and
``sibling`` accuracy is the erase's unwanted spread.

Per example: the write region occupies positions ``[0, R)`` with
``R = T - 3N``, divided into ``R // 3`` three-token write slots (slot
``i`` = positions ``3i, 3i+1, 3i+2``); the final ``R % 3`` positions are
filler. A pair is written as ``[e, f, value]`` in one slot (twice, in
two slots, if updated). The query block occupies the last ``3N``
positions: every pair once, in random order, as ``[e, f, answer_slot]``;
the answer slot's input token is filler (the model must recall the value
rather than read it). The label at the answer slot of pair ``(e, f)`` is
the pair's **latest** value; ``old_labels`` carries ``v_old`` at the
answer slots of updated pairs (else ``-100``), ``key_class`` the class
code and ``write_distance`` the token distance ``3 (s_2 - s_1)`` between
an updated pair's two value tokens (else ``-1``); both ``-100`` / ``-1``
elsewhere. Labels follow the repo's unshifted convention (``labels[p]``
is predicted from ``p - 1``).

Token ranges (``V`` = ``vocab_size``, ``F`` = ``reserved_filler``,
``M`` = ``num_fields``):

- ``[1, V/2 - F - M)`` entity tokens (distinct within an example);
- ``[V/2 - F - M, V/2 - F)`` the fixed field tokens (field ``j`` is
  ``V/2 - F - M + j``);
- ``[V/2 - F, V/2)`` reserved filler;
- ``[V/2, V)`` values (every value written in an example is distinct).

Token ``0`` is unused. The reserved filler range is disjoint from the
entity, field and value ranges, so no filler token (and no field token)
can ever equal a key or value -- filler never looks like a write.

The multi-query ``[e, f, answer]`` recall format follows Zoology (Arora
et al., 2023, arXiv 2312.04927, ``zoology/data/multiquery_ar.py``); the
per-entity updates follow that repo's ``forgetting_mqar``.

Implementation: per-example and deterministic. ``generate_entity_field``
handles the restart loop (an example whose placement exceeds its
``max_placement_attempts`` budget is regenerated, keeping its ``E`` so
hard-to-place ``E`` values are not replaced by easier ones); each example
is built by a sequential procedure over the slot grid with its own
draws from the shared per-call :class:`numpy.random.Generator`. The same
``(config, num_examples, seed, num_entities)`` therefore always produces
a bit-identical batch.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass

import numpy as np
import torch

from ._sampling import _check_sampler_floor, _distinct_ints

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
"""Label reserved for non-answer positions (repo convention)."""

NO_CLASS = -1
"""key_class value at non-answer positions."""
NO_DISTANCE = -1
"""write_distance value at non-answer positions."""

_MAX_EXAMPLE_RESTARTS = 100
"""Hard cap on example restarts (safety net; never observed at the
default settings; tight configs that pass the construction-time check can
still need restarts)."""


class EntityFieldClass(enum.IntEnum):
    """Placement classes, encoded in ``key_class`` at answer positions.

    Codes ``0``-``2`` are the updated pairs' geometry; ``3``-``5`` are
    never rewritten and distinguish how close a binding sits to an update.
    """

    SAME = 0
    ADJACENT = 1
    DISTANT = 2
    SIBLING = 3
    SIBLING_LATE = 4
    UNTOUCHED = 5


ENTITY_FIELD_CLASSES: tuple[str, ...] = tuple(c.name.lower() for c in EntityFieldClass)
"""Metric key fragments per entity-field class code, in class order:
``('same', 'adjacent', 'distant', 'sibling', 'sibling_late',
'untouched')``. Exported next to the generator so the metrics and
analysis take class names from the task."""


def _round_half_up(x: float) -> int:
    """Round half away from zero (used for the update count)."""
    return int(math.floor(x + 0.5))


@dataclass(frozen=True)
class EntityFieldConfig:
    """Configuration for the entity-field generator.

    Args:
        vocab_size: vocabulary size ``V``. Entity tokens come from
            ``[1, V/2 - F - M)``, the fixed field tokens from
            ``[V/2 - F - M, V/2 - F)``, filler from the reserved range
            ``[V/2 - F, V/2)`` and values from ``[V/2, V)`` (``F`` =
            ``reserved_filler``, ``M`` = ``num_fields``).
        seq_len: sequence length ``T``.
        num_fields: fields ``M`` per entity; ``N = E * M`` pairs per
            example.
        num_entities: the slice values: number of entities ``E`` drawn
            uniformly per example. Typical grids: ``E in {d_k / (2 M),
            d_k / M, 2 d_k / M}`` for ``N`` on the same scale as the
            overwrite task's ``K``.
        update_fraction: share of entities that get exactly one field
            rewritten (rounded half up).
        placement_probs: class assignment probabilities for the updated
            pairs, in ``(same, adjacent, distant)`` order. Must sum to 1.
        block_size: placement block size ``B`` in tokens (``B = 16`` in
            the default settings); a write's block is the block of its
            value token.
        distant_min_blocks: minimum block distance for the ``distant``
            class.
        reserved_filler: number ``F`` of token ids at ``[V/2 - F, V/2)``
            reserved exclusively for filler. The layout depends on the
            reserved filler (field tokens sit just below it), so ``F >=
            1`` is required and there is no non-reserved fallback.
        max_placement_attempts: per-updated-pair retry budget when
            drawing the first write slot ``s_1``; an example is
            restarted when a pair exhausts it.

    Note:
        Feasibility is checked for EVERY slice value ``E`` in
        ``num_entities`` at construction: the largest ``E`` with every
        update ``distant`` must fit (``N + U`` write slots, ``U`` =
        ``round_half_up(update_fraction * E)``, within ``(T - 3N) // 3``
        slots), the write region must span at least
        ``distant_min_blocks + 1`` blocks, ``E`` distinct entity tokens
        must remain and ``N + U`` distinct values must fit in the value
        half.
    """

    vocab_size: int = 8192
    seq_len: int = 256
    num_fields: int = 4
    num_entities: tuple[int, ...] = (2, 4, 8)
    update_fraction: float = 0.75
    placement_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    block_size: int = 16
    distant_min_blocks: int = 4
    reserved_filler: int = 512
    max_placement_attempts: int = 100

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {self.vocab_size}")
        # Provide enough headroom for entity tokens, field tokens and the
        # reserved filler: entity ids come from [1, V/2 - F - M); field 0
        # is V/2 - F - M. Requiring V/2 - F - M >= 2 keeps at least one
        # entity id (and field 0 a valid token above all entity ids).
        if self.vocab_size // 2 - self.reserved_filler - self.num_fields < 2:
            raise ValueError(
                f"vocab_size/reserved_filler/num_fields leave no room for "
                f"entity and field tokens: need vocab_size // 2 - "
                f"reserved_filler - num_fields >= 2, got "
                f"{self.vocab_size // 2 - self.reserved_filler - self.num_fields}"
            )
        if self.reserved_filler < 1:
            raise ValueError(
                f"reserved_filler must be >= 1 (the layout always uses the "
                f"reserved filler range); got {self.reserved_filler}"
            )
        if self.reserved_filler > self.vocab_size // 2:
            raise ValueError(
                f"reserved_filler ({self.reserved_filler}) must be <= "
                f"vocab_size // 2 = {self.vocab_size // 2} so the filler "
                f"range [V/2 - F, V/2) is non-empty"
            )
        if self.seq_len < 6:
            raise ValueError(f"seq_len must be >= 6, got {self.seq_len}")
        if self.num_fields < 1:
            raise ValueError(f"num_fields must be >= 1, got {self.num_fields}")
        if not self.num_entities:
            raise ValueError("num_entities must be non-empty")
        duplicates = sorted(
            {e for e in self.num_entities if self.num_entities.count(e) > 1}
        )
        if duplicates:
            raise ValueError(f"duplicate num_entities entries: {duplicates}")
        if not math.isfinite(self.update_fraction) or not (
            0.0 <= self.update_fraction <= 1.0
        ):
            raise ValueError(
                f"update_fraction must be in [0, 1], got {self.update_fraction}"
            )
        if len(self.placement_probs) != 3:
            raise ValueError(
                f"placement_probs must have 3 entries "
                f"(same, adjacent, distant), got {self.placement_probs}"
            )
        if any(not math.isfinite(p) for p in self.placement_probs):
            raise ValueError(
                f"placement_probs entries must be finite, got "
                f"{self.placement_probs}"
            )
        if any(p < 0 for p in self.placement_probs):
            raise ValueError(
                f"placement_probs entries must be >= 0, got {self.placement_probs}"
            )
        if abs(sum(self.placement_probs) - 1.0) > 1e-9:
            raise ValueError(
                f"placement_probs must sum to 1, got {sum(self.placement_probs)}"
            )
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
        for entry in self.num_entities:
            _validate_entity_count(self, entry)

    @property
    def slices(self) -> tuple[int, ...]:
        """The per-example slice values ``N = E * M`` (sorted), matching
        each example's ``slice_id``."""
        return tuple(sorted(e * self.num_fields for e in self.num_entities))


def _validate_placement_geometry(
    config: EntityFieldConfig, n_entities: int, num_slots: int, u: int
) -> None:
    """Reject class mixtures the write region's slot grid cannot place.

    Slot ``i``'s value token is at ``3i + 2``, so it falls in block
    ``(3i + 2) // B``. A ``same`` pair needs two slots in one block, so
    a block with ``c`` slots offers ``c // 2`` disjoint pairs; an
    ``adjacent`` pair needs a slot in some block and another in the next
    block (any slot there has a larger position). The ``distant`` class
    is covered by the block-span check in :func:`_validate_entity_count`.

    Args:
        config: task configuration.
        n_entities: the entity count ``E`` of this slice.
        num_slots: the write-region slot count.
        u: the number of updated pairs ``U``.

    Raises:
        ValueError: If a class with a positive probability cannot be
            placed for this slice and block size.
    """
    p_same, p_adjacent, _ = config.placement_probs
    slot_blocks = [(3 * i + 2) // config.block_size for i in range(num_slots)]
    if p_same > 0:
        counts: dict[int, int] = {}
        for b in slot_blocks:
            counts[b] = counts.get(b, 0) + 1
        same_pairs = sum(c // 2 for c in counts.values())
        if same_pairs < u:
            raise ValueError(
                f"infeasible num_entities entry {n_entities} with block_size "
                f"{config.block_size}: the {num_slots}-slot write region offers "
                f"only {same_pairs} disjoint same-block slot pairs, fewer than "
                f"the U = {u} updated pairs the 'same' class needs"
            )
    if p_adjacent > 0:
        blocks = set(slot_blocks)
        if not any(b + 1 in blocks for b in blocks):
            raise ValueError(
                f"infeasible num_entities entry {n_entities} with block_size "
                f"{config.block_size}: no two consecutive blocks hold a value "
                f"slot, so the 'adjacent' class cannot be placed"
            )


def _validate_entity_count(config: EntityFieldConfig, n_entities: int) -> None:
    """Feasibility and sampler checks for a fixed entity count ``e``.

    The all-distant worst case needs ``N + U`` write slots (``N = e *
    M`` pairs, ``U = round_half_up(update_fraction * e)`` updated pairs
    taking two slots each) within ``(T - 3N) // 3`` slots; the write
    region must span ``distant_min_blocks + 1`` blocks; ``e`` distinct
    entity tokens must fit in ``[1, V/2 - F - M)``; and ``N + U``
    distinct values must fit in the value half ``[V/2, V)``. The slot
    grid must also be able to place every class with a positive
    probability (see :func:`_validate_placement_geometry`). The two
    without-replacement draws (``e`` entity ids, ``N + U`` values) must
    also clear the 1% per-round success floor, so a valid config never
    stalls the redraw loop.
    """
    if n_entities < 1:
        raise ValueError(f"num_entities entries must be >= 1, got {n_entities}")
    m = config.num_fields
    n = n_entities * m
    u = _round_half_up(config.update_fraction * n_entities)
    if u > n:
        raise ValueError(
            f"update count {u} for {n_entities} entities exceeds the {n} pairs"
        )
    r = config.seq_len - 3 * n
    if r < 0:
        raise ValueError(
            f"infeasible num_entities entry {n_entities}: N = {n} pairs need "
            f"3N = {3 * n} query tokens but seq_len = {config.seq_len}"
        )
    num_slots = r // 3
    if n + u > num_slots:
        raise ValueError(
            f"infeasible num_entities entry {n_entities}: N + U = {n} + {u} = "
            f"{n + u} write slots needed (updated pairs take two) but the "
            f"{r}-token write region holds only {num_slots} slots"
        )
    if num_slots < 1:
        raise ValueError(
            f"infeasible num_entities entry {n_entities}: the write region "
            f"holds no slots"
        )
    last_value_pos = 3 * (num_slots - 1) + 2
    if last_value_pos // config.block_size + 1 < config.distant_min_blocks + 1:
        raise ValueError(
            f"infeasible num_entities entry {n_entities}: the {r}-token write "
            f"region spans fewer than distant_min_blocks + 1 = "
            f"{config.distant_min_blocks + 1} blocks"
        )
    _validate_placement_geometry(config, n_entities, num_slots, u)
    n_entity_pool = (
        config.vocab_size // 2 - config.reserved_filler - config.num_fields - 1
    )
    if n_entities > n_entity_pool:
        raise ValueError(
            f"infeasible num_entities entry {n_entities}: entity tokens come "
            f"from [1, {n_entity_pool + 1}), which has only {n_entity_pool} "
            f"ids, fewer than {n_entities} distinct entities"
        )
    _check_sampler_floor(
        n_entity_pool,
        n_entities,
        f"num_entities entry {n_entities}",
        "entity",
        "",
        "increase vocab_size or reduce reserved_filler",
    )
    n_values = n + u
    n_value_pool = config.vocab_size // 2
    if n_values > n_value_pool:
        raise ValueError(
            f"infeasible num_entities entry {n_entities}: N + U = {n_values} "
            f"distinct values are needed but the value pool "
            f"[vocab_size // 2, vocab_size) has only {n_value_pool} ids"
        )
    _check_sampler_floor(
        n_value_pool,
        n_values,
        f"num_entities entry {n_entities}",
        "value",
        "",
        "increase vocab_size",
    )


def _block_of(position: int, block_size: int) -> int:
    """The block index of a token ``position``: ``position // B``.

    A write slot ``i`` holds its value at ``3i + 2``; a write's block is
    the block of that value token.
    """
    return position // block_size


def _try_example(
    config: EntityFieldConfig, rng: np.random.Generator, n_entities: int
) -> dict[str, np.ndarray] | None:
    """Generate one example, or ``None`` to trigger a restart.

    Pure and deterministic in ``rng``. See the module docstring for the
    format and classes.

    Args:
        config: task configuration.
        rng: numpy Generator (the shared per-call stream).
        n_entities: the number of entities ``E`` for this example.

    Returns:
        The per-example arrays (``input_ids``, ``labels``,
        ``old_labels``, ``key_class``, ``write_distance``, all ``(T,)``
        int64, plus a scalar ``slice_id``), or ``None`` when placement
        failed after an updated pair exhausted ``max_placement_attempts``.
    """
    vocab = config.vocab_size
    m = config.num_fields
    f = config.reserved_filler
    seq = config.seq_len
    block = config.block_size
    dmin = config.distant_min_blocks
    probs = np.asarray(config.placement_probs, dtype=np.float64)
    probs = probs / probs.sum()
    half = vocab // 2
    field_offset = half - f - m

    n = n_entities * m
    r = seq - 3 * n
    num_slots = r // 3
    u = _round_half_up(config.update_fraction * n_entities)

    # Pairs are indexed row-major by (entity, field): pi = e * M + f.
    entity_ids = _distinct_ints(rng, 1, field_offset, (1, n_entities))[0]
    # N base values (one per pair) + U update values, all distinct.
    values = _distinct_ints(rng, half, vocab, (1, n + u))[0]
    v0 = values[:n]  # original value per pair (v_old when updated)
    v_new = values[n:]  # update value per updated pair

    if u > 0:
        # A permutation prefix is distinct by construction (no rejection).
        upd_entities = rng.permutation(n_entities)[:u]
        upd_fields = rng.integers(0, m, size=u)
        upd_pairs = upd_entities * m + upd_fields
        upd_classes = rng.choice(
            np.asarray(
                [
                    int(EntityFieldClass.SAME),
                    int(EntityFieldClass.ADJACENT),
                    int(EntityFieldClass.DISTANT),
                ]
            ),
            size=u,
            p=probs,
        )
    else:
        upd_entities = np.empty(0, dtype=np.int64)
        upd_pairs = np.empty(0, dtype=np.int64)
        upd_classes = np.empty(0, dtype=np.int64)
    upd_pair_set = set(int(p) for p in upd_pairs.tolist())

    value_curr = v0.copy()
    for j, pi in enumerate(upd_pairs):
        value_curr[int(pi)] = v_new[j]

    # ---- Placement ----
    free = list(range(num_slots))
    slots: dict[int, tuple[int, ...]] = {}

    def _class_ok(b1: int, b2: int, cls: int) -> bool:
        """Whether write blocks ``b1 < b2`` satisfy class ``cls``."""
        if cls == int(EntityFieldClass.SAME):
            return b2 == b1
        if cls == int(EntityFieldClass.ADJACENT):
            return b2 == b1 + 1
        # DISTANT
        return b2 - b1 >= dmin

    # Updated pairs first, in random order.
    for j in rng.permutation(u):
        pi = int(upd_pairs[j])
        cls = int(upd_classes[j])
        placed_s2 = False
        for _ in range(config.max_placement_attempts):
            if not free:
                return None
            s1 = int(free[rng.integers(0, len(free))])
            b1 = _block_of(3 * s1 + 2, block)
            cands = [
                s
                for s in free
                if s > s1 and _class_ok(b1, _block_of(3 * s + 2, block), cls)
            ]
            if cands:
                s2 = int(cands[rng.integers(0, len(cands))])
                slots[pi] = (s1, s2)
                free.remove(s1)
                free.remove(s2)
                placed_s2 = True
                break
        if not placed_s2:
            return None  # this updated pair exhausted its budget: restart

    # Every other (non-updated) pair in a uniformly random free slot.
    other = [pi for pi in range(n) if pi not in upd_pair_set]
    for j in rng.permutation(len(other)):
        pi = other[j]
        if not free:
            return None
        s = int(free[rng.integers(0, len(free))])
        slots[pi] = (s,)
        free.remove(s)

    # ---- Class assignment ----
    entity_update_s2: dict[int, int] = {}
    for j, pe in enumerate(upd_entities):
        s1, s2 = slots[int(upd_pairs[j])]
        entity_update_s2[int(pe)] = s2
    pair_class = np.full(n, int(EntityFieldClass.UNTOUCHED), dtype=np.int64)
    for k, pi in enumerate(upd_pairs):
        pair_class[int(pi)] = int(upd_classes[k])
    for pi in range(n):
        e = pi // m
        if e in entity_update_s2 and pi not in upd_pair_set:
            s = slots[pi][0]
            pair_class[pi] = (
                int(EntityFieldClass.SIBLING)
                if s < entity_update_s2[e]
                else int(EntityFieldClass.SIBLING_LATE)
            )

    # ---- Writing ----
    input_ids = np.zeros(seq, dtype=np.int64)
    labels = np.full(seq, IGNORE_INDEX, dtype=np.int64)
    old_labels = np.full(seq, IGNORE_INDEX, dtype=np.int64)
    key_class = np.full(seq, NO_CLASS, dtype=np.int64)
    write_distance = np.full(seq, NO_DISTANCE, dtype=np.int64)

    for pi in range(n):
        e_tok = int(entity_ids[pi // m])
        f_tok = int(field_offset + (pi % m))
        ss = slots[pi]
        if len(ss) == 2:
            s1, s2 = ss
            input_ids[3 * s1 : 3 * s1 + 3] = (e_tok, f_tok, int(v0[pi]))
            input_ids[3 * s2 : 3 * s2 + 3] = (e_tok, f_tok, int(value_curr[pi]))
        else:
            s = ss[0]
            input_ids[3 * s : 3 * s + 3] = (e_tok, f_tok, int(value_curr[pi]))

    # Query block: every pair once, random order, as [e, f, answer].
    for q, pi in enumerate(rng.permutation(n)):
        qpos = seq - 3 * n + 3 * q
        e_tok = int(entity_ids[pi // m])
        f_tok = int(field_offset + (pi % m))
        a = qpos + 2
        input_ids[qpos : qpos + 2] = (e_tok, f_tok)
        labels[a] = int(value_curr[pi])
        key_class[a] = int(pair_class[pi])
        is_updated = pi in upd_pair_set
        if is_updated:
            s1, s2 = slots[pi]
            old_labels[a] = int(v0[pi])
            write_distance[a] = 3 * (s2 - s1)

    # ---- Filler (reserved range, disjoint from every key and value) ----
    fill_pos = np.nonzero(input_ids == 0)[0]
    input_ids[fill_pos] = rng.integers(half - f, half, size=fill_pos.size)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "old_labels": old_labels,
        "key_class": key_class,
        "write_distance": write_distance,
        "slice_id": np.int64(n),
    }


def generate_entity_field(
    config: EntityFieldConfig,
    num_examples: int,
    seed: int,
    num_entities: int | None = None,
) -> dict[str, torch.Tensor]:
    """Generate a batch of entity-field examples (per-example, deterministic).

    Args:
        config: task configuration.
        num_examples: number of examples to generate.
        seed: seed for ``np.random.default_rng``. Pure and deterministic:
            the same ``(config, num_examples, seed, num_entities)`` always
            produces a bit-identical batch.
        num_entities: if given, every example uses exactly this many
            entities ``E`` (a fixed slice) instead of drawing one from
            ``config.num_entities``.

    Returns:
        Dict of stacked int64 tensors, all of shape ``(num_examples, T)``
        except ``slice_id`` of shape ``(num_examples,)``:

        - ``input_ids``: token sequences.
        - ``labels``: unshifted; latest value at answer positions, else
          ``-100``.
        - ``slice_id``: the number of pairs ``N = E * M``.
        - ``old_labels``: ``v_old`` at answer positions of updated pairs,
          else ``-100``.
        - ``key_class``: class code at answer positions (see
          :class:`EntityFieldClass`), else ``-1``.
        - ``write_distance``: token distance ``3 (s2 - s1)`` between an
          updated pair's two value tokens, else ``-1``.

    Raises:
        ValueError: if ``num_examples < 1`` or ``num_entities`` is
            infeasible for ``config``.
        RuntimeError: if placement fails for an example after
            ``_MAX_EXAMPLE_RESTARTS`` restarts (never observed at the
            default settings; tight configs that pass the
            construction-time check can still need restarts).
    """
    if num_examples < 1:
        raise ValueError(f"num_examples must be >= 1, got {num_examples}")
    if num_entities is not None:
        _validate_entity_count(config, num_entities)

    rng = np.random.default_rng(seed)
    seq = config.seq_len
    names = ("input_ids", "labels", "old_labels", "key_class", "write_distance")
    arrays: dict[str, np.ndarray] = {
        "input_ids": np.zeros((num_examples, seq), dtype=np.int64),
        "labels": np.full((num_examples, seq), IGNORE_INDEX, dtype=np.int64),
        "old_labels": np.full((num_examples, seq), IGNORE_INDEX, dtype=np.int64),
        "key_class": np.full((num_examples, seq), NO_CLASS, dtype=np.int64),
        "write_distance": np.full((num_examples, seq), NO_DISTANCE, dtype=np.int64),
    }
    slice_id = np.zeros(num_examples, dtype=np.int64)

    choices = np.asarray(config.num_entities, dtype=np.int64)
    if num_entities is not None:
        per_example = np.full(num_examples, num_entities, dtype=np.int64)
    else:
        # Draw each example's E once, before the restart loop, and keep it
        # across restarts (like the overwrite task's K).
        per_example = choices[rng.integers(0, choices.size, size=num_examples)]

    pending = np.arange(num_examples, dtype=np.int64)
    tries = np.zeros(num_examples, dtype=np.int64)
    while pending.size:
        tries[pending] += 1
        if (tries[pending] > _MAX_EXAMPLE_RESTARTS).any():
            raise RuntimeError(
                f"entity_field: placement failed after {_MAX_EXAMPLE_RESTARTS} "
                f"restarts; the config is likely infeasible: {config}"
            )
        batch: dict[str, np.ndarray] = {
            name: np.zeros((pending.size, seq), dtype=np.int64) for name in names
        }
        batch_slice = np.zeros(pending.size, dtype=np.int64)
        failed: list[int] = []
        for local_i, ex_idx in enumerate(pending):
            ex = _try_example(config, rng, int(per_example[ex_idx]))
            if ex is None:
                failed.append(local_i)
            else:
                batch_slice[local_i] = int(ex["slice_id"])
                for name in names:
                    batch[name][local_i] = ex[name]
        ok_mask = np.ones(pending.size, dtype=bool)
        ok_mask[failed] = False
        done = pending[ok_mask]
        for name in names:
            arrays[name][done] = batch[name][ok_mask]
        slice_id[done] = batch_slice[ok_mask]
        pending = np.asarray([pending[li] for li in failed], dtype=np.int64)

    out = {name: torch.from_numpy(arrays[name]) for name in names}
    out["slice_id"] = torch.from_numpy(slice_id)
    return out
