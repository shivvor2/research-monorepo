"""Repeated-key overwrite generator with placement classes.

Procedural generator for the repeated-key overwrite task: ``K`` distinct
keys, each belonging to one **placement class** controlling the geometry
of its writes:

- ``same``: ``(k, v_old)`` then ``(k, v_new)`` in the same ``block_size``
  token block;
- ``adjacent``: the two writes sit in blocks differing by exactly 1;
- ``distant``: the two writes sit at least ``distant_min_blocks`` blocks
  apart;
- ``single``: one write ``(k, v)``, never overwritten.

Per example: the write region occupies positions ``[0, R)`` with
``R = T - 2K``, divided into ``R / 2`` two-token pair slots (slot ``i`` =
positions ``2i``, ``2i + 1``); the query block occupies the last ``2K``
positions as ``[key, answer_slot]`` pairs, each key once. Keys are drawn
distinct from ``[1, V/2 - F)`` (``F = reserved_filler``), values distinct
from ``[V/2, V)`` (two per
double-write key with ``v_old != v_new`` since they are distinct draws,
one per single key). The label at the answer slot of key ``k`` is the
**latest** value; ``old_labels`` carries ``v_old`` at the answer slots of
double-write keys, ``key_class`` the class code and ``write_distance`` the
token distance ``2 (s_2 - s_1)`` between the two writes (``-1`` for single
keys); both ``-100`` / ``-1`` elsewhere. Labels follow the repo's unshifted
convention (``labels[p]`` is predicted from ``p - 1``).

Unlike MQAR, filler is drawn from ``[1, V)`` **excluding this example's
keys and values** (rejection sampling), so a filler token can never act as
an extra write to a queried key. Roughly a quarter of filler two-token
pairs are then indistinguishable from a real write (a key-range token
followed by a value-range token). With ``reserved_filler=F > 0`` the key
half is split further: keys come from ``[1, V/2 - F)``, values from
``[V/2, V)`` as usual, and filler is drawn uniformly from the *reserved*
range ``[V/2 - F, V/2)`` only. A filler token can then never equal any key
or value, so no filler pair can mimic a write.

Implementation: BATCH-VECTORIZED (throughput). ``generate_overwrite``
generates all examples of a call — and of each restart round — in
vectorized numpy passes; per-example placement state lives in a boolean
mask over the pair slots. The per-example random procedure is:

- ``K``: uniform over ``config.num_keys`` (or fixed via ``num_keys``);
- classes: iid categorical draws from ``class_probs``;
- keys / values: uniform draws without replacement (rejection on
  duplicates within a row);
- placement: keys processed in index order ``t = 0, 1, ...`` — the
  intended "process keys in random order" is distributionally equivalent
  because the classes are iid (the class sequence in processing order has
  the same law in any order) and the (key, value) pairs are exchangeable.
  ``s_1`` is uniform over the free pair slots, ``s_2`` uniform over the
  free slots satisfying the class block rule given ``s_1`` (the slots of a
  block are contiguous, so the valid slots form a slice ``[a, b)``);
  ``s_1`` is redrawn up to ``max_placement_attempts`` times, after which
  the whole example is restarted (classes, keys, values and placement
  redrawn; the example's ``K`` is kept across restarts, so hard-to-place
  ``K`` values are not replaced by easier ones; at most
  ``_MAX_EXAMPLE_RESTARTS`` restarts per example);
- query block: each key once, in uniform random order (a uniform
  permutation per example);
- filler: uniform over ``[1, V)`` excluding the example's keys and
  values (default), or uniform over the reserved range
  ``[V/2 - F, V/2)`` with ``reserved_filler=F > 0``, which is disjoint
  from every key and value.

Because the batch implementation draws its randomness in a different
order than a per-example loop, its random streams are its own: generator
+ seed reproduce every example bit-exactly. A literal sequential
reference procedure lives in the test suite
(``test_vectorized_matches_reference_distribution``) and is compared
against this implementation distributionally.
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
"""Hard cap on example restarts (safety net; never observed at the
pilot grid settings; tight configs that pass the construction-time check
can still need restarts (e.g. all-distant K=42 at T=256 restarts ~79% of
examples))."""

_MAX_SAMPLING_ROUNDS = 10_000
"""Hard cap on redraw rounds of :func:`_distinct_ints` before a
RuntimeError. The configuration-time sampler bounds keep valid configs
far below this; a hit means a caller bypassed them (e.g. an impossible
draw)."""


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
        vocab_size: vocabulary size ``V``. Keys come from ``[1, V/2)``
            (or ``[1, V/2 - F)`` with ``reserved_filler=F > 0``), values
            from ``[V/2, V)``.
        seq_len: sequence length ``T`` (must be even).
        num_keys: the slice values: number of keys ``K`` drawn uniformly per
            example. Typical grids: ``K in {d_k / 2, d_k, 2 d_k}``.
        class_probs: class assignment probabilities per key, in
            ``(same, adjacent, distant, single)`` order. Must sum to 1.
        block_size: placement block size ``B`` in tokens (``B = C = 16`` in
            the pilot).
        distant_min_blocks: minimum block distance for the ``distant``
            class.
        max_placement_attempts: per-key retry budget when drawing the first
            write slot ``s_1``; an example is restarted when a key exhausts
            it.
        reserved_filler: number ``F`` of token ids at the top of the key
            half, ``[V/2 - F, V/2)``, reserved exclusively for filler.
            ``0`` (default): current behaviour, unchanged bit for bit —
            filler is drawn from ``[1, V)`` excluding each example's own
            keys and values (rejection sampling), and existing seeds
            reproduce existing data. ``F > 0``: keys come from
            ``[1, V/2 - F)`` (values unchanged from ``[V/2, V)``) and
            filler comes from the reserved range only, so no filler token
            can ever equal a key or value and no draw needs rejection.

    Note:
        ``reserved_filler`` must satisfy ``0 <= F < V/2 - 1``. The
        without-replacement samplers (keys from the key half, values from
        the value half) must not stall: every slice value ``K`` needs at
        least ``K`` distinct key ids, and the expected number of redraws
        per row must stay bounded, which requires
        ``V/2 - 1 - F >= K (K - 1) / 2`` key ids and ``V/2 >= K (2 K - 1)``
        value ids (per-row success probability at least about ``1/e``).
    """

    vocab_size: int = 8192
    seq_len: int = 256
    num_keys: tuple[int, ...] = (8, 16, 32)
    class_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    block_size: int = 16
    distant_min_blocks: int = 4
    max_placement_attempts: int = 100
    reserved_filler: int = 0

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
        duplicates = sorted({k for k in self.num_keys if self.num_keys.count(k) > 1})
        if duplicates:
            raise ValueError(f"duplicate num_keys entries: {duplicates}")
        if self.reserved_filler < 0:
            raise ValueError(
                f"reserved_filler must be >= 0, got {self.reserved_filler}"
            )
        # Keys are drawn from [1, V/2 - F): at least one id must remain
        # (F < V/2 - 1); the per-slice feasibility and sampler-stall checks
        # live in _validate_key_count, which runs next for every entry.
        if self.reserved_filler > self.vocab_size // 2 - 2:
            raise ValueError(
                f"reserved_filler ({self.reserved_filler}) must be < "
                f"vocab_size // 2 - 1 = {self.vocab_size // 2 - 1} so at "
                f"least one key id remains"
            )
        # Feasibility for EVERY entry (an all-distant assignment must fit:
        # 4K write tokens + 2K query tokens <= T, and the write region must
        # span at least distant_min_blocks + 1 blocks).
        for entry in self.num_keys:
            _validate_key_count(self, entry)
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
    """Feasibility and sampler-stall checks for a fixed key count ``k``.

    Placement first (the all-distant worst case needs ``4K + 2K`` tokens;
    the write region must span ``distant_min_blocks + 1`` blocks), then the
    without-replacement samplers: ``k`` distinct keys from the ``n_key =
    V/2 - 1 - F``-id key pool, and ``2K`` distinct values from the
    ``n_val = V/2``-id value pool. Drawing by redrawing whole rows succeeds
    per round with probability about ``exp(-K(K-1)/(2n))`` (keys) and
    ``exp(-2K(2K-1)/V)`` (values); requiring ``n >= d(d-1)/2`` keeps the
    expected redraws per row below about ``e``.
    """
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
    n_key = config.vocab_size // 2 - 1 - config.reserved_filler
    if n_key < k:
        raise ValueError(
            f"infeasible num_keys entry {k}: keys come from "
            f"[1, vocab_size // 2 - reserved_filler) = [1, {n_key + 1}), "
            f"which has only {n_key} ids, fewer than {k} distinct keys"
        )
    if n_key < k * (k - 1) // 2:
        raise ValueError(
            f"num_keys entry {k} would stall the key sampler: drawing "
            f"{k} distinct ids from the {n_key}-id key pool succeeds per "
            f"round with probability ~exp(-K(K-1)/(2n)); keep n >= "
            f"K(K-1)/2 = {k * (k - 1) // 2} key ids (success >= ~1/e) by "
            f"increasing vocab_size or reducing reserved_filler"
        )
    n_val = config.vocab_size // 2
    if n_val < 2 * k:
        raise ValueError(
            f"infeasible num_keys entry {k}: 2K = {2 * k} distinct values "
            f"are needed but the value pool [vocab_size // 2, vocab_size) "
            f"has only V/2 = {n_val} ids"
        )
    if n_val < k * (2 * k - 1):
        raise ValueError(
            f"num_keys entry {k} would stall the value sampler: drawing "
            f"2K = {2 * k} distinct values from the {n_val}-id value pool "
            f"succeeds per round with probability "
            f"~exp(-2K(2K-1)/V); keep V/2 >= K(2K-1) = "
            f"{k * (2 * k - 1)} value ids (success >= ~1/e) by increasing "
            f"vocab_size"
        )


def _distinct_ints(
    rng: np.random.Generator, low: int, high: int, shape: tuple[int, int]
) -> np.ndarray:
    """Uniform draws without replacement, per row.

    Each row of the result holds ``shape[1]`` distinct values from
    ``[low, high)``. Rows are redrawn (rejection) until distinct, which
    preserves the uniform law over ordered distinct rows.

    Args:
        rng: numpy Generator.
        low: inclusive lower bound.
        high: exclusive upper bound.
        shape: ``(rows, draws)``.

    Returns:
        Int64 array of ``shape`` whose rows hold distinct values.

    Shape:
        ``(shape)``.
    """
    out = np.empty(shape, dtype=np.int64)
    pending = np.arange(shape[0])
    rounds = 0
    while pending.size:
        rounds += 1
        if rounds > _MAX_SAMPLING_ROUNDS:
            raise RuntimeError(
                f"_distinct_ints: could not draw {shape[1]} distinct values "
                f"from the range [{low}, {high}) in {_MAX_SAMPLING_ROUNDS} "
                f"rounds"
            )
        out[pending] = rng.integers(low, high, size=(pending.size, shape[1]))
        ordered = np.sort(out[pending], axis=1)
        duplicate = np.any(ordered[:, 1:] == ordered[:, :-1], axis=1)
        pending = pending[duplicate]
    return out


def _generate_example_batch(
    config: OverwriteConfig, rng: np.random.Generator, k_arr: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Generate a batch of examples in vectorized numpy passes.

    Args:
        config: task configuration, including the filler mode
            (``reserved_filler``; see :class:`OverwriteConfig`).
        rng: numpy Generator (shared across the whole batch).
        k_arr: ``(n,)`` int64 key counts per row (already drawn or fixed).

    Returns:
        Tuple ``(arrays, success)``: ``arrays`` maps the five per-example
        array names (``input_ids``, ``labels``, ``old_labels``,
        ``key_class``, ``write_distance``) to ``(n, T)`` int64 arrays;
        ``success`` is the ``(n,)`` bool mask of rows whose placement
        succeeded (failed rows exhausted their per-key budget and must be
        restarted by the caller).

    Shape:
        ``arrays[*]: (n, T)``; ``success: (n,)``.
    """
    n = int(k_arr.size)
    seq_len, vocab = config.seq_len, config.vocab_size
    block, dmin = config.block_size, config.distant_min_blocks
    probs = np.asarray(config.class_probs, dtype=np.float64)
    probs = probs / probs.sum()

    k_max = int(k_arr.max())
    write_region = seq_len - 2 * k_arr  # (n,), even (T even, k int)
    n_slots = write_region // 2  # (n,)
    slots_max = int(n_slots.max())
    # Slot s occupies positions 2s, 2s + 1 -> block (2 s) // B; the slots of
    # a block are contiguous: block b covers slots [lo[b], lo[b+1]) with
    # lo[b] = ceil(b * B / 2) = (b * B + 1) // 2.
    block_of_slot = (2 * np.arange(slots_max, dtype=np.int64)) // block
    n_blocks_max = int(block_of_slot[-1]) + 1
    block_lo = np.array(
        [(b * block + 1) // 2 for b in range(n_blocks_max + dmin + 3)],
        dtype=np.int64,
    )
    n_blocks = block_of_slot[n_slots - 1] + 1  # (n,) per-row block count

    valid_keys = np.arange(k_max, dtype=np.int64)[None, :] < k_arr[:, None]
    classes = rng.choice(4, size=(n, k_max), p=probs)
    keys = _distinct_ints(rng, 1, vocab // 2 - config.reserved_filler, (n, k_max))
    values = _distinct_ints(rng, vocab // 2, vocab, (n, 2 * k_max))
    # Query order (each key once, in random order): a uniform permutation
    # per row. Argsort of the row's uniforms with invalid keys
    # masked to +inf: the first K_r entries are then exactly the valid key
    # indices in uniform random order.
    uniforms = rng.random((n, k_max))
    query_order = np.argsort(np.where(valid_keys, uniforms, np.inf), axis=1)
    q_slot = np.empty((n, k_max), dtype=np.int64)
    q_rows, q_js = np.nonzero(valid_keys)
    # Key query_order[r, j] goes to query slot j (j < K_r).
    q_slot[q_rows, query_order[q_rows, q_js]] = q_js

    # Placement state: free pair slots per row.
    free_mask = np.arange(slots_max, dtype=np.int64)[None, :] < n_slots[:, None]
    free_count = n_slots.astype(np.int64).copy()
    s1_arr = np.full((n, k_max), -1, dtype=np.int64)
    s2_arr = np.full((n, k_max), -1, dtype=np.int64)
    slot_arr = np.full((n, k_max), -1, dtype=np.int64)
    success = np.ones(n, dtype=bool)

    for t in range(k_max):
        active = np.flatnonzero(k_arr > t)
        if active.size == 0:
            break
        cls_t = classes[active, t]
        single_rows = active[cls_t == int(OverwriteClass.SINGLE)]

        # Single writes: one uniform free slot.
        if single_rows.size:
            counts = free_count[single_rows]
            ranks = (rng.random(single_rows.size) * counts).astype(np.int64)
            cum = np.cumsum(free_mask[single_rows], axis=1, dtype=np.int64)
            # The (rank)-th free slot (0-based) is the first p with
            # #free in [0..p] == rank + 1.
            pos = (cum == (ranks + 1)[:, None]).argmax(axis=1)
            free_mask[single_rows, pos] = False
            free_count[single_rows] -= 1
            slot_arr[single_rows, t] = pos

        # Double writes: s1 uniform over free slots, s2 uniform over the
        # class-valid candidates given s1; retry s1 on failure.
        remaining = active[cls_t != int(OverwriteClass.SINGLE)]
        for _attempt in range(config.max_placement_attempts):
            if remaining.size == 0:
                break
            m = remaining.size
            counts = free_count[remaining]
            ranks = (rng.random(m) * counts).astype(np.int64)
            cum = np.cumsum(free_mask[remaining], axis=1, dtype=np.int64)
            # cum[:, j] = # free slots in [0..j]; cum0[:, j] = # free in
            # [0, j), the prefix-sum form used for slice counts.
            cum0 = np.zeros((m, slots_max + 1), dtype=np.int64)
            cum0[:, 1:] = cum
            s1 = (cum == (ranks + 1)[:, None]).argmax(axis=1)
            b1 = block_of_slot[s1]
            cls_r = classes[remaining, t]
            nb_r = n_blocks[remaining]
            ns_r = n_slots[remaining]
            # Candidate slice [a, b) per class; the empty sentinel makes
            # a > b so the count is <= 0.
            empty = slots_max + 1
            same_a = s1 + 1
            same_b = np.minimum(block_lo[b1 + 1], ns_r)
            adj_ok = b1 + 1 < nb_r
            adj_a = np.where(adj_ok, block_lo[b1 + 1], empty)
            adj_b = np.minimum(np.where(adj_ok, block_lo[b1 + 2], 0), ns_r)
            dist_ok = b1 + dmin < nb_r
            dist_a = np.where(dist_ok, block_lo[b1 + dmin], empty)
            dist_b = np.where(dist_ok, ns_r, 0)
            a = np.where(
                cls_r == int(OverwriteClass.SAME),
                same_a,
                np.where(cls_r == int(OverwriteClass.ADJACENT), adj_a, dist_a),
            )
            b = np.where(
                cls_r == int(OverwriteClass.SAME),
                same_b,
                np.where(cls_r == int(OverwriteClass.ADJACENT), adj_b, dist_b),
            )
            a_c = np.minimum(a, slots_max)
            b_c = np.minimum(b, slots_max)
            local = np.arange(m)
            cand_count = cum0[local, b_c] - cum0[local, a_c]
            ok = cand_count > 0
            if ok.any():
                rows_ok = remaining[ok]
                local_ok = np.flatnonzero(ok)
                c_ranks = (rng.random(rows_ok.size) * cand_count[ok]).astype(np.int64)
                # The c-th free slot in [a, b) (0-based) is the first p with
                # #free in [0..p] == #free in [0, a) + c + 1.
                target = cum0[local_ok, a_c[ok]] + c_ranks + 1
                s2 = (cum[ok] == target[:, None]).argmax(axis=1)
                s1_arr[rows_ok, t] = s1[ok]
                s2_arr[rows_ok, t] = s2
                free_mask[rows_ok, s1[ok]] = False
                free_mask[rows_ok, s2] = False
                free_count[rows_ok] -= 2
            remaining = remaining[~ok]
        if remaining.size:
            # Per-key placement budget exhausted: restart these rows.
            success[remaining] = False

    # ---- Write tokens and query block (vectorized scatter) ----
    input_ids = np.zeros((n, seq_len), dtype=np.int64)
    labels = np.full((n, seq_len), IGNORE_INDEX, dtype=np.int64)
    old_labels = np.full((n, seq_len), IGNORE_INDEX, dtype=np.int64)
    key_class = np.full((n, seq_len), NO_CLASS, dtype=np.int64)
    write_distance = np.full((n, seq_len), NO_DISTANCE, dtype=np.int64)
    occupied = np.zeros((n, seq_len), dtype=bool)

    is_double = valid_keys & (classes != int(OverwriteClass.SINGLE))
    is_single = valid_keys & (classes == int(OverwriteClass.SINGLE))

    rows_d, ts_d = np.nonzero(is_double)
    if rows_d.size:
        s1_d = s1_arr[rows_d, ts_d]
        s2_d = s2_arr[rows_d, ts_d]
        input_ids[rows_d, 2 * s1_d] = keys[rows_d, ts_d]
        input_ids[rows_d, 2 * s1_d + 1] = values[rows_d, 2 * ts_d]
        input_ids[rows_d, 2 * s2_d] = keys[rows_d, ts_d]
        input_ids[rows_d, 2 * s2_d + 1] = values[rows_d, 2 * ts_d + 1]
        occupied[rows_d, 2 * s1_d] = True
        occupied[rows_d, 2 * s1_d + 1] = True
        occupied[rows_d, 2 * s2_d] = True
        occupied[rows_d, 2 * s2_d + 1] = True
    rows_s, ts_s = np.nonzero(is_single)
    if rows_s.size:
        slot_s = slot_arr[rows_s, ts_s]
        input_ids[rows_s, 2 * slot_s] = keys[rows_s, ts_s]
        input_ids[rows_s, 2 * slot_s + 1] = values[rows_s, 2 * ts_s]
        occupied[rows_s, 2 * slot_s] = True
        occupied[rows_s, 2 * slot_s + 1] = True

    # Query block: key t at [write_region + 2 * q_slot[t]], answer at +1.
    rows_q, ts_q = np.nonzero(valid_keys)
    q_key_pos = write_region[rows_q] + 2 * q_slot[rows_q, ts_q]
    ans_pos = q_key_pos + 1
    input_ids[rows_q, q_key_pos] = keys[rows_q, ts_q]
    occupied[rows_q, q_key_pos] = True
    labels[rows_q, ans_pos] = np.where(
        is_double[rows_q, ts_q],
        values[rows_q, 2 * ts_q + 1],
        values[rows_q, 2 * ts_q],
    )
    key_class[rows_q, ans_pos] = classes[rows_q, ts_q]
    dbl_q = is_double[rows_q, ts_q]
    if dbl_q.any():
        rows_dq = rows_q[dbl_q]
        ts_dq = ts_q[dbl_q]
        old_labels[rows_dq, ans_pos[dbl_q]] = values[rows_dq, 2 * ts_dq]
        write_distance[rows_dq, ans_pos[dbl_q]] = 2 * (
            s2_arr[rows_dq, ts_dq] - s1_arr[rows_dq, ts_dq]
        )

    # ---- Filler ----
    fill_rows, fill_cols = np.nonzero(~occupied)
    if config.reserved_filler > 0:
        # Reserved mode: uniform over the reserved range [V/2 - F, V/2)
        # only, which is disjoint from every key and value, so a single
        # draw needs no rejection.
        tokens = rng.integers(
            vocab // 2 - config.reserved_filler, vocab // 2, size=fill_rows.size
        )
    else:
        # Default mode: uniform over [1, V) excluding this example's keys
        # and values (rejection; a (rows, vocab) lookup table is O(K) per
        # row and cheap at the pilot sizes).
        forbidden = np.zeros((n, vocab), dtype=bool)
        forbidden[rows_q, keys[rows_q, ts_q]] = True
        forbidden[rows_q, values[rows_q, 2 * ts_q]] = True
        forbidden[rows_q, values[rows_q, 2 * ts_q + 1]] = True
        tokens = np.empty(fill_rows.size, dtype=np.int64)
        pending_fill = np.arange(fill_rows.size)
        while pending_fill.size:
            tokens[pending_fill] = rng.integers(1, vocab, size=pending_fill.size)
            rejected = forbidden[fill_rows[pending_fill], tokens[pending_fill]]
            pending_fill = pending_fill[rejected]

    input_ids[fill_rows, fill_cols] = tokens

    arrays = {
        "input_ids": input_ids,
        "labels": labels,
        "old_labels": old_labels,
        "key_class": key_class,
        "write_distance": write_distance,
    }
    return arrays, success


def generate_overwrite(
    config: OverwriteConfig,
    num_examples: int,
    seed: int,
    num_keys: int | None = None,
) -> dict[str, torch.Tensor]:
    """Generate a batch of repeated-key overwrite examples (batch-vectorized).

    Args:
        config: task configuration.
        num_examples: number of examples ``N`` to generate.
        seed: seed for ``np.random.default_rng``. Pure and deterministic:
            the same ``(config, num_examples, seed, num_keys)`` always
            produces a bit-identical batch.
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
        - ``write_distance``: token distance ``2 (s2 - s1)`` between the
          two writes at answer positions of double-write keys, else ``-1``.

    Raises:
        ValueError: if ``num_examples < 1`` or ``num_keys`` is infeasible for
            ``config.seq_len``.
        RuntimeError: if placement fails for an example after
            ``_MAX_EXAMPLE_RESTARTS`` restarts (never observed at the
            pilot grid settings; tight configs that pass the
            construction-time check can still need restarts, e.g.
            all-distant K=42 at T=256 restarts ~79% of examples).

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
    names = ("input_ids", "labels", "old_labels", "key_class", "write_distance")
    arrays: dict[str, np.ndarray] = {
        "input_ids": np.zeros((num_examples, seq_len), dtype=np.int64),
        "labels": np.full((num_examples, seq_len), IGNORE_INDEX, dtype=np.int64),
        "old_labels": np.full((num_examples, seq_len), IGNORE_INDEX, dtype=np.int64),
        "key_class": np.full((num_examples, seq_len), NO_CLASS, dtype=np.int64),
        "write_distance": np.full((num_examples, seq_len), NO_DISTANCE, dtype=np.int64),
    }
    slice_id = np.zeros(num_examples, dtype=np.int64)

    num_keys_arr = np.asarray(config.num_keys, dtype=np.int64)
    # Each example's K is drawn ONCE, before the restart loop, and kept
    # across restarts: redrawing K for the pending rows would replace
    # hard-to-place K values by easier ones whenever restarts happen,
    # biasing the slice mix (all-distant K=42 at T=256 restarts ~79% of
    # examples; with redrawn K its share collapsed from 0.5 to ~0.19).
    k_all = (
        num_keys_arr[rng.integers(0, num_keys_arr.size, size=num_examples)]
        if num_keys is None
        else np.full(num_examples, num_keys, dtype=np.int64)
    )
    pending = np.arange(num_examples, dtype=np.int64)
    tries = np.zeros(num_examples, dtype=np.int64)
    while pending.size:
        tries[pending] += 1
        if (tries[pending] > _MAX_EXAMPLE_RESTARTS).any():
            raise RuntimeError(
                f"overwrite: placement failed after {_MAX_EXAMPLE_RESTARTS} "
                f"restarts; the config is likely infeasible: {config}"
            )
        k_arr = k_all[pending]
        batch, ok = _generate_example_batch(config, rng, k_arr)
        done = pending[ok]
        for name in names:
            arrays[name][done] = batch[name][ok]
        slice_id[done] = k_arr[ok]
        pending = pending[~ok]

    out = {name: torch.from_numpy(arrays[name]) for name in names}
    out["slice_id"] = torch.from_numpy(slice_id)
    return out
