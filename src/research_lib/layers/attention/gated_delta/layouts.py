"""Merge schedules for the multi-state gated delta-rule layouts.

Pure functions, unit-tested on their own (``test_layouts.py``).  No state,
no parameters, no in-place operations: every function returns a *new* slot
stack built with ``torch.cat`` / indexing (rule 7 of the implementation
overview — autograd saved the old tensors).

The three multi-state layouts differ *only* in their merge schedule
(``05_derivation_multi_state.md`` §1):

- ``ws`` (windowed uniform, transported): ``W + 2`` slots.  Slot 0 = the
  current block (the only slot that receives writes), slots ``1..W`` = the
  blocks ``1..W`` back, slot ``W + 1`` = everything older.  At every block
  boundary (each ``B`` tokens) the stack shifts one up and the window's
  last slot drains into ``rest``.
- ``fs`` (frozen control, windowed only): same slots and shift as ``ws``,
  but slots ``>= 1`` are frozen (never transported) once their block ends.
  The shift lives in the same loop; freezing is a flag, not a subclass.
- ``ll`` (Fenwick / log-linear, transported): ``bitlen(t_max) + 1`` slots.
  Level 0 is the sentinel (the current token), level ``l >= 1`` has bucket
  size ``2^(l-1)`` and is active iff bit ``l-1`` of the position is set
  (Log-Linear §3.1-3.2; fla's ``construct_level_mask``).  At every position
  a Fenwick carry merges the low levels into the next one.

The Fenwick conventions are 0-based with the sentinel at level 0 (V5,
resolved: ``05`` §6.2): at position ``t`` the active levels partition
``[0, t]`` into one bucket per set bit of ``t``.

Run tests with::

    pytest src/research_lib/layers/attention/gated_delta/tests/test_layouts.py -v
"""

import logging
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class Layout(str, Enum):
    """Slot-stack layout of the multi-state layer (``str`` enum, plain
    strings accepted in constructors, precedent ``AttentionType``).

    - ``single``: one state, no schedule — the GDN-2 reference function.
    - ``ws``: windowed uniform, transported; ``window + 2`` slots.
    - ``fs``: windowed, frozen control (same slots/shift as ``ws``; slots
      ``>= 1`` are never transported).
    - ``ll``: Fenwick / log-linear, transported; ``bitlen(t_max) + 1`` slots.
    """

    SINGLE = "single"
    WS = "ws"
    FS = "fs"
    LL = "ll"


def num_slots(layout: Layout | str, window: int = 0, t_max: int = 0) -> int:
    """Number of state slots for a layout.

    Args:
        layout (Layout | str): One of ``single`` / ``ws`` / ``fs`` / ``ll``.
        window (int): Window size ``W`` (required for ``ws``/``fs``;
            ``window + 2`` slots: 0 = current block, ``1..W`` = blocks
            back, ``W + 1`` = rest).
        t_max (int): Longest total sequence the cache will hold (required
            for ``ll``; ``bitlen(t_max) + 1`` slots, ``05`` §6.5 — the
            earlier ``ceil(log2) + 1`` is one short when the final state is
            kept).

    Returns:
        int: The slot count ``L``.

    Raises:
        ValueError: Unknown layout, or missing/invalid ``window``/``t_max``.
    """
    layout = Layout(layout)
    if layout is Layout.SINGLE:
        return 1
    if layout in (Layout.WS, Layout.FS):
        if window < 1:
            raise ValueError(f"window must be >= 1 for layout {layout!s}, got {window}")
        return window + 2
    # Layout.LL
    if t_max < 1:
        raise ValueError(f"t_max must be >= 1 for layout 'll', got {t_max}")
    return int(t_max).bit_length() + 1


def lssb(x: int) -> int:
    """Index of the lowest set bit of ``x`` (``x >= 1``).

    ``lssb(1) = 0``, ``lssb(2) = 1``, ``lssb(12) = 2``.

    Args:
        x (int): A positive integer.

    Returns:
        int: The 0-based index of the lowest set bit.

    Raises:
        ValueError: If ``x < 1``.
    """
    if x < 1:
        raise ValueError(f"lssb requires x >= 1, got {x}")
    return (x & -x).bit_length() - 1


def ll_level(t: int, s: int) -> int:
    """Fenwick level of key position ``s`` as seen from query position ``t``.

    Level 0 iff ``s == t`` (the sentinel).  For ``s < t``, level ``l >= 1``
    is the unique one with bit ``l - 1`` of ``t`` set and
    ``h - 2^(l-1) <= s < h`` where ``h = t - (t mod 2^(l-1))``
    (``05`` §6.2; the same condition as fla's ``construct_level_mask``).

    Args:
        t (int): Query position (absolute, 0-based).
        s (int): Key position, ``s <= t``.

    Returns:
        int: The level in ``0..bitlen(t)``.

    Raises:
        ValueError: If ``s > t``.
    """
    if s == t:
        return 0
    if s > t:
        raise ValueError(f"ll_level requires s <= t, got s={s}, t={t}")
    lvl = 1
    while True:
        size = 1 << (lvl - 1)
        if (t >> (lvl - 1)) & 1:
            h = t - (t % size)
            lo = h - size
            if lo <= s < h:
                return lvl
        lvl += 1
        if lvl > 63:  # pragma: no cover - guards s <= t violations of the invariant
            raise RuntimeError(f"ll_level did not terminate for s={s}, t={t}")


def ll_level_matrix(positions: torch.Tensor) -> torch.Tensor:
    """Level of every (query, key) pair in a chunk.

    Args:
        positions (Tensor): 1-D ``LongTensor`` of absolute positions,
            shape ``(C,)``.  The chunk must be contiguous and increasing
            (in-chunk pairs then only take levels ``0..k`` with
            ``C = 2^k``, but this function does not require it).

    Returns:
        Tensor: ``(C, C)`` ``LongTensor`` with ``out[t, s] = ll_level(t, s)``
        for ``s <= t`` (row = query) and ``-1`` above the diagonal.
    """
    if positions.dim() != 1:
        raise ValueError(f"positions must be 1-D, got shape {tuple(positions.shape)}")
    C = positions.shape[0]
    pos = positions.tolist()
    out = torch.full((C, C), -1, dtype=torch.long, device=positions.device)
    for r in range(C):
        for c in range(r + 1):
            out[r, c] = ll_level(pos[r], pos[c])
    return out


def ws_shift(S: torch.Tensor, window: int) -> torch.Tensor:
    """Functional block-boundary shift on the slot axis (dim -3).

    ``rest' = rest + S[W]``; ``S'[w] = S[w-1]`` for ``w = 1..W``;
    ``S'[0] = 0``.  A new bucket is *never* copied into slot 0 (the new
    block starts empty; writes land there during the block).

    Args:
        S (Tensor): Slot stack of shape ``(..., L, d_k, d_v)`` with
            ``L = window + 2`` (slot axis is dim ``-3``).
        window (int): Window size ``W``.

    Returns:
        Tensor: The shifted stack, same shape, a new tensor (the input is
        untouched).

    Raises:
        ValueError: If ``S`` does not have ``window + 2`` slots.
    """
    if S.shape[-3] != window + 2:
        raise ValueError(
            f"ws_shift expects {window + 2} slots (window={window}), "
            f"got {S.shape[-3]}"
        )
    zero = torch.zeros_like(S[..., 0:1, :, :])
    merged = S[..., window : window + 1, :, :] + S[..., window + 1 : window + 2, :, :]
    return torch.cat([zero, S[..., :window, :, :], merged], dim=-3)


def ll_token_merge(S: torch.Tensor, next_position: int) -> torch.Tensor:
    """Fenwick carry for arriving at ``next_position`` (>= 1); functional.

    With ``j = lssb(next_position)``: slot ``j + 1`` receives the sum of
    slots ``0..j``, and slots ``0..j`` are zeroed.  The *old* content of
    slot ``j + 1`` is discarded: by the Fenwick invariant it is necessarily
    empty (bit ``j`` of ``next_position - 1`` is 0, so level ``j + 1`` was
    not active at ``next_position - 1``).

    Args:
        S (Tensor): Slot stack of shape ``(..., L, d_k, d_v)``.
        next_position (int): The absolute position being arrived at
            (``>= 1``; a merge happens at every arrival).

    Returns:
        Tensor: The merged stack, same shape, a new tensor.

    Raises:
        ValueError: If ``next_position < 1`` or the slot stack cannot hold
            level ``j + 1`` (``t_max`` too small; the cache's documented
            capacity limit).
    """
    j = lssb(next_position)
    L = S.shape[-3]
    if j + 1 > L - 1:
        raise ValueError(
            f"ll_token_merge at position {next_position} needs level {j + 1}, "
            f"but the slot stack has only {L} levels (t_max too small)"
        )
    merged = S[..., : j + 1, :, :].sum(dim=-3).unsqueeze(-3)
    zeros = torch.zeros_like(S[..., : j + 1, :, :])
    return torch.cat([zeros, merged, S[..., j + 2 :, :, :]], dim=-3)


def ll_chunk_merge(
    S: torch.Tensor, chunk_state: torch.Tensor, chunk_counter: int, k: int
) -> torch.Tensor:
    """Fold a finished chunk into the coarse levels; functional.

    The chunk's own content (``chunk_state``) is a full bucket of size
    ``C = 2^k``, i.e. level ``k + 1``.  With ``c = chunk_counter`` (the
    number of finished chunks) and ``m = lssb(c)``:

    - slot ``k + 1 + m`` receives ``chunk_state + sum`` of slots
      ``k + 1 .. k + m``;
    - slots ``k + 1 .. k + m`` are zeroed;
    - slots ``0..k`` (the intra-chunk levels) are left as they are — in the
      canonical cache form at a chunk boundary they are already zero.

    The old content of slot ``k + 1 + m`` is discarded (necessarily empty,
    by the same Fenwick invariant as :func:`ll_token_merge`, applied to the
    chunk counter).

    Args:
        S (Tensor): Slot stack of shape ``(..., L, d_k, d_v)``; slot axis
            is dim ``-3``.
        chunk_state (Tensor): The finished chunk's content, shape
            ``(..., d_k, d_v)``.
        chunk_counter (int): Number of finished chunks including this one
            (``c >= 1``).
        k (int): ``log2(chunk_size)``.

    Returns:
        Tensor: The merged stack, same shape, a new tensor.

    Raises:
        ValueError: If the slot stack cannot hold level ``k + 1 + m``.
    """
    if chunk_counter < 1:
        raise ValueError(f"chunk_counter must be >= 1, got {chunk_counter}")
    m = lssb(chunk_counter)
    L = S.shape[-3]
    if k + 1 + m > L - 1:
        raise ValueError(
            f"ll_chunk_merge at chunk counter {chunk_counter} needs level "
            f"{k + 1 + m}, but the slot stack has only {L} levels (t_max too small)"
        )
    merged = chunk_state + S[..., k + 1 : k + 1 + m, :, :].sum(dim=-3)
    merged = merged.unsqueeze(-3)
    zeros = torch.zeros_like(S[..., k + 1 : k + 1 + m, :, :])
    return torch.cat(
        [S[..., : k + 1, :, :], zeros, merged, S[..., k + m + 2 :, :, :]], dim=-3
    )
