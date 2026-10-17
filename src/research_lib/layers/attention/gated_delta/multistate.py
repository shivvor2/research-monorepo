"""Multi-state gated delta-rule layer (WS / FS / LL layouts).

**Reference implementation** (naive PyTorch + autograd, no Triton, no
hand-written backward), like the single-state
:class:`~research_lib.layers.attention.gated_delta.reference.GatedDeltaReference`
it builds on.  One class, :class:`MultistateGatedDelta`, covers every
variant: ``gate_mode`` × ``layout`` × ``lambda_mode`` are constructor
arguments (rule 1; keeps the λ≡1 reduction test meaningful).  With
``layout="single"`` the layer is the *same function* as
``GatedDeltaReference`` — same projections submodule (``self.projections``),
same parameter names, no λ module, same functional calls — so a reference
``state_dict`` loads with ``strict=True`` and gives identical outputs.

The math (``05_derivation_multi_state.md``):

- Every slot is transported by the same transition
  ``A_t = (I - k (b*k)^T) Diag(e^g)``; the write ``k (w*v)^T`` lands in
  slot 0 only; the other slots receive only the erase
  (``X^(l>=1) = -W S^(l)``, §2.2 Step 3 — *never* the full chunk write).
- The readout is ``o_t = scale * sum_l lambda_t^(l) (S_t^(l))^T q_t``
  (read after write), then the shared output head (per-head RMSNorm ×
  output gate × output projection).
- Layouts differ only in their merge schedule (``layouts.py``): ``ws`` /
  ``fs`` shift at every block boundary; ``ll`` runs a Fenwick carry at
  every position (recurrent) and every chunk (chunkwise; ``05`` §7: the
  level-masked WY matrix intra-chunk, transported coarse levels
  inter-chunk).

Canonical cache form ("pre-arrival at position ``p``", spec §2.2):
``cache.state`` after consuming tokens ``[0, p)`` holds the buckets of
that history as token ``p`` will see them *before its own write* — every
merge or shift that happens at position ``p`` has already been applied,
and slot 0 holds bucket 0's pre-write content (zero for ``ll`` always,
and for ``ws``/``fs`` when ``p ≡ 0 mod B``).  Both paths end with "apply
the schedule for the next position", so they produce identical caches and
hand off trivially (tests M13, M14, M19).

Alignment policy (deterministic, not an automatic mode switch): in
``mode="chunk"`` the multi-state layouts require chunks to start at
``position ≡ 0 mod C``; a misaligned start is processed recurrently up to
the next multiple, and a ragged tail recurrently.  ``layout="single"``
calls :func:`functional.chunk_scan` directly (a short last chunk is fine
there).

Run tests with::

    pytest src/research_lib/layers/attention/gated_delta/tests/test_multistate.py -v
"""

import logging
from typing import NamedTuple

import torch
import torch.nn as nn

from research_lib.layers.attention.gated_delta.cache import GatedDeltaCache
from research_lib.layers.attention.gated_delta.functional import (
    ChunkLocals,
    GateMode,
    _compute_dtype,
    apply_gate_mode,
    chunk_local_terms,
    chunk_scan,
    recurrent_scan,
)
from research_lib.layers.attention.gated_delta.layouts import (
    Layout,
    ll_chunk_merge,
    ll_offset_level_matrix,
    ll_token_merge,
    num_slots,
    ws_shift,
)
from research_lib.layers.attention.gated_delta.mixing import LambdaMixer, LambdaMode
from research_lib.layers.attention.gated_delta.projections import GatedDeltaProjections

logger = logging.getLogger(__name__)


def _check_scan_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    lam: torch.Tensor,
    L: int,
) -> tuple[int, int]:
    """Shared shape validation for the multi-state scans; returns ``(B, T)``."""
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    if k.shape != q.shape or b.shape != q.shape or g.shape != q.shape:
        raise ValueError(
            "q, k, b, g must share the (B,T,H,dk) shape; got "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, b={tuple(b.shape)}, "
            f"g={tuple(g.shape)}"
        )
    if v.shape != (B, T, H, dv) or w.shape != (B, T, H, dv):
        raise ValueError(
            f"v and w must share the (B,T,H,dv) shape; got v={tuple(v.shape)}, "
            f"w={tuple(w.shape)}"
        )
    if lam.shape != (B, T, H, L):
        raise ValueError(
            f"lam must be (B,T,H,L) = {(B, T, H, L)}, got {tuple(lam.shape)}"
        )
    return B, T


def _multistate_recurrent_step(
    S: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    g_t: torch.Tensor,
    b_t: torch.Tensor,
    w_t: torch.Tensor,
    lam_t: torch.Tensor,
    frozen_slots: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One canonical recurrent step at position ``p``: transport, write, read.

    Args:
        S (Tensor): Canonical slot stack ``(B, H, L, d_k, d_v)`` *before*
            token ``p``'s transport.
        q_t/k_t/g_t/b_t (Tensor): ``(B, H, d_k)`` per-token tensors.
        v_t/w_t (Tensor): ``(B, H, d_v)`` per-token tensors.
        lam_t (Tensor): ``(B, H, L)`` λ for this token.
        frozen_slots (bool): ``True`` for layout ``fs`` (transport slot 0
            only; slots ``>= 1`` stay frozen).
        scale (float): Output scale.

    Returns:
        tuple: ``(o_t, S_after_write)`` — this token's output and the
        post-write slot stack (the caller applies the schedule for the
        next position to obtain the canonical cache).
    """
    # Transport, decay first (the locked convention: decay, then erase,
    # then write).  The erase is the *same* rank-1 update for every slot
    # (the write goes to slot 0 only), so it is computed once per slot as
    # an outer product against the decayed stack:
    #   A_t S^(l) = Diag(e^g) S^(l) - k ((b*k)^T Diag(e^g) S^(l))^T.
    # FS: slots >= 1 are frozen (no decay, no erase) and stay untouched.
    if frozen_slots:
        Sd0 = S[..., :1, :, :] * g_t.exp()[..., None, :, None]  # (B,H,1,dk,dv)
        read0 = (Sd0 * (b_t * k_t)[:, :, None, :, None]).sum(dim=-2)  # (B,H,1,dv)
        erase0 = (k_t[..., :, None] * read0).unsqueeze(-3)  # (B,H,1,dk,dv)
        z = (w_t * v_t)[..., None, :]  # (B,H,1,dv)
        write0 = (k_t[..., :, None] * z).unsqueeze(-3)  # (B,H,1,dk,dv)
        S_new = torch.cat([Sd0 - erase0 + write0, S[..., 1:, :, :]], dim=-3)
    else:
        Sd = S * g_t.exp()[..., None, :, None]  # decay all slots
        read = (Sd * (b_t * k_t)[:, :, None, :, None]).sum(dim=-2)  # (B,H,L,dv)
        erase = torch.einsum("bhk,bhlv->bhlkv", k_t, read)  # (B,H,L,dk,dv)
        Sd = Sd - erase  # the erase acts on every slot
        z = w_t * v_t  # the write lands in slot 0 only, with the raw z
        write0 = (k_t[..., :, None] * z[..., None, :]).unsqueeze(-3)  # (B,H,1,dk,dv)
        S_new = torch.cat([Sd[..., :1, :, :] + write0, Sd[..., 1:, :, :]], dim=-3)
    assert S_new.shape == S.shape, "slot-stack shape invariant"
    # Read after write, mixed with lambda: o = scale * sum_l lam_l (S_l^T q).
    # q: (B,H,dk) -> (B,H,1,dk,1); per-slot read (S_l^T q): sum over dk.
    per_slot = (S_new * q_t[:, :, None, :, None]).sum(dim=-2)  # (B, H, L, dv)
    o_t = scale * (lam_t[..., None] * per_slot).sum(dim=-2)
    return o_t, S_new


def multistate_recurrent_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    lam: torch.Tensor,
    layout: Layout | str,
    window: int,
    block_size: int,
    initial_state: torch.Tensor | None = None,
    position: int = 0,
    scale: float = 1.0,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token multi-state scan, canonical cache form.

    Per step at absolute position ``p`` (spec §2.3): transport every slot
    (FS: slot 0 only), write slot 0, read with λ, then apply the schedule
    for ``p + 1`` (``ws_shift`` at block boundaries; ``ll_token_merge`` at
    every position).  The returned state is the canonical cache ("the
    buckets as token ``p + 1`` sees them before its own write").

    Used for the recurrent path, for decoding, and (by
    :func:`multistate_chunk_scan`) for misaligned starts and ragged tails.
    The numerical core runs with autocast disabled in the compute dtype
    (at least fp32 by default; fp64 inputs stay fp64).

    Args:
        q (Tensor): ``(B, T, H, d_k)`` queries (caller normalizes).
        k (Tensor): ``(B, T, H, d_k)`` keys.
        v (Tensor): ``(B, T, H, d_v)`` values.
        g (Tensor): ``(B, T, H, d_k)`` log-decay.
        b (Tensor): ``(B, T, H, d_k)`` erase gate.
        w (Tensor): ``(B, T, H, d_v)`` write gate.
        lam (Tensor): ``(B, T, H, L)`` mixing weights (any nonnegative
            values; computed from the layer input by :class:`LambdaMixer`).
        layout (Layout | str): ``ws`` / ``fs`` / ``ll`` (``single`` goes
            through ``functional.chunk_scan`` / ``functional.recurrent_scan``).
        window (int): WS/FS window ``W`` (ignored for ``ll``).
        block_size (int): WS/FS block size ``B`` (ignored for ``ll``).
        initial_state (Tensor | None): Canonical slot stack
            ``(B, H, L, d_k, d_v)`` or ``None`` (zeros).
        position (int): Absolute position of the *first* token of this
            call (the schedule depends on it).
        scale (float): Output scale.
        compute_dtype (torch.dtype | None): Overrides the precision policy.

    Returns:
        tuple: ``(o, canonical_state)`` with ``o`` of shape ``(B, T, H, d_v)``
        in the input dtype and the canonical state of shape
        ``(B, H, L, d_k, d_v)`` in the compute dtype.
    """
    layout = Layout(layout)
    if layout is Layout.SINGLE:
        raise ValueError(
            "layout='single' uses functional.chunk_scan / functional.recurrent_scan"
        )
    L = lam.shape[-1]
    B, T = _check_scan_inputs(q, k, v, g, b, w, lam, L)
    H, dk, dv = q.shape[2], q.shape[3], v.shape[-1]
    cd = _compute_dtype(q, compute_dtype)
    with torch.autocast(device_type=q.device.type, enabled=False):
        q_ = q.transpose(1, 2).to(cd)
        k_ = k.transpose(1, 2).to(cd)
        v_ = v.transpose(1, 2).to(cd)
        g_ = g.transpose(1, 2).to(cd)
        b_ = b.transpose(1, 2).to(cd)
        w_ = w.transpose(1, 2).to(cd)
        lam_ = lam.transpose(1, 2).to(cd)
        if initial_state is None:
            S = q_.new_zeros(B, H, L, dk, dv)
        else:
            if initial_state.shape != (B, H, L, dk, dv):
                raise ValueError(
                    f"initial_state must be (B,H,L,dk,dv) = {(B, H, L, dk, dv)}, "
                    f"got {tuple(initial_state.shape)}"
                )
            S = initial_state.to(cd)
        frozen = layout is Layout.FS
        outputs = []
        for t in range(T):
            p = position + t
            o_t, S = _multistate_recurrent_step(
                S,
                q_[:, :, t],
                k_[:, :, t],
                v_[:, :, t],
                g_[:, :, t],
                b_[:, :, t],
                w_[:, :, t],
                lam_[:, :, t],
                frozen,
                scale,
            )
            outputs.append(o_t)
            # Schedule for the next position (canonical form): shifts and
            # merges commute with transport, so applying them here is the
            # verified convention (a block's state is frozen as of its last
            # token for FS).
            if layout in (Layout.WS, Layout.FS):
                if (p + 1) % block_size == 0:
                    S = ws_shift(S, window)
            else:  # LL: Fenwick carry at every arrival.
                S = ll_token_merge(S, p + 1)
        if outputs:
            o = torch.stack(outputs, dim=2)  # (B, H, T, dv)
        else:
            o = q_.new_zeros(B, H, 0, dv)
        o = o.transpose(1, 2).to(q.dtype)
        return o, S


def _ws_fs_chunk_step(
    S: torch.Tensor,
    q_c: torch.Tensor,
    loc: ChunkLocals,
    lam_c: torch.Tensor,
    layout: Layout,
    window: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One WS/FS chunk step (spec §2.4), from an aligned chunk start.

    ``X^(0) = NZ - W S^(0)`` (kept general even though the canonical cache
    has ``S^(0) = 0`` at a block boundary); ``X^(l>=1) = -W S^(l)`` (the
    erase only — writes land in bucket 0).  For ``fs``, the frozen slots
    are read with plain ``Q`` (nothing decays a frozen state) and not
    updated.  Ends with the shift (the schedule for the next position,
    which is a block boundary).

    Args:
        S (Tensor): Canonical slot stack ``(B, H, L, d_k, d_v)`` at the
            chunk start.
        q_c (Tensor): Chunk queries ``(B, H, C, d_k)`` (plain; used for
            the FS frozen reads).
        loc (ChunkLocals): Phase-1 quantities for this chunk.
        lam_c (Tensor): ``(B, H, C, L)`` λ for this chunk.
        layout (Layout): ``WS`` or ``FS``.
        window (int): Window ``W``.
        scale (float): Output scale.

    Returns:
        tuple: ``(O_chunk, S_next)`` — the chunk output ``(B, H, C, d_v)``
        and the canonical slot stack for the next position.
    """
    B, Hc, C, dk = q_c.shape
    # X^(0) = NZ - W S^(0); X^(l>=1) = -W S^(l).
    X0 = loc.NZ - loc.W @ S[..., 0, :, :]  # (B, H, C, dv)
    Xr = -(loc.W.unsqueeze(2) @ S[..., 1:, :, :])  # (B, H, L-1, C, dv)
    X = torch.cat([X0.unsqueeze(2), Xr], dim=2)  # (B, H, L, C, dv)
    # O^(l) = Qg S^(l) + Aqk X^(l);  S_next^(l) = Diag(gC) S^(l) + M^T X^(l).
    O_lvl = loc.Qg.unsqueeze(2) @ S + loc.Aqk.unsqueeze(2) @ X  # (B,H,L,C,dv)
    S_next = (
        loc.gC.view(B, Hc, 1, dk, 1) * S + loc.M.transpose(-1, -2).unsqueeze(2) @ X
    )  # (B, H, L, dk, dv)
    if layout is Layout.FS:
        # Frozen slots: plain read (Q, no decay, no erase), state unchanged.
        O_frozen = q_c.unsqueeze(2) @ S[..., 1:, :, :]  # (B, H, L-1, C, dv)
        O_lvl = torch.cat([O_lvl[..., :1, :, :], O_frozen], dim=-3)
        S_next = torch.cat([S_next[..., :1, :, :], S[..., 1:, :, :]], dim=-3)
    # Read-then-mix with lambda.
    O_chunk = scale * torch.einsum("bhcl,bhlcd->bhcd", lam_c, O_lvl)
    # The schedule for the next position (a block boundary).
    return O_chunk, ws_shift(S_next, window)


class LlChunkPrecomputed(NamedTuple):
    """State-independent per-chunk LL quantities, batched over full chunks.

    Computed once per call (Phase 1 for all chunks in one
    :func:`chunk_local_terms` call, then the derived quantities batched
    over the chunk axis), so the per-chunk Python loop only ever touches
    state-dependent tensors.  Index with ``[:, :, n]`` for chunk ``n``.

    Fields (all ``(B, H, N, ...)``):
        o_intra: ``(C, d_v)`` — the intra-chunk output
            ``((Aqk N) ⊙ Λ^H) Z`` (fine levels ``0..k``).
        r_mat: ``(C, d_k)`` — the coarse-level read operator
            ``R = Qg - Aqk W``.
        f_mat: ``(d_k, d_k)`` — the coarse-level transport
            ``F = Diag(γ^C) - M^T W``.
        s_chunk: ``(d_k, d_v)`` — the chunk's own content at the chunk end,
            ``M^T NZ`` (a level-``k+1`` bucket).
    """

    o_intra: torch.Tensor
    r_mat: torch.Tensor
    f_mat: torch.Tensor
    s_chunk: torch.Tensor


def _ll_precompute(
    loc_full: ChunkLocals,
    w_f: torch.Tensor,
    v_f: torch.Tensor,
    lam_full: torch.Tensor,
    levels: torch.Tensor,
) -> LlChunkPrecomputed:
    """Batch the state-independent LL quantities over all full chunks.

    One gather for ``Λ^H`` (from the in-chunk level matrix), one matmul
    for ``Aqk N``, and the intra-chunk output
    ``((Aqk N) ⊙ Λ^H) Z`` — each a single batched call over
    ``(B, H, N, ...)`` — plus the coarse-level read/transport operators
    and the chunk content.  The per-chunk loop then only does the
    state-dependent work (``05`` §7.3: coarse levels are
    transported-only buckets read through ``R``).

    Args:
        loc_full (ChunkLocals): Phase-1 terms for all chunks, ``(B, H, N,
            ...)``.
        w_f (Tensor): ``(B, H, N, C, d_v)`` chunk-major write gate.
        v_f (Tensor): ``(B, H, N, C, d_v)`` chunk-major values.
        lam_full (Tensor): ``(B, H, N, C, L)`` λ for all chunks.
        levels (Tensor): ``(C, C)`` in-chunk offset level matrix (levels of
            in-chunk pairs, which depend only on the offsets for an aligned
            chunk), ``-1`` above the diagonal, already on the target device.

    Returns:
        LlChunkPrecomputed: the batched quantities (see the NamedTuple).
    """
    B, Hc, N, C = loc_full.Aqk.shape[:4]
    idx = levels.clamp(min=0).view(1, 1, 1, C, C).expand(B, Hc, N, C, C)
    lam_h = lam_full.gather(-1, idx)  # (B,H,N,C,C)
    lam_h = lam_h * (levels >= 0).to(lam_h.dtype).view(1, 1, 1, C, C)  # 0 above diag
    z_full = w_f * v_f  # (B,H,N,C,dv)
    aqk_n = loc_full.Aqk @ loc_full.N  # (B,H,N,C,C) exact pairwise coefficients
    o_intra = (aqk_n * lam_h) @ z_full  # (B,H,N,C,dv)
    r_mat = loc_full.Qg - loc_full.Aqk @ loc_full.W  # (B,H,N,C,dk)
    dk = loc_full.Qg.shape[-1]
    eye = torch.eye(dk, dtype=loc_full.Qg.dtype, device=loc_full.Qg.device)
    f_mat = (
        loc_full.gC.unsqueeze(-1) * eye - loc_full.M.transpose(-1, -2) @ loc_full.W
    )  # (B,H,N,dk,dk)
    s_chunk = loc_full.M.transpose(-1, -2) @ loc_full.NZ  # (B,H,N,dk,dv)
    return LlChunkPrecomputed(
        o_intra=o_intra, r_mat=r_mat, f_mat=f_mat, s_chunk=s_chunk
    )


def _ll_chunk_step(
    S: torch.Tensor,
    pre: LlChunkPrecomputed,
    lam_c: torch.Tensor,
    position: int,
    chunk_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """State-dependent part of one LL chunk step (spec §2.4 / ``05`` §7).

    Every state-independent quantity (the intra-chunk output
    ``((Aqk N) ⊙ Λ^H) Z``, the coarse read operator ``R``, the coarse
    transport ``F``, the chunk's own content ``M^T NZ``) arrives
    precomputed and batched over chunks (:func:`_ll_precompute`); this
    step only touches state-dependent tensors.

    Args:
        S (Tensor): Canonical slot stack ``(B, H, L, d_k, d_v)`` at the
            chunk start (slots ``0..k`` are zero there, by the canonical
            form at a chunk boundary).
        pre (LlChunkPrecomputed): The precomputed quantities for this
            chunk (already sliced from the batched tuple).
        lam_c (Tensor): ``(B, H, C, L)`` λ for this chunk.
        position (int): Absolute position of the chunk start.
        chunk_size (int): ``C = 2^k``.
        scale (float): Output scale.

    Returns:
        tuple: ``(O_chunk, S_next)`` — the chunk output ``(B, H, C, d_v)``
        and the canonical slot stack for the next position.
    """
    k = chunk_size.bit_length() - 1
    # ---- inter-chunk: coarse levels are transported-only buckets -------
    O_inter = torch.einsum(
        "bhcl,bhlcd->bhcd",
        lam_c[..., k + 1 :],
        pre.r_mat.unsqueeze(2) @ S[..., k + 1 :, :, :],
    )
    O_chunk = scale * (pre.o_intra + O_inter)
    # ---- end of chunk: transport, then a Fenwick step over chunks ------
    S = torch.cat(
        [S[..., : k + 1, :, :], pre.f_mat.unsqueeze(2) @ S[..., k + 1 :, :, :]],
        dim=-3,
    )
    S = ll_chunk_merge(S, pre.s_chunk, (position + chunk_size) // chunk_size, k)
    return O_chunk, S


def multistate_chunk_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    lam: torch.Tensor,
    layout: Layout | str,
    window: int,
    block_size: int,
    initial_state: torch.Tensor | None = None,
    position: int = 0,
    chunk_size: int = 16,
    scale: float = 1.0,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunkwise multi-state scan with the deterministic alignment policy.

    ``ws``/``fs`` require ``block_size == chunk_size`` (validated by the
    layer; asserted here per chunk), so every chunk starts at a block
    boundary where the canonical slot 0 is zero.  ``ll`` requires
    ``chunk_size = 2^k``.  A misaligned start (``position % chunk_size !=
    0``) is processed recurrently up to the next multiple of ``chunk_size``,
    and a ragged tail recurrently — a documented path, not an automatic
    mode switch.  The returned state is the canonical cache, identical to
    :func:`multistate_recurrent_scan`'s.

    The numerical core runs with autocast disabled in the compute dtype
    (at least fp32 by default; fp64 inputs stay fp64).

    Args:
        q (Tensor): ``(B, T, H, d_k)`` queries.
        k (Tensor): ``(B, T, H, d_k)`` keys.
        v (Tensor): ``(B, T, H, d_v)`` values.
        g (Tensor): ``(B, T, H, d_k)`` log-decay.
        b (Tensor): ``(B, T, H, d_k)`` erase gate.
        w (Tensor): ``(B, T, H, d_v)`` write gate.
        lam (Tensor): ``(B, T, H, L)`` mixing weights.
        layout (Layout | str): ``ws`` / ``fs`` / ``ll``.
        window (int): WS/FS window ``W`` (ignored for ``ll``).
        block_size (int): WS/FS block size ``B`` (ignored for ``ll``).
        initial_state (Tensor | None): Canonical slot stack
            ``(B, H, L, d_k, d_v)`` or ``None`` (zeros).
        position (int): Absolute position of the first token of this call.
        chunk_size (int): Chunk length ``C``.
        scale (float): Output scale.
        compute_dtype (torch.dtype | None): Overrides the precision policy.

    Returns:
        tuple: ``(o, canonical_state)`` with ``o`` of shape ``(B, T, H, d_v)``
        in the input dtype and the canonical state of shape
        ``(B, H, L, d_k, d_v)`` in the compute dtype.
    """
    layout = Layout(layout)
    if layout is Layout.SINGLE:
        raise ValueError(
            "layout='single' uses functional.chunk_scan / functional.recurrent_scan"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if layout is Layout.LL and chunk_size & (chunk_size - 1):
        raise ValueError(
            f"layout='ll' requires chunk_size to be a power of two, got {chunk_size}"
        )
    if layout in (Layout.WS, Layout.FS) and block_size != chunk_size:
        raise ValueError(
            f"layout={layout.value!r} requires block_size == chunk_size, got "
            f"block_size={block_size}, chunk_size={chunk_size}"
        )
    L = lam.shape[-1]
    B, T = _check_scan_inputs(q, k, v, g, b, w, lam, L)
    H, dk, dv = q.shape[2], q.shape[3], v.shape[-1]
    cd = _compute_dtype(q, compute_dtype)
    with torch.autocast(device_type=q.device.type, enabled=False):
        if initial_state is None:
            S = q.new_zeros(B, H, L, dk, dv, dtype=cd)
        else:
            if initial_state.shape != (B, H, L, dk, dv):
                raise ValueError(
                    f"initial_state must be (B,H,L,dk,dv) = {(B, H, L, dk, dv)}, "
                    f"got {tuple(initial_state.shape)}"
                )
            S = initial_state.to(cd)
        pos = position
        out_parts: list[torch.Tensor] = []
        # ---- head: recurrently to the next chunk boundary ----------------
        # Capped at T: a call whose input ends before the next boundary
        # consumes the whole input recurrently.  Without the cap,
        # n_head > T would advance `pos` past the input, n_full would go
        # negative (and -1 is truthy, so the full-chunk reshape would run
        # and raise), and t_tail = -1 would reprocess the last token.
        n_head = min((-pos) % chunk_size, T)
        if n_head:
            o_h, S = multistate_recurrent_scan(
                q[:, :n_head],
                k[:, :n_head],
                v[:, :n_head],
                g[:, :n_head],
                b[:, :n_head],
                w[:, :n_head],
                lam[:, :n_head],
                layout,
                window,
                block_size,
                initial_state=S,
                position=pos,
                scale=scale,
                compute_dtype=compute_dtype,
            )
            out_parts.append(o_h)
            pos += n_head
        # ---- full chunks: Phase 1 batched over all of them -----------------
        # One chunk_local_terms call on (B, H, N, C, ·) (as functional.chunk_scan
        # does): transpose, cast, then split the time axis into chunks (the
        # reshape copies the non-contiguous transpose). The state-independent
        # LL quantities are batched over the chunk axis too (_ll_precompute),
        # so the per-chunk loop below only ever touches state-dependent tensors.
        n_full = (T - n_head) // chunk_size
        if n_full:
            t0 = n_head
            t1 = n_head + n_full * chunk_size
            sl = slice(t0, t1)
            q_f = q[:, sl].transpose(1, 2).to(cd).reshape(B, H, n_full, chunk_size, dk)
            k_f = k[:, sl].transpose(1, 2).to(cd).reshape(B, H, n_full, chunk_size, dk)
            g_f = g[:, sl].transpose(1, 2).to(cd).reshape(B, H, n_full, chunk_size, dk)
            b_f = b[:, sl].transpose(1, 2).to(cd).reshape(B, H, n_full, chunk_size, dk)
            v_f = v[:, sl].transpose(1, 2).to(cd).reshape(B, H, n_full, chunk_size, dv)
            w_f = w[:, sl].transpose(1, 2).to(cd).reshape(B, H, n_full, chunk_size, dv)
            loc_full = chunk_local_terms(q_f, k_f, v_f, g_f, b_f, w_f, compute_dtype)
            lam_full = (
                lam[:, sl].transpose(1, 2).to(cd).reshape(B, H, n_full, chunk_size, L)
            )  # (B,H,N,C,L)
            pre_full: LlChunkPrecomputed | None = None
            if layout is Layout.LL:
                # Canonical-form check, ONCE per call (it used to live in
                # the per-chunk step, costing a device sync per chunk):
                # the state entering the FIRST full chunk (after the
                # recurrent head) must have zero fine slots 0..k — they are
                # intra-chunk quantities, recomputed from a zero state each
                # chunk (05 §7.1), so a nonzero cache here is not in the
                # canonical form (rule 8: assert, never silently ignore).
                # Canonical-form check, ONCE per call, and only where it
                # can fail: on a caller-supplied cache entering the first
                # full chunk directly (initial_state given, no recurrent
                # head). A fresh state is zeros by construction, and a
                # state produced by the recurrent head is canonical by
                # construction (its merges run at every arrival), so
                # checking either would be a device readback for nothing.
                # After the first chunk the invariant holds by
                # construction anyway: the chunk step only touches the
                # coarse slots (>= k+1) and the merge zeroes the folded
                # coarse slots; the fine slots pass through untouched.
                k_lvl = chunk_size.bit_length() - 1
                if initial_state is not None and n_head == 0:
                    fine = S[..., : k_lvl + 1, :, :]
                    if fine.numel() and fine.abs().max() > 0:
                        raise ValueError(
                            "layout='ll' chunk path requires the canonical "
                            f"cache form at a chunk start: slots 0..{k_lvl} "
                            "(the intra-chunk levels) must be zero, got "
                            "nonzero entries; continue from a cache "
                            "produced by a previous call (or zeros)"
                        )
                # In-chunk levels depend only on the offsets for an aligned
                # chunk (05 §7.1): fine levels come from the low k bits —
                # which are the bits of the offsets, since p0 ≡ 0 (mod C) —
                # and coarse levels never contain tokens of the current
                # chunk. One cached (C, C) offset matrix per chunk size
                # (ll_offset_level_matrix), moved to the device once for
                # this call; never a per-chunk (or per-entry) computation.
                # non_blocking: the cached source tensor is never mutated,
                # so the H2D copy may stay off the sync path (a blocking
                # .to would stream-sync once per call; see sync_check).
                levels = ll_offset_level_matrix(chunk_size).to(
                    q.device, non_blocking=True
                )
                pre_full = _ll_precompute(loc_full, w_f, v_f, lam_full, levels)
            for n in range(n_full):
                p0 = position + t0 + n * chunk_size  # absolute chunk start
                assert p0 % chunk_size == 0, "chunk alignment invariant"
                loc_n = ChunkLocals(*(f[:, :, n] for f in loc_full))  # (B,H,C,·)
                lam_c = lam_full[:, :, n]  # (B,H,C,L)
                if layout in (Layout.WS, Layout.FS):
                    O_c, S = _ws_fs_chunk_step(
                        S, q_f[:, :, n], loc_n, lam_c, layout, window, scale
                    )
                else:
                    assert pre_full is not None  # set for LL just above
                    pre_n = LlChunkPrecomputed(
                        o_intra=pre_full.o_intra[:, :, n],
                        r_mat=pre_full.r_mat[:, :, n],
                        f_mat=pre_full.f_mat[:, :, n],
                        s_chunk=pre_full.s_chunk[:, :, n],
                    )
                    O_c, S = _ll_chunk_step(S, pre_n, lam_c, p0, chunk_size, scale)
                out_parts.append(O_c.transpose(1, 2))  # (B, C, H, dv)
                pos += chunk_size
        # ---- ragged tail: recurrently --------------------------------------
        t_tail = n_head + n_full * chunk_size
        if t_tail < T:
            o_t, S = multistate_recurrent_scan(
                q[:, t_tail:],
                k[:, t_tail:],
                v[:, t_tail:],
                g[:, t_tail:],
                b[:, t_tail:],
                w[:, t_tail:],
                lam[:, t_tail:],
                layout,
                window,
                block_size,
                initial_state=S,
                position=position + t_tail,
                scale=scale,
                compute_dtype=compute_dtype,
            )
            out_parts.append(o_t)
        if out_parts:
            # The head/tail parts are already in the input dtype (the
            # recurrent scan casts); the full-chunk parts are in the
            # compute dtype.  Cast the concatenation to the input dtype —
            # the documented contract, and what the recurrent scan
            # returns (for bf16 inputs the cat would otherwise promote
            # to fp32).
            o = torch.cat(out_parts, dim=1).to(q.dtype)
        else:
            o = q.new_zeros(B, 0, H, dv)
        return o, S


class MultistateGatedDelta(nn.Module):
    r"""Multi-state gated delta-rule token mixer (WS / FS / Fenwick layouts).

    Implements the transported multi-state memory of
    ``05_derivation_multi_state.md``: a stack of ``L`` state buckets
    ``(B, H, L, d_k, d_v)``, every bucket transported by the GDN-2
    transition, writes landing in slot 0 only, merged per the layout's
    schedule, and read out per token as

    .. math::

        o_t = \mathrm{scale} \cdot \sum_\ell \lambda_t^{(h,\ell)}
              (S_t^{(\ell)})^T q_t

    followed by the shared output head (per-head RMSNorm × output gate ×
    output projection).  At ``λ ≡ 1`` the layer is *exactly* GDN-2 (the
    bucket states sum to the single state, by linearity), which is the
    initialization (D9) and the M06 reduction test.

    Variants are init args (one class, no subclasses): ``gate_mode``
    (GDN-2 / Table 5 ablations / KDA / GDN), ``layout``
    (``single`` / ``ws`` / ``fs`` / ``ll``), ``lambda_mode``
    (``static`` / ``lin`` / ``mlp``).  ``layout="single"`` has no λ module
    and is functionally identical to
    :class:`~research_lib.layers.attention.gated_delta.reference.GatedDeltaReference`.

    **Reference implementation** (naive PyTorch + autograd): correctness
    baselines and small synthetic experiments, not production-scale
    training — for that use fla's kernels.

    Args:
        hidden_size (int): Model dimension :math:`d`.
        num_heads (int): Number of heads ``H`` (the pilot uses
            :math:`H_v = H`).
        head_dim (int): Key head dimension :math:`d_k`.
        expand_v (float): :math:`d_v = d_k \cdot \mathrm{expand\_v}`.
            Defaults to 1.0.
        gate_mode (str | GateMode): ``"full"`` (GDN-2, default),
            ``"b_only"``, ``"w_only"``, ``"kda"``, ``"gdn"``.
        layout (str | Layout): ``"ws"`` (default), ``"fs"``, ``"ll"``,
            ``"single"`` (the GDN-2 reference function).
        lambda_mode (str | LambdaMode): ``"static"`` / ``"lin"`` /
            ``"mlp"`` (default).  Ignored for ``layout="single"``.
        lambda_mlp_hidden (int | None): MLP hidden width (defaults to
            ``hidden_size``).
        window (int): WS/FS window ``W``. Defaults to 8.
        block_size (int | None): WS/FS block size ``B``; ``None`` means
            ``chunk_size`` (so ``B = C``, D13).
        t_max (int): Longest total sequence the cache will hold (the
            longest **eval** length), used only for ``ll``'s slot count
            ``bitlen(t_max) + 1`` (D23). Defaults to 4096.
        mode (str): Forward path: ``"chunk"`` (default) or ``"recurrent"``;
            overridable per call.  No automatic switching.
        chunk_size (int): Chunk length ``C``. Defaults to 16.
        use_short_conv (bool): Short causal conv (k=4, SiLU) on
            ``q``/``k``/``v``. Defaults to ``True``.
        conv_size (int): Short-conv kernel size. Defaults to 4.
        output_gate_activation (str): ``"silu"`` (paper, default) or
            ``"sigmoid"`` (fla parity).
        low_rank_gates (bool): Low-rank log-decay projection. Defaults to
            ``False``.
        scale (float | None): Output scale; ``None`` means
            :math:`d_k^{-1/2}` (the per-head output RMSNorm cancels any
            constant scale).
        norm_eps (float): Epsilon for the per-head output RMSNorm. Defaults
            to 1e-5.
        init_scheme (str): ``"paper"`` (default).
        collect_stats (bool): If ``True``, each forward stores
            ``self.latest_stats`` (λ mean/std per slot and the erase
            strength) under ``torch.no_grad``. Defaults to ``False``.
        compute_dtype (torch.dtype | None): Overrides the precision policy
            of the recurrence core. Defaults to ``None`` (at least fp32).
        device: Optional torch device.
        dtype: Optional torch dtype.

    Shape:
        - Input: :math:`(B, T, d)`
        - Output: :math:`(B, T, d)`

    Examples::

        >>> layer = MultistateGatedDelta(hidden_size=64, num_heads=4,
        ...                             head_dim=16, layout="ll", t_max=256)
        >>> y, cache = layer(torch.randn(2, 32, 64), use_cache=True)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        expand_v: float = 1.0,
        gate_mode: str = "full",
        layout: str = "ws",
        lambda_mode: str = "mlp",
        lambda_mlp_hidden: int | None = None,
        window: int = 8,
        block_size: int | None = None,
        t_max: int = 4096,
        mode: str = "chunk",
        chunk_size: int = 16,
        use_short_conv: bool = True,
        conv_size: int = 4,
        output_gate_activation: str = "silu",
        low_rank_gates: bool = False,
        scale: float | None = None,
        norm_eps: float = 1e-5,
        init_scheme: str = "paper",
        collect_stats: bool = False,
        compute_dtype: torch.dtype | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if mode not in ("chunk", "recurrent"):
            raise ValueError(f"mode must be 'chunk' or 'recurrent', got {mode!r}")
        try:
            self.layout = Layout(layout)
        except ValueError as e:
            raise ValueError(
                f"layout must be one of {[item.value for item in Layout]}, got {layout!r}"
            ) from e
        try:
            self.gate_mode = GateMode(gate_mode)
        except ValueError as e:
            raise ValueError(
                f"gate_mode must be one of {[m.value for m in GateMode]}, "
                f"got {gate_mode!r}"
            ) from e
        if self.layout is not Layout.SINGLE:
            try:
                lm = LambdaMode(lambda_mode)
            except ValueError as e:
                raise ValueError(
                    f"lambda_mode must be one of {[m.value for m in LambdaMode]}, "
                    f"got {lambda_mode!r}"
                ) from e
            self.lambda_mode = lm
        else:
            self.lambda_mode = None  # type: ignore[assignment]
        self.mode = mode
        self.chunk_size = chunk_size
        self.window = window
        self.block_size = chunk_size if block_size is None else block_size
        self.t_max = t_max
        self.scale = 1.0 / (head_dim**0.5) if scale is None else float(scale)
        self.compute_dtype = compute_dtype
        self.collect_stats = collect_stats
        self.latest_stats: dict | None = None

        if self.layout is Layout.SINGLE:
            self.num_slots = 1
        else:
            self.num_slots = num_slots(self.layout, window=window, t_max=t_max)
        if self.layout in (Layout.WS, Layout.FS):
            if self.block_size != chunk_size:
                raise ValueError(
                    f"layout={self.layout.value!r} requires block_size == chunk_size"
                    f" (got block_size={self.block_size}, chunk_size={chunk_size}); "
                    "the chunk must start at a block boundary so slot 0 is zero"
                )
        elif self.layout is Layout.LL:
            if chunk_size & (chunk_size - 1):
                raise ValueError(
                    f"layout='ll' requires chunk_size to be a power of two, got "
                    f"{chunk_size} (the Fenwick levels must align with chunks, "
                    "'05' §7.1)"
                )
            k = chunk_size.bit_length() - 1
            if self.num_slots < k + 2:
                raise ValueError(
                    f"layout='ll' with chunk_size={chunk_size} needs at least "
                    f"{k + 2} slots (levels 0..{k + 1}), but t_max={t_max} gives "
                    f"{self.num_slots}; increase t_max"
                )

        # Same submodule name as GatedDeltaReference so state dicts are
        # interchangeable (test M06).
        self.projections = GatedDeltaProjections(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            expand_v=expand_v,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            output_gate_activation=output_gate_activation,
            low_rank_gates=low_rank_gates,
            norm_eps=norm_eps,
            init_scheme=init_scheme,
            device=device,
            dtype=dtype,
        )
        if self.layout is Layout.SINGLE:
            self.lambda_mixer = None
        else:
            # LambdaMixer accepts the plain string (validated above).
            self.lambda_mixer = LambdaMixer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_levels=self.num_slots,
                mode=lambda_mode,
                mlp_hidden=lambda_mlp_hidden,
                device=device,
                dtype=dtype,
            )

    def state_bytes(self) -> int:
        """Recurrent-state bytes of this layer.

        The canonical slot stack in fp32:
        ``num_heads * num_slots * d_k * d_v * 4`` (the single-state
        layer is the ``L = 1`` case).

        Returns:
            int: Bytes of one layer's recurrent state.
        """
        proj = self.projections
        return proj.num_heads * self.num_slots * proj.head_k_dim * proj.head_v_dim * 4

    def extra_repr(self) -> str:
        return (
            f"layout={self.layout.value}, gate_mode={self.gate_mode.value}, "
            f"lambda_mode={self.lambda_mode.value if self.lambda_mode else None}, "
            f"num_slots={self.num_slots}, window={self.window}, "
            f"block_size={self.block_size}, chunk_size={self.chunk_size}, "
            f"mode={self.mode}"
        )

    def _maybe_collect_stats(
        self, b: torch.Tensor, k: torch.Tensor, lam: torch.Tensor | None
    ) -> None:
        """Store λ per-slot statistics and the erase strength (flag 3(b),(c))."""
        if not self.collect_stats:
            return
        with torch.no_grad():
            stats: dict = {
                # mean over tokens of sum_i b_{t,i} k_{t,i}^2, after
                # gate-mode scalarization; scalar.
                "erase_strength": (b * k.square())
                .sum(dim=-1)
                .mean()
                .detach(),
            }
            if lam is not None:
                stats["lambda_mean"] = lam.mean(dim=(0, 1)).detach()  # (H, L)
                stats["lambda_std"] = lam.std(dim=(0, 1)).detach()  # (H, L)
            self.latest_stats = stats

    def forward(
        self,
        x: torch.Tensor,
        cache: GatedDeltaCache | None = None,
        use_cache: bool = False,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, GatedDeltaCache | None]:
        """Run the layer.

        Args:
            x (Tensor): Input of shape :math:`(B, T, d)`.
            cache (GatedDeltaCache | None): Cache from a previous call
                (canonical state + conv caches + position), or ``None``.
            use_cache (bool): If ``True``, return the updated cache.
            mode (str | None): Overrides ``self.mode`` for this call
                (``"chunk"`` or ``"recurrent"``).

        Returns:
            tuple: ``(y, new_cache)``; ``new_cache`` is ``None`` unless
            ``use_cache=True``.  The cache state is the canonical slot
            stack ``(B, H, L, d_k, d_v)`` (``(B, H, d_k, d_v)`` for
            ``layout="single"``), in the compute dtype.
        """
        path = self.mode if mode is None else mode
        if path not in ("chunk", "recurrent"):
            raise ValueError(f"mode must be 'chunk' or 'recurrent', got {path!r}")
        B = x.shape[0]
        H, dk, dv = (
            self.projections.num_heads,
            self.projections.head_k_dim,
            self.projections.head_v_dim,
        )
        start_pos = cache.position if cache is not None else 0
        if cache is not None:
            if self.layout is Layout.SINGLE:
                expected = (B, H, dk, dv)
            else:
                expected = (B, H, self.num_slots, dk, dv)
            if cache.state.shape != expected:
                raise ValueError(
                    f"cache.state must be {expected}, got {tuple(cache.state.shape)}"
                )
        q, k, v, b, w, g, conv_cache = self.projections.project(
            x,
            conv_cache=cache.conv_state if cache is not None else None,
            use_cache=use_cache,
        )
        b, w, g = apply_gate_mode(b, w, g, self.gate_mode)
        lam = None
        if self.layout is Layout.SINGLE:
            initial_state = cache.state if cache is not None else None
            if path == "chunk":
                o, S = chunk_scan(
                    q,
                    k,
                    v,
                    g,
                    b,
                    w,
                    initial_state=initial_state,
                    chunk_size=self.chunk_size,
                    scale=self.scale,
                    compute_dtype=self.compute_dtype,
                )
            else:
                o, S = recurrent_scan(
                    q,
                    k,
                    v,
                    g,
                    b,
                    w,
                    initial_state=initial_state,
                    scale=self.scale,
                    compute_dtype=self.compute_dtype,
                )
        else:
            assert self.lambda_mixer is not None  # only None for layout='single'
            lam = self.lambda_mixer(x)
            initial_state = cache.state if cache is not None else None
            if path == "chunk":
                o, S = multistate_chunk_scan(
                    q,
                    k,
                    v,
                    g,
                    b,
                    w,
                    lam,
                    self.layout,
                    self.window,
                    self.block_size,
                    initial_state=initial_state,
                    position=start_pos,
                    chunk_size=self.chunk_size,
                    scale=self.scale,
                    compute_dtype=self.compute_dtype,
                )
            else:
                o, S = multistate_recurrent_scan(
                    q,
                    k,
                    v,
                    g,
                    b,
                    w,
                    lam,
                    self.layout,
                    self.window,
                    self.block_size,
                    initial_state=initial_state,
                    position=start_pos,
                    scale=self.scale,
                    compute_dtype=self.compute_dtype,
                )
        self._maybe_collect_stats(b, k, lam)
        y = self.projections.output(o, x)
        new_cache = None
        if use_cache:
            new_cache = GatedDeltaCache(
                state=S,
                conv_state=conv_cache,
                position=start_pos + x.shape[1],
            )
        return y, new_cache
