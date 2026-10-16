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
    transition,
)
from research_lib.layers.attention.gated_delta.layouts import (
    Layout,
    ll_chunk_merge,
    ll_level_matrix,
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


def _ll_chunk_step(
    S: torch.Tensor,
    loc: ChunkLocals,
    Z: torch.Tensor,
    lam_c: torch.Tensor,
    position: int,
    chunk_size: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One LL chunk step (spec §2.4 / ``05`` §7), from an aligned chunk start.

    Intra-chunk (fine levels ``0..k``): the exact pairwise coefficient
    matrix ``Aqk N`` masked by the level-λ matrix ``Λ^H`` applied to the
    chunk writes ``Z = w ⊙ v``.  Inter-chunk (coarse levels ``>= k+1``):
    transported-only buckets read with ``R = Qg - Aqk W``.  At the chunk
    end: transport the coarse levels with ``F = Diag(gC) - M^T W``, then a
    Fenwick carry on the chunk counter folds this chunk's content
    (``M^T NZ``, a level-``k+1`` bucket) into the coarse levels.

    Args:
        S (Tensor): Canonical slot stack ``(B, H, L, d_k, d_v)`` at the
            chunk start (slots ``0..k`` are zero there, by the canonical
            form at a chunk boundary).
        loc (ChunkLocals): Phase-1 quantities for this chunk.
        Z (Tensor): Chunk writes ``(B, H, C, d_v)`` (``w ⊙ v``).
        lam_c (Tensor): ``(B, H, C, L)`` λ for this chunk.
        position (int): Absolute position of the chunk start.
        chunk_size (int): ``C = 2^k``.
        scale (float): Output scale.

    Returns:
        tuple: ``(O_chunk, S_next)`` — the chunk output ``(B, H, C, d_v)``
        and the canonical slot stack for the next position.
    """
    B, Hc, C = loc.Aqk.shape[:3]
    dk = loc.Qg.shape[-1]
    k = chunk_size.bit_length() - 1
    # Canonical-form invariant: at a chunk boundary the fine levels
    # (slots 0..k) are zero — they are intra-chunk quantities, recomputed
    # from a zero state each chunk (05 §7.1).  A cache violating this is
    # not in the canonical form (rule 8: assert, never silently ignore).
    if (S[..., : k + 1, :, :].abs().max() if S.numel() else 0) > 0:
        raise ValueError(
            "layout='ll' chunk path requires the canonical cache form at a "
            f"chunk start: slots 0..{k} (the intra-chunk levels) must be "
            "zero, got nonzero entries; continue from a cache produced by "
            "a previous call (or zeros)"
        )
    # ---- intra-chunk: the H mask on the WY-transformed matrix ----------
    positions = position + torch.arange(C, device=S.device)
    levels = ll_level_matrix(positions)  # (C, C), -1 above the diagonal
    idx = levels.clamp(min=0).view(1, 1, C, C).expand(B, Hc, C, C)
    LamH = lam_c.gather(-1, idx)  # (B, H, C, C)
    LamH = LamH * (levels >= 0).to(LamH.dtype).view(1, 1, C, C)  # 0 above diag
    AqkN = loc.Aqk @ loc.N  # (B, H, C, C) exact pairwise coefficients
    O_intra = (AqkN * LamH) @ Z  # (B, H, C, dv)
    # ---- inter-chunk: coarse levels are transported-only buckets -------
    R = loc.Qg - loc.Aqk @ loc.W  # (B, H, C, dk)
    O_inter = torch.einsum(
        "bhcl,bhlcd->bhcd",
        lam_c[..., k + 1 :],
        R.unsqueeze(2) @ S[..., k + 1 :, :, :],
    )
    O_chunk = scale * (O_intra + O_inter)
    # ---- end of chunk: transport, then a Fenwick step over chunks ------
    eye = torch.eye(dk, dtype=S.dtype, device=S.device)
    F = loc.gC.unsqueeze(-1) * eye - loc.M.transpose(-1, -2) @ loc.W  # (B,H,dk,dk)
    S = torch.cat(
        [S[..., : k + 1, :, :], F.unsqueeze(2) @ S[..., k + 1 :, :, :]], dim=-3
    )
    S_chunk = loc.M.transpose(-1, -2) @ loc.NZ  # (B, H, dk, dv)
    S = ll_chunk_merge(S, S_chunk, (position + chunk_size) // chunk_size, k)
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
        n_head = (-pos) % chunk_size
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
        # ---- full chunks ---------------------------------------------------
        n_full = (T - n_head) // chunk_size
        for n in range(n_full):
            t0 = n_head + n * chunk_size  # local index within this call
            p0 = position + t0  # absolute chunk start
            assert p0 % chunk_size == 0, "chunk alignment invariant"
            sl = slice(t0, t0 + chunk_size)
            q_c = q[:, sl].transpose(1, 2).to(cd).unsqueeze(2)  # (B,H,1,C,dk)
            k_c = k[:, sl].transpose(1, 2).to(cd).unsqueeze(2)
            v_c = v[:, sl].transpose(1, 2).to(cd).unsqueeze(2)
            g_c = g[:, sl].transpose(1, 2).to(cd).unsqueeze(2)
            b_c = b[:, sl].transpose(1, 2).to(cd).unsqueeze(2)
            w_c = w[:, sl].transpose(1, 2).to(cd).unsqueeze(2)
            loc = chunk_local_terms(q_c, k_c, v_c, g_c, b_c, w_c, compute_dtype)
            loc = ChunkLocals(*(f[:, :, 0] for f in loc))  # squeeze N=1
            lam_c = lam[:, sl].transpose(1, 2).to(cd)  # (B,H,C,L)
            if layout in (Layout.WS, Layout.FS):
                O_c, S = _ws_fs_chunk_step(
                    S, q_c[:, :, 0], loc, lam_c, layout, window, scale
                )
            else:
                Z = w_c[:, :, 0] * v_c[:, :, 0]  # (B, H, C, dv)
                O_c, S = _ll_chunk_step(S, loc, Z, lam_c, p0, chunk_size, scale)
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
            o = torch.cat(out_parts, dim=1)
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
