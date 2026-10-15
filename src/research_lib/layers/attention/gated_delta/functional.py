"""Pure functional core for the gated delta-rule family (GDN-2 / KDA / Gated DeltaNet).

This module is the **single source of truth** for the recurrence math (see
``impl/01_spec_gated_delta.md`` §4).  Both :class:`~research_lib.layers.attention.gated_delta.reference.GatedDeltaReference`
(single state, PR 1) and the multi-state layer (PR 2) call these functions;
neither re-implements the math.

The recurrence (per batch element and head), with ``q_t, k_t`` L2-normalized:

.. math::

    S_t = (I - k_t (b_t \\odot k_t)^T)\\, \\mathrm{Diag}(e^{g_t})\\, S_{t-1}
          + k_t (w_t \\odot v_t)^T,
    \\qquad o_t = \\mathrm{scale} \\cdot S_t^T q_t

- :math:`g_t \\le 0` is the log-decay (:math:`d_k` channels),
  :math:`b_t \\in [0,1]^{d_k}` the erase gate, :math:`w_t \\in [0,1]^{d_v}`
  the write gate (GDN-2, arXiv:2605.22791, Eq. 10).
- **Read after write**: :math:`o_t` uses :math:`S_t` (token ``t``'s own write
  included).
- **Decay, then erase, then write**.
- Equivalent working form (used in code):
  :math:`S' = \\mathrm{Diag}(e^{g_t}) S_{t-1}`,
  :math:`u_t = w_t \\odot v_t - S'^T (b_t \\odot k_t)`,
  :math:`S_t = S' + k_t u_t^T`.

Gate ablations (GDN-2 paper Table 5) scalarize the *channel* structure at
runtime while keeping the projections unchanged; :func:`apply_gate_mode`
implements the table.

Layout conventions (fla's):
    - per-token tensors: ``(B, T, H, D)``;
    - states: ``(B, H, d_k, d_v)``;
    - internally the scans transpose to ``(B, H, T, D)`` and chunk-major
      ``(B, H, N, C, D)``.

Precision policy (explicit in code, never left to type promotion):
    gates, log-decay, cumulative sums, every ``exp``, the L2 normalization
    and the triangular solve run in ``max(fp32, input dtype)`` (fp64 in
    tests); the recurrent state is fp32 (fp64 in tests); outputs are cast
    back to the input dtype at the end.  See :func:`_compute_dtype`.  The
    scan functions additionally disable ``torch.autocast`` around their
    numerical cores, so bf16-mixed training cannot silently lower the
    recurrence's precision (the paper's App. D and fla's kernels keep the
    core in fp32); a ``compute_dtype`` argument overrides the policy for
    degradation experiments.

Log space by default: :math:`\\mathbf A_{qk}` and :math:`\\mathbf L` are built
by subtracting cumulative log-decays, masking disallowed entries with
``-inf`` *before* ``exp``, and contracting (never form ``K / exp(G)``).
The teaching split form is allowed only inside tests.

Reference for the chunk form: ``04_derivation_single_state.md`` §2.11
(Eqs. 20–25 of the paper in the guide's letters) and
``impl/verify_derivations.py`` (the executable reference).
"""

import logging
from enum import Enum
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)


def _compute_dtype(
    x: torch.Tensor, compute_dtype: torch.dtype | None = None
) -> torch.dtype:
    """Resolve the dtype every precision-sensitive operation runs in.

    With ``compute_dtype=None`` (the default policy), fp64 stays fp64 (used
    by the fp64 test suite) and fp32, bf16 and fp16 are promoted to fp32, so
    the recurrence core always runs in **at least fp32**, as in the GDN-2
    paper (App. D.1-D.4) and fla's kernels.

    Args:
        x (Tensor): Any tensor whose dtype defines the default policy.
        compute_dtype (torch.dtype | None): Explicit override of the policy
            (e.g. ``torch.bfloat16`` for degradation experiments).  The
            caller owns the numerical consequences; there is no checking
            that the override is *wider* than the input dtype.

    Returns:
        torch.dtype: ``compute_dtype`` if given, else ``max(fp32, x.dtype)``.
    """
    if compute_dtype is not None:
        return compute_dtype
    if x.dtype == torch.float64:
        return torch.float64
    return torch.float32


class GateMode(str, Enum):
    """Runtime gate treatment; all five modes share the same projections.

    Modes (GDN-2 paper Table 5, plus the KDA / GDN anchors):

    - ``full``: gates as projected (GDN-2 itself).
    - ``b_only``: erase gate keeps its channels, write gate is scalarized
      (``mean`` over :math:`d_v`, broadcast back).
    - ``w_only``: write gate keeps its channels, erase gate is scalarized
      (``mean`` over :math:`d_k`, broadcast back).
    - ``kda``: both gates tied to the same scalar
      :math:`\\beta = \\mathrm{mean}_c(b)` (broadcast to both axes); KDA.
    - ``gdn``: as ``kda``, and additionally the log-decay is scalarized
      (``mean`` over :math:`d_k`, broadcast); Gated DeltaNet.

    Scalarization happens at *runtime* only: parameters (and therefore
    ``state_dict`` keys and parameter counts) are identical across modes.
    """

    FULL = "full"
    B_ONLY = "b_only"
    W_ONLY = "w_only"
    KDA = "kda"
    GDN = "gdn"


def apply_gate_mode(
    b: torch.Tensor, w: torch.Tensor, g: torch.Tensor, gate_mode: GateMode | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scalarize the gates per the gate-ablation table (GDN-2 paper Table 5).

    Args:
        b (Tensor): Erase gate of shape ``(B, T, H, d_k)``.
        w (Tensor): Write gate of shape ``(B, T, H, d_v)``.
        g (Tensor): Log-decay of shape ``(B, T, H, d_k)``.
        gate_mode (GateMode | str): One of ``full`` / ``b_only`` / ``w_only``
            / ``kda`` / ``gdn`` (plain strings accepted).

    Returns:
        tuple: ``(b, w, g)`` with the same shapes as the inputs.

    Notes:
        In ``kda`` and ``gdn`` modes the tied scalar is
        :math:`\\beta = \\mathrm{mean}_c(b)` (mean over the erase-gate channel
        axis of that head), used for both the erase and the write gate; in
        ``gdn`` mode the decay additionally becomes
        :math:`\\mathrm{mean}_c(g)` broadcast.  In ``kda``/``gdn`` mode the
        ``w`` projection receives no gradient (intended: the gate ablations
        fix the parameter count).
    """
    mode = GateMode(gate_mode)
    if mode is GateMode.FULL:
        return b, w, g
    if mode is GateMode.B_ONLY:
        w_scalar = w.mean(dim=-1, keepdim=True)
        return b, w_scalar.expand(w.shape), g
    if mode is GateMode.W_ONLY:
        b_scalar = b.mean(dim=-1, keepdim=True)
        return b_scalar.expand(b.shape), w, g
    if mode is GateMode.KDA:
        beta = b.mean(dim=-1, keepdim=True)
        return beta.expand(b.shape), beta.expand(w.shape), g
    if mode is GateMode.GDN:
        beta = b.mean(dim=-1, keepdim=True)
        g_scalar = g.mean(dim=-1, keepdim=True)
        return beta.expand(b.shape), beta.expand(w.shape), g_scalar.expand(g.shape)
    raise ValueError(f"unknown gate mode: {gate_mode!r}")  # pragma: no cover


def transition(
    S: torch.Tensor, k: torch.Tensor, b: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """Apply the transition operator :math:`A_t = (I - k (b \\odot k)^T) \\mathrm{Diag}(e^g)` to S.

    Decay first, then the erase; the write is *not* included (this is the
    primitive of the recurrence, test G01).

    Args:
        S (Tensor): State of shape ``(..., d_k, d_v)``.
        k (Tensor): Key of shape ``(..., d_k)``.
        b (Tensor): Erase gate of shape ``(..., d_k)``.
        g (Tensor): Log-decay of shape ``(..., d_k)``.

    Returns:
        Tensor: ``A_t S`` of shape ``(..., d_k, d_v)``, in the compute dtype.
    """
    cd = _compute_dtype(S)
    S = S.to(cd)
    k = k.to(cd)
    b = b.to(cd)
    g = g.to(cd)
    # S' = Diag(e^g) S
    Sd = S * g.exp()[..., :, None]
    # read = S'^T (b * k)  -> (..., d_v)
    read = (Sd * (b * k)[..., :, None]).sum(dim=-2)
    # A S = S' - k (S'^T (b*k))^T
    return Sd - k[..., :, None] * read[..., None, :]


def recurrent_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    scale: float = 1.0,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token scan of the GDN-2 recurrence.

    Used for tests and decoding; never for training (it stores a state per
    token through autograd).

    The numerical core runs with autocast **disabled**, in the compute
    dtype, so a surrounding ``torch.autocast`` region (e.g. bf16-mixed
    training) cannot silently lower the recurrence's precision; autocast
    stays active everywhere outside this function.  Inputs keep whatever
    dtype the caller produced; outputs are cast back to the input dtype.

    Args:
        q (Tensor): Queries of shape ``(B, T, H, d_k)`` (fla layout; the
            caller is responsible for L2 normalization).
        k (Tensor): Keys of shape ``(B, T, H, d_k)``.
        v (Tensor): Values of shape ``(B, T, H, d_v)``.
        g (Tensor): Log-decay of shape ``(B, T, H, d_k)``.
        b (Tensor): Erase gate of shape ``(B, T, H, d_k)``.
        w (Tensor): Write gate of shape ``(B, T, H, d_v)``.
        initial_state (Tensor | None): Initial state of shape
            ``(B, H, d_k, d_v)`` or ``None`` (zeros).
        scale (float): Output scale.
        compute_dtype (torch.dtype | None): Overrides the precision policy
            for the recurrence core.  ``None`` (the default) keeps the core
            in at least fp32 (fp64 inputs stay fp64), as in the GDN-2 paper
            and fla's kernels; an explicit dtype (e.g. ``torch.bfloat16``)
            is meant for degradation experiments.

    Returns:
        tuple: ``(o, final_state)`` with ``o`` of shape ``(B, T, H, d_v)`` in
        the input dtype and ``final_state`` of shape ``(B, H, d_k, d_v)`` in
        the compute dtype.
    """
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    cd = _compute_dtype(q, compute_dtype)
    if k.shape != q.shape or b.shape != q.shape or g.shape != q.shape:
        raise ValueError(
            f"q/k/b/g must all be (B,T,H,dk); got q={tuple(q.shape)}, "
            f"k={tuple(k.shape)}, b={tuple(b.shape)}, g={tuple(g.shape)}"
        )
    if v.shape != (B, T, H, dv) or w.shape != (B, T, H, dv):
        raise ValueError(
            f"v/w must be (B,T,H,dv); got v={tuple(v.shape)}, w={tuple(w.shape)}"
        )
    # Autocast off for the recurrence core: the surrounding region (e.g.
    # bf16-mixed training) keeps its speed, but this function's numerics
    # always run in the compute dtype.
    with torch.autocast(device_type=q.device.type, enabled=False):
        # internal layout (B, H, T, D), compute dtype
        q_ = q.transpose(1, 2).to(cd)
        k_ = k.transpose(1, 2).to(cd)
        v_ = v.transpose(1, 2).to(cd)
        g_ = g.transpose(1, 2).to(cd)
        b_ = b.transpose(1, 2).to(cd)
        w_ = w.transpose(1, 2).to(cd)
        if initial_state is None:
            S = q_.new_zeros(B, H, dk, dv)
        else:
            if initial_state.shape != (B, H, dk, dv):
                raise ValueError(
                    f"initial_state must be (B,H,dk,dv) = {(B, H, dk, dv)}, "
                    f"got {tuple(initial_state.shape)}"
                )
            S = initial_state.to(cd)
        outputs = []
        for t in range(T):
            # decay
            Sd = S * g_[:, :, t].exp()[..., None]
            # gated read, then u = w*v - S'^T (b*k)
            read = (Sd * (b_[:, :, t] * k_[:, :, t])[..., :, None]).sum(dim=-2)
            u = w_[:, :, t] * v_[:, :, t] - read
            # write
            S = Sd + k_[:, :, t][..., None] * u[..., None, :]
            # read after write
            outputs.append(scale * (q_[:, :, t][..., :, None] * S).sum(dim=-2))
        o = torch.stack(outputs, dim=2)  # (B, H, T, dv)
        o = o.transpose(1, 2).to(q.dtype)
        return o, S


class ChunkLocals(NamedTuple):
    """Phase-1 (state-independent) quantities, per chunk, chunk-major.

    All fields use the chunk-major layout ``(B, H, N, C, ...)`` where ``C``
    is the chunk length.  ``04_derivation_single_state.md`` §2.11 has the
    derivations; the letters are the guide's (not the paper's).

    Fields:
        Qg: ``(B, H, N, C, d_k)``  -- :math:`\\mathbf Q \\odot e^{G}`.
        Aqk: ``(B, H, N, C, C)``  -- :math:`\\mathbf A_{qk}`, lower
            triangular **including** the diagonal, built in log space.
        N: ``(B, H, N, C, C)``  -- :math:`(I + \\mathbf L)^{-1}`, unit
            lower-triangular (needed by the log-linear multi-state layout).
        NZ: ``(B, H, N, C, d_v)``  -- :math:`\\mathbf N (\\mathbf w \\odot \\mathbf V)`.
        W: ``(B, H, N, C, d_k)``  -- :math:`\\mathbf N \\mathbf K_{gb}` with
            :math:`\\mathbf K_{gb} = \\Gamma \\odot \\mathbf B \\odot \\mathbf K`.
        M: ``(B, H, N, C, d_k)``  -- :math:`\\mathbf K \\odot e^{G_{last} - G}`.
        gC: ``(B, H, N, d_k)``  -- :math:`e^{G_{last}}` (chunk-end decay).
    """

    Qg: torch.Tensor
    Aqk: torch.Tensor
    N: torch.Tensor
    NZ: torch.Tensor
    W: torch.Tensor
    M: torch.Tensor
    gC: torch.Tensor


def chunk_local_terms(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    compute_dtype: torch.dtype | None = None,
) -> ChunkLocals:
    """Compute Phase-1 (chunk-local) terms for all chunks at once.

    No state is involved; every quantity depends only on the chunk's inputs.
    The masked exponentials are built in **log space**: cumulative log-decays
    are subtracted, disallowed entries are filled with ``-inf`` *before*
    ``exp`` (so they are exact zeros, keeping the backward pass clean), and
    the contraction follows.  The ``(..., C, C, d_k)`` intermediate is fine
    at pilot scale; see ``04`` §2.9 "Getting the matmul back" for the
    sub-chunking optimization deferred to the kernel work.

    The numerical core runs with autocast **disabled**, in the compute
    dtype (at least fp32 by default, as in the GDN-2 paper and fla's
    kernels), so a surrounding ``torch.autocast`` region cannot silently
    lower Phase 1's precision.

    Args:
        q (Tensor): ``(B, H, N, C, d_k)`` chunk-major queries.
        k (Tensor): ``(B, H, N, C, d_k)`` chunk-major keys.
        v (Tensor): ``(B, H, N, C, d_v)`` chunk-major values.
        g (Tensor): ``(B, H, N, C, d_k)`` chunk-major log-decay.
        b (Tensor): ``(B, H, N, C, d_k)`` chunk-major erase gate.
        w (Tensor): ``(B, H, N, C, d_v)`` chunk-major write gate.
        compute_dtype (torch.dtype | None): Overrides the precision policy
            (``None`` = at least fp32; an explicit dtype is for degradation
            experiments).

    Returns:
        ChunkLocals: the named tuple of Phase-1 tensors, in the compute
        dtype.
    """
    cd = _compute_dtype(q, compute_dtype)
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _chunk_local_terms_impl(q, k, v, g, b, w, cd)


def _chunk_local_terms_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    cd: torch.dtype,
) -> ChunkLocals:
    """Autocast-disabled implementation of :func:`chunk_local_terms`."""
    q = q.to(cd)
    k = k.to(cd)
    v = v.to(cd)
    g = g.to(cd)
    b = b.to(cd)
    w = w.to(cd)
    N, C = q.shape[2], q.shape[3]
    dk, dv = q.shape[-1], v.shape[-1]
    assert q.shape == (q.shape[0], q.shape[1], N, C, dk), "shape assert"
    assert k.shape == q.shape, "shape assert"
    assert b.shape == q.shape and g.shape == q.shape, "shape assert"
    assert v.shape == (q.shape[0], q.shape[1], N, C, dv), "shape assert"
    assert w.shape == v.shape, "shape assert"

    # G_r = inclusive within-chunk cumsum of g (G_0 = g_0); restarts per chunk.
    G = g.cumsum(dim=3)  # (B,H,N,C,dk)

    # ---- log-space score matrices --------------------------------------
    # D[..., r, i, a] = G[r, a] - G[i, a]; exponents are <= 0 on the kept
    # (lower-triangular) entries, so exp is safe; masked entries are -inf
    # *before* exp so they become exact zeros.
    D = G[..., :, None, :] - G[..., None, :, :]  # (B,H,N,C,C,dk)
    tril_incl = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    tril_strict = tril_incl.tril(-1)
    tril_incl = tril_incl.unsqueeze(-1)  # (C, C, 1) to broadcast over dk
    tril_strict = tril_strict.unsqueeze(-1)
    E_causal = D.masked_fill(~tril_incl, float("-inf")).exp()
    E_strict = D.masked_fill(~tril_strict, float("-inf")).exp()
    # A_qk[r, i] = sum_a q[r,a] k[i,a] e^{G_r - G_i}, i <= r
    Aqk = torch.einsum("...ra,...ria,...ia->...ri", q, E_causal, k)
    # L[r, i] = sum_a (b*k)[r,a] k[i,a] e^{G_r - G_i}, i < r (strict)
    L = torch.einsum("...ra,...ria,...ia->...ri", b * k, E_strict, k)

    # ---- gated right-hand sides and the one solve ----------------------
    Z = w * v  # (B,H,N,C,dv)
    Kgb = G.exp() * (b * k)  # (B,H,N,C,dk)
    eye = torch.eye(C, dtype=cd, device=q.device)
    # One unit lower-triangular solve against [Z, K_gb, I]:
    # [N Z, N K_gb, N]; N is needed by the log-linear multi-state layout.
    rhs = torch.cat([Z, Kgb, eye.expand(*Z.shape[:-1], C)], dim=-1)
    eye_L = eye.expand(*L.shape)
    sol = torch.linalg.solve_triangular(eye_L + L, rhs, upper=False, unitriangular=True)
    NZ = sol[..., :dv]
    W = sol[..., dv : dv + dk]
    N_inv = sol[..., dv + dk :]

    # ---- tail quantities ------------------------------------------------
    G_last = G[..., -1, :]  # (B,H,N,dk)
    Qg = q * G.exp()
    M = k * (G[..., -1:, :] - G).exp()

    return ChunkLocals(Qg=Qg, Aqk=Aqk, N=N_inv, NZ=NZ, W=W, M=M, gC=G_last.exp())


def chunk_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 16,
    scale: float = 1.0,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunkwise forward: Phase 1 batched over chunks, Phase 2 sequential over chunks.

    Same outputs as :func:`recurrent_scan`.  ``T`` need not be a multiple of
    ``chunk_size``: the last chunk may be shorter (all formulas hold for any
    chunk length; the short tail is processed with its own Phase-1 call).

    The numerical core (Phase 1 and Phase 2) runs with autocast
    **disabled**, in the compute dtype (at least fp32 by default, as in the
    GDN-2 paper and fla's kernels), so a surrounding ``torch.autocast``
    region (e.g. bf16-mixed training) cannot silently lower the recurrence's
    precision; autocast stays active everywhere outside this function.
    Inputs keep whatever dtype the caller produced; outputs are cast back
    to the input dtype.

    Phase 2, per chunk (``04`` §2.11):

    .. math::

        \\mathbf U = \\mathbf{NZ} - \\mathbf W S, \\qquad
        \\mathbf O = \\mathrm{scale} \\cdot (\\mathbf Q_g S + \\mathbf A_{qk} \\mathbf U),
        \\qquad S \\leftarrow \\mathrm{Diag}(\\gamma^C) S + \\mathbf M^T \\mathbf U

    The state is reassigned each step (functional updates, no in-place ops).

    Args:
        q (Tensor): Queries of shape ``(B, T, H, d_k)``.
        k (Tensor): Keys of shape ``(B, T, H, d_k)``.
        v (Tensor): Values of shape ``(B, T, H, d_v)``.
        g (Tensor): Log-decay of shape ``(B, T, H, d_k)``.
        b (Tensor): Erase gate of shape ``(B, T, H, d_k)``.
        w (Tensor): Write gate of shape ``(B, T, H, d_v)``.
        initial_state (Tensor | None): Initial state ``(B, H, d_k, d_v)`` or
            ``None`` (zeros).
        chunk_size (int): Chunk length ``C``. Defaults to 16.
        scale (float): Output scale.
        compute_dtype (torch.dtype | None): Overrides the precision policy
            for the recurrence core.  ``None`` (the default) keeps the core
            in at least fp32 (fp64 inputs stay fp64), as in the GDN-2 paper
            and fla's kernels; an explicit dtype (e.g. ``torch.bfloat16``)
            is meant for degradation experiments.

    Returns:
        tuple: ``(o, final_state)`` with ``o`` of shape ``(B, T, H, d_v)`` in
        the input dtype and ``final_state`` of shape ``(B, H, d_k, d_v)`` in
        the compute dtype.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    cd = _compute_dtype(q, compute_dtype)
    if k.shape != q.shape or b.shape != q.shape or g.shape != q.shape:
        raise ValueError("q, k, b, g must share the (B,T,H,dk) shape")
    if v.shape != (B, T, H, dv) or w.shape != (B, T, H, dv):
        raise ValueError("v and w must share the (B,T,H,dv) shape")
    # Autocast off for the recurrence core (see the docstring).  The
    # compute_dtype override is threaded through to Phase 1 so an explicit
    # narrow dtype is not re-promoted by chunk_local_terms' default policy.
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _chunk_scan_impl(
            q, k, v, g, b, w, initial_state, chunk_size, scale, cd, compute_dtype
        )


def _chunk_scan_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
    chunk_size: int,
    scale: float,
    cd: torch.dtype,
    compute_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autocast-disabled implementation of :func:`chunk_scan`."""
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    # internal layout (B, H, T, D), compute dtype
    q_ = q.transpose(1, 2).to(cd)
    k_ = k.transpose(1, 2).to(cd)
    v_ = v.transpose(1, 2).to(cd)
    g_ = g.transpose(1, 2).to(cd)
    b_ = b.transpose(1, 2).to(cd)
    w_ = w.transpose(1, 2).to(cd)
    if initial_state is None:
        S = q_.new_zeros(B, H, dk, dv)
    else:
        if initial_state.shape != (B, H, dk, dv):
            raise ValueError(
                f"initial_state must be (B,H,dk,dv) = {(B, H, dk, dv)}, "
                f"got {tuple(initial_state.shape)}"
            )
        S = initial_state.to(cd)

    C = chunk_size
    n_full, tail = divmod(T, C)

    # ---- Phase 1: batched over the full chunks, tail separately ---------
    def to_chunks(t: torch.Tensor, d: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        full = t[:, :, : n_full * C].reshape(B, H, n_full, C, d)
        tl = t[:, :, n_full * C :].reshape(B, H, 1, tail, d) if tail else None
        return full, tl

    qf, qt = to_chunks(q_, dk)
    kf, kt = to_chunks(k_, dk)
    vf, vt = to_chunks(v_, dv)
    gf, gt = to_chunks(g_, dk)
    bf, bt = to_chunks(b_, dk)
    wf, wt = to_chunks(w_, dv)

    loc_list: list[ChunkLocals] = []
    if n_full > 0:
        loc_full = chunk_local_terms(qf, kf, vf, gf, bf, wf, compute_dtype)
        loc_list.extend(
            ChunkLocals(*(field[:, :, n] for field in loc_full)) for n in range(n_full)
        )
    if tail:
        loc_tail = chunk_local_terms(qt, kt, vt, gt, bt, wt, compute_dtype)
        loc_list.append(ChunkLocals(*(field[:, :, 0] for field in loc_tail)))

    # ---- Phase 2: sequential over chunks --------------------------------
    outputs = []
    for loc in loc_list:
        # (B, H, C, d_v) each; S is (B, H, d_k, d_v)
        U = loc.NZ - loc.W @ S
        O_chunk = scale * (loc.Qg @ S + loc.Aqk @ U)
        S = loc.gC[..., :, None] * S + loc.M.transpose(-1, -2) @ U
        outputs.append(O_chunk)

    o = torch.cat(outputs, dim=2) if outputs else q_.new_zeros(B, H, 0, dv)
    o = o.transpose(1, 2).to(q.dtype)
    return o, S
