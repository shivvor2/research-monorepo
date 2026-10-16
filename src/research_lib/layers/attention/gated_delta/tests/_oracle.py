"""Test-only brute-force oracles for the gated delta-rule recurrence.

Deliberately naive and *independent* of
``research_lib.layers.attention.gated_delta.functional``: every quantity is
written out with explicit matrices and literal Python loops, in the
recurrence's own letters.  If the functional core and the oracle disagree,
the functional core (or the transcription into this file) is wrong — never
the test tolerance.

All functions operate on a single (batch, head) pair: ``q``/``k``/``g``/``b``
are ``(T, d_k)``, ``v``/``w`` are ``(T, d_v)``, states are
``(d_k, d_v)``.  Loop over batch/head in the tests.

The multi-state oracle (`multistate_oracle`, `ms_ll_level`) mirrors the
brute-force oracle of ``impl/verify_derivations.py`` (``oracle``): for
every ``t`` and ``s <= t`` the write ``P_{(s,t]} k_s z_s^T`` is placed into
the bucket that ``s`` occupies as seen from ``t``.  It is an independent
transcription of ``05_derivation_multi_state.md`` §1–§6.
"""

import torch


def literal_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    S0: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GDN-2 recurrence with explicit matrices.

    :math:`S_t = (I - k_t (b_t \\odot k_t)^T)\\, \\mathrm{Diag}(e^{g_t})\\,
    S_{t-1} + k_t (w_t \\odot v_t)^T`, then :math:`o_t = \\mathrm{scale}\\,
    S_t^T q_t`.

    Args:
        q (Tensor): ``(T, d_k)``.
        k (Tensor): ``(T, d_k)``.
        v (Tensor): ``(T, d_v)``.
        g (Tensor): ``(T, d_k)`` log-decay.
        b (Tensor): ``(T, d_k)`` erase gate.
        w (Tensor): ``(T, d_v)`` write gate.
        S0 (Tensor): ``(d_k, d_v)`` initial state.
        scale (float): Output scale.

    Returns:
        tuple: ``(o, S_T)`` with ``o`` of shape ``(T, d_v)``.
    """
    T, dk = q.shape
    A = torch.eye(dk, dtype=q.dtype, device=q.device)
    S = S0.clone()
    o = []
    for t in range(T):
        erase = torch.outer(k[t], b[t] * k[t])
        decay = torch.diag(g[t].exp())
        write = torch.outer(k[t], w[t] * v[t])
        S = (A - erase) @ decay @ S + write
        o.append(scale * (q[t] @ S))
    return torch.stack(o), S


def literal_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    S0: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA recurrence with explicit matrices.

    :math:`S_t = (I - \\beta_t k_t k_t^T)\\, \\mathrm{Diag}(e^{g_t})\\,
    S_{t-1} + \\beta_t k_t v_t^T` (scalar :math:`\\beta_t` per token).

    Args:
        q (Tensor): ``(T, d_k)``.
        k (Tensor): ``(T, d_k)``.
        v (Tensor): ``(T, d_v)``.
        g (Tensor): ``(T, d_k)`` log-decay.
        beta (Tensor): ``(T,)`` scalar gates.
        S0 (Tensor): ``(d_k, d_v)`` initial state.
        scale (float): Output scale.

    Returns:
        tuple: ``(o, S_T)`` with ``o`` of shape ``(T, d_v)``.
    """
    T, dk = q.shape
    A = torch.eye(dk, dtype=q.dtype, device=q.device)
    S = S0.clone()
    o = []
    for t in range(T):
        erase = beta[t] * torch.outer(k[t], k[t])
        decay = torch.diag(g[t].exp())
        write = beta[t] * torch.outer(k[t], v[t])
        S = (A - erase) @ decay @ S + write
        o.append(scale * (q[t] @ S))
    return torch.stack(o), S


def literal_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_scalar: torch.Tensor,
    beta: torch.Tensor,
    S0: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaNet recurrence with explicit matrices.

    :math:`S_t = \\alpha_t (I - \\beta_t k_t k_t^T)\\, S_{t-1} + \\beta_t
    k_t v_t^T` with scalar decay :math:`\\alpha_t = e^{g_t}` per token.

    Args:
        q (Tensor): ``(T, d_k)``.
        k (Tensor): ``(T, d_k)``.
        v (Tensor): ``(T, d_v)``.
        g_scalar (Tensor): ``(T,)`` scalar log-decay.
        beta (Tensor): ``(T,)`` scalar gates.
        S0 (Tensor): ``(d_k, d_v)`` initial state.
        scale (float): Output scale.

    Returns:
        tuple: ``(o, S_T)`` with ``o`` of shape ``(T, d_v)``.
    """
    T, dk = q.shape
    A = torch.eye(dk, dtype=q.dtype, device=q.device)
    S = S0.clone()
    o = []
    for t in range(T):
        erase = beta[t] * torch.outer(k[t], k[t])
        write = beta[t] * torch.outer(k[t], v[t])
        S = g_scalar[t].exp() * ((A - erase) @ S) + write
        o.append(scale * (q[t] @ S))
    return torch.stack(o), S


def _transition_matrix(
    k: torch.Tensor, b: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """A = (I - k (b*k)^T) Diag(e^g) as an explicit (dk, dk) matrix."""
    dk = k.shape[-1]
    A = torch.eye(dk, dtype=k.dtype, device=k.device)
    return (A - torch.outer(k, b * k)) @ torch.diag(g.exp())


def brute_force_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Brute-force unrolled state: explicit P-product sums, zero initial state.

    :math:`S_t = \\sum_{s \\le t} P_{(s,t]}\\, k_s z_s^T` with
    :math:`P_{(s,t]} = A_t A_{t-1} \\cdots A_{s+1}` (empty product = ``I``)
    and :math:`z_s = w_s \\odot v_s`; :math:`o_t = \\mathrm{scale}\\,
    S_t^T q_t`.

    Args:
        q (Tensor): ``(T, d_k)``.
        k (Tensor): ``(T, d_k)``.
        v (Tensor): ``(T, d_v)``.
        g (Tensor): ``(T, d_k)`` log-decay.
        b (Tensor): ``(T, d_k)`` erase gate.
        w (Tensor): ``(T, d_v)`` write gate.
        scale (float): Output scale.

    Returns:
        tuple: ``(o, S_T)`` with ``o`` of shape ``(T, d_v)``.
    """
    T, dk = q.shape
    dv = v.shape[-1]
    A = [_transition_matrix(k[t], b[t], g[t]) for t in range(T)]
    z = w * v
    o = []
    S = torch.zeros(dk, dv, dtype=q.dtype, device=q.device)
    for t in range(T):
        S = torch.zeros(dk, dv, dtype=q.dtype, device=q.device)
        for s in range(t + 1):
            P = torch.eye(dk, dtype=q.dtype, device=q.device)
            for j in range(s + 1, t + 1):
                P = A[j] @ P
            S = S + P @ torch.outer(k[s], z[s])
        o.append(scale * (q[t] @ S))
    return torch.stack(o), S


# ============================================================================
# Multi-state oracle (PR 2)
# ============================================================================


def ms_ll_level(t: int, s: int) -> int:
    """Fenwick level of key ``s`` as seen from query ``t``; literal loop.

    Independently transcribed from ``05_derivation_multi_state.md`` §6.2
    (do *not* import ``layouts.ll_level`` — the oracle must stay
    independent of the code under test).  Level 0 iff ``s == t``;
    otherwise scan levels upward until the range rule hits.
    """
    if s == t:
        return 0
    lvl = 1
    while True:
        size = 1 << (lvl - 1)
        if (t >> (lvl - 1)) & 1:
            hi = t - (t % size)
            lo = hi - size
            if lo <= s < hi:
                return lvl
        lvl += 1


def multistate_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    lam: torch.Tensor,
    layout: str,
    block_size: int | None = None,
    window: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Brute-force multi-state oracle, zero initial state.

    For every ``t`` and ``s <= t``, the write ``P_{(s,t]} k_s z_s^T`` is
    placed into the bucket ``s`` occupies as seen from ``t``:

    - ``ws``: slot ``min(t//B - s//B, W + 1)``;
    - ``fs``: same slot, but transported only until the end of ``s``'s
      block: :math:`P_{(s, e_s]}` with
      :math:`e_s = \\min(t, (s//B + 1)B - 1)`;
    - ``ll``: slot ``ms_ll_level(t, s)``.

    Args:
        q (Tensor): ``(T, d_k)``.
        k (Tensor): ``(T, d_k)``.
        v (Tensor): ``(T, d_v)``.
        g (Tensor): ``(T, d_k)`` log-decay.
        b (Tensor): ``(T, d_k)`` erase gate.
        w (Tensor): ``(T, d_v)`` write gate.
        lam (Tensor): ``(T, L)`` mixing weights.
        layout (str): ``"ws"`` / ``"fs"`` / ``"ll"`` / ``"single"``.
        block_size (int | None): Block size ``B`` for ``ws``/``fs``.
        window (int | None): Window ``W`` for ``ws``/``fs``.

    Returns:
        tuple: ``(o, S_buckets)`` with ``o`` of shape ``(T, d_v)`` and
        ``S_buckets`` of shape ``(T, L, d_k, d_v)`` — the post-write
        bucket states at every ``t`` (``o_t = sum_l lam[t, l]
        (S_buckets[t, l])^T q_t``).
    """
    T, dk = q.shape
    dv = v.shape[-1]
    L = lam.shape[1]
    if layout == "single":
        o, _ = brute_force_oracle(q, k, v, g, b, w)
        return o, torch.empty(0, 0, dk, dv, dtype=q.dtype, device=q.device)
    if layout in ("ws", "fs") and (block_size is None or window is None):
        raise ValueError(
            f"layout={layout!r} needs block_size and window, got "
            f"block_size={block_size}, window={window}"
        )
    # Narrow for the closures below (they only run for ws/fs).
    Bsz = block_size if block_size is not None else 1
    Wsz = window if window is not None else 0
    A = [_transition_matrix(k[t], b[t], g[t]) for t in range(T)]
    z = w * v

    def P(s: int, t: int) -> torch.Tensor:
        """P_(s, t] = A_t ... A_{s+1}; identity if s == t."""
        M = torch.eye(dk, dtype=q.dtype, device=q.device)
        for r in range(s + 1, t + 1):
            M = A[r] @ M
        return M

    def bucket(t: int, s: int) -> int:
        if layout == "ll":
            return ms_ll_level(t, s)
        off = t // Bsz - s // Bsz
        return off if off <= Wsz else Wsz + 1

    out = torch.zeros(T, dv, dtype=q.dtype, device=q.device)
    Sb = torch.zeros(T, L, dk, dv, dtype=q.dtype, device=q.device)
    for t in range(T):
        for s in range(t + 1):
            if layout == "fs":
                end = min(t, (s // Bsz + 1) * Bsz - 1)
                Pm = P(s, end)  # frozen at the end of s's block
            else:
                Pm = P(s, t)
            Sb[t, bucket(t, s)] += Pm @ torch.outer(k[s], z[s])
        for lvl in range(L):
            out[t] += lam[t, lvl] * (Sb[t, lvl].T @ q[t])
    return out, Sb
