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
