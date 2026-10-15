"""Input/output projections for the gated delta-rule layers.

One module shared by :class:`~research_lib.layers.attention.gated_delta.reference.GatedDeltaReference`
(single state, PR 1) and the multi-state layer (PR 2), so both have
identical parameter names and the gate-mode ablations keep parameter counts
fixed.

Projections (GDN-2 paper §3.5, App. C.1, D.5; checked against fla's
``GatedDeltaNet2`` layer on 2026-10-13, see ``04_derivation_single_state.md``
§2.10):

===================  =======================================================  ==============
output               computation                                              shape
===================  =======================================================  ==============
``q, k``             Linear -> CausalShortConv1d (k=4, SiLU) -> reshape ->    (B,T,H,dk)
                     L2-normalize over ``d_k`` in fp32 (eps 1e-6)
``v``                Linear -> short conv (SiLU)                              (B,T,H,dv)
``b``                sigmoid(Linear)                                          (B,T,H,dk)
``w``                sigmoid(Linear)                                          (B,T,H,dv)
``g``                ``-e^{A_log} * softplus(f(x) + dt_bias)`` in fp32        (B,T,H,dk)
output               per-head RMSNorm(dv) of the mixer output, times          (B,T,d)
                     ``act(Linear(x))``, then Linear back to ``d``
===================  =======================================================  ==============

Initialization (``init_scheme="paper"``, GDN-2 paper D.5): every ``nn.Linear``
weight is Xavier-uniform with gain :math:`2^{-2.5}` and zero biases;
``A_log`` is :math:`\\log U(1,16)` per head and ``dt_bias`` the inverse
softplus of :math:`dt \\sim \\exp(U(\\log 10^{-3}, \\log 10^{-1}))` clamped
at :math:`10^{-4}` (fla convention).  ``A_log`` and ``dt_bias`` are marked
``_no_weight_decay = True``.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from research_lib.layers.short_conv import CausalShortConv1d

logger = logging.getLogger(__name__)

_OUTPUT_GATE_ACTIVATIONS = {"silu": F.silu, "sigmoid": torch.sigmoid}


class HeadRMSNorm(nn.Module):
    r"""RMSNorm over the last dimension with a per-head affine weight.

    Applies :math:`x / \\sqrt{\\mathrm{mean}(x^2) + \\varepsilon}` along the
    head dimension, then a learnable per-head scale.  Used for the GDN-2
    output norm (paper D.5: "the output is passed through an RMSNorm").

    Args:
        num_heads (int): Number of heads ``H``.
        head_dim (int): Normalized dimension ``d_v``.
        eps (float): Variance epsilon. Defaults to 1e-5.

    Shape:
        - Input: :math:`(B, T, H, d_v)`
        - Output: same shape, in the compute dtype.

    Examples::

        >>> norm = HeadRMSNorm(num_heads=2, head_dim=8)
        >>> x = torch.randn(3, 5, 2, 8)
        >>> y = norm(x)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(num_heads, head_dim, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in the compute dtype (fp32 unless the input is fp64).
        cd = torch.float64 if x.dtype == torch.float64 else torch.float32
        x32 = x.to(cd)
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        return x32 * torch.rsqrt(var + self.eps) * self.weight.to(cd)


class GatedDeltaProjections(nn.Module):
    r"""Shared input/output projections for the gated delta-rule layers.

    Args:
        hidden_size (int): Model dimension :math:`d`.
        num_heads (int): Number of key heads ``H`` (the pilot uses
            :math:`H_v = H`; grouped value heads are out of scope).
        head_dim (int): Key head dimension :math:`d_k`.
        expand_v (float): Value head dimension is
            :math:`d_v = \\lfloor d_k \\cdot \\mathrm{expand\\_v} \\rfloor`.
            Defaults to 1.0.
        use_short_conv (bool): If ``True`` (default), ``q``/``k``/``v`` get a
            causal depthwise conv (kernel ``conv_size``) + SiLU; if ``False``,
            SiLU alone.
        conv_size (int): Short-conv kernel size. Defaults to 4.
        output_gate_activation (str): ``"silu"`` (paper, default) or
            ``"sigmoid"`` (fla parity).
        low_rank_gates (bool): If ``True``, the log-decay projection
            :math:`f` is low-rank: ``Linear(d, dv) -> Linear(dv, H*dk)``.
            Defaults to ``False``.
        norm_eps (float): Epsilon for the per-head output RMSNorm. Defaults
            to 1e-5.
        init_scheme (str): ``"paper"`` (only supported scheme).
        device: Optional torch device.
        dtype: Optional torch dtype.

    Shape:
        - Input ``x``: :math:`(B, T, d)`
        - ``project`` returns ``q, k`` of :math:`(B, T, H, d_k)`,
          ``g, b`` of :math:`(B, T, H, d_k)`, ``v, w`` of
          :math:`(B, T, H, d_v)`.
        - ``output`` returns :math:`(B, T, d)`.

    Examples::

        >>> proj = GatedDeltaProjections(hidden_size=32, num_heads=2, head_dim=8)
        >>> x = torch.randn(2, 16, 32)
        >>> q, k, v, b, w, g, cache = proj.project(x)
        >>> y = proj.output(torch.randn(2, 16, 2, 8), x)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        output_gate_activation: str = "silu",
        low_rank_gates: bool = False,
        norm_eps: float = 1e-5,
        init_scheme: str = "paper",
        device=None,
        dtype=None,
    ):
        super().__init__()
        head_v_dim = int(head_dim * expand_v)
        if head_v_dim != head_dim * expand_v:
            raise ValueError(
                f"head_dim * expand_v = {head_dim} * {expand_v} is not an integer"
            )
        if output_gate_activation not in _OUTPUT_GATE_ACTIVATIONS:
            raise ValueError(
                f"output_gate_activation must be one of "
                f"{sorted(_OUTPUT_GATE_ACTIVATIONS)}, got {output_gate_activation!r}"
            )
        if init_scheme != "paper":
            raise ValueError(f"init_scheme must be 'paper', got {init_scheme!r}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = head_v_dim
        self.use_short_conv = use_short_conv
        self.output_gate_activation = output_gate_activation

        d, H, dk, dv = hidden_size, num_heads, head_dim, head_v_dim
        f = {"device": device, "dtype": dtype}
        self.q_proj = nn.Linear(d, H * dk, bias=False, **f)
        self.k_proj = nn.Linear(d, H * dk, bias=False, **f)
        self.v_proj = nn.Linear(d, H * dv, bias=False, **f)
        self.b_proj = nn.Linear(d, H * dk, bias=False, **f)
        self.w_proj = nn.Linear(d, H * dv, bias=False, **f)
        if low_rank_gates:
            self.g_proj = nn.Sequential(
                nn.Linear(d, dv, bias=False, **f),
                nn.Linear(dv, H * dk, bias=False, **f),
            )
        else:
            self.g_proj = nn.Linear(d, H * dk, bias=False, **f)
        # Log-decay parameters: A_log per (key) head, dt_bias per key channel.
        self.A_log = nn.Parameter(torch.empty(H, device=device, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.empty(H, dk, device=device, dtype=dtype))
        # Output branch: per-head RMSNorm, SiLU gate from x, final projection.
        self.norm = HeadRMSNorm(H, dv, eps=norm_eps, device=device, dtype=dtype)
        self.out_gate_proj = nn.Linear(d, H * dv, bias=False, **f)
        self.out_proj = nn.Linear(H * dv, d, bias=False, **f)

        if use_short_conv:
            self.conv_q = CausalShortConv1d(
                H * dk, kernel_size=conv_size, activation="silu", **f
            )
            self.conv_k = CausalShortConv1d(
                H * dk, kernel_size=conv_size, activation="silu", **f
            )
            self.conv_v = CausalShortConv1d(
                H * dv, kernel_size=conv_size, activation="silu", **f
            )

        self._reset_parameters()

    @torch.no_grad()
    def _reset_parameters(self) -> None:
        """Initialize per the GDN-2 paper D.5 (Xavier gain 2^-2.5) and fla."""
        gain = 2.0**-2.5
        linears = [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.b_proj,
            self.w_proj,
            self.out_gate_proj,
            self.out_proj,
        ]
        if isinstance(self.g_proj, nn.Sequential):
            linears.extend(self.g_proj)
        else:
            linears.append(self.g_proj)
        for m in linears:
            nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.ones_(self.norm.weight)
        # A_log = log U(1, 16) per head (fla convention).
        nn.init.uniform_(self.A_log, math.log(1.0), math.log(16.0))
        # dt_bias = softplus^{-1}(dt), dt = exp(U(log 1e-3, log 1e-1)) >= 1e-4.
        dt = torch.empty_like(self.dt_bias)
        dt.uniform_(math.log(1e-3), math.log(1e-1)).exp_().clamp_(min=1e-4)
        self.dt_bias.data.copy_(torch.log(torch.expm1(dt)))
        # fla convention: exclude these from weight decay.
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]

    def project(
        self,
        x: torch.Tensor,
        conv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ]:
        """Project the layer input into the recurrence's inputs.

        Args:
            x (Tensor): Layer input of shape :math:`(B, T, d)`.
            conv_cache (tuple | None): Short-conv caches ``(q, k, v)`` from a
                previous call, or ``None`` for a fresh start.
            use_cache (bool): If ``True``, return the new conv caches.

        Returns:
            tuple: ``(q, k, v, b, w, g, new_conv_cache)`` with shapes
            :math:`(B,T,H,d_k)`, :math:`(B,T,H,d_k)`, :math:`(B,T,H,d_v)`,
            :math:`(B,T,H,d_k)`, :math:`(B,T,H,d_v)`, :math:`(B,T,H,d_k)` and
            the conv-cache tuple or ``None``.  ``q``, ``k``, ``g``, ``b`` and
            ``w`` are computed in fp32 (fp64 if the module is fp64); ``v``
            keeps the module dtype.
        """
        B, T, d = x.shape
        if d != self.hidden_size:
            raise ValueError(f"x has hidden size {d}, layer expects {self.hidden_size}")
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        cd = torch.float64 if x.dtype == torch.float64 else torch.float32

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.use_short_conv:
            qc, kc, vc = conv_cache if conv_cache is not None else (None, None, None)
            q, new_qc = self.conv_q(q, cache=qc, output_final_state=use_cache)
            k, new_kc = self.conv_k(k, cache=kc, output_final_state=use_cache)
            v, new_vc = self.conv_v(v, cache=vc, output_final_state=use_cache)
            new_conv_cache = (new_qc, new_kc, new_vc) if use_cache else None
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)
            new_conv_cache = None

        # L2-normalize q, k over d_k in fp32 (eps 1e-6).
        q = F.normalize(q.view(B, T, H, dk).to(cd), dim=-1, eps=1e-6)
        k = F.normalize(k.view(B, T, H, dk).to(cd), dim=-1, eps=1e-6)
        v = v.view(B, T, H, dv)
        # Gates and log-decay in fp32 (precision policy).
        b = torch.sigmoid(self.b_proj(x).view(B, T, H, dk).to(cd))
        w = torch.sigmoid(self.w_proj(x).view(B, T, H, dv).to(cd))
        f = self.g_proj(x)
        g = -torch.exp(self.A_log.view(1, 1, H, 1).to(cd)) * F.softplus(
            f.view(B, T, H, dk).to(cd) + self.dt_bias.view(1, 1, H, dk).to(cd)
        )
        return q, k, v, b, w, g, new_conv_cache

    def output(self, o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply the output norm, output gate and final projection.

        Args:
            o (Tensor): Mixer output of shape :math:`(B, T, H, d_v)`.
            x (Tensor): The layer input (the gate reads it), shape
                :math:`(B, T, d)`.

        Returns:
            Tensor: Layer output of shape :math:`(B, T, d)`.
        """
        B, T, d = x.shape
        H, dv = self.num_heads, self.head_v_dim
        if o.shape != (B, T, H, dv):
            raise ValueError(
                f"o must be (B,T,H,dv) = {(B, T, H, dv)}, got {tuple(o.shape)}"
            )
        cd = torch.float64 if x.dtype == torch.float64 else torch.float32
        normed = self.norm(o)  # compute dtype
        act = _OUTPUT_GATE_ACTIVATIONS[self.output_gate_activation]
        gate = act(self.out_gate_proj(x).view(B, T, H, dv))
        h = (normed * gate.to(cd)).to(x.dtype)
        return self.out_proj(h.reshape(B, T, H * dv))
