"""Lambda mixing weights for the multi-state gated delta-rule layer.

λ is a per-head, per-slot scalar weight applied to each bucket's read
(``05_derivation_multi_state.md`` §8): ``o_t = scale * sum_l lambda_t^(l)
(S_t^(l))^T q_t``.  All three parameterizations (``static`` / ``lin`` /
``mlp``) are nonnegative through a softplus and are **exactly 1 at
initialization** (D9, the baseline-anchored no-op: at λ≡1 the multi-state
model *is* GDN-2, by the bucket-sum invariant, so the only new gradient
direction at init is ∂L/∂λ).

The source disagreement on λ (Log-Linear §3.1 "a linear function of
``x_t``"; AMD describes it as input-independent and proposes an MLP; fla's
port of Log-Linear Mamba-2 uses ``softplus(L * linear(x))`` per head) is
the reason there are three modes (D5).

``nonlinearity`` is a constructor field because V2 (the original release's
choice) is still open: the init bias is computed as the *inverse* of the
nonlinearity at 1, so a later nonlinearity keeps λ≡1 at init.  Only
``softplus`` is implemented (D21).

Run tests with::

    pytest src/research_lib/layers/attention/gated_delta/tests/test_mixing.py -v
"""

import logging
import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LambdaMode(str, Enum):
    """λ parameterization (``str`` enum, plain strings accepted).

    - ``static``: ``softplus(theta)`` with ``theta`` a learned
      ``(H, L)`` table, broadcast over the batch/time (the capacity
      control; AMD's reading).
    - ``lin``: ``softplus(s * (W_lambda x + c))`` (the fla/Log-Linear form).
    - ``mlp``: ``softplus(MLP(x))`` with ``MLP = Linear -> GELU ->
      Linear`` (AMD's proposal).
    """

    STATIC = "static"
    LIN = "lin"
    MLP = "mlp"


# softplus and its inverse at y: softplus^{-1}(y) = log(e^y - 1); at y=1 this
# is log(e - 1) = 0.5413... (D21).  A later nonlinearity registers its
# inverse here and keeps the λ≡1 init.
_NONLINEARITIES = {
    "softplus": (F.softplus, lambda y: math.log(math.expm1(y))),
}


class LambdaMixer(nn.Module):
    r"""Compute the per-head, per-slot mixing weights λ from the layer input.

    λ≡1 exactly at initialization for every mode (the bias terms are the
    inverse nonlinearity of 1); λ is nonnegative in every mode; λ is
    computed from the **layer input** ``x`` (the hidden state entering the
    mixer), like Log-Linear.  All λ parameters are marked
    ``_no_weight_decay = True`` (fla convention).

    Modes (``05`` §8.3):

    =========  =====================================================  =============================================
    mode       λ                                                      parameters
    =========  =====================================================  =============================================
    ``static``  :math:`\mathrm{softplus}(\theta_{h,\ell})`            :math:`\theta \in \mathbb{R}^{H\times L}`
    ``lin``     :math:`\mathrm{softplus}(s\,(W_\lambda x + c))`       ``Linear(d, H*L)`` (weight 0, bias ``c``),
                                                                       :math:`s \in \mathbb{R}^{H\times L}` (init 1)
    ``mlp``     :math:`\mathrm{softplus}(\mathrm{MLP}(x))`            ``Linear(d, m) -> GELU -> Linear(m, H*L)``,
                                                                       last layer weight 0, bias ``c``
    =========  =====================================================  =============================================

    Args:
        hidden_size (int): Model dimension ``d`` of the layer input.
        num_heads (int): Number of heads ``H``.
        num_levels (int): Number of state slots ``L`` (from
            :func:`~research_lib.layers.attention.gated_delta.layouts.num_slots`).
        mode (str | LambdaMode): ``"static"`` / ``"lin"`` / ``"mlp"``.
            Defaults to ``"mlp"``.
        mlp_hidden (int | None): Hidden width of the MLP mode; defaults to
            ``hidden_size``.
        nonlinearity (str): ``"softplus"`` (the only implemented choice;
            the field exists so a later nonlinearity keeps the λ≡1 init).
        device: Optional torch device.
        dtype: Optional torch dtype.

    Shape:
        - Input: :math:`(B, T, d)`
        - Output: :math:`(B, T, H, L)`, nonnegative, in the compute dtype
          (fp32, or fp64 when the input is fp64).

    Examples::

        >>> mixer = LambdaMixer(hidden_size=32, num_heads=2, num_levels=10)
        >>> lam = mixer(torch.randn(2, 16, 32))
        >>> lam.shape
        torch.Size([2, 16, 2, 10])
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_levels: int,
        mode: str = "mlp",
        mlp_hidden: int | None = None,
        nonlinearity: str = "softplus",
        device=None,
        dtype=None,
    ):
        super().__init__()
        try:
            self.mode = LambdaMode(mode)
        except ValueError as e:
            raise ValueError(
                f"mode must be one of {[m.value for m in LambdaMode]}, got {mode!r}"
            ) from e
        if nonlinearity not in _NONLINEARITIES:
            raise ValueError(
                f"nonlinearity must be one of {sorted(_NONLINEARITIES)}, "
                f"got {nonlinearity!r}"
            )
        self.nonlinearity = nonlinearity
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_levels = num_levels

        act, act_inv = _NONLINEARITIES[nonlinearity]
        self._act = act
        init_bias = float(
            act_inv(1.0)
        )  # act(init_bias) == 1 exactly-ish -> λ≡1 at init

        d, H, L = hidden_size, num_heads, num_levels
        f = {"device": device, "dtype": dtype}
        if self.mode is LambdaMode.STATIC:
            self.theta = nn.Parameter(
                torch.full((H, L), init_bias, device=device, dtype=dtype)
            )
            params = [self.theta]
        elif self.mode is LambdaMode.LIN:
            self.lambda_proj = nn.Linear(d, H * L, bias=True, **f)
            nn.init.zeros_(self.lambda_proj.weight)
            nn.init.constant_(self.lambda_proj.bias, init_bias)
            self.scale = nn.Parameter(torch.ones(H, L, device=device, dtype=dtype))
            params = [self.lambda_proj.weight, self.lambda_proj.bias, self.scale]
        else:  # MLP
            m = mlp_hidden if mlp_hidden is not None else hidden_size
            if m < 1:
                raise ValueError(f"mlp_hidden must be >= 1, got {m}")
            self.mlp_in = nn.Linear(d, m, bias=True, **f)
            self.act = nn.GELU()
            self.mlp_out = nn.Linear(m, H * L, bias=True, **f)
            # hidden layer: default init; last layer: weight 0, bias init_bias
            nn.init.zeros_(self.mlp_out.weight)
            nn.init.constant_(self.mlp_out.bias, init_bias)
            params = [
                self.mlp_in.weight,
                self.mlp_in.bias,
                self.mlp_out.weight,
                self.mlp_out.bias,
            ]
        # fla convention: exclude λ parameters from weight decay.
        for p in params:
            p._no_weight_decay = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute λ.

        Args:
            x (Tensor): Layer input of shape :math:`(B, T, d)`.

        Returns:
            Tensor: λ of shape :math:`(B, T, H, L)`, nonnegative, in the
            compute dtype (fp32, or fp64 when ``x`` is fp64 — the same
            policy as the projections).
        """
        B, T, d = x.shape
        if d != self.hidden_size:
            raise ValueError(f"x has hidden size {d}, mixer expects {self.hidden_size}")
        H, L = self.num_heads, self.num_levels
        cd = torch.float64 if x.dtype == torch.float64 else torch.float32
        if self.mode is LambdaMode.STATIC:
            lam = self._act(self.theta.to(cd))  # (H, L)
            lam = lam.view(1, 1, H, L).expand(B, T, H, L)
        elif self.mode is LambdaMode.LIN:
            logits = self.lambda_proj(x).to(cd).view(B, T, H, L)
            lam = self._act(self.scale.to(cd) * logits)
        else:  # MLP
            h = self.act(self.mlp_in(x))
            logits = self.mlp_out(h).to(cd).view(B, T, H, L)
            lam = self._act(logits)
        return lam
