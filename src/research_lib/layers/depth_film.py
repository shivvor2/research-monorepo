"""Depth-conditioned Feature-wise Linear Modulation (DepthFiLM)."""

import torch
import torch.nn as nn


class DepthFiLM(nn.Module):
    r"""Depth-conditioned Feature-wise Linear Modulation (DepthFiLM).

    Generates per-channel scale and shift vectors conditioned on the current
    recurrent-loop index :math:`t`.  At every loop iteration the same base
    sub-layer (e.g. attention or FFN) is executed, but its output is
    re-scaled and shifted by a lookup that depends on :math:`t`.  This lets
    a *single* set of shared weights implement functionally distinct
    operations at different recurrent depths without introducing per-loop
    parameter growth.

    The modulation is initialized to the identity
    (:math:`\text{scale}\approx 0,\;\text{shift}\approx 0`) so that early
    training behaves as though the loop-depth signal were absent, letting
    the model learn to *earn* the depth conditioning gradually.

    Formally, for an input feature map :math:`x` of shape
    :math:`(B, T, C)` and loop index :math:`t`:

    .. math::
        \gamma_t,\beta_t &= \text{MLP}\bigl(\text{Embed}(t)\bigr) \\
        \text{FiLM}(x,t) &= x \odot (1+\gamma_t) + \beta_t

    where :math:`\odot` is broadcast element-wise multiplication and the
    MLP is a tiny two-layer network.

    Args:
        dim (int): Channel dimension :math:`C` of the feature map being
            conditioned.
        max_loops (int): Number of discrete loop indices to embed.
            Indices larger than ``max_loops-1`` are safely clamped to the
            last entry, which enables *depth extrapolation* (inference with
            more loops than were seen during training).
        hidden (int, optional): Hidden dimension of the FiLM generator MLP.
            Default: ``64``.

    Shape:
        - Input: :math:`(B, T, C)` where :math:`C = \text{dim}`.
        - Output: same shape as input.

    Examples::

        >>> film = DepthFiLM(dim=512, max_loops=8)
        >>> x = torch.randn(2, 128, 512)
        >>> out = film(x, loop_t=3)   # condition on the 4-th loop step

    References:
        * Perez et al., "FiLM: Visual Reasoning with a General Conditioning
          Layer", 2017.

    .. note::
        This module is designed to sit **between** a sub-layer computation
        (attention or FFN) and its residual addition, not inside the
        sub-layer itself.  The wrapped attention/FFN module requires no
        modification.
    """

    def __init__(self, dim: int, max_loops: int, hidden: int = 64):
        super().__init__()
        self.max_loops = max_loops
        self.dim = dim
        self.depth_emb = nn.Embedding(max_loops, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Pre-allocate index buffer to avoid creating a new tensor every call.
        self.register_buffer("_idx", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        r"""Depth-condition the input tensor.

        Args:
            x (Tensor): Feature map of shape :math:`(B, T, C)`.
            loop_t (int): Current loop iteration index (0-based).

        Returns:
            Tensor: Modulated feature map, same shape as ``x``.
        """
        t_idx = min(loop_t, self.max_loops - 1)
        self._idx.fill_(t_idx)
        d = self.depth_emb(self._idx)
        scale, shift = self.mlp(d).chunk(2, dim=-1)
        return x * (1 + scale) + shift
