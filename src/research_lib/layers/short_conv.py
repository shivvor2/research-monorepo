"""Causal short convolution used by gated delta-rule token mixers.

The GDN-2 architecture (arXiv:2605.22791, §3.5) applies a small causal
depthwise convolution (kernel size 4, SiLU) to the ``q``, ``k`` and ``v``
projections before they enter the recurrence, mixing a few most-recent
tokens into each position's representation.  The same primitive appears in
the Gated DeltaNet / Kimi Linear lineage (fla's ``short_conv``).

This module is generic (it lives in ``layers/``, not inside
``layers/attention/gated_delta/``) so it can be reused by other mixers.

Run tests with::

    pytest src/research_lib/layers/tests/test_short_conv.py -v
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_ACTIVATIONS = {"silu": F.silu, "sigmoid": torch.sigmoid, None: None, "identity": None}


class CausalShortConv1d(nn.Module):
    r"""Causal depthwise 1-D convolution over the token axis with an explicit cache.

    The input has layout ``(B, T, C)``: tokens along dim 1.  The convolution is
    depthwise (``groups = C``) with no padding, and causality is achieved by
    left-padding the token axis with the caller-supplied cache (or zeros).  At
    time ``t`` the output sees inputs ``x[t-K+1 .. t]`` only.

    The cache holds the last ``K-1`` **pre-conv** inputs with layout
    ``(B, C, K-1)``; it is what must be carried across sequential calls
    (prefill/decode handoffs).  A ``None`` cache is equivalent to zeros, so a
    fresh call starts cleanly.

    Args:
        channels (int): Number of channels ``C`` (equal to the depthwise
            group count).
        kernel_size (int): Convolution kernel size ``K``. Defaults to 4.
        bias (bool): Whether the depthwise conv has a bias. Defaults to
            ``False``.
        activation (str | None): Activation applied after the convolution.
            One of ``"silu"``, ``"sigmoid"``, ``None`` (identity).
            Defaults to ``"silu"`` (GDN-2 paper §3.5).
        device: Optional torch device for the conv weight.
        dtype: Optional torch dtype for the conv weight.

    Shape:
        - x: :math:`(B, T, C)`
        - cache: :math:`(B, C, K-1)` or ``None``
        - Returns: ``y`` of shape :math:`(B, T, C)`` and, if
          ``output_final_state=True``, the new cache of shape
          :math:`(B, C, K-1)`` (else ``None``).

    Examples::

        >>> conv = CausalShortConv1d(channels=8, kernel_size=4)
        >>> x = torch.randn(2, 10, 8)
        >>> y, cache = conv(x, output_final_state=True)   # prefill
        >>> y2, cache2 = conv(torch.randn(2, 1, 8), cache=cache,
        ...                   output_final_state=True)     # decode step
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        bias: bool = False,
        activation: str | None = "silu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(map(str, _ACTIVATIONS))}, "
                f"got {activation!r}"
            )
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            padding=0,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the causal short convolution.

        Args:
            x (Tensor): Input of shape :math:`(B, T, C)`.
            cache (Tensor | None): Previous pre-conv inputs of shape
                :math:`(B, C, K-1)`, or ``None`` for a fresh start.
            output_final_state (bool): If ``True``, return the new cache
                (the last ``K-1`` pre-conv inputs of ``[cache, x]``).

        Returns:
            tuple: ``(y, new_cache)`` with ``y`` of shape :math:`(B, T, C)`
            and ``new_cache`` of shape :math:`(B, C, K-1)` or ``None``.
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (B, T, C), got shape {tuple(x.shape)}")
        B, T, C = x.shape
        if self.channels != C:
            raise ValueError(f"x has {C} channels, layer expects {self.channels}")
        K = self.kernel_size
        # (B, C, T) for the conv; cache is already (B, C, K-1).
        xt = x.transpose(1, 2)
        if cache is None:
            pad = xt.new_zeros(B, C, K - 1)
        else:
            if cache.shape != (B, C, K - 1):
                raise ValueError(
                    f"cache must be (B, C, K-1) = {(B, C, K - 1)}, "
                    f"got {tuple(cache.shape)}"
                )
            pad = cache
        padded = torch.cat([pad, xt], dim=-1)  # (B, C, K-1+T)
        y = self.conv(padded)  # (B, C, T): (K-1+T) - K + 1 = T
        y = y.transpose(1, 2)  # (B, T, C)
        act = _ACTIVATIONS[self.activation]
        if act is not None:
            y = act(y)
        # The new cache is the last K-1 entries of [cache, x]; this is also
        # correct when T < K-1 (part of the old cache is retained).  K=1
        # needs an explicit empty cache: -(K-1) is -0, so the slice would
        # return the *whole* padded tensor.  The slice is cloned so the
        # cache does not keep the entire prefill's storage alive.
        if not output_final_state:
            new_cache = None
        elif K > 1:
            new_cache = padded[..., -(K - 1) :].clone()
        else:
            new_cache = padded.new_empty(B, C, 0)
        return y, new_cache

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, kernel_size={self.kernel_size}, "
            f"activation={self.activation!r}"
        )
