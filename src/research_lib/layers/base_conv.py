"""Zoology's BaseConv short convolution mixer.

``y = conv_k(u) ⊙ (W u + c) + u`` — a depthwise causal convolution of
width ``k`` (kernel 3 in the Zoology MQAR harness) whose output gates a
linear projection of the input, on a residual connection
(``zoology/mixers/base_conv.py``).  This is the layer-0 mixer of the
two-layer Zoology-style language model used by the multi-state GDN-2
pilot: the layer under test is the
gated-delta mixer in layer 1, exactly as Zoology's ``Hybrid`` alternates
mixers by layer index.

The causal convolution reuses
:class:`~research_lib.layers.short_conv.CausalShortConv1d` with
``activation=None`` (the gating branch provides the nonlinearity).

Run tests with::

    pytest src/research_lib/layers/tests/test_base_conv.py -v
"""

import logging

import torch
import torch.nn as nn

from research_lib.layers.short_conv import CausalShortConv1d

logger = logging.getLogger(__name__)


class BaseConv(nn.Module):
    r"""Zoology's BaseConv: a gated depthwise causal convolution mixer.

    .. math::

        y = \mathrm{conv}_k(u) \odot (W u + c) + u

    Args:
        d_model (int): Model dimension ``d``.
        kernel_size (int): Causal convolution width ``k``. Defaults to 3
            (Zoology's MQAR harness).
        device: Optional torch device.
        dtype: Optional torch dtype.

    Shape:
        - Input: :math:`(B, T, d)`
        - Output: :math:`(B, T, d)`

    Examples::

        >>> mixer = BaseConv(d_model=32)
        >>> y = mixer(torch.randn(2, 16, 32))
        >>> y.shape
        torch.Size([2, 16, 32])
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        self.d_model = d_model
        self.kernel_size = kernel_size
        # Depthwise causal conv, no activation (the gate branch nonlinearity
        # is the elementwise product with the linear branch).
        self.conv = CausalShortConv1d(
            channels=d_model,
            kernel_size=kernel_size,
            bias=False,
            activation=None,
            device=device,
            dtype=dtype,
        )
        # The gating branch W u + c (bias on: it is the "+c" of the formula).
        self.gate = nn.Linear(d_model, d_model, bias=True, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated short convolution.

        Args:
            x (Tensor): Input of shape :math:`(B, T, d)`.

        Returns:
            Tensor: Output of shape :math:`(B, T, d)`.
        """
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"x must be (B, T, d_model={self.d_model}), "
                f"got shape {tuple(x.shape)}"
            )
        conv_out, _ = self.conv(x)  # (B, T, d), no cache needed
        return conv_out * self.gate(x) + x

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, kernel_size={self.kernel_size}"
