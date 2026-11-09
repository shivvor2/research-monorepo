"""A Swish-gated linear unit (SwiGLU) feed-forward block.

Computes ``w3(silu(w1 x) * w2 x)`` with no biases and an inner width
rounded up to a multiple (the convention of the MAD backbone — Poli et
al., 2024, "Mechanistic Design and Scaling of Hybrid Architectures",
arXiv:2403.17844, ``mad/model/layers/mlp.py``).

Run tests with::

    pytest src/research_lib/layers/tests/test_swiglu.py -v
"""

from __future__ import annotations

import torch
from torch import nn


class SwiGLU(nn.Module):
    r"""A bias-free SwiGLU block: :math:`W_3(\mathrm{SiLU}(W_1 x) \odot
    W_2 x)`.

    Args:
        d_model: Input and output width ``d``.
        hidden_dim: Inner width; when omitted it is MAD's default
            ``multiple_of * ceil((8 d / 3) / multiple_of)`` (352 at
            ``d_model=128``, ``multiple_of=16``; e.g. 18 at ``d_model=7``,
            ``multiple_of=3``), rounded **up** to a multiple of
            ``multiple_of`` only then.
        multiple_of: Round the default inner width to a multiple of this.
        device: Optional torch device.
        dtype: Optional torch dtype.

    Shape:
        - Input: ``(..., d_model)``
        - Output: ``(..., d_model)``
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        multiple_of: int = 16,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        if multiple_of < 1:
            raise ValueError(f"multiple_of must be >= 1, got {multiple_of}")
        f = {"device": device, "dtype": dtype}
        if hidden_dim is None:
            inner = int(8 * d_model / 3)
            hidden_dim = multiple_of * ((inner + multiple_of - 1) // multiple_of)
        elif hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        self.d_model = d_model
        self.hidden_dim = int(hidden_dim)
        self.w1 = nn.Linear(d_model, self.hidden_dim, bias=False, **f)
        self.w2 = nn.Linear(d_model, self.hidden_dim, bias=False, **f)
        self.w3 = nn.Linear(self.hidden_dim, d_model, bias=False, **f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``x (Tensor)`` of shape ``(..., d)``; returns ``(..., d)``."""
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))
