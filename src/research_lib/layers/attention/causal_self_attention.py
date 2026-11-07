"""A causal self-attention wrapper over RotaryMultiheadAttention.

:class:`RotaryMultiheadAttention` has ``nn.MultiheadAttention``'s
interface (separate query / key / value tensors and masks), not a
mixer's. :class:`CausalSelfAttention` is a small wrapper with the mixer
interface a Zoology-style harness expects: it takes ``(B, T, d)``, runs
causal self-attention (query = key = value = input) through PyTorch's
scaled-dot-product attention, and returns ``(B, T, d)``. It is the
positive-control mixer of a synthetic recall project: a model that
should solve the task, so that a failure can be blamed on the task
rather than on the gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from research_lib.layers.attention.rotatory_mha import RotaryMultiheadAttention


class CausalSelfAttention(nn.Module):
    """Causal self-attention with the ``(B, T, d) → (B, T, d)`` mixer
    interface.

    Args:
        d_model: Model width ``d`` (also the head-count product).
        num_heads: Number of attention heads (``d_model`` divisible).
        rotary_dim: Rotary embedding dimension; ``None`` uses the full
            head width (``d_model // num_heads``).
        dropout: Dropout probability on the attention output weights
            (0 by default; off during evaluation).

    Shape:
        - Input: ``(B, T, d_model)``
        - Output: ``(B, T, d_model)``

    Examples:
        >>> attn = CausalSelfAttention(32, 2)
        >>> y = attn(torch.randn(2, 16, 32))
        >>> y.shape
        torch.Size([2, 16, 32])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rotary_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be a positive multiple of num_heads "
                f"({num_heads})"
            )
        head_width = d_model // num_heads
        if rotary_dim is not None and rotary_dim > head_width:
            raise ValueError(
                f"rotary_dim ({rotary_dim}) cannot exceed the head width "
                f"({head_width}); the rotary dimension rotates within a head"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.rotary_dim = rotary_dim
        resolved = rotary_dim if rotary_dim is not None else head_width
        if resolved % 2 != 0:
            raise ValueError(
                f"the rotary dimension must be even to pair sine/cosine rotary "
                f"embeddings, got d_model={d_model}, num_heads={num_heads}, "
                f"rotary_dim={resolved}; pick an even rotary dimension or an "
                f"even head width"
            )
        if resolved == 2:
            # rotary-embedding-torch 0.9.1 divides by the rotary frequency
            # stride, which is zero for dim == 2 (ZeroDivisionError).
            raise ValueError(
                f"a rotary dimension of 2 (d_model={d_model}, num_heads="
                f"{num_heads}, rotary_dim={rotary_dim}) fails inside "
                f"rotary-embedding-torch at the locked version; pick a larger "
                f"even rotary dimension or width"
            )
        self.attention = RotaryMultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rotary_dim=rotary_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal self-attention over the full sequence.

        Args:
            x (Tensor): Token states of shape ``(B, T, d_model)``.

        Returns:
            Tensor: The attention output of shape ``(B, T, d_model)``.
        """
        out, _ = self.attention(x, x, x, is_causal=True, need_weights=False)
        return out
