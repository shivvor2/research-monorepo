from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding as _RotaryEmbedding


class RotaryEmbedding(nn.Module):
    """Wrapper to match flash_attn's RotaryEmbedding API using lucidrains'."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self._rotary = _RotaryEmbedding(dim=dim, theta=base)
        self.interleaved = interleaved

    def forward(self, x: torch.Tensor, seqlen_offset: int = 0) -> torch.Tensor:
        # x: (B, S, H, D)
        return self._rotary.rotate_queries_or_keys(x, offset=seqlen_offset)
