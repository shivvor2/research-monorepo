from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ..layers.activations import SquaredReLU
from ..layers.attention import RotaryMultiheadAttention
from ..layers.feed_forward import FeedForward
from ..layers.logit_clipping import TanhSoftCapping
from .config import NanoGPTConfig


class TransformerBlock(nn.Module):
    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        self.norm_1 = nn.RMSNorm(config.n_embd)
        self.attn = RotaryMultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            use_xpos=True,
            bias=config.bias,
            batch_first=True,
        )
        self.norm_2 = nn.RMSNorm(config.n_embd)
        self.ff = FeedForward(
            in_features=config.n_embd,
            hidden_features=config.ff_dim,
            activation=SquaredReLU(),
            bias=config.bias,
            dropout=config.dropout,
        )

    # Apply casual masking to attention?
    # To do sparse attention, apply a left diagonal mask
    def forward(
        self,
        x,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        # Attention and Post norm
        # Replace Residuals with mHC later
        attn_output, _ = self.attn(
            x,
            x,
            x,  # Self-attention: Q=K=V
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,  # Use SDPA for efficiency
            is_causal=True,  # Causal masking for decoder
        )
        x_1 = x + attn_output
        x_1 = self.norm_1(x_1)
        x = x + x_1

        # Position-Wise Feed Forward and post norm
        x = self.norm_2(self.ff(x))

        return x


class ModdedNanoGPT(nn.Module):
    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.n_embd,
            padding_idx=config.padding_idx,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(config=config) for _ in range(config.n_layer)]
        )

        self.norm_f = nn.RMSNorm(config.n_embd)
        self.output = nn.Linear(
            in_features=config.n_embd, out_features=config.vocab_size, bias=config.bias
        )
        self.logit_clipping = TanhSoftCapping()

    def forward(self, x, attn_mask=None):

        # Embeddings
        x = self.embedding(x)

        for b in self.blocks:
            x = b(x, attn_mask)

        x = self.norm_f(x)
        x = self.output(x)
        x = self.logit_clipping(x)

        return x
