import torch
import torch.nn as nn
import torch.optim as optim
from rotary_embedding_torch import RotaryEmbedding

from ..layers.activations import SquaredReLU
from ..layers.attention import RotaryMultiheadAttention
from ..layers.feed_forward import FeedForward
from ..layers.logit_clipping import TanhSoftCapping
from .config import NanoGPTConfig


class TransformerBlock(nn.Module):
    def __init__(self, config: NanoGPTConfig):
        super.__init__()
        self.norm_1 = nn.RMSNorm(config.n_embd)
        self.attn = RotaryMultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_heads,
            dropout=config.dropout,
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
    def forward(self, x, attn_mask=None):
        # Attention and Post norm
        # Replace Residuals with mHC later
        x_1 = self.attn(x, x, x, attn_mask=attn_mask, is_casual=True)
        x_1 = self.norm1(x)
        x = x + x_1

        # Position-Wise Feed Forward and post norm
        x_2 = self.ff(x)
        x_2 = self.norm_2(x)
        x = x + x_2

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

        # TODO Initialize Casual Mask by default (prevent model from seeing future tokens)
        #

        for b in self.blocks:
            x = b(x, attn_mask)

        x = self.norm_f(x)
        x = self.output(x)
        x = self.logit_clipping(x)

        return x
