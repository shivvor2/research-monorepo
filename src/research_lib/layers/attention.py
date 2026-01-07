from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding


class RotaryMultiHeadAttention(nn.Module):
    """Custom MHA with Rotary Embeddings"""

    def __init__(
        self,
        config: NanoGPTConfig,
        embed_dim: int,
        num_heads: Any,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads

        self.W_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.W_o = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # RoPE dimension should be head_dim, not n_embd!
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
        self.dropout = config.dropout

    def forward(self, x):
        B, seq_len, _ = x.shape

        q = self.W_q(x).view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings BEFORE attention
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Efficient attention with causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.W_o(attn_output)
