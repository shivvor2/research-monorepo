"""
Drop-in replacement for nn.MultiheadAttention with Rotary Position Embeddings.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding


class RotaryMultiheadAttention(nn.Module):
    """
    Multi-Head Attention with Rotary Position Embeddings (RoPE).

    API designed to closely match nn.MultiheadAttention for drop-in replacement.
    Uses F.scaled_dot_product_attention for efficient computation.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that embed_dim will
                   be split across num_heads (i.e. each head will have dimension
                   embed_dim // num_heads).
        dropout: Dropout probability on attn_output_weights. Default: 0.0.
        bias: If True, adds bias to input/output projection layers. Default: True.
        kdim: Total number of features for keys. Default: None (uses kdim=embed_dim).
        vdim: Total number of features for values. Default: None (uses vdim=embed_dim).
        batch_first: If True, then the input and output tensors are provided
                     as (batch, seq, feature). Default: True.
        device: Device to place parameters on. Default: None.
        dtype: Data type for parameters. Default: None.
        rotary_dim: Dimension for rotary embeddings. Default: None (uses head_dim).
        rotary_base: Base for rotary embedding frequencies. Default: 10000.0.
        rotary_interleaved: Whether to use interleaved rotary style. Default: False.

    Examples:
        >>> # Basic usage
        >>> rotary_attn = RotaryMultiheadAttention(embed_dim=512, num_heads=8)
        >>> attn_output, attn_weights = rotary_attn(query, key, value)

        >>> # Self-attention with causal masking (most common for decoders)
        >>> attn_output, _ = rotary_attn(x, x, x, is_causal=True, need_weights=False)

        >>> # Drop-in replacement for nn.MultiheadAttention
        >>> # Before:
        >>> # self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
        >>> # After:
        >>> self.attn = RotaryMultiheadAttention(512, 8, batch_first=True)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # RoPE-specific arguments (lucidrains implementation)
        rotary_dim: Optional[int] = None,
        rotary_base: float = 10000.0,
        use_xpos: bool = False,  # NEW: length extrapolation
        xpos_scale_base: int = 512,  # NEW: xpos scale base
        interpolate_factor: float = 1.0,  # NEW: context extension (NTK-aware)
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # For self-attention optimization
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        factory_kwargs = {"device": device, "dtype": dtype}

        if self._qkv_same_embed_dim:
            # Fused QKV projection for self-attention (more efficient)
            self.in_proj_weight = nn.Parameter(
                torch.empty(3 * embed_dim, embed_dim, **factory_kwargs)
            )
            if bias:
                self.in_proj_bias = nn.Parameter(
                    torch.empty(3 * embed_dim, **factory_kwargs)
                )
            else:
                self.register_parameter("in_proj_bias", None)

            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None
        else:
            # Separate projections for cross-attention
            self.register_parameter("in_proj_weight", None)
            self.register_parameter("in_proj_bias", None)

            self.q_proj_weight = nn.Parameter(
                torch.empty(embed_dim, embed_dim, **factory_kwargs)
            )
            self.k_proj_weight = nn.Parameter(
                torch.empty(embed_dim, self.kdim, **factory_kwargs)
            )
            self.v_proj_weight = nn.Parameter(
                torch.empty(embed_dim, self.vdim, **factory_kwargs)
            )

            if bias:
                self.q_proj_bias = nn.Parameter(
                    torch.empty(embed_dim, **factory_kwargs)
                )
                self.k_proj_bias = nn.Parameter(
                    torch.empty(embed_dim, **factory_kwargs)
                )
                self.v_proj_bias = nn.Parameter(
                    torch.empty(embed_dim, **factory_kwargs)
                )
            else:
                self.register_parameter("q_proj_bias", None)
                self.register_parameter("k_proj_bias", None)
                self.register_parameter("v_proj_bias", None)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        # Rotary embeddings
        rotary_dim = rotary_dim if rotary_dim is not None else self.head_dim
        self.rotary_emb = RotaryEmbedding(
            dim=rotary_dim,
            theta=rotary_base,  # 'theta' not 'base'
            # interleaved is not a param here - lucidrains uses rotate_half style by default
            cache_if_possible=True,
            cache_max_seq_len=8192,
            seq_before_head_dim=False,  # We use (B, H, S, D) format internally
            use_xpos=use_xpos,  # NEW: support for xpos
            xpos_scale_base=xpos_scale_base,  # NEW: xpos scale base
            interpolate_factor=interpolate_factor,  # NEW: for context length extension
        )
        self.use_xpos = use_xpos

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier uniform initialization."""
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
            if self.in_proj_bias is not None:
                nn.init.zeros_(self.in_proj_bias)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            if self.q_proj_bias is not None:
                nn.init.zeros_(self.q_proj_bias)
                nn.init.zeros_(self.k_proj_bias)
                nn.init.zeros_(self.v_proj_bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_dim: int = -2,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys using lucidrains' implementation.

        Args:
            q: Queries of shape (B, num_heads, L, head_dim)
            k: Keys of shape (B, num_heads, S, head_dim)
            seq_dim: Dimension where sequence length is. Default -2 for (B, H, S, D)
            offset: Position offset for KV-cache inference

        Returns:
            Tuple of rotated (q, k)
        """
        if self.use_xpos:
            # xpos requires both q and k to be rotated together for proper scaling
            q, k = self.rotary_emb.rotate_queries_and_keys(q, k, seq_dim=seq_dim)
        else:
            # Standard RoPE - can rotate independently
            q = self.rotary_emb.rotate_queries_or_keys(
                q, seq_dim=seq_dim, offset=offset
            )
            k = self.rotary_emb.rotate_queries_or_keys(
                k, seq_dim=seq_dim, offset=offset
            )

        return q, k

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        seqlen_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention outputs using query, key, and value embeddings with
        rotary position embeddings.

        Args:
            query: Query embeddings of shape (L, N, E) when batch_first=False
                   or (N, L, E) when batch_first=True.
            key: Key embeddings of shape (S, N, E) when batch_first=False
                 or (N, S, E) when batch_first=True.
            value: Value embeddings of shape (S, N, E) when batch_first=False
                   or (N, S, E) when batch_first=True.
            key_padding_mask: If specified, a mask of shape (N, S) indicating which
                              elements within key to ignore for attention.
                              True = ignore, False = attend.
            need_weights: If True, returns attn_output_weights. Set to False for
                          best performance with SDPA. Default: True.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain
                       positions. Shape (L, S) or (N*num_heads, L, S).
                       True = ignore, False = attend (for bool masks).
            average_attn_weights: If True, returns attention weights averaged across
                                  heads. Default: True.
            is_causal: If True, applies a causal mask. Default: False.
            seqlen_offset: Position offset for rotary embeddings (for KV-cache).
                           Only used when use_xpos=False. Default: 0.

        Returns:
            attn_output: Attention outputs of shape (N, L, E) when batch_first=True,
                         or (L, N, E) when batch_first=False.
            attn_output_weights: Only returned when need_weights=True.
                                 Shape (N, L, S) if average_attn_weights=True,
                                 else (N, num_heads, L, S).
        """
        # Handle batch_first
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # Project Q, K, V
        if self._qkv_same_embed_dim and query is key and key is value:
            # Self-attention with fused projection
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            qkv = qkv.view(batch_size, tgt_len, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        elif self._qkv_same_embed_dim:
            # Different Q, K, V but same dimensions - use fused weight
            w_q, w_k, w_v = self.in_proj_weight.chunk(3)
            b_q, b_k, b_v = (
                self.in_proj_bias.chunk(3)
                if self.in_proj_bias is not None
                else (None, None, None)
            )
            q = F.linear(query, w_q, b_q)
            k = F.linear(key, w_k, b_k)
            v = F.linear(value, w_v, b_v)

            q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(
                1, 2
            )
            k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(
                1, 2
            )
            v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(
                1, 2
            )
        else:
            # Cross-attention with separate projections
            q = F.linear(query, self.q_proj_weight, getattr(self, "q_proj_bias", None))
            k = F.linear(key, self.k_proj_weight, getattr(self, "k_proj_bias", None))
            v = F.linear(value, self.v_proj_weight, getattr(self, "v_proj_bias", None))

            q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(
                1, 2
            )
            k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(
                1, 2
            )
            v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(
                1, 2
            )

        # Apply rotary embeddings to Q and K
        q, k = self._apply_rotary(q, k, seq_dim=-2, offset=seqlen_offset)

        # Prepare attention mask
        combined_mask = self._prepare_attention_mask(
            attn_mask, key_padding_mask, q.dtype, batch_size, tgt_len, src_len, q.device
        )

        # Compute attention
        dropout_p = self.dropout if self.training else 0.0

        if need_weights:
            attn_output, attn_weights = self._attention_with_weights(
                q, k, v, combined_mask, is_causal, dropout_p, average_attn_weights
            )
        else:
            # Use efficient SDPA (no attention weights returned)
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=combined_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
            attn_weights = None

        # Reshape: (B, num_heads, L, head_dim) -> (B, L, embed_dim)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, tgt_len, self.embed_dim)
        )

        # Output projection
        attn_output = self.out_proj(attn_output)

        # Handle batch_first for output
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_weights

    def _prepare_attention_mask(
        self,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
        batch_size: int,
        tgt_len: int,
        src_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Prepare and combine attention masks for SDPA."""
        combined_mask = None

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                combined_mask = torch.zeros(attn_mask.shape, dtype=dtype, device=device)
                combined_mask.masked_fill_(attn_mask, float("-inf"))
            else:
                combined_mask = attn_mask.to(dtype=dtype)

            # Expand 2D mask to 4D: (L, S) -> (1, 1, L, S)
            if combined_mask.dim() == 2:
                combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)
            # Expand 3D mask: (N*H, L, S) -> (N, H, L, S)
            elif combined_mask.dim() == 3:
                combined_mask = combined_mask.view(
                    batch_size, self.num_heads, tgt_len, src_len
                )

        if key_padding_mask is not None:
            # key_padding_mask: (N, S) -> (N, 1, 1, S)
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            padding_mask_float = torch.zeros(
                padding_mask.shape, dtype=dtype, device=device
            )
            padding_mask_float.masked_fill_(padding_mask, float("-inf"))

            if combined_mask is not None:
                combined_mask = combined_mask + padding_mask_float
            else:
                combined_mask = padding_mask_float

        return combined_mask

    def _attention_with_weights(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        combined_mask: Optional[torch.Tensor],
        is_causal: bool,
        dropout_p: float,
        average_attn_weights: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention with explicit weight computation."""
        tgt_len = q.shape[2]
        src_len = k.shape[2]

        scale = 1.0 / (self.head_dim**0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if combined_mask is not None:
            attn_weights = attn_weights + combined_mask

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(tgt_len, src_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights.masked_fill_(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_dropped = F.dropout(
            attn_weights, p=dropout_p, training=self.training
        )

        attn_output = torch.matmul(attn_weights_dropped, v)

        # Prepare output weights
        if average_attn_weights:
            output_weights = attn_weights.mean(dim=1)  # (N, L, S)
        else:
            output_weights = attn_weights  # (N, num_heads, L, S)

        return attn_output, output_weights

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"dropout={self.dropout}, "
            f"bias={self.in_proj_bias is not None or self.out_proj.bias is not None}, "
            f"batch_first={self.batch_first}, "
            f"head_dim={self.head_dim}, "
            f"kdim={self.kdim}, "
            f"vdim={self.vdim})"
        )
