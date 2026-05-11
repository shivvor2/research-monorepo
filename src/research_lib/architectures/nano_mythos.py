"""
NanoMythos: Hybrid Attention + Recurrent Depth + Attention Residuals.

This architecture combines:
    1. Hybrid attention (KDA linear attention + MHA full attention)
    2. Recurrent depth (weight-shared loop phase with LTI + ACT)
    3. Attention Residuals (replacing standard residual connections across depth)

The model is structured in three phases:
    - Prelude: non-looping transformer sublayers, residual-free, AttnRes-managed
    - Loop: recurrent transformer blocks (each block iterates internally and
      produces ONE output that becomes a single AttnRes depth layer)
    - Coda: non-looping transformer sublayers, residual-free, AttnRes-managed

Key design principle:
    The recurrent block's internal iterations are opaque to AttnRes. Each
    recurrent block contributes exactly 1 depth layer (its final output),
    just like a standard transformer block contributes 2 depth layers (attn + MLP).
    The intermediates inside the recurrent loop are NOT exposed because they are
    partial computations converging toward one output, not independent layer
    contributions.

AttnRes depth layer count:
    n_prelude_blocks * 2 + n_loop_blocks + n_coda_blocks * 2

Dependencies:
    - flash_attn_res (pip install flash-attn-res)
    - fla (pip install flash-linear-attention)
    - research_lib (this package)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from fla.layers import KimiDeltaAttention
from flash_attn_res.ops.phase_1 import phase_1_batched_attention_triton_op
from flash_attn_res.ops.phase_2 import phase_2_online_softmax_merge_triton_op
from torch.utils.checkpoint import checkpoint

from ..layers.activations import SquaredReLU
from ..layers.attention import RotaryMultiheadAttention
from ..layers.feed_forward import FeedForward
from ..layers.logit_clipping import TanhSoftCapping
from ..layers.recurrent_block import (
    RecurrentBlock,
    RecurrentTransformerBlock,
    SelfAttentionWrapper,
)
from .config import NanoMythosConfig

EPS = 1e-6


# ---------------------------------------------------------------------------
# Attention pattern generation
# ---------------------------------------------------------------------------


def build_attention_pattern(n_blocks: int, linear_to_full_ratio: int) -> List[str]:
    """Generate the attention type sequence for a phase.

    Args:
        n_blocks: Number of transformer blocks in this phase.
        linear_to_full_ratio: Number of KDA blocks per 1 MHA block.
            The repeating unit is [KDA]*ratio + [MHA].

    Returns:
        List of 'kda' or 'mha' strings, length n_blocks.
        Edge case: if n_blocks is not a multiple of (ratio + 1),
        the final block is forced to MHA.

    Examples:
        >>> build_attention_pattern(6, 2)
        ['kda', 'kda', 'mha', 'kda', 'kda', 'mha']
        >>> build_attention_pattern(5, 2)
        ['kda', 'kda', 'mha', 'kda', 'mha']
        >>> build_attention_pattern(4, 1)
        ['kda', 'mha', 'kda', 'mha']
    """
    if n_blocks == 0:
        return []

    period = linear_to_full_ratio + 1
    pattern = []
    for i in range(n_blocks):
        pos_in_period = i % period
        if pos_in_period < linear_to_full_ratio:
            pattern.append("kda")
        else:
            pattern.append("mha")

    # Edge case: force last block to MHA if phase doesn't end cleanly
    if n_blocks % period != 0:
        pattern[-1] = "mha"

    return pattern


# ---------------------------------------------------------------------------
# Residual-free sublayers (for prelude/coda, AttnRes-managed)
# ---------------------------------------------------------------------------


class KDASublayer(nn.Module):
    """Residual-free KDA attention sublayer for AttnRes.

    Wraps fla's KimiDeltaAttention. Applies RMSNorm before attention.
    Returns raw attention output (no residual addition).
    """

    def __init__(self, config: NanoMythosConfig, layer_idx: int = 0):
        super().__init__()
        self.norm = nn.RMSNorm(config.n_embd)
        self.attn = KimiDeltaAttention(
            hidden_size=config.n_embd,
            expand_v=config.kda_expand_v,
            head_dim=config.kda_head_dim,
            num_heads=config.n_embd // config.kda_head_dim,
            mode=config.kda_mode,
            use_short_conv=config.kda_use_short_conv,
            conv_size=config.kda_conv_size,
            layer_idx=layer_idx,
        )

    def forward(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        normed = self.norm(h)
        out, _, _ = self.attn(normed)
        return out


class MHASublayer(nn.Module):
    """Residual-free MHA (full attention) sublayer for AttnRes.

    Wraps RotaryMultiheadAttention with RMSNorm pre-norm.
    Returns raw attention output (no residual addition).
    """

    def __init__(self, config: NanoMythosConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.n_embd)
        self.attn = RotaryMultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.dropout,
            use_xpos=True,
            bias=config.bias,
            batch_first=True,
        )

    def forward(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        normed = self.norm(h)
        out, _ = self.attn(
            normed,
            normed,
            normed,
            key_padding_mask=None,
            attn_mask=kwargs.get("attn_mask", None),
            need_weights=False,
            is_causal=True,
        )
        return out


class MLPSublayer(nn.Module):
    """Residual-free MLP sublayer for AttnRes."""

    def __init__(self, config: NanoMythosConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.n_embd)
        self.ff = FeedForward(
            in_features=config.n_embd,
            hidden_features=config.ff_dim,
            activation=SquaredReLU(),
            bias=config.bias,
            dropout=config.dropout,
        )

    def forward(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.ff(self.norm(h))


# ---------------------------------------------------------------------------
# Recurrent block wrapper for the loop phase
# ---------------------------------------------------------------------------


def _build_recurrent_block(
    config: NanoMythosConfig,
    attn_type: str,
    layer_idx: int,
) -> RecurrentBlock:
    """Build a single RecurrentBlock with the specified attention type.

    The RecurrentBlock wraps a RecurrentTransformerBlock (which has attn + MLP
    with internal residuals, FiLM conditioning, etc.) and provides the LTI,
    ACT, and loop-index embedding shell.

    Args:
        config: Model config.
        attn_type: 'kda' or 'mha'.
        layer_idx: Layer index for KDA cache keying.

    Returns:
        A RecurrentBlock instance.
    """
    # Build the attention module
    if attn_type == "kda":
        attention = KimiDeltaAttention(
            hidden_size=config.n_embd,
            expand_v=config.kda_expand_v,
            head_dim=config.kda_head_dim,
            num_heads=config.n_embd // config.kda_head_dim,
            mode=config.kda_mode,
            use_short_conv=config.kda_use_short_conv,
            conv_size=config.kda_conv_size,
            layer_idx=layer_idx,
        )
        # KDA's forward signature: (hidden_states, attention_mask, ...)
        # Wrap it so RecurrentTransformerBlock can call it as attn(x, **kwargs)
        attention = _KDAWrapper(attention)
    else:
        attention = SelfAttentionWrapper(
            RotaryMultiheadAttention(
                embed_dim=config.n_embd,
                num_heads=config.n_head,
                dropout=config.dropout,
                use_xpos=True,
                bias=config.bias,
                batch_first=True,
            )
        )

    feedforward = FeedForward(
        in_features=config.n_embd,
        hidden_features=config.ff_dim,
        activation=SquaredReLU(),
        bias=config.bias,
        dropout=config.dropout,
    )

    core_block = RecurrentTransformerBlock(
        dim=config.n_embd,
        attention=attention,
        feedforward=feedforward,
        max_loop_iters=config.max_loop_iters,
        film_hidden=config.film_hidden,
        dropout=config.dropout,
        use_film=True,
    )

    return RecurrentBlock(
        dim=config.n_embd,
        core_block=core_block,
        max_loop_iters=config.max_loop_iters,
        act_threshold=config.act_threshold,
        use_lti=config.use_lti,
        loop_dim_fraction=config.loop_dim_fraction,
        loop_theta=config.loop_theta,
    )


class _KDAWrapper(nn.Module):
    """Wraps KimiDeltaAttention to accept a single input tensor x.

    KDA's forward expects (hidden_states, attention_mask, past_key_values, ...).
    This wrapper calls it as self-attention with the appropriate signature so
    it can be used inside RecurrentTransformerBlock.
    """

    def __init__(self, kda: KimiDeltaAttention):
        super().__init__()
        self.kda = kda

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        out, _, _ = self.kda(x)
        return out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class NanoMythosAttnRes(nn.Module):
    """NanoMythos: Hybrid Attention + Recurrent Depth + Attention Residuals.

    Architecture overview:
        Embedding → [AttnRes over all depth layers] → RMSNorm → Linear → Logit cap

    AttnRes depth layers:
        - Prelude: n_prelude_blocks * 2 layers (attn sublayer + MLP sublayer each)
        - Loop: n_loop_blocks layers (each recurrent block = 1 layer)
        - Coda: n_coda_blocks * 2 layers (attn sublayer + MLP sublayer each)
        Total: n_prelude_blocks*2 + n_loop_blocks + n_coda_blocks*2

    The loop phase uses RecurrentBlock from research_lib.layers.recurrent_block.
    Each RecurrentBlock iterates internally (LTI + ACT + FiLM) and produces
    exactly ONE output. This output is what AttnRes sees as a single depth layer.
    The internal loop iterations are fully opaque to AttnRes.
    """

    def __init__(self, config: NanoMythosConfig):
        super().__init__()
        self.config = config

        # --- Embedding ---
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.n_embd,
            padding_idx=config.padding_idx,
        )

        # --- Build attention patterns for each phase ---
        prelude_pattern = build_attention_pattern(
            config.n_prelude_blocks, config.linear_to_full_ratio
        )
        loop_pattern = build_attention_pattern(
            config.n_loop_blocks, config.linear_to_full_ratio
        )
        coda_pattern = build_attention_pattern(
            config.n_coda_blocks, config.linear_to_full_ratio
        )

        # --- Prelude sublayers (residual-free, AttnRes-managed) ---
        # Each block contributes 2 sublayers: attn + MLP
        self.prelude_layers = nn.ModuleList()
        layer_idx = 0
        for attn_type in prelude_pattern:
            if attn_type == "kda":
                self.prelude_layers.append(KDASublayer(config, layer_idx=layer_idx))
            else:
                self.prelude_layers.append(MHASublayer(config))
            self.prelude_layers.append(MLPSublayer(config))
            layer_idx += 1

        # --- Loop phase (RecurrentBlocks, each = 1 AttnRes depth layer) ---
        self.loop_blocks = nn.ModuleList()
        for attn_type in loop_pattern:
            self.loop_blocks.append(
                _build_recurrent_block(config, attn_type, layer_idx)
            )
            layer_idx += 1

        # --- Coda sublayers (residual-free, AttnRes-managed) ---
        self.coda_layers = nn.ModuleList()
        for attn_type in coda_pattern:
            if attn_type == "kda":
                self.coda_layers.append(KDASublayer(config, layer_idx=layer_idx))
            else:
                self.coda_layers.append(MHASublayer(config))
            self.coda_layers.append(MLPSublayer(config))
            layer_idx += 1

        # --- Output head ---
        self.norm_f = nn.RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.logit_clipping = TanhSoftCapping()

        # --- AttnRes parameters ---
        n_prelude_sublayers = len(self.prelude_layers)  # n_prelude_blocks * 2
        n_loop_layers = len(self.loop_blocks)  # n_loop_blocks (1 each)
        n_coda_sublayers = len(self.coda_layers)  # n_coda_blocks * 2

        self._n_prelude_sublayers = n_prelude_sublayers
        self._n_loop_layers = n_loop_layers
        self._n_coda_sublayers = n_coda_sublayers

        n_total_depth_layers = n_prelude_sublayers + n_loop_layers + n_coda_sublayers

        # Paper: "All pseudo-query vectors must be initialized to zero."
        self.pseudo_queries = nn.Parameter(
            torch.zeros(n_total_depth_layers, config.n_embd)
        )

        # AttnRes block size for checkpointing
        self.block_size = config.effective_attnres_block_size

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input token IDs [B, T].
            attn_mask: Optional attention mask for MHA sublayers.
            n_loops: Override number of recurrent iterations (for inference
                depth extrapolation). Defaults to config.max_loop_iters.

        Returns:
            Logits tensor [B, T, vocab_size].
        """
        x = self.embedding(x)  # [B, T, D] — b0 in AttnRes paper

        eps = EPS
        block_size = self.block_size
        kwargs = {"attn_mask": attn_mask, "is_causal": True}

        n_prelude = self._n_prelude_sublayers
        n_loop = self._n_loop_layers
        n_coda = self._n_coda_sublayers
        n_loops_actual = n_loops if n_loops is not None else self.config.max_loop_iters

        # All depth layers are processed through a unified AttnRes loop.
        # We build one flat list of "layer functions" that the AttnRes block
        # loop iterates over.

        blocks = [x]  # b0 = embedding output

        # Assemble all pseudo-queries and layer callables in order
        all_queries = self.pseudo_queries  # [n_total, D]

        # Build a dispatch list: each entry is a callable(h) -> output
        # For prelude/coda: residual-free sublayer
        # For loop: RecurrentBlock (takes h as input, returns single output)
        layer_fns: List = []

        # Prelude sublayers
        for i in range(n_prelude):
            layer = self.prelude_layers[i]
            layer_fns.append(lambda h, _layer=layer: _layer(h, **kwargs))

        # Loop blocks (each RecurrentBlock is 1 depth layer)
        for i in range(n_loop):
            block = self.loop_blocks[i]
            # RecurrentBlock.forward(h, e, n_loops, **kwargs)
            # h = AttnRes-aggregated input, e = frozen copy for LTI
            layer_fns.append(
                lambda h, _block=block, _n=n_loops_actual: _block(
                    h, h.detach(), n_loops=_n, **kwargs
                )
            )

        # Coda sublayers
        for i in range(n_coda):
            layer = self.coda_layers[i]
            layer_fns.append(lambda h, _layer=layer: _layer(h, **kwargs))

        n_total = len(layer_fns)
        assert n_total == n_prelude + n_loop + n_coda

        # --- Unified AttnRes block loop ---
        for block_start in range(0, n_total, block_size):
            num_queries = min(block_size, n_total - block_start)
            bq = all_queries[block_start : block_start + num_queries]

            def run_attnres_block(
                bq_arg,
                *prev_blocks,
                _block_start=block_start,
                _num_queries=num_queries,
            ):
                values = torch.stack(prev_blocks, dim=0)
                phase1_out, phase1_lse = phase_1_batched_attention_triton_op(
                    values, bq_arg, eps
                )
                # Cast to values dtype because flash-attn-res triton kernels
                # Hardcodes output to bf16, no penalty since we train in bf16 anyways
                # See https://github.com/catswe/flash-attention-residuals/blob/main/src/flash_attn_res/ops/phase_1.py#L25
                phase1_out = phase1_out.to(values.dtype)
                bq_list = bq_arg.unbind(0)
                p1_out_list = phase1_out.unbind(0)
                p1_lse_list = phase1_lse.unbind(0)

                curr_block = None
                for i in range(_num_queries):
                    layer_fn = layer_fns[_block_start + i]
                    if i == 0:
                        h_in = p1_out_list[i]
                        out = layer_fn(h_in)
                        curr_block = out
                    else:
                        # Same triton kernel output dtype problem
                        h_in = phase_2_online_softmax_merge_triton_op(
                            curr_block,
                            bq_list[i],
                            p1_out_list[i],
                            p1_lse_list[i],
                            eps,
                        ).to(values.dtype)
                        out = layer_fn(h_in)
                        curr_block = curr_block + out
                return curr_block

            curr_block = checkpoint(run_attnres_block, bq, *blocks, use_reentrant=False)
            blocks.append(curr_block)

        # --- Final AttnRes readout ---
        final_out, _ = phase_1_batched_attention_triton_op(
            torch.stack(blocks, dim=0),
            self.pseudo_queries[-1:],
            eps,
        )
        # Same triton kernel output dtype problem
        x = final_out.squeeze(0).to(blocks[0].dtype)

        # --- Output head ---
        x = self.norm_f(x)
        x = self.output(x)
        x = self.logit_clipping(x)
        return x
