"""Configuration for NanoMythos architecture (AttnRes + Recurrent + Hybrid Attention)."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NanoMythosConfig:
    """Configuration for the NanoMythos model.

    The model is structured in three phases:
        1. Prelude  — non-looping, residual-free sublayers (AttnRes-managed)
        2. Loop     — RecurrentBlocks (each produces 1 AttnRes depth layer)
        3. Coda     — non-looping, residual-free sublayers (AttnRes-managed)

    Each prelude/coda transformer block contributes 2 AttnRes depth layers
    (attn sublayer + MLP sublayer). Each loop RecurrentBlock contributes 1
    AttnRes depth layer (its final output after all internal iterations).

    Total AttnRes depth layers:
        n_prelude_blocks * 2 + n_loop_blocks + n_coda_blocks * 2

    Attention mixing is controlled by ``linear_to_full_ratio``: for every
    ``linear_to_full_ratio`` blocks using KDA, 1 block uses MHA. If the
    phase length is not a multiple of (ratio + 1), the final block in
    that phase is forced to MHA.
    """

    # --- Core model dimensions ---
    vocab_size: int = 50257
    block_size: int = 1024
    n_embd: int = 768
    n_head: int = 12
    ff_dim: int = 2048
    bias: bool = False
    dropout: float = 0.0
    padding_idx: int = 0

    # --- Phase structure ---
    n_prelude_blocks: int = 2
    n_loop_blocks: int = 2
    n_coda_blocks: int = 2

    # --- Attention mixing ---
    linear_to_full_ratio: int = 1

    # --- Recurrent loop settings (for loop phase blocks only) ---
    max_loop_iters: int = 5
    act_threshold: float = 0.99
    use_lti: bool = True
    loop_dim_fraction: int = 8
    loop_theta: float = 10000.0
    film_hidden: int = 64

    # --- AttnRes settings ---
    attnres_block_size: Optional[int] = None

    # --- KDA-specific settings ---
    kda_head_dim: int = 128
    kda_expand_v: float = 1.0
    kda_use_short_conv: bool = True
    kda_conv_size: int = 4
    kda_mode: str = "chunk"

    @property
    def n_attnres_depth_layers(self) -> int:
        """Total AttnRes depth layers across all phases."""
        return self.n_prelude_blocks * 2 + self.n_loop_blocks + self.n_coda_blocks * 2

    @property
    def effective_attnres_block_size(self) -> int:
        """Computed AttnRes block size for checkpointing.

        Targets ~8 AttnRes blocks total, as recommended by the paper.
        """
        if self.attnres_block_size is not None:
            return self.attnres_block_size
        return max(1, self.n_attnres_depth_layers // 8)
