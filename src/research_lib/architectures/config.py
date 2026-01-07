from dataclasses import dataclass
from typing import List, Union


@dataclass
class NanoGPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    ff_dim: int = 2048  # n_embd * 8/3 ?
    bias: bool = False
    dropout: float = 0.0
    padding_idx: int = 0
