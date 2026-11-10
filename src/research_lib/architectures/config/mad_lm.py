"""Configuration dataclass for the MAD-style language model."""

from dataclasses import dataclass


@dataclass
class MadLMConfig:
    r"""Configuration for :class:`~research_lib.architectures.mad_lm.MadLM`.

    Architecture fields only. Which sequence mixer goes into each mixer
    block is not a config concern: the model takes already-built mixer
    modules (``seq_mixers``); the experiment builds them.

    Args:
        vocab_size: Vocabulary size ``V``.
        d_model: Model dimension :math:`d`.
        norm_eps: Epsilon of every LayerNorm.
        swiglu_multiple_of: Round the SwiGLU inner width to a multiple of
            this (the MAD convention).

    Raises:
        ValueError: Non-positive ``d_model`` / ``vocab_size``.
    """

    vocab_size: int = 16
    d_model: int = 128
    norm_eps: float = 1e-5
    swiglu_multiple_of: int = 16

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
