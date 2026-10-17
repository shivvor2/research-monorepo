"""Configuration dataclass for the Zoology-style language model."""

from dataclasses import dataclass
from enum import Enum


class StateMixer(str, Enum):
    """What follows the sequence mixer in each block.

    - ``NONE``: identity (the sequence mixer is the whole block).
    - ``MLP``: a feed-forward MLP with hidden size ``ff_mult * d_model``.
    """

    NONE = "none"
    MLP = "mlp"


@dataclass
class ZoologyLMConfig:
    r"""Configuration for :class:`~research_lib.architectures.zoology_lm.ZoologyLM`.

    Architecture fields only. Which sequence mixer goes into each block
    is not a config concern: the model takes already-built mixer modules
    (``seq_mixers``); the experiment builds them (e.g.
    :func:`~research_lib.architectures.zoology_lm.zoology_mixers` with a
    ``make_mixer`` factory).

    Args:
        vocab_size: Vocabulary size ``V``.
        d_model: Model dimension :math:`d`.
        state_mixer: :class:`StateMixer` member or its string value
            (Hydra passes strings); ``"none"`` by default.
        ff_mult: MLP hidden multiplier (``hidden = ff_mult * d_model``),
            used only when ``state_mixer`` is ``MLP``.
        norm_eps: Epsilon of every RMSNorm.

    Raises:
        ValueError: Unknown ``state_mixer`` string, or non-positive
            ``d_model`` / ``ff_mult``.
    """

    vocab_size: int = 8192
    d_model: int = 32
    # The enum member or its string value (coerced in __post_init__, so
    # Hydra's strings work).
    state_mixer: StateMixer | str = StateMixer.NONE
    ff_mult: int = 4
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        self.state_mixer = StateMixer(self.state_mixer)
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.ff_mult < 1:
            raise ValueError(f"ff_mult must be >= 1, got {self.ff_mult}")
