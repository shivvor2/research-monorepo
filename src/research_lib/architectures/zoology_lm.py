"""Zoology-style language-model harness with a pluggable sequence mixer.

The harness (checked against the Zoology code, ``zoology/model.py`` /
``mixers/hybrid.py``): token embedding, **no positional embedding**,
pre-norm RMSNorm residual blocks, an optional state mixer, a final
RMSNorm and an untied ``nn.Linear`` LM head.  ``forward(input_ids)``
returns logits only; there is no generation API.

:class:`ZoologyLM` is mixer-agnostic: it receives one already-built
module per block (``seq_mixers``, so ``n_layer = len(seq_mixers)``), like
:class:`~research_lib.layers.recurrent_block.RecurrentTransformerBlock`
receives its attention module.  Each mixer maps ``(B, T, d_model)`` to
``(B, T, d_model)``, or to a tuple whose **first** element is that (the
block keeps only the first element, so mixers that also return a cache
need no wrapper).  Parameter names are ``blocks.{i}.mixer.*``, so
checkpoints and the weight-decay patterns are unaffected by who builds
the mixer.

:func:`zoology_mixers` lays out the blocks for the Zoology layer
patterns (:class:`LayerPattern`): even blocks a ``BaseConv`` and odd
blocks ``make_mixer()`` (``ZOOLOGY``), or ``make_mixer()`` in every block
(``MIXER_X2``).  Building a specific mixer is the experiment's job: it
passes its own ``make_mixer`` factory (e.g. constructing
:class:`~research_lib.layers.attention.MultistateGatedDelta` from its
own config), so nothing in this file needs editing when a new mixer
arrives.

Run tests with::

    pytest src/research_lib/architectures/tests/test_zoology_lm.py -v
"""

import logging
from collections.abc import Callable, Sequence
from enum import Enum

import torch
import torch.nn as nn

from research_lib.layers.base_conv import BaseConv
from research_lib.layers.feed_forward import FeedForward

from .config.zoology_lm import StateMixer, ZoologyLMConfig

logger = logging.getLogger(__name__)


class LayerPattern(str, Enum):
    """How the sequence mixers are laid out over the blocks.

    - ``ZOOLOGY``: even blocks ``BaseConv``, odd blocks the given mixer
      (Zoology's ``Hybrid([BaseConv, mixer])`` picks by ``layer_idx % 2``).
    - ``MIXER_X2``: the given mixer in every block (the name is kept:
      configs use it).
    """

    ZOOLOGY = "zoology"
    MIXER_X2 = "mixer_x2"


def zoology_mixers(
    n_layer: int,
    d_model: int,
    make_mixer: Callable[[], nn.Module],
    pattern: LayerPattern | str = LayerPattern.ZOOLOGY,
    conv_kernel: int = 3,
    device=None,
    dtype=None,
) -> list[nn.Module]:
    """One already-built sequence mixer per block, per the layer pattern.

    Args:
        n_layer: Number of blocks (>= 1).
        d_model: Model dimension (the BaseConv width).
        make_mixer: Factory building one mixer (called once per mixer
            block; the caller's config decides what it builds).
        pattern: :class:`LayerPattern` member or its string value
            (Hydra passes strings).
        conv_kernel: Kernel width of the BaseConv blocks.
        device: Optional torch device for the BaseConv blocks.
        dtype: Optional torch dtype for the BaseConv blocks.

    Returns:
        list of ``n_layer`` modules, position ``i`` being block ``i``'s
        mixer.

    Raises:
        ValueError: ``n_layer < 1`` or an unknown ``pattern`` string.
    """
    pattern = LayerPattern(pattern)
    if n_layer < 1:
        raise ValueError(f"n_layer must be >= 1, got {n_layer}")
    mixers: list[nn.Module] = []
    for i in range(n_layer):
        if pattern is LayerPattern.ZOOLOGY and i % 2 == 0:
            mixers.append(
                BaseConv(
                    d_model=d_model,
                    kernel_size=conv_kernel,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            mixers.append(make_mixer())
    return mixers


class _ZoologyBlock(nn.Module):
    """One pre-norm residual block: sequence mixer, then optional MLP."""

    def __init__(
        self,
        seq_mixer: nn.Module,
        d_model: int,
        norm_eps: float,
        state_mixer: StateMixer | str = StateMixer.NONE,
        ff_mult: int = 4,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.mixer = seq_mixer
        self.state_mixer = StateMixer(state_mixer)
        if self.state_mixer is StateMixer.MLP:
            self.norm2 = nn.RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
            # FeedForward takes no factory kwargs; move it instead.
            self.mlp = FeedForward(d_model, hidden_features=ff_mult * d_model)
            if device is not None or dtype is not None:
                self.mlp.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``x (Tensor)`` of shape ``(B, T, d)``; returns ``(B, T, d)``."""
        out = self.mixer(self.norm1(x))
        if isinstance(out, tuple):  # a mixer may return (y, cache); keep y
            out = out[0]
        x = x + out
        if self.state_mixer is StateMixer.MLP:
            x = x + self.mlp(self.norm2(x))
        return x


class ZoologyLM(nn.Module):
    r"""Zoology-style causal LM over already-built sequence mixers.

    Token embedding (no positional embedding) → ``n_layer`` pre-norm
    residual blocks (one per ``seq_mixers`` entry) → final ``nn.RMSNorm``
    → untied ``nn.Linear`` LM head.  Embeddings and head are initialised
    from :math:`\mathcal N(0, 0.02)`; the mixers keep their own
    initialisation.

    Args:
        config (ZoologyLMConfig): The architecture configuration.
        seq_mixers (Sequence[nn.Module]): One already-built mixer per
            block, so ``n_layer = len(seq_mixers)``.  Mixer contract: the
            output is ``(B, T, d_model)``, or a tuple whose first element
            is that (only the first element is kept).
        device: Optional torch device.
        dtype: Optional torch dtype.

    Shape:
        - Input: :math:`(B, T)` token ids
        - Output: :math:`(B, T, V)` logits

    Examples::

        >>> from research_lib.architectures.config import ZoologyLMConfig
        >>> mixers = zoology_mixers(2, 16, lambda: BaseConv(16))
        >>> model = ZoologyLM(ZoologyLMConfig(vocab_size=64, d_model=16), mixers)
        >>> logits = model(torch.zeros(2, 8, dtype=torch.long))
        >>> logits.shape
        torch.Size([2, 8, 64])
    """

    def __init__(
        self,
        config: ZoologyLMConfig,
        seq_mixers: Sequence[nn.Module],
        device=None,
        dtype=None,
    ):
        super().__init__()
        mixers = list(seq_mixers)
        if not mixers:
            raise ValueError("seq_mixers must contain at least one module")
        self.config = config
        d = config.d_model
        f = {"device": device, "dtype": dtype}

        self.embedding = nn.Embedding(config.vocab_size, d, **f)
        self.blocks = nn.ModuleList(
            [
                _ZoologyBlock(
                    seq_mixer=mixer,
                    d_model=d,
                    norm_eps=config.norm_eps,
                    state_mixer=config.state_mixer,
                    ff_mult=config.ff_mult,
                    **f,
                )
                for mixer in mixers
            ]
        )
        self.norm_f = nn.RMSNorm(d, eps=config.norm_eps, **f)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False, **f)
        # Init: embeddings and head normal(0, 0.02); norms ones (the
        # nn.RMSNorm default); the mixers keep their own init.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The final hidden states, after the final RMSNorm, before the head.

        The trunk (embedding, mixers, final norm) runs on every position;
        only :meth:`head` can be restricted to a subset of positions (the
        answer-position head of the synthetic pilot evaluates it only at
        answer positions, which is exact: nothing after the head mixes
        positions and the loss/metrics read only those positions).

        Args:
            input_ids (Tensor): Token ids of shape :math:`(B, T)`.

        Returns:
            Tensor: Hidden states of shape :math:`(B, T, d)`.
        """
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.norm_f(x)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the untied LM head to hidden states of any leading shape.

        Args:
            hidden (Tensor): Hidden states of shape :math:`(..., d)`.

        Returns:
            Tensor: Logits of shape :math:`(..., V)`.
        """
        return self.lm_head(hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the logits for every position.

        Args:
            input_ids (Tensor): Token ids of shape :math:`(B, T)`.

        Returns:
            Tensor: Logits of shape :math:`(B, T, V)`.
        """
        return self.head(self.hidden_states(input_ids))
