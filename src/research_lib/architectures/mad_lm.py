"""MAD-style language-model harness with a pluggable sequence mixer.

The harness follows MAD's backbone (Poli et al., 2024, "Mechanistic
Design and Scaling of Hybrid Architectures", arXiv:2403.17844,
``mad/model/language_model.py``): a token embedding with **no position
embedding**, and for each sequence mixer two residual blocks each with
pre-norm LayerNorm — ``x = x + mixer(norm(x))`` then ``x = x +
SwiGLU(norm(x))`` — a final LayerNorm and a ``nn.Linear`` LM head with a
bias.  With two mixers this is MAD's ``[mixer, swiglu, mixer, swiglu]``.

:class:`MadLM` is mixer-agnostic: it receives one already-built module
per mixer position (``seq_mixers``), like :class:`~research_lib.architectures.zoology_lm.ZoologyLM`.
Each mixer maps ``(B, T, d_model)`` to ``(B, T, d_model)``, or to a tuple
whose **first** element is that (the block keeps only the first element).
The MAD backbone's weight initialisation is applied to the backbone
modules only (embedding ``normal(0, 0.02)``; the head's and the SwiGLU's
linear biases zeroed; each LayerNorm at its default); the mixers keep
their own initialisation, so the layer under test is identical across
harnesses.

Run tests with::

    pytest src/research_lib/architectures/tests/test_mad_lm.py -v
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import nn

from research_lib.layers.swiglu import SwiGLU

from .config.mad_lm import MadLMConfig

logger = logging.getLogger(__name__)


class _MadResidualBlock(nn.Module):
    """One mixer block of the MAD backbone: mixer then SwiGLU, each pre-norm
    (LayerNorm), with ``x = x + layer(norm(x))``."""

    def __init__(
        self,
        seq_mixer: nn.Module,
        d_model: int,
        norm_eps: float,
        swiglu_multiple_of: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        f = {"device": device, "dtype": dtype}
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps, **f)
        self.mixer = seq_mixer
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps, **f)
        self.swiglu = SwiGLU(
            d_model, multiple_of=swiglu_multiple_of, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``x (Tensor)`` of shape ``(B, T, d)``; returns ``(B, T, d)``."""
        out = self.mixer(self.norm1(x))
        if isinstance(out, tuple):  # a mixer may return (y, cache); keep y
            out = out[0]
        x = x + out
        x = x + self.swiglu(self.norm2(x))
        return x


class MadLM(nn.Module):
    r"""MAD-style causal LM over already-built sequence mixers.

    Token embedding (no positional embedding) → ``2 * len(seq_mixers)``
    residual blocks (``[mixer, swiglu]`` per mixer) → final ``nn.LayerNorm``
    → ``nn.Linear`` LM head (bias ``True``).  Embeddings initialised from
    :math:`\mathcal N(0, 0.02)`; linear biases zeroed; each LayerNorm at
    its default; the mixers keep their own initialisation.

    Args:
        config (MadLMConfig): The architecture configuration.
        seq_mixers (Sequence[nn.Module]): One already-built mixer per
            mixer position.  Mixer contract: the output is
            ``(B, T, d_model)``, or a tuple whose first element is that.
        device: Optional torch device.
        dtype: Optional torch dtype.

    Shape:
        - Input: :math:`(B, T)` token ids
        - Output: :math:`(B, T, V)` logits

    Examples::

        >>> from research_lib.architectures.config import MadLMConfig
        >>> from torch import nn as _nn
        >>> model = MadLM(MadLMConfig(vocab_size=16, d_model=32),
        ...               [_nn.Identity()])
        >>> logits = model(torch.zeros(2, 8, dtype=torch.long))
        >>> logits.shape
        torch.Size([2, 8, 16])
    """

    def __init__(
        self,
        config: MadLMConfig,
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
                _MadResidualBlock(
                    seq_mixer=mixer,
                    d_model=d,
                    norm_eps=config.norm_eps,
                    swiglu_multiple_of=config.swiglu_multiple_of,
                    device=device,
                    dtype=dtype,
                )
                for mixer in mixers
            ]
        )
        self.norm_f = nn.LayerNorm(d, eps=config.norm_eps, **f)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=True, **f)
        # MAD's backbone init, applied to the backbone modules only:
        # embeddings normal(0, 0.02); linear biases zeroed; LayerNorms at
        # their defaults.  The mixers keep their own init.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The final hidden states, after the final LayerNorm, before the head.

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
        """Apply the LM head to hidden states of any leading shape.

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
