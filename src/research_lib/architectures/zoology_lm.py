"""Zoology-style two-layer language model with a pluggable gated-delta mixer.

The pilot harness (``06_experiment_protocol.md`` §2), checked against the
Zoology code (``zoology/model.py``, ``mixers/hybrid.py``): a small causal
LM with **no positional embedding**, ``n_layer`` pre-norm residual blocks
whose sequence mixer is
:class:`~research_lib.layers.base_conv.BaseConv` (kernel 3) in even
layers and the gated-delta layer under test
(:class:`~research_lib.layers.attention.MultistateGatedDelta`) in odd
layers — Zoology's ``Hybrid([BaseConv, mixer])`` picks the mixer by
``layer_idx % 2``, so its "two-layer" recurrent models are one conv layer
plus one recurrent layer, which also keeps attribution clean.  The
``"mixer_x2"`` pattern puts the gated-delta mixer in every block (the
calibration alternative).

The gated-delta mixer is **always** :class:`MultistateGatedDelta` — with
``layout="single"`` it is the single-state reference layer — so every row
of the experiment grid is one class and gate modes / layouts / λ variants
are constructor arguments.

Run tests with::

    pytest src/research_lib/architectures/tests/test_zoology_lm.py -v
"""

import logging

import torch
import torch.nn as nn

from research_lib.layers.attention import MultistateGatedDelta
from research_lib.layers.base_conv import BaseConv
from research_lib.layers.feed_forward import FeedForward

from .config.zoology_lm import ZoologyLMConfig

logger = logging.getLogger(__name__)

_LAYER_PATTERNS = ("zoology", "mixer_x2")
_STATE_MIXERS = ("none", "mlp")


class _ZoologyBlock(nn.Module):
    """One pre-norm residual block: sequence mixer, then optional MLP."""

    def __init__(
        self,
        seq_mixer: nn.Module,
        d_model: int,
        norm_eps: float,
        state_mixer: str = "none",
        ff_mult: int = 4,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.mixer = seq_mixer
        self.state_mixer = state_mixer
        if state_mixer == "mlp":
            self.norm2 = nn.RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
            # FeedForward takes no factory kwargs; move it instead.
            self.mlp = FeedForward(d_model, hidden_features=ff_mult * d_model)
            if device is not None or dtype is not None:
                self.mlp.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``x (Tensor)`` of shape ``(B, T, d)``; returns ``(B, T, d)``."""
        out = self.mixer(self.norm1(x))
        if isinstance(out, tuple):  # gated-delta layers return (y, cache)
            out = out[0]
        x = x + out
        if self.state_mixer == "mlp":
            x = x + self.mlp(self.norm2(x))
        return x


class ZoologyLM(nn.Module):
    r"""Two-layer Zoology-style causal LM with a gated-delta layer under test.

    Token embedding (no positional embedding) → ``n_layer`` pre-norm
    residual blocks → final ``nn.RMSNorm`` → untied ``nn.Linear`` LM
    head.  No generation API; ``forward(input_ids)`` returns logits only.

    Args:
        config (ZoologyLMConfig): The model configuration.

    Shape:
        - Input: :math:`(B, T)` token ids
        - Output: :math:`(B, T, V)` logits

    Examples::

        >>> from research_lib.architectures.config import ZoologyLMConfig
        >>> model = ZoologyLM(ZoologyLMConfig(vocab_size=64, d_model=16,
        ...                                   t_max=32))
        >>> logits = model(torch.zeros(2, 8, dtype=torch.long))
        >>> logits.shape
        torch.Size([2, 8, 64])
    """

    def __init__(self, config: ZoologyLMConfig, device=None, dtype=None):
        super().__init__()
        self.config = config
        d = config.d_model
        f = {"device": device, "dtype": dtype}

        self.embedding = nn.Embedding(config.vocab_size, d, **f)
        self.blocks = nn.ModuleList(
            [
                _ZoologyBlock(
                    seq_mixer=self._make_seq_mixer(i, **f),
                    d_model=d,
                    norm_eps=config.norm_eps,
                    state_mixer=config.state_mixer,
                    ff_mult=config.ff_mult,
                    **f,
                )
                for i in range(config.n_layer)
            ]
        )
        self.norm_f = nn.RMSNorm(d, eps=config.norm_eps, **f)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False, **f)
        # Init: embeddings and head normal(0, 0.02); norms ones (the
        # nn.RMSNorm default); the gated-delta layers and BaseConv keep
        # their own init.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _make_seq_mixer(self, layer_idx: int, **f) -> nn.Module:
        """The sequence mixer of block ``layer_idx``.

        Args:
            layer_idx (int): Block index.
            **f: Optional torch ``device`` / ``dtype`` factory kwargs.

        Returns:
            The mixer module (:class:`BaseConv` or
            :class:`MultistateGatedDelta`).
        """
        c = self.config
        if c.layer_pattern == "zoology" and layer_idx % 2 == 0:
            return BaseConv(d_model=c.d_model, kernel_size=c.base_conv_kernel, **f)
        head_dim = c.head_dim if c.head_dim is not None else c.d_model // c.num_heads
        return MultistateGatedDelta(
            hidden_size=c.d_model,
            num_heads=c.num_heads,
            head_dim=head_dim,
            expand_v=c.expand_v,
            gate_mode=c.gate_mode,
            layout=c.layout,
            lambda_mode=c.lambda_mode,
            window=c.window,
            chunk_size=c.chunk_size,
            t_max=c.t_max,
            mode=c.mixer_mode,
            output_gate_activation=c.output_gate_activation,
            collect_stats=c.collect_stats,
            **f,
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the logits for every position.

        Args:
            input_ids (Tensor): Token ids of shape :math:`(B, T)`.

        Returns:
            Tensor: Logits of shape :math:`(B, T, V)`.
        """
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    # ------------------------------------------------------------------ #
    # Gated-delta helpers
    # ------------------------------------------------------------------ #

    def gated_delta_layers(self) -> list[MultistateGatedDelta]:
        """The gated-delta mixer layers, in block order.

        Returns:
            list of :class:`MultistateGatedDelta` (every one is present
            also with ``layout="single"``; empty only for a
            ``"zoology"``-pattern model with ``n_layer == 1``).
        """
        return [
            b.mixer for b in self.blocks if isinstance(b.mixer, MultistateGatedDelta)
        ]

    def set_collect_stats(self, flag: bool) -> None:
        """Toggle statistics collection on every gated-delta layer.

        Args:
            flag (bool): New value of ``collect_stats``.
        """
        for layer in self.gated_delta_layers():
            layer.collect_stats = flag

    def state_bytes_per_mixer(self) -> int:
        """Recurrent-state bytes of one gated-delta layer.

        ``num_heads * num_slots * d_k * d_v * 4`` (the canonical slot
        stack in fp32); the single-state layer is the ``L = 1`` case.

        Returns:
            int: Bytes of one layer's recurrent state.

        Raises:
            RuntimeError: If the model has no gated-delta layer.
        """
        layers = self.gated_delta_layers()
        if not layers:
            raise RuntimeError("state_bytes_per_mixer: no gated-delta layer found")
        layer = layers[0]
        proj = layer.projections
        return proj.num_heads * layer.num_slots * proj.head_k_dim * proj.head_v_dim * 4
