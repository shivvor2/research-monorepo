"""The project's harness builders: architecture + mixers from a config.

A harness builds the full model (architecture and its already-built
sequence mixers) from the resolved config. The positive-control harness
both pilots use is Zoology (:class:`ZoologyLM`) with a mixer chosen by
:class:`MixerKind` — the gated-delta layer under test, or causal
self-attention as the expected-to-do-well control.
"""

from __future__ import annotations

from enum import Enum

from omegaconf import DictConfig

from research_lib.architectures.config import ZoologyLMConfig
from research_lib.architectures.zoology_lm import ZoologyLM, zoology_mixers
from research_lib.layers.attention import CausalSelfAttention, MultistateGatedDelta


class MixerKind(str, Enum):
    """The mixer a Zoology harness builds for each block.

    - ``GATED_DELTA``: the multi-state gated-delta layer under test
      (:class:`~research_lib.layers.attention.MultistateGatedDelta`).
    - ``ATTENTION``: causal self-attention, the positive control a model
      is expected to solve the task with
      (:class:`~research_lib.layers.attention.CausalSelfAttention`).
    """

    GATED_DELTA = "gated_delta"
    ATTENTION = "attention"


def build_zoology(
    cfg: DictConfig, *, mixer: MixerKind | str = MixerKind.GATED_DELTA
) -> ZoologyLM:
    """Build the Zoology harness with the chosen mixer.

    The mixer choice is :class:`MixerKind` (or its string value); the
    gated-delta mixer is configured from ``cfg.model``, while attention
    uses the model width and head count directly.

    Args:
        cfg: The resolved config (``model.*``).
        mixer: The :class:`MixerKind` for the sequence mixers.

    Returns:
        A :class:`ZoologyLM` ready to train.

    Raises:
        ValueError: An unknown mixer kind.
    """
    c = cfg.model
    head_dim = c.head_dim
    head_dim = int(head_dim) if head_dim is not None else c.d_model // c.num_heads
    kind = MixerKind(mixer)

    def make_mixer():
        if kind is MixerKind.GATED_DELTA:
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
            )
        if kind is MixerKind.ATTENTION:
            return CausalSelfAttention(c.d_model, c.num_heads, rotary_dim=head_dim)
        raise ValueError(f"unknown mixer {kind}")

    mixers = zoology_mixers(
        n_layer=c.n_layer,
        d_model=c.d_model,
        make_mixer=make_mixer,
        pattern=c.layer_pattern,
        conv_kernel=c.base_conv_kernel,
    )
    model_config = ZoologyLMConfig(
        vocab_size=c.vocab_size,
        d_model=c.d_model,
        state_mixer=c.state_mixer,
        ff_mult=c.ff_mult,
    )
    return ZoologyLM(model_config, mixers)
