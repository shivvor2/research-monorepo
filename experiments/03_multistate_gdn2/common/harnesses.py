"""The project's harness builders: architecture + mixers from a config.

A harness builds the full model (architecture and its already-built
sequence mixers) from the resolved config. The positive-control harness
both pilots use is Zoology (:class:`ZoologyLM`) with a mixer chosen by
:class:`MixerKind` — the gated-delta layer under test, or causal
self-attention as the expected-to-do-well control.
"""

from __future__ import annotations

from enum import Enum

from omegaconf import DictConfig, OmegaConf

from research_lib.architectures.config import MadLMConfig, ZoologyLMConfig
from research_lib.architectures.mad_lm import MadLM
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


def build_zoology(cfg: DictConfig) -> ZoologyLM:
    """Build the Zoology harness with the mixer the run selects.

    The mixer is read from ``cfg.model.mixer`` (the run's own choice):
    ``gated_delta`` (the default, and the only value the first pilot's
    config can produce) builds the layer under test, ``attention`` builds
    the expected-to-do-well control. The gated-delta mixer is configured
    from ``cfg.model``; attention uses the model width and head count
    directly (its rotary dimension is the full head width — the
    gated-delta ``head_dim`` key must not set it).

    Args:
        cfg: The resolved config (``model.*``).

    Returns:
        A :class:`ZoologyLM` ready to train.

    Raises:
        ValueError: An unknown ``model.mixer`` value.
    """
    c = cfg.model
    head_dim = c.head_dim
    head_dim = int(head_dim) if head_dim is not None else c.d_model // c.num_heads
    kind = MixerKind(OmegaConf.select(cfg, "model.mixer", default="gated_delta"))

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
            return CausalSelfAttention(c.d_model, c.num_heads)
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


def build_mad(cfg: DictConfig) -> MadLM:
    """Build the MAD harness with the mixer the run selects.

    Reads ``cfg.model.mixer`` (``gated_delta``, the default, or
    ``attention``), and builds ``cfg.model.num_mixers`` independent mixer
    modules, each followed by a SwiGLU block inside :class:`MadLM`.  The
    gated-delta mixer is configured from ``cfg.model``; attention is
    :class:`CausalSelfAttention` with rotary on the full head width (its
    head count comes from the same ``cfg.model.num_heads`` field).

    Args:
        cfg: The resolved config (``model.*``).

    Returns:
        A :class:`MadLM` ready to train.

    Raises:
        ValueError: An unknown ``model.mixer`` value.
    """
    c = cfg.model
    d = int(c.d_model)
    # The gated-delta GLA layout uses 8 heads of width 16 at d=128; MAD's
    # attention control is a different layout: 16 heads of width 8, rotary
    # on the full head width (``configs/layers/mh-attention.yml``).
    num_heads = int(OmegaConf.select(cfg, "model.num_heads", default=8))
    attention_heads = int(
        OmegaConf.select(cfg, "model.attention_num_heads", default=16)
    )
    num_mixers = int(c.get("num_mixers", 2))
    kind = MixerKind(OmegaConf.select(cfg, "model.mixer", default="gated_delta"))

    def make_mixer():
        if kind is MixerKind.GATED_DELTA:
            # The gated-delta GLA layout's own head count and width: derive
            # (and validate the divisibility of) them only for the gated
            # mixer, so an attention run is not refused over a setting it
            # does not use.
            if d % num_heads != 0:
                raise ValueError(
                    f"model.d_model ({d}) must be a multiple of "
                    "model.num_heads "
                    f"({num_heads})"
                )
            head_dim = d // num_heads
            return MultistateGatedDelta(
                hidden_size=d,
                num_heads=num_heads,
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
            # The attention positive control: derive its head layout only
            # for attention (a gated-delta run must not be refused over it).
            if d % attention_heads != 0:
                raise ValueError(
                    f"model.d_model ({d}) must be a multiple of "
                    f"model.attention_num_heads ({attention_heads})"
                )
            return CausalSelfAttention(
                d, attention_heads, rotary_dim=d // attention_heads
            )
        raise ValueError(f"unknown mixer {kind}")

    mixers = [make_mixer() for _ in range(num_mixers)]
    model_config = MadLMConfig(
        vocab_size=int(c.vocab_size),
        d_model=d,
    )
    return MadLM(model_config, mixers)
