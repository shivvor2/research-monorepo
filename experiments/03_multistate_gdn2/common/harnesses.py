"""The project's harness builders: architecture + mixers from a config.

A harness builds the full model (architecture and its already-built
sequence mixers) from the resolved config. The positive-control harness
both pilots use is Zoology (:class:`ZoologyLM`) with a mixer chosen by
:class:`MixerKind` — the gated-delta layer under test, or causal
self-attention as the expected-to-do-well control.
"""

from __future__ import annotations

from enum import Enum

import torch.nn as nn
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


class AttentionQkNorm(str, Enum):
    """Query/key normalisation for the attention positive control.

    - ``RMS``: an ``RMSNorm`` per head on the query and key projections
      (the :class:`~research_lib.layers.attention.CausalSelfAttention`
      default).
    - ``NONE``: no query/key normalisation — plain scaled-dot-product
      attention, as MAD's attention applies.
    """

    RMS = "rms"
    NONE = "none"


def resolve_attention_qk_norm(cfg: DictConfig) -> type[nn.Module] | None:
    """The query/key norm module (or ``None``) for the attention control.

    Reads ``model.attention_qk_norm``; an absent key means the default, an
    ``RMSNorm`` (the behaviour before the switch existed). Only the values
    of :class:`AttentionQkNorm` are accepted; anything else is rejected so
    a typo is loud.
    """
    value = OmegaConf.select(cfg, "model.attention_qk_norm", default="rms")
    try:
        choice = AttentionQkNorm(str(value))
    except ValueError:
        raise ValueError(
            "model.attention_qk_norm must be one of "
            f"{[v.value for v in AttentionQkNorm]} (got {value!r})"
        ) from None
    return nn.RMSNorm if choice is AttentionQkNorm.RMS else None


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
    gated-delta mixer is configured from ``cfg.model`` (its head count is
    ``model.num_heads``, the head dimension derived as ``d_model //
    num_heads``); attention is :class:`CausalSelfAttention` with rotary on
    the full head width and its own head count,
    ``model.attention_num_heads`` (default 16, so its head width is
    ``d_model // attention_num_heads``).  Query/key normalisation follows
    ``model.attention_qk_norm`` (default ``rms``); the MAD control sets it
    to ``none``, as MAD's attention applies no query/key norm.

    Args:
        cfg: The resolved config (``model.*``).

    Returns:
        A :class:`MadLM` ready to train.

    Raises:
        ValueError: An unknown ``model.mixer`` value, or a ``d_model``
            not divisible by the selected mixer's head count (``num_heads``
            for gated-delta, ``attention_num_heads`` for attention).
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
                d,
                attention_heads,
                rotary_dim=d // attention_heads,
                qk_norm=resolve_attention_qk_norm(cfg),
            )
        raise ValueError(f"unknown mixer {kind}")

    mixers = [make_mixer() for _ in range(num_mixers)]
    model_config = MadLMConfig(
        vocab_size=int(c.vocab_size),
        d_model=d,
    )
    return MadLM(model_config, mixers)
