"""Config resolvers for this project's OmegaConf configs.

Registers ``padded_generated_length(size, chunk)``: ``size + 1`` rounded up to a
multiple of ``chunk``. The fixed-set (MAD-style) task uses it to derive
its padded sequence length from ``seq_len`` and the harness's chunk size,
so the run length and ``t_max`` follow the length rule with one source of
truth instead of a hand-set ``max_seq_len``.

Reference the resolver from a YAML value as ``${padded_generated_length:<size>,
<chunk>}``. The registration is idempotent; importing this module (the
single-source override, ``01_train.py``) registers it before Hydra
composes any config that uses it.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

_RESOLVER_NAME = "padded_generated_length"


def _padded_generated_length(size: Any, chunk: Any) -> int:
    """``size + 1`` rounded up to a multiple of ``chunk``.

    Args:
        size: The sequence length ``T`` (the generator returns ``T + 1``
            tokens).
        chunk: The chunk size (a positive integer).

    Returns:
        The first multiple of ``chunk`` at or above ``size + 1``.
    """
    n = int(size) + 1
    c = int(chunk)
    if c < 1:
        raise ValueError(f"chunk_size must be >= 1, got {c}")
    return ((n + c - 1) // c) * c


def register_conf_resolvers() -> None:
    """Register ``padded_generated_length`` (idempotent)."""
    if not OmegaConf.has_resolver(_RESOLVER_NAME):
        # ``register_new_resolver`` (not the legacy ``register_resolver``):
        # this resolver's arguments are themselves interpolations
        # (``${padded_generated_length:${task.seq_len}, ${model.chunk_size}}``), which
        # only the new resolver form supports.
        OmegaConf.register_new_resolver(_RESOLVER_NAME, _padded_generated_length)


register_conf_resolvers()
