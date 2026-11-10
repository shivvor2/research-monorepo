"""This project's harness registry (and the task config-type lookup).

The train script builds its model through the harness registry instead of
an ``if harness == ...`` ladder; a harness that ships with a task
registers itself here (and an existing task can be paired with a new
one). Task *config types*
and slice/class-name facts come from the general task registry in
``research_lib`` — :func:`config_type_of_task` is a thin delegation so
nobody re-declares them in the project layer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from research_lib.data.synthetic import MacroTokenAccuracy, TokenAccuracy
from research_lib.data.synthetic.registry import require_task

from .harnesses import build_mad, build_zoology

logger = logging.getLogger(__name__)

# A harness is ``Callable[[cfg], nn.Module]``: it builds the full model
# (architecture plus mixers) from the resolved config.
HarnessBuilder = Callable[[Any], Any]

HARNESSES: dict[str, HarnessBuilder] = {
    "zoology": build_zoology,
    "mad": build_mad,
}
"""The registered harnesses: harness name -> model builder."""


def config_type_of_task(name: str) -> type:
    """The config type of a registered task (from the general registry).

    Raises:
        ValueError: Unknown task name.
    """
    return require_task(name).config_type


def require_harness(name: str) -> HarnessBuilder:
    """The registered harness builder for ``name``.

    Raises:
        ValueError: Unknown harness name.
    """
    try:
        return HARNESSES[name]
    except KeyError:
        raise ValueError(
            f"unknown harness {name!r}; registered: {', '.join(sorted(HARNESSES))}"
        ) from None


def register_harness(name: str, builder: HarnessBuilder) -> None:
    """Register a harness builder under ``name`` (a harness that ships
    with a task registers itself here)."""
    HARNESSES[name] = builder


# The extra evaluation metrics a task can request by name (a closed set,
# keyed by Enum-style string names): ``{name: (build_fn,)}`` where the
# build function takes ``num_classes`` (the task's vocab size).  A task
# lists its extras in its config; :func:`build_extra_metrics` resolves
# them here so the train script holds no metric names.
EXTRA_METRICS: dict[str, Callable[[int], Any]] = {
    "macro_token_accuracy": lambda num_classes: MacroTokenAccuracy(
        num_classes=num_classes
    ),
    "token_accuracy": lambda _num_classes: TokenAccuracy(),
}
"""The named extra metrics, each a ``(num_classes) -> metric`` builder."""


def build_extra_metrics(names: list[str] | tuple[str, ...], vocab_size: int) -> tuple:
    """The extra metrics a task's config requested, built from ``names``.

    Args:
        names: The metric names (a task-config list; a non-registered
            name raises).
        vocab_size: The task's vocabulary size (feeds the metrics that
            need ``num_classes``).

    Returns:
        A tuple of built metrics (empty when ``names`` is empty).

    Raises:
        ValueError: An unknown metric name.
    """
    unknown = [n for n in names if n not in EXTRA_METRICS]
    if unknown:
        raise ValueError(
            f"unknown extra metric(s) {sorted(set(unknown))}; "
            f"known: {sorted(EXTRA_METRICS)}"
        )
    return tuple(EXTRA_METRICS[n](int(vocab_size)) for n in names)
