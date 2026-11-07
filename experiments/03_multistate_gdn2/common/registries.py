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
