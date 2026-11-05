"""This project's task and harness registries.

The train script builds its task config through the task registry and its
model through the harness registry, instead of an ``if task == ...`` /
``if harness == ...`` ladder. A task PR adds one entry to each registry
and the pairing is fixed by config: a task config names its harness via
``defaults``, so the two cannot be crossed by accident.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from research_lib.data.synthetic.mqar import MQARConfig
from research_lib.data.synthetic.overwrite import OverwriteConfig

logger = logging.getLogger(__name__)

# A harness is ``Callable[[cfg], nn.Module]``: it builds the full model
# (architecture plus mixers) from the resolved config.
HarnessBuilder = Callable[[Any], Any]

TASK_CONFIG_TYPES: dict[str, type] = {
    "mqar": MQARConfig,
    "overwrite": OverwriteConfig,
}
"""The config type per registered task."""

HARNESSES: dict[str, HarnessBuilder] = {}
"""The registered harnesses: harness name -> model builder."""


def config_type_of_task(name: str) -> type:
    """The config type of a registered task.

    Raises:
        ValueError: Unknown task name.
    """
    try:
        return TASK_CONFIG_TYPES[name]
    except KeyError:
        raise ValueError(
            f"unknown task {name!r}; registered: {', '.join(sorted(TASK_CONFIG_TYPES))}"
        ) from None


def register_task(name: str, config_type: type) -> None:
    """Register a task's config type (a task PR adds its task here and in
    the general task registry)."""
    TASK_CONFIG_TYPES[name] = config_type


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
    """Register a harness builder under ``name`` (MQAR and overwrite use
    the first pilot's Zoology harness; a task PR with its own architecture
    registers it here)."""
    HARNESSES[name] = builder