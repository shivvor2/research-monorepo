"""The synthetic-task registry: one place that maps a task name to its config
type, generator and slice convention.

A task is registered with its config type, its batch generator and the
keyword its generator takes for a single slice value in a fixed
evaluation set. The datamodule and any project resolve a task through
this registry instead of an ``if task == ...`` ladder, so a new task is
added here once. The slice values of a config always come from its
``slices`` property, per the config convention.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

SliceValues = Callable[[Any], tuple[int, ...]]
"""Maps a config to its slice values (single source: ``config.slices``)."""


@dataclass(frozen=True)
class TaskSpec:
    """One registered synthetic task.

    Args:
        name: The registry name (passed to the datamodule as ``task``).
        config_type: The config class for the task (validated against
            the datamodule's ``task_config``).
        generator: The batch generator (``generator(config,
            num_examples, seed, **fixed_kwargs) -> dict``).
        slice_kwarg: The generator keyword for one explicit slice value
            in a fixed evaluation set (``None`` when slices are always
            taken from ``config.slices``).
        slice_arg_of: Optional map from a slice value (as returned by
            ``config.slices``) to the fixed argument the generator
            takes for that slice, ``callable(slice_value, config)``. A
            task whose slice values are a *derived* quantity (e.g. the
            entity-field task's ``N = E * M`` pairs, whose generator
            takes ``num_entities`` ``= N // M``) supplies this;
            ``None`` means the slice value is the argument as is.
        class_names: The task's metric key fragments per ``key_class``
            code, for :class:`~research_lib.data.synthetic.RecallMetrics`
            (``None`` uses the overwrite names).
    """

    name: str
    config_type: type
    generator: Callable[..., dict[str, Any]]
    slice_kwarg: str | None = None
    slice_arg_of: Callable[[int, Any], int] | None = None
    class_names: tuple[str, ...] | None = None

    def slice_values(self, config: Any) -> tuple[int, ...]:
        """The task's slice values from a config (via ``config.slices``)."""
        return tuple(int(v) for v in config.slices)  # type: ignore[attr-defined]


# Imported lazily below to avoid a cycle: the generators import only
# torch/typing, while this module is imported by the datamodule.
def _mqar_spec():
    from .mqar import MQARConfig, generate_mqar

    return TaskSpec("mqar", MQARConfig, generate_mqar, slice_kwarg="num_kv_pairs")


def _overwrite_spec():
    from .overwrite import OVERWRITE_CLASSES, OverwriteConfig, generate_overwrite

    return TaskSpec(
        "overwrite",
        OverwriteConfig,
        generate_overwrite,
        slice_kwarg="num_keys",
        class_names=OVERWRITE_CLASSES,
    )


def _entity_field_spec():
    from .entity_field import (
        ENTITY_FIELD_CLASSES,
        EntityFieldConfig,
        generate_entity_field,
    )

    # Slice values are the number of pairs N = E * M (on the same scale as
    # the overwrite task's K); the generator's fixed argument is the entity
    # count E = N // M.
    return TaskSpec(
        "entity_field",
        EntityFieldConfig,
        generate_entity_field,
        slice_kwarg="num_entities",
        slice_arg_of=lambda slice_value, config: slice_value // config.num_fields,
        class_names=ENTITY_FIELD_CLASSES,
    )


TASKS: dict[str, TaskSpec] = {
    "mqar": _mqar_spec(),
    "overwrite": _overwrite_spec(),
    "entity_field": _entity_field_spec(),
}
"""The registered synthetic tasks, keyed by name."""


def require_task(name: str) -> TaskSpec:
    """The resolved :class:`TaskSpec` for ``name``, or raise.

    Raises:
        ValueError: Unknown task name.
    """
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(
            f"unknown task {name!r}; registered tasks: {', '.join(sorted(TASKS))}"
        ) from None


def generators() -> dict[str, Callable]:
    """The generator per registered task name."""
    return {name: spec.generator for name, spec in TASKS.items()}


def config_types() -> dict[str, type]:
    """The config type per registered task name."""
    return {name: spec.config_type for name, spec in TASKS.items()}


def slice_values_of(task: str, config: Any) -> tuple[int, ...]:
    """The slice values of ``config`` under ``task`` (from ``config.slices``)."""
    return require_task(task).slice_values(config)
