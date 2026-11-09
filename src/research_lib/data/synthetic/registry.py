"""The synthetic-task registry: one place that maps a task name to its config
type, generator and slice convention.

A task is registered with its config type, its training regime and the
functions that produce its batches. The datamodule and any project
resolve a task through this registry instead of an ``if task == ...``
ladder, so a new task is added here once. The slice values of a config
always come from its ``slices`` property, per the config convention.

Two training regimes are supported:

- ``STREAMING``: fresh batches from a seeded, unbounded generator
  (``generator(config, num_examples, seed, **fixed_kwargs) -> dict``),
  as the streaming datamodule expects.
- ``FIXED_SET``: a fixed materialised set presented per epoch
  (``dataset_fn(config, num_examples, is_training, rng) -> (inputs,
  targets)`` and ``row_fn(config, pad_to)`` returning the row mapping),
  as the fixed-set datamodule expects.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SliceValues = Callable[[Any], tuple[int, ...]]
"""Maps a config to its slice values (single source: ``config.slices``)."""

# A batch generator: ``(config, num_examples, seed, **fixed_kwargs) -> dict``.
Generator = Callable[..., dict[str, Any]]
# A fixed-set dataset function: ``(config, num_examples, is_training, rng)
# -> (inputs, targets)`` each stacked over instances.
DatasetFn = Callable[..., tuple[np.ndarray, np.ndarray]]
# A row-mapping factory: ``(config, pad_to)`` returns a callable
# ``(inputs, targets, is_training) -> one batch row dict``.
RowFnFactory = Callable[..., Callable[..., dict[str, Any]]]


class TaskRegime(str, Enum):
    """How a task's training set is produced.

    - ``STREAMING``: unlimited fresh batches from a seeded generator (the
      streaming datamodule's regime).
    - ``FIXED_SET``: a fixed set materialised once and presented as a new
      permutation per epoch.
    """

    STREAMING = "streaming"
    FIXED_SET = "fixed_set"


@dataclass(frozen=True)
class TaskSpec:
    """One registered synthetic task.

    Args:
        name: The registry name (passed to the datamodule as ``task``).
        config_type: The config class for the task (validated against
            the datamodule's ``task_config``).
        regime: The training regime (:class:`TaskRegime`).
        generator: The streaming batch generator (``generator(config,
            num_examples, seed, **fixed_kwargs) -> dict``); required only
            for a ``STREAMING`` task.
        dataset_fn: The fixed-set dataset function (``(config,
            num_examples, is_training, rng) -> (inputs, targets)``);
            required only for a ``FIXED_SET`` task.
        row_fn: The fixed-set row-mapping factory (``(config, pad_to)``
            returning ``(inputs, targets, is_training) -> row``);
            required only for a ``FIXED_SET`` task.
        slice_kwarg: The generator keyword for one explicit slice value
            in a fixed evaluation set (``None`` when slices are always
            taken from ``config.slices``).
        class_names: The task's per-class key fragments (``None`` when
            the task has no classes).
        slice_arg_of: Optional map from a slice value (as returned by
            ``config.slices``) to the fixed argument a streaming task's
            generator takes for that slice, ``callable(slice_value,
            config)``. A task whose slice values are a *derived*
            quantity (e.g. the entity-field ``N = E * M`` pairs, whose
            generator takes ``num_entities`` ``= N // M``) supplies
            this; a fixed-set task always takes its slices from
            ``config.slices``, so it cannot.

    Raises:
        ValueError: The ``regime`` is unknown, or the provided functions
            do not match it, or a fixed-set task declares a
            ``slice_arg_of``.
    """

    name: str
    config_type: type
    generator: Generator | None = None
    slice_kwarg: str | None = None
    class_names: tuple[str, ...] | None = None
    slice_arg_of: Callable[[int, Any], int] | None = None
    regime: TaskRegime | str = TaskRegime.STREAMING
    dataset_fn: DatasetFn | None = None
    row_fn: RowFnFactory | None = None

    def __post_init__(self) -> None:
        regime = TaskRegime(self.regime)
        object.__setattr__(self, "regime", regime)
        streaming = regime is TaskRegime.STREAMING
        if streaming:
            if self.generator is None:
                raise ValueError(
                    f"task {self.name!r} is streaming but has no generator"
                )
            if self.dataset_fn is not None or self.row_fn is not None:
                raise ValueError(
                    f"task {self.name!r} is streaming but declares fixed-set functions"
                )
        else:
            if self.dataset_fn is None or self.row_fn is None:
                raise ValueError(
                    f"task {self.name!r} is fixed-set but lacks dataset_fn/row_fn"
                )
            if self.generator is not None:
                raise ValueError(
                    f"task {self.name!r} is fixed-set but declares a streaming generator"
                )
            if self.slice_kwarg is not None:
                raise ValueError(
                    f"task {self.name!r} is fixed-set and cannot take a slice_kwarg"
                )
            if self.slice_arg_of is not None:
                raise ValueError(
                    f"task {self.name!r} is fixed-set and cannot take a slice_arg_of"
                )

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


def _mad_fuzzy_spec():
    from .mad_fuzzy import MadFuzzyConfig, fuzzy_recall_dataset, mad_batch_row

    def row_fn(config, pad_to):
        """The row mapping bound to the config and the padded length."""

        def _row(inputs, targets, is_training):
            return mad_batch_row(config, inputs, targets, pad_to, bool(is_training))

        return _row

    return TaskSpec(
        "mad_fuzzy",
        MadFuzzyConfig,
        regime=TaskRegime.FIXED_SET,
        dataset_fn=fuzzy_recall_dataset,
        row_fn=row_fn,
    )


TASKS: dict[str, TaskSpec] = {
    "mqar": _mqar_spec(),
    "overwrite": _overwrite_spec(),
    "entity_field": _entity_field_spec(),
    "mad_fuzzy": _mad_fuzzy_spec(),
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


def generators() -> dict[str, Generator]:
    """The streaming generator per registered streaming task name.

    Fixed-set tasks are excluded: they have no batch generator.
    """
    return {
        name: spec.generator
        for name, spec in TASKS.items()
        if spec.regime is TaskRegime.STREAMING
    }


def config_types() -> dict[str, type]:
    """The config type per registered task name."""
    return {name: spec.config_type for name, spec in TASKS.items()}


def slice_values_of(task: str, config: Any) -> tuple[int, ...]:
    """The slice values of ``config`` under ``task`` (from ``config.slices``)."""
    return require_task(task).slice_values(config)
