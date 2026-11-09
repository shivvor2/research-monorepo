"""Synthetic recall tasks: procedural MQAR, repeated-key overwrite and
entity-field.

Public names:
    - :class:`MQARConfig`, :func:`generate_mqar` (Zoology-format MQAR)
    - :class:`OverwriteConfig`, :class:`OverwriteClass`,
      :func:`generate_overwrite` (repeated-key overwrite with placement
      classes)
    - :class:`EntityFieldConfig`, :class:`EntityFieldClass`,
      :func:`generate_entity_field`, :data:`ENTITY_FIELD_CLASSES`
      (entity-field with per-entity update classes)
    - :class:`SyntheticRecallDataModule` (Lightning DataModule streaming
      fresh training batches and fixed validation/test sets)
    - :class:`RecallMetrics`, :func:`answer_predictions` (answer-position
      accuracy, per-slice, per-class, old-value leakage)
"""

from .datamodule import SyntheticRecallDataModule
from .entity_field import (
    ENTITY_FIELD_CLASSES,
    EntityFieldClass,
    EntityFieldConfig,
    generate_entity_field,
)
from .metrics import OVERWRITE_CLASSES, RecallMetrics, answer_predictions
from .mqar import MQARConfig, generate_mqar
from .overwrite import OverwriteClass, OverwriteConfig, generate_overwrite
from .registry import (
    TASKS,
    TaskSpec,
    config_types,
    generators,
    require_task,
    slice_values_of,
)

__all__ = [
    "ENTITY_FIELD_CLASSES",
    "EntityFieldClass",
    "EntityFieldConfig",
    "MQARConfig",
    "OverwriteClass",
    "OverwriteConfig",
    "OVERWRITE_CLASSES",
    "RecallMetrics",
    "SyntheticRecallDataModule",
    "TASKS",
    "TaskSpec",
    "answer_predictions",
    "config_types",
    "generate_entity_field",
    "generate_mqar",
    "generate_overwrite",
    "generators",
    "require_task",
    "slice_values_of",
]
