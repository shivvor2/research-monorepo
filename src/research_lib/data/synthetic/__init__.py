"""Synthetic recall tasks: procedural MQAR, repeated-key overwrite,
entity-field and MAD-style fuzzy in-context recall.

Public names:
    - :class:`MQARConfig`, :func:`generate_mqar` (Zoology-format MQAR)
    - :class:`OverwriteConfig`, :class:`OverwriteClass`,
      :func:`generate_overwrite` (repeated-key overwrite with placement
      classes)
    - :class:`EntityFieldConfig`, :class:`EntityFieldClass`,
      :func:`generate_entity_field`, :data:`ENTITY_FIELD_CLASSES`
      (entity-field with per-entity update classes)
    - :class:`MadFuzzyConfig`, :func:`fuzzy_recall_instance`,
      :func:`fuzzy_recall_dataset`, :func:`mad_batch_row`,
      :func:`mad_reference_split` (MAD-style fuzzy in-context recall)
    - :class:`MacroTokenAccuracy`, :class:`TokenAccuracy` (MAD-style
      token accuracy metrics)
    - :class:`SyntheticRecallDataModule` (Lightning DataModule streaming
      fresh training batches and fixed validation/test sets)
    - :class:`FixedSetRecallDataModule` (lightning DataModule for tasks
      trained on a fixed materialised set)
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
from .fixed_set import FixedSetRecallDataModule
from .mad_fuzzy import (
    MadFuzzyConfig,
    fuzzy_recall_dataset,
    fuzzy_recall_instance,
    mad_batch_row,
    mad_reference_split,
)
from .mad_metrics import MacroTokenAccuracy, TokenAccuracy
from .metrics import OVERWRITE_CLASSES, RecallMetrics, answer_predictions
from .mqar import MQARConfig, generate_mqar
from .overwrite import OverwriteClass, OverwriteConfig, generate_overwrite
from .registry import (
    TASKS,
    TaskRegime,
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
    "FixedSetRecallDataModule",
    "OVERWRITE_CLASSES",
    "TASKS",
    "MQARConfig",
    "MacroTokenAccuracy",
    "MadFuzzyConfig",
    "OverwriteClass",
    "OverwriteConfig",
    "RecallMetrics",
    "SyntheticRecallDataModule",
    "TaskRegime",
    "TaskSpec",
    "TokenAccuracy",
    "answer_predictions",
    "config_types",
    "fuzzy_recall_dataset",
    "fuzzy_recall_instance",
    "generate_entity_field",
    "generate_mqar",
    "generate_overwrite",
    "generators",
    "mad_batch_row",
    "mad_reference_split",
    "require_task",
    "slice_values_of",
]
