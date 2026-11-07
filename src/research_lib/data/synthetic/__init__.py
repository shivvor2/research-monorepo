"""Synthetic recall tasks: procedural MQAR and repeated-key overwrite.

Public names:
    - :class:`MQARConfig`, :func:`generate_mqar` (Zoology-format MQAR)
    - :class:`OverwriteConfig`, :class:`OverwriteClass`,
      :func:`generate_overwrite` (repeated-key overwrite with placement
      classes)
    - :class:`SyntheticRecallDataModule` (Lightning DataModule streaming
      fresh training batches and fixed validation/test sets)
    - :class:`RecallMetrics`, :func:`answer_predictions` (answer-position
      accuracy, per-slice, per-class, old-value leakage)
"""

from .datamodule import SyntheticRecallDataModule
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
    TaskSpec,
    config_types,
    generators,
    require_task,
    slice_values_of,
)

__all__ = [
    "OVERWRITE_CLASSES",
    "TASKS",
    "FixedSetRecallDataModule",
    "MQARConfig",
    "MacroTokenAccuracy",
    "MadFuzzyConfig",
    "OverwriteClass",
    "OverwriteConfig",
    "RecallMetrics",
    "SyntheticRecallDataModule",
    "TaskSpec",
    "TokenAccuracy",
    "answer_predictions",
    "config_types",
    "fuzzy_recall_dataset",
    "fuzzy_recall_instance",
    "generate_mqar",
    "generate_overwrite",
    "generators",
    "mad_batch_row",
    "mad_reference_split",
    "require_task",
    "slice_values_of",
]
