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
from .metrics import RecallMetrics, answer_predictions
from .mqar import MQARConfig, generate_mqar
from .overwrite import OverwriteClass, OverwriteConfig, generate_overwrite

__all__ = [
    "MQARConfig",
    "OverwriteClass",
    "OverwriteConfig",
    "RecallMetrics",
    "SyntheticRecallDataModule",
    "answer_predictions",
    "generate_mqar",
    "generate_overwrite",
]
