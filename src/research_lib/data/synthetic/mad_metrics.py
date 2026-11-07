"""MAD-style accuracy metrics (macro token accuracy and plain token).

The task's reported metric is MAD's macro accuracy
(:class:`MacroTokenAccuracy`): ``torchmetrics``
``MulticlassAccuracy(num_classes, ignore_index=-100)`` at its **default**
(macro) averaging, accumulated over the scored positions of one whole
evaluation pass.  The macro mean is over token classes of each class's
recall, and classes absent from both targets and predictions are excluded
(such computation is delegated to ``torchmetrics`` itself so the
definition cannot drift; the torchmetrics version is recorded per run).
:class:`TokenAccuracy` is the plain fraction of scored tokens predicted
correctly.

Both implement the module's extra-metric protocol (reset / update /
compute) and consume the same answer-position predictions as the base
``RecallMetrics``: ``(B, T - 1)`` predicted tokens and the batch's
unshifted ``labels`` (a position is scored when ``labels[:, 1:] != -100``).

Run tests with::

    pytest src/research_lib/data/synthetic/tests/test_mad_metrics.py -v
"""

from __future__ import annotations

from torch import Tensor

from .mqar import IGNORE_INDEX


class MacroTokenAccuracy:
    """Accumulate MAD's macro token accuracy over one evaluation pass.

    Args:
        num_classes: Vocabulary size (the number of token classes).
        ignore_index: The label value marking non-scored positions.
    """

    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX):
        from torchmetrics.classification import MulticlassAccuracy

        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self._metric = MulticlassAccuracy(
            num_classes=self.num_classes, ignore_index=self.ignore_index
        )

    def reset(self) -> None:
        """Zero the accumulated confusion counts."""
        self._metric.reset()

    def update(self, predictions: Tensor, batch: dict[str, Tensor]) -> None:
        """Accumulate one batch at its scored positions.

        Args:
            predictions: ``(B, T - 1)`` predicted next tokens (only the
                scored positions are used).
            batch: Batch with unshifted ``labels`` of shape ``(B, T)``; a
                position is scored when ``labels[:, 1:] != ignore_index``.
        """
        targets = batch["labels"][:, 1:]
        mask = targets != self.ignore_index
        self._metric.to(predictions.device)
        self._metric.update(predictions[mask], targets[mask])

    def compute(self) -> dict[str, float]:
        """The macro accuracy, as the single key ``acc_mad``."""
        return {"acc_mad": float(self._metric.compute())}


class TokenAccuracy:
    """Accumulate the plain fraction of scored tokens over one pass.

    Args:
        ignore_index: The label value marking non-scored positions.
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX):
        self.ignore_index = int(ignore_index)
        self._correct = 0
        self._total = 0

    def reset(self) -> None:
        """Zero the counters."""
        self._correct = 0
        self._total = 0

    def update(self, predictions: Tensor, batch: dict[str, Tensor]) -> None:
        """Accumulate correct/total at the scored positions.

        Args:
            predictions: ``(B, T - 1)`` predicted next tokens.
            batch: Batch with unshifted ``labels``.
        """
        targets = batch["labels"][:, 1:]
        mask = targets != self.ignore_index
        correct = int((predictions[mask] == targets[mask]).sum())
        self._correct += correct
        self._total += int(mask.sum())

    def compute(self) -> dict[str, float]:
        """The token accuracy, as the single key ``acc_token``."""
        return {"acc_token": float(self._correct / self._total) if self._total else 0.0}
