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

Of the two, only :class:`TokenAccuracy` keeps ``update`` free of host
synchronization (no ``.item()``, ``int()`` or ``.tolist()``):
:class:`MacroTokenAccuracy` delegates to ``torchmetrics``, whose own
argument validation calls ``torch.unique`` on every update.  Both
``compute`` methods SUM-all-reduce across ranks when a distributed
process group with world size > 1 is initialised, so the readings are
global; :class:`TokenAccuracy` reduces copies of its counters so a
repeated ``compute`` does not double-count, and every rank joins the
collective (its counters are created in :meth:`reset`).

Run tests with::

    pytest src/research_lib/data/synthetic/tests/test_mad_metrics.py -v
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._shared import distributed_reduce_device
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
        """Accumulate one batch, ignoring non-scored positions.

        The predictions and the shifted targets are flattened and handed
        to ``torchmetrics``; ``ignore_index`` drops every position whose
        target is ``-100`` (in the answer-position path the model also
        marks those predictions with ``-1``, which is never scored).

        Args:
            predictions: ``(B, T - 1)`` predicted next tokens.
            batch: Batch with unshifted ``labels`` of shape ``(B, T)``.
        """
        targets = batch["labels"][:, 1:]
        self._metric.to(predictions.device)
        self._metric.update(
            predictions.reshape(-1).contiguous(), targets.reshape(-1).contiguous()
        )

    def compute(self) -> dict[str, float]:
        """The macro accuracy, as the single key ``acc_mad``.

        Under an NCCL process group the metric is moved to the CUDA device
        first: a rank that saw no batch keeps it on CPU, and torchmetrics'
        cross-rank reduction rejects CPU state under NCCL.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_backend() == torch.distributed.Backend.NCCL:
                self._metric.to(torch.device("cuda", torch.cuda.current_device()))
        return {"acc_mad": float(self._metric.compute())}


class TokenAccuracy:
    """Accumulate the plain fraction of scored tokens over one pass.

    Args:
        ignore_index: The label value marking non-scored positions.
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX):
        self.ignore_index = int(ignore_index)
        self.reset()

    def reset(self) -> None:
        """Zero the counters (on CPU; moved to the predictions' device on the
        first update).  Zeroing here means a rank that never sees a batch
        still creates a counter, so it always joins the ``compute``
        all-reduce instead of skipping it and stalling the other ranks."""
        self._correct = torch.zeros((), dtype=torch.long)
        self._total = torch.zeros((), dtype=torch.long)

    def update(self, predictions: Tensor, batch: dict[str, Tensor]) -> None:
        """Accumulate correct/total at the scored positions (no host sync).

        Args:
            predictions: ``(B, T - 1)`` predicted next tokens.
            batch: Batch with unshifted ``labels``.
        """
        targets = batch["labels"][:, 1:]
        mask = targets != self.ignore_index
        device = predictions.device
        if self._correct.device != device:
            self._correct = self._correct.to(device)
            self._total = self._total.to(device)
        self._correct += ((predictions == targets) & mask).sum()
        self._total += mask.sum()

    def compute(self) -> dict[str, float]:
        """The token accuracy, as the single key ``acc_token``.

        The counters are reduced / read as copies, so calling this twice
        (a logged value and a record collector) does not double the counts.
        Under an NCCL process group the copies are moved to the CUDA device
        the backend needs before the all-reduce: a rank that saw no batch
        keeps CPU counters, and NCCL rejects CPU tensors.
        """
        correct = self._correct.clone()
        total = self._total.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = distributed_reduce_device(correct)
            if device != correct.device:
                correct = correct.to(device)
                total = total.to(device)
            torch.distributed.all_reduce(correct, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        total_int = int(total)
        return {"acc_token": float(correct) / total_int if total_int else 0.0}
