"""Lightning training module for the multi-state GDN-2 synthetic pilot.

A :class:`~research_lib.training.modules.DualOptimizerModule` for
:class:`~research_lib.architectures.zoology_lm.ZoologyLM` trained on
MQAR / repeated-key overwrite:

- **Loss on answer positions only** (labels are ``-100`` except at
  answers — the inherited ``compute_loss`` is reused unchanged), with a
  ``loss_on="all"`` switch for full-sequence cross-entropy
  (:class:`LossOn`).
- **Answer-position head** (``answer_head=True``, the default): under
  ``loss_on="answers"`` the LM head is evaluated only at the answer
  positions (via
  :func:`~research_lib.training.losses.answer_position_logits`), which
  is exact — same loss, gradients and metrics in exact arithmetic; the
  trunk still runs on every position. ``answer_head=False`` restores
  the full head (for checks only; it changes the run identity).
- **Validation / test:** answer accuracy, per slice and (for overwrite)
  per placement class, via
  :class:`~research_lib.data.synthetic.RecallMetrics`.

Per-layer mixer statistics are collected by
:class:`~research_lib.training.callbacks.LayerStatsCallback`, not by
this module.

Run tests with::

    pytest src/research_lib/training/tests/test_mgdn2_synthetic_recall_module.py -v
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import Enum
from typing import Any, cast

import torch
import torch.nn.functional as F

from research_lib.data.synthetic.metrics import RecallMetrics
from research_lib.training.losses import (
    AnswerLogits,
    answer_cross_entropy,
    answer_position_logits,
)
from research_lib.training.modules.dual_optimizer import DualOptimizerModule

logger = logging.getLogger(__name__)


class LossOn(str, Enum):
    """Where the training loss is computed.

    - ``ANSWERS``: cross-entropy on answer positions only (labels are
      ``-100`` elsewhere, exactly as Zoology).
    - ``ALL``: full-sequence cross-entropy on ``input_ids``.
    """

    ANSWERS = "answers"
    ALL = "all"


class MGDN2SyntheticRecallModule(DualOptimizerModule):
    """Train a :class:`ZoologyLM` on the multi-state GDN-2 synthetic tasks.

    **Scope.** This module is task-specific: it is the synthetic-recall
    training module of the multi-state GDN-2 project (MQAR and the
    repeated-key overwrite task, via :class:`~research_lib.data.synthetic`
    batches and :class:`~research_lib.data.synthetic.RecallMetrics`).
    Convention: in ``research_lib``, a class that inherits directly from
    :class:`lightning.pytorch.LightningModule` is a general building
    block; a class that inherits from one of ``research_lib``'s own
    modules (as this one inherits from
    :class:`~research_lib.training.modules.DualOptimizerModule`) is a
    specific instance and may be tied to one task or project. A general
    synthetic-task module is deferred until a second experiment needs
    one; when it exists, :class:`MGDN2SyntheticRecallModule` will become
    that module with fixed init arguments (see issue #8).

    The parent's ``training_step`` calls ``self.forward(**batch)`` and the
    synthetic batches carry extra keys (``labels``, ``slice_id``,
    ``batch_index``, …), so :meth:`forward` is overridden to pass only
    ``input_ids`` to the model.

    Args:
        slice_values: The slice values of the task (``D`` for MQAR, ``K``
            for overwrite; from ``task_config.slices``).
        with_classes: Accumulate class-conditioned metrics;
            ``None`` auto-detects per batch (``True`` for overwrite).
        loss_on: :class:`LossOn` member or its string value (Hydra passes
            strings); ``"answers"`` is the default.
        answer_head: Evaluate the LM head only at answer positions under
            ``loss_on="answers"`` (the default; exact, see the module
            docstring). Requires the model to provide ``hidden_states``
            and ``head`` methods.
        **kwargs: Forwarded to
            :class:`~research_lib.training.modules.DualOptimizerModule`
            (model, optimizer and schedule configs, target modules, …).

    Raises:
        ValueError: Unknown ``loss_on`` string.
        TypeError: ``answer_head=True`` with ``loss_on="answers"``
        and the model lacks ``hidden_states`` or ``head``.
    """

    def __init__(
        self,
        *,
        slice_values: Sequence[int],
        with_classes: bool | None = None,
        loss_on: LossOn | str = LossOn.ANSWERS,
        answer_head: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.loss_on = LossOn(loss_on)
        self.answer_head = bool(answer_head)
        if self.answer_head and self.loss_on is LossOn.ANSWERS:
            # The answer path calls hidden_states/head directly, so they
            # are required there; loss_on="all" never calls them.
            missing = [
                name
                for name in ("hidden_states", "head")
                if not callable(getattr(self.model, name, None))
            ]
            if missing:
                raise TypeError(
                    f"answer_head=True requires the model to provide "
                    f"{', '.join(missing)}; {type(self.model).__name__} does not"
                )
        self.slice_values = tuple(slice_values)
        self.with_classes = with_classes
        self._val_metrics: RecallMetrics | None = None
        self._test_metrics: RecallMetrics | None = None

    # ------------------------------------------------------------------ #
    # Forward / loss
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | AnswerLogits:  # type: ignore[override]
        """Forward pass; every batch key except ``input_ids`` is ignored.

        Under ``loss_on="answers"`` with ``answer_head=True`` and
        ``labels`` given, returns :class:`AnswerLogits` (the head
        evaluated only at the answer positions); otherwise returns the
        full ``(B, T, V)`` logits.

        Args:
            input_ids (Tensor): Token ids of shape ``(B, T)``.
            labels (Tensor | None): Unshifted labels of shape ``(B, T)``;
                the parent's ``training_step`` passes the whole batch, so
                this is set in training.
            **kwargs: Ignored (``slice_id``, ``batch_index``, …).

        Returns:
            Tensor: Logits of shape ``(B, T, V)``, or :class:`AnswerLogits`
            under the answer-head path.
        """
        if self.loss_on is LossOn.ANSWERS and self.answer_head and labels is not None:
            # The constructor checked that the model provides both methods;
            # the static type of ``self.model`` (nn.Module) does not know them.
            model = cast(Any, self.model)
            return answer_position_logits(
                model.hidden_states(input_ids), model.head, labels
            )
        return self.model(input_ids)

    def compute_loss(
        self, model_output: torch.Tensor | AnswerLogits, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """The loss under ``self.loss_on``.

        Args:
            model_output: Logits ``(B, T, V)`` or :class:`AnswerLogits`
                (the answer-head path).
            batch: The full batch dict.

        Returns:
            Scalar loss: answer-positions-only CE (``"answers"``; the
            inherited computation) or full-sequence CE (``"all"``).
        """
        if isinstance(model_output, AnswerLogits):
            return answer_cross_entropy(model_output)
        if self.loss_on is LossOn.ANSWERS:
            return super().compute_loss(model_output, batch)
        logits = model_output
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["input_ids"][..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    # ------------------------------------------------------------------ #
    # Validation / test
    # ------------------------------------------------------------------ #

    def on_validation_epoch_start(self) -> None:
        self._val_metrics = RecallMetrics(
            self.slice_values, with_classes=self.with_classes
        )

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """One validation batch: loss + RecallMetrics update.

        Args:
            batch: Batch dict with ``input_ids``, ``labels``, ``slice_id``
                (and for overwrite ``old_labels`` / ``key_class``).
            batch_idx: Index of the batch.

        Returns:
            The validation loss tensor.
        """
        with torch.no_grad():
            out = self.forward(batch["input_ids"], labels=batch["labels"])
            loss = self.compute_loss(out, batch)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        assert self._val_metrics is not None  # set in on_validation_epoch_start
        if isinstance(out, AnswerLogits):
            self._val_metrics.update_answers(out.logits, batch)
        else:
            self._val_metrics.update(out, batch)
        return loss

    def on_validation_epoch_end(self) -> None:
        assert self._val_metrics is not None
        for key, value in self._val_metrics.compute().items():
            self.log(f"val/{key}", value)

    def on_test_epoch_start(self) -> None:
        self._test_metrics = RecallMetrics(
            self.slice_values, with_classes=self.with_classes
        )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One test batch: loss + RecallMetrics update.

        Args:
            batch: As in :meth:`validation_step`.
            batch_idx: Index of the batch.

        Returns:
            The test loss tensor.
        """
        with torch.no_grad():
            out = self.forward(batch["input_ids"], labels=batch["labels"])
            loss = self.compute_loss(out, batch)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        assert self._test_metrics is not None  # set in on_test_epoch_start
        if isinstance(out, AnswerLogits):
            self._test_metrics.update_answers(out.logits, batch)
        else:
            self._test_metrics.update(out, batch)
        return loss

    def on_test_epoch_end(self) -> None:
        assert self._test_metrics is not None
        for key, value in self._test_metrics.compute().items():
            self.log(f"test/{key}", value)
