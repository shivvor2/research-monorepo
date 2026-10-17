"""Lightning training module for the synthetic recall tasks.

A :class:`~research_lib.training.modules.DualOptimizerModule` for
:class:`~research_lib.architectures.zoology_lm.ZoologyLM` trained on
MQAR / repeated-key overwrite (``06_experiment_protocol.md``):

- **Loss on answer positions only** (labels are ``-100`` except at
  answers — the inherited ``compute_loss`` is reused unchanged), with a
  ``loss_on="all"`` config switch for full-sequence cross-entropy.
- **Validation / test:** answer accuracy, per slice and (for overwrite)
  per placement class, via
  :class:`~research_lib.data.synthetic.RecallMetrics`.
- **Mixer statistics:** with ``collect_mixer_stats`` (default), the
  gated-delta layers collect λ / erase statistics during the validation
  epoch only; the per-slot means are logged as
  ``mixer/layer{i}/lambda_mean/slot{l}`` etc.

The module's two optimizer groups are built by
:func:`build_weight_decay_configs` (``grouped``: weight decay on the
matrix group only; ``uniform``: Zoology's decay-everything setup).

Run tests with::

    pytest src/research_lib/training/tests/test_synthetic_recall_module.py -v
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from research_lib.data.synthetic.metrics import RecallMetrics
from research_lib.training.configs import OptimizerConfig
from research_lib.training.modules.dual_optimizer import DualOptimizerModule

logger = logging.getLogger(__name__)

_LOSS_ON = ("answers", "all")
_DECAY_SCHEMES = ("grouped", "uniform")


def build_weight_decay_configs(
    decay_scheme: str = "grouped",
    *,
    weight_decay: float = 0.1,
    betas: Sequence[float] = (0.9, 0.95),
    optimizer_class: type[Optimizer] = torch.optim.AdamW,
) -> tuple[OptimizerConfig, OptimizerConfig]:
    """The two optimizer configs of the pilot's weight-decay schemes.

    Both groups use the same optimizer class and betas; the LR is 0 at
    construction (the schedule sets the peak for step 0).

    Args:
        decay_scheme: ``"grouped"`` (default) — ``weight_decay`` applies
            to the matrix group (the 2-D linear weights) only; the vector
            group (embeddings, LM head, norms, biases, conv filters,
            ``A_log``/``dt_bias``, λ-mixer parameters) is exempt.
            ``"uniform"`` — ``weight_decay`` applies to every parameter in
            both groups (Zoology's setup, which decays everything).
        weight_decay: The matrix group's weight decay (the protocol's
            0.1; the vector group's value under ``uniform``).
        betas: Adam betas for both groups.
        optimizer_class: The optimizer class for both groups (the
            protocol's AdamW).

    Returns:
        tuple: ``(matrix_optimizer_config, vector_optimizer_config)``.

    Raises:
        ValueError: Unknown ``decay_scheme``.
    """
    if decay_scheme not in _DECAY_SCHEMES:
        raise ValueError(
            f"decay_scheme must be one of {_DECAY_SCHEMES}, got {decay_scheme!r}"
        )
    betas = tuple(betas)
    vector_weight_decay = weight_decay if decay_scheme == "uniform" else 0.0
    matrix_optimizer_config = OptimizerConfig(
        optimizer_class,
        {"lr": 0.0, "weight_decay": weight_decay, "betas": betas},
    )
    vector_optimizer_config = OptimizerConfig(
        optimizer_class,
        {"lr": 0.0, "weight_decay": vector_weight_decay, "betas": betas},
    )
    return matrix_optimizer_config, vector_optimizer_config


class SyntheticRecallModule(DualOptimizerModule):
    """Train a :class:`ZoologyLM` on the synthetic recall tasks.

    The parent's ``training_step`` calls ``self.forward(**batch)`` and the
    synthetic batches carry extra keys (``labels``, ``slice_id``,
    ``batch_index``, …), so :meth:`forward` is overridden to pass only
    ``input_ids`` to the model.

    Args:
        slice_values: The slice values of the task (``D`` for MQAR, ``K``
            for overwrite; from ``task_config.slices``).
        with_classes: Accumulate class-conditioned metrics;
            ``None`` auto-detects per batch (``True`` for overwrite).
        loss_on: ``"answers"`` (default) — cross-entropy on answer
            positions only, exactly as Zoology; ``"all"`` — full-sequence
            cross-entropy on ``input_ids`` (the protocol's alternative).
        collect_mixer_stats: If ``True`` (default), enable
            ``collect_stats`` on the model's gated-delta layers for the
            validation epoch only and log the averaged λ / erase
            statistics.
        **kwargs: Forwarded to
            :class:`~research_lib.training.modules.DualOptimizerModule`
            (model, optimizer and schedule configs, target modules, …).

    Raises:
        ValueError: Unknown ``loss_on``.
    """

    def __init__(
        self,
        *,
        slice_values: Sequence[int],
        with_classes: bool | None = None,
        loss_on: str = "answers",
        collect_mixer_stats: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if loss_on not in _LOSS_ON:
            raise ValueError(f"loss_on must be one of {_LOSS_ON}, got {loss_on!r}")
        self.slice_values = tuple(slice_values)
        self.with_classes = with_classes
        self.loss_on = loss_on
        self.collect_mixer_stats = collect_mixer_stats
        self._val_metrics: RecallMetrics | None = None
        self._test_metrics: RecallMetrics | None = None
        # Per-layer accumulators of ``latest_stats`` dicts for the epoch:
        # layer index -> list of dicts (one entry per validation batch).
        self._mixer_stats: dict[int, list[dict[str, torch.Tensor]]] = {}

    # ------------------------------------------------------------------ #
    # Forward / loss
    # ------------------------------------------------------------------ #

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass; every batch key except ``input_ids`` is ignored.

        Args:
            input_ids (Tensor): Token ids of shape ``(B, T)``.
            **kwargs: Ignored (``labels``, ``slice_id``, ``batch_index``,
                … — the parent's ``training_step`` passes the whole batch).

        Returns:
            Tensor: Logits of shape ``(B, T, V)``.
        """
        return self.model(input_ids)

    def compute_loss(
        self, model_output: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """The loss under ``self.loss_on``.

        Args:
            model_output: Logits ``(B, T, V)``.
            batch: The full batch dict.

        Returns:
            Scalar loss: answer-positions-only CE (``"answers"``; the
            inherited computation) or full-sequence CE (``"all"``).
        """
        if self.loss_on == "answers":
            return super().compute_loss(model_output, batch)
        logits = model_output
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["input_ids"][..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    # ------------------------------------------------------------------ #
    # Mixer statistics (validation epoch only)
    # ------------------------------------------------------------------ #

    def _collect_mixer_stats(self) -> None:
        """Snapshot every gated-delta layer's ``latest_stats`` for this batch."""
        if not self.collect_mixer_stats:
            return
        model = self.model
        layers = getattr(model, "gated_delta_layers", None)
        if not callable(layers):
            return
        for i, layer in enumerate(model.gated_delta_layers()):  # type: ignore[attr-defined]
            stats = layer.latest_stats
            if stats is not None:
                self._mixer_stats.setdefault(i, []).append(stats)

    def _log_mixer_stats(self, prefix: str) -> None:
        """Log the batch-averaged λ / erase statistics (``prefix="val"``)."""
        if not self._mixer_stats:
            return
        for i, entries in self._mixer_stats.items():
            # Average over validation batches, then over heads.
            keys = entries[0].keys()
            avg = {
                key: torch.stack(
                    [e[key].float() for e in entries if key in e], dim=0
                ).mean(dim=0)
                for key in keys
            }
            for key, value in avg.items():
                if key.startswith("lambda_"):
                    per_slot = value.mean(dim=0)  # average over heads -> (L,)
                    for slot in range(per_slot.numel()):
                        self.log(
                            f"mixer/layer{i}/{key}/slot{slot}", float(per_slot[slot])
                        )
                else:
                    self.log(f"mixer/layer{i}/{key}", float(value))

    def _set_collect_stats(self, flag: bool) -> None:
        model = self.model
        setter = getattr(model, "set_collect_stats", None)
        if callable(setter):
            model.set_collect_stats(flag)  # type: ignore[attr-defined]
            if flag:
                self._mixer_stats = {}

    # ------------------------------------------------------------------ #
    # Validation / test
    # ------------------------------------------------------------------ #

    def on_validation_epoch_start(self) -> None:
        self._val_metrics = RecallMetrics(
            self.slice_values, with_classes=self.with_classes
        )
        self._set_collect_stats(self.collect_mixer_stats)

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
            logits = self.forward(batch["input_ids"])
            loss = self.compute_loss(logits, batch)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        assert self._val_metrics is not None  # set in on_validation_epoch_start
        self._val_metrics.update(logits, batch)
        self._collect_mixer_stats()
        return loss

    def on_validation_epoch_end(self) -> None:
        assert self._val_metrics is not None
        for key, value in self._val_metrics.compute().items():
            self.log(f"val/{key}", value)
        self._log_mixer_stats("val")
        self._set_collect_stats(False)

    def on_test_epoch_start(self) -> None:
        self._test_metrics = RecallMetrics(
            self.slice_values, with_classes=self.with_classes
        )
        self._set_collect_stats(self.collect_mixer_stats)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One test batch: loss + RecallMetrics update.

        Args:
            batch: As in :meth:`validation_step`.
            batch_idx: Index of the batch.

        Returns:
            The test loss tensor.
        """
        with torch.no_grad():
            logits = self.forward(batch["input_ids"])
            loss = self.compute_loss(logits, batch)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        assert self._test_metrics is not None  # set in on_test_epoch_start
        self._test_metrics.update(logits, batch)
        self._collect_mixer_stats()
        return loss

    def on_test_epoch_end(self) -> None:
        assert self._test_metrics is not None
        for key, value in self._test_metrics.compute().items():
            self.log(f"test/{key}", value)
        self._log_mixer_stats("test")
        self._set_collect_stats(False)
