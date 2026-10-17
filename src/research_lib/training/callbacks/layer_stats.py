"""Layer-statistics callback for any layer that collects its own stats.

A layer follows the two-attribute convention of :class:`StatsProvider`: it
keeps ``collect_stats`` (the layer collects while ``True``) and
``latest_stats`` (a mapping filled by each forward while collecting).
:class:`~research_lib.layers.attention.MultistateGatedDelta` satisfies it;
so does any other layer that adopts the convention.

:class:`LayerStatsCallback` enables collection on every provider for the
validation and test epochs only (training steps are unaffected), averages
each statistics key over the epoch's batches, passes the result through a
``summarize`` function and logs every returned entry as
``{prefix}/layer{i}/{name}`` (``i`` = the provider's index among the
providers, in ``named_modules()`` order).

Run tests with::

    pytest src/research_lib/training/callbacks/tests/test_layer_stats.py -v
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import lightning as L
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@runtime_checkable
class StatsProvider(Protocol):
    """A layer that collects its own statistics on demand.

    Attributes:
        collect_stats: The layer collects while ``True``.
        latest_stats: Filled by each forward while collecting, else
            ``None``.
    """

    collect_stats: bool
    latest_stats: Mapping[str, Tensor] | None


def mean_summary(key: str, value: Tensor) -> dict[str, float]:
    """The default summarizer: one entry, the tensor's mean.

    Args:
        key: The statistics key (e.g. ``"erase_strength"``).
        value: The batch-averaged statistics tensor.

    Returns:
        dict: ``{key: mean}`` as a float.
    """
    return {key: float(value.float().mean())}


class LayerStatsCallback(L.Callback):
    """Collect and log per-layer statistics during validation and test.

    Collection is on only during the validation and test epochs: the
    providers are found and enabled at each epoch start, snapshotted after
    every batch, and disabled at the epoch end, so training steps are
    unaffected.

    Args:
        prefix: Metric prefix; entries are logged as
            ``{prefix}/layer{i}/{name}``.
        summarize: Maps ``(key, batch-averaged tensor)`` to a mapping of
            ``{name: value}`` entries to log. The default
            :func:`mean_summary` logs the tensor's mean under the key
            itself.
        sync_dist: Passed through to ``pl_module.log`` (average across
            ranks under DDP instead of logging rank-local values).

    Examples::

        >>> # Keep the multi-state pilot's metric names:
        >>> def per_slot_summary(key, value):
        ...     if key.startswith("lambda_"):
        ...         per_slot = value.float().mean(dim=0)
        ...         return {f"{key}/slot{l}": float(v) for l, v in enumerate(per_slot)}
        ...     return {key: float(value.float().mean())}
        >>> LayerStatsCallback(prefix="mixer", summarize=per_slot_summary)  # doctest: +SKIP
    """

    def __init__(
        self,
        prefix: str = "mixer",
        summarize: Callable[[str, Tensor], Mapping[str, Any]] = mean_summary,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.summarize = summarize
        self.sync_dist = sync_dist
        self._providers: list[StatsProvider] = []
        self._snapshots: list[list[dict[str, Tensor]]] = []

    # ------------------------------------------------------------------ #
    # Epoch lifecycle
    # ------------------------------------------------------------------ #

    def _start_epoch(self, pl_module: L.LightningModule) -> None:
        """Find the providers (named_modules order), enable and clear."""
        self._providers = [
            module
            for _, module in pl_module.named_modules()
            if isinstance(module, StatsProvider)
        ]
        self._snapshots = [[] for _ in self._providers]
        for module in self._providers:
            module.collect_stats = True

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._start_epoch(pl_module)

    def on_test_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._start_epoch(pl_module)

    # ------------------------------------------------------------------ #
    # Per-batch snapshot
    # ------------------------------------------------------------------ #

    def _snapshot(self) -> None:
        for i, module in enumerate(self._providers):
            stats = module.latest_stats
            if stats is None:
                continue
            self._snapshots[i].append(
                {key: value.detach().clone() for key, value in stats.items()}
            )

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._snapshot()

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._snapshot()

    # ------------------------------------------------------------------ #
    # Epoch end: average, summarize, log, disable
    # ------------------------------------------------------------------ #

    def _end_epoch(self, pl_module: L.LightningModule) -> None:
        for i, module in enumerate(self._providers):
            module.collect_stats = False
            snapshots = self._snapshots[i]
            if not snapshots:
                continue
            for key in snapshots[0]:
                values = [
                    snapshot[key].float() for snapshot in snapshots if key in snapshot
                ]
                if not values:
                    continue
                averaged = torch.stack(values, dim=0).mean(dim=0)
                for name, value in self.summarize(key, averaged).items():
                    pl_module.log(
                        f"{self.prefix}/layer{i}/{name}",
                        float(value),
                        sync_dist=self.sync_dist,
                    )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._end_epoch(pl_module)

    def on_test_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._end_epoch(pl_module)
