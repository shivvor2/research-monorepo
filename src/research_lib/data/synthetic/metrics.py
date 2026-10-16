"""Answer-position metrics for the synthetic recall tasks.

Accuracy/leakage are computed at **answer positions only**, under the repo
shift convention: the prediction for answer position ``p`` is
``argmax(logits[p - 1])`` and the target is ``labels[p]``
(``DualOptimizerModule.compute_loss`` shifts the same way with
``ignore_index=-100``; ``labels[:, 0]`` is never an answer position for
either task).

``RecallMetrics`` accumulates over evaluation batches:

- ``'acc'``: overall answer accuracy (over answers, not averaged per
  sequence);
- ``'acc/slice={n}'``: accuracy per slice value (``D`` for MQAR, ``K`` for
  overwrite);
- overwrite only: ``'acc/class={c}'``, ``'acc/slice={n}/class={c}'``,
  ``'leak/class={c}'``, ``'leak/slice={n}/class={c}'`` and
  ``'other/class={c}'`` where

  - ``leak`` is the fraction of answers whose prediction equals ``v_old``
    (old-value leakage; separates stale recall from random errors),
  - ``other = 1 - acc - leak``.

Class keys use the lowercase :class:`~research_lib.data.synthetic.overwrite.OverwriteClass`
names (``same``, ``adjacent``, ``distant``, ``single``); cells with no
answers are OMITTED rather than emitted as 0.0 (a 0.0 would be
indistinguishable from a real zero accuracy downstream), so classes and
(slice, class) cells that never occur are absent from the dict.

All counts live in ONE long tensor of shape ``(num_slices, 5, 3)`` (axis 1:
the four placement classes plus a "no class" bucket used when class metrics
are off, i.e. MQAR; axis 2: answers / correct / leaked).
:meth:`RecallMetrics.update` is free of host synchronization (no
``.item()``, ``int(tensor)``, ``.tolist()`` or ``.unique()``; the
accumulation uses ``index_add_`` — ``torch.bincount`` was measured to
synchronize the CUDA stream under ``torch.cuda.set_sync_debug_mode``).
Answers whose ``slice_id`` is not among the configured ``slice_values`` are
counted in a separate on-device counter and reported by
:meth:`RecallMetrics.compute` (``ValueError``). Under a distributed
process group with world size > 1, ``compute()`` first SUM-all-reduces the
count tensor (and the unknown-slice counter) across ranks, so the metrics
are global. Computation stays on the logits' device; ``compute()`` returns
Python floats.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

from .mqar import IGNORE_INDEX
from .overwrite import OverwriteClass

logger = logging.getLogger(__name__)

CLASS_NAMES: tuple[str, ...] = tuple(c.name.lower() for c in OverwriteClass)
"""Metric key fragments per class code: ('same', 'adjacent', 'distant', 'single')."""

_NO_CLASS_BUCKET = 4
"""Column of the count tensor's class axis used when class metrics are off."""


def answer_predictions(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract answer-position predictions under the repo shift convention.

    Args:
        logits: model logits of shape ``(B, T, V)``.
        labels: unshifted labels of shape ``(B, T)`` with ``-100`` at
            non-answer positions; ``labels[p]`` is predicted from position
            ``p - 1``.

    Returns:
        Tuple ``(preds, targets, mask)``, each of shape ``(B, T - 1)``:

        - ``preds``: ``logits[:, :-1].argmax(-1)`` (predicted token at each
          shifted position),
        - ``targets``: ``labels[:, 1:]``,
        - ``mask``: ``targets != -100`` (answer positions).

    Shape:
        logits: ``(B, T, V)``, labels: ``(B, T)``; outputs ``(B, T - 1)``.
    """
    if logits.shape[0] != labels.shape[0] or logits.shape[1] != labels.shape[1]:
        raise ValueError(
            f"batch mismatch: logits {tuple(logits.shape)} vs labels "
            f"{tuple(labels.shape)}"
        )
    preds = logits[:, :-1, :].argmax(dim=-1)
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX
    return preds, targets, mask


class RecallMetrics:
    """Accumulator for one evaluation pass over recall batches.

    Call :meth:`update` with each batch and its logits, then
    :meth:`compute` for the metric dict. For overwrite batches
    (``key_class``/``old_labels`` present) the class-conditioned keys are
    accumulated too. All counts live in one long tensor of shape
    ``(num_slices, 5, 3)`` on the logits' device; ``update()`` performs no
    host synchronization, and ``compute()`` SUM-all-reduces the counts
    across ranks when a distributed process group with world size > 1 is
    initialized (so the metrics are global).

    Args:
        slice_values: the slice values that can occur in the batches
            (``D`` for MQAR, ``K`` for overwrite; the datamodule knows them
            from ``config.slices``). Per-slice keys are derived for exactly
            these values; answers carrying any other ``slice_id`` are
            counted separately and reported (``ValueError``) by
            :meth:`compute`.
        with_classes: accumulate class-conditioned metrics. ``None``
            (default) auto-detects per batch from the ``key_class`` key.

    Raises:
        ValueError: empty ``slice_values``, negative or duplicate entries.

    Example:
        >>> metrics = RecallMetrics(slice_values=(4, 8, 16))
        >>> metrics.update(logits, batch)  # doctest: +SKIP
        >>> metrics.compute()["acc"]  # doctest: +SKIP
        0.83
    """

    CLASS_NAMES = CLASS_NAMES

    def __init__(
        self,
        slice_values: Sequence[int],
        with_classes: bool | None = None,
    ) -> None:
        values = tuple(slice_values)
        if not values:
            raise ValueError("slice_values must be non-empty")
        if any(v < 0 for v in values):
            raise ValueError(f"slice_values entries must be >= 0, got {values}")
        duplicates = sorted({v for v in values if values.count(v) > 1})
        if duplicates:
            raise ValueError(f"duplicate slice_values entries: {duplicates}")
        self._slice_values = values
        self._num_slices = len(values)
        self._with_classes = with_classes
        # slice VALUE -> row of the count tensor (-1 = unknown value).
        lookup = torch.full((max(values) + 1,), -1, dtype=torch.long)
        for row, value in enumerate(values):
            lookup[value] = row
        self._slice_row_lookup = lookup
        self.reset()

    def reset(self) -> None:
        """Zero all counters."""
        # One long tensor of shape (num_slices, 5, 3): axis 1 = the four
        # placement classes + the "no class" bucket (class metrics off),
        # axis 2 = (answers, correct, leaked). Created lazily on the device
        # of the first update; the unknown-slice counter is a scalar.
        self._counts: torch.Tensor | None = None
        self._unknown: torch.Tensor | None = None
        self._has_class_batches = False

    # ------------------------------------------------------------------ #
    # Update (no host synchronization)
    # ------------------------------------------------------------------ #

    def update(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> None:
        """Accumulate one evaluation batch (free of host synchronization).

        Args:
            logits: ``(B, T, V)`` model logits.
            batch: batch dict with ``labels`` and ``slice_id``; for
                overwrite also ``old_labels`` and ``key_class`` (class
                metrics are accumulated iff ``key_class`` is present, or
                always when ``with_classes=True``). Answers whose
                ``slice_id`` is not among ``slice_values`` are counted in
                the unknown-slice counter (reported by :meth:`compute`).
        """
        preds, targets, mask = answer_predictions(logits, batch["labels"])
        device = preds.device
        if self._counts is None or self._unknown is None:
            self._counts = torch.zeros(
                (self._num_slices, 5, 3), dtype=torch.long, device=device
            )
            self._unknown = torch.zeros((), dtype=torch.long, device=device)
        correct = (preds == targets) & mask

        with_classes = self._with_classes
        if with_classes is None:
            with_classes = "key_class" in batch
        if with_classes:
            if "key_class" not in batch:
                raise KeyError("with_classes=True requires 'key_class' in the batch")
            if "old_labels" not in batch:
                raise KeyError("class metrics require 'old_labels' in the batch")
            # IGNORE_INDEX / NO_CLASS can never equal a vocab-id prediction,
            # so leaked is well defined at every masked position.
            old_targets = batch["old_labels"].to(device)[:, 1:]
            leak = (preds == old_targets) & mask
            key_pos = batch["key_class"].to(device)[:, 1:]
            # Garbage class codes at masked positions are harmless (zero
            # weight below); clamp keeps the combined index in range.
            class_col = key_pos.clamp(min=0, max=_NO_CLASS_BUCKET)
            self._has_class_batches = True
        else:
            leak = torch.zeros_like(mask)
            class_col = torch.full_like(targets, _NO_CLASS_BUCKET)

        # Slice rows (-1 = a slice value not among slice_values; values
        # outside the lookup's range are forced unknown without a sync).
        lookup = self._slice_row_lookup
        if lookup.device != device:
            lookup = lookup.to(device)
            self._slice_row_lookup = lookup
        ids = batch["slice_id"].to(device)
        in_range = (ids >= 0) & (ids < lookup.numel())
        rows = lookup[ids.clamp(min=0, max=lookup.numel() - 1)]
        rows = torch.where(in_range, rows, torch.full_like(rows, -1))
        slice_pos = rows[:, None].expand_as(targets)
        known = slice_pos >= 0
        row_pos = slice_pos.clamp_min(0)  # unknown rows excluded by weights
        self._unknown += (mask & ~known).sum()

        # Combined index (row * 5 + class) at answer positions; accumulate
        # the three counters with index_add_ (sync-free; torch.bincount
        # synchronizes the CUDA stream).
        counted = mask & known
        idx = (row_pos * 5 + class_col).reshape(-1)
        flat = self._counts.view(-1)  # (num_slices * 5 * 3,) contiguous view
        flat.index_add_(0, idx * 3 + 0, counted.reshape(-1).long())
        flat.index_add_(0, idx * 3 + 1, (correct & known).reshape(-1).long())
        flat.index_add_(0, idx * 3 + 2, (leak & known).reshape(-1).long())

    # ------------------------------------------------------------------ #
    # Compute
    # ------------------------------------------------------------------ #

    def _global_counts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(counts, unknown) with a SUM all-reduce across ranks when a
        distributed process group with world size > 1 is initialized.

        Reduces clones, so the accumulator itself is unchanged and further
        updates / repeated compute() calls stay correct.
        """
        assert self._counts is not None and self._unknown is not None
        counts, unknown = self._counts, self._unknown
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            counts = counts.clone()
            unknown = unknown.clone()
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(unknown, op=torch.distributed.ReduceOp.SUM)
        return counts, unknown

    def compute(self) -> dict[str, float]:
        """Return the accumulated metrics as Python floats.

        Under a distributed process group with world size > 1 the counts
        (and the unknown-slice counter) are SUM-all-reduced across ranks
        first, so the metrics are global.

        Keys: ``'acc'``; ``'acc/slice={n}'`` per observed slice. If class
        metrics were accumulated: ``'acc/class={c}'``, ``'leak/class={c}'``,
        ``'other/class={c}'`` per observed class and
        ``'acc/slice={n}/class={c}'``, ``'leak/slice={n}/class={c}'``,
        ``'other/slice={n}/class={c}'`` per observed (slice, class) cell.

        Cells with no answers are OMITTED (never emitted as 0.0, which
        would be indistinguishable from a real zero accuracy downstream):
        classes and (slice, class) cells that never occur are therefore
        absent from the dict, and downstream consumers must treat a missing
        key as "not measured", not as zero.

        Raises:
            ValueError: if any accumulated answer carried a ``slice_id``
                outside ``slice_values``.
        """
        if self._counts is None or self._unknown is None:
            counts = torch.zeros((self._num_slices, 5, 3), dtype=torch.long)
            unknown = torch.zeros((), dtype=torch.long)
        else:
            counts, unknown = self._global_counts()
        num_unknown = int(unknown)
        if num_unknown != 0:
            raise ValueError(
                f"{num_unknown} answers carried slice values outside "
                f"slice_values={self._slice_values}"
            )

        out: dict[str, float] = {}

        def _ratio(stats: torch.Tensor) -> tuple[float, float, float]:
            answers = int(stats[0])
            if answers == 0:
                return 0.0, 0.0, 0.0
            acc = int(stats[1]) / answers
            leak = int(stats[2]) / answers
            return acc, leak, 1.0 - acc - leak

        # Empty slice cell: omit (cannot occur for valid batches).
        # Class never observed: omit. Empty (slice, class) cell: omit.
        acc, _, _ = _ratio(counts.sum(dim=(0, 1)))
        out["acc"] = acc
        for row, value in enumerate(self._slice_values):
            stats = counts[row].sum(dim=0)
            if int(stats[0]) == 0:
                continue
            acc_n, _, _ = _ratio(stats)
            out[f"acc/slice={value}"] = acc_n
        if self._has_class_batches:
            for c in range(len(CLASS_NAMES)):
                stats = counts[:, c, :].sum(dim=0)
                if int(stats[0]) == 0:
                    continue
                name = CLASS_NAMES[c]
                acc_c, leak_c, other_c = _ratio(stats)
                out[f"acc/class={name}"] = acc_c
                out[f"leak/class={name}"] = leak_c
                out[f"other/class={name}"] = other_c
            for row, value in enumerate(self._slice_values):
                for c in range(len(CLASS_NAMES)):
                    stats = counts[row, c, :]
                    if int(stats[0]) == 0:
                        continue
                    name = CLASS_NAMES[c]
                    acc_nc, leak_nc, other_nc = _ratio(stats)
                    out[f"acc/slice={value}/class={name}"] = acc_nc
                    out[f"leak/slice={value}/class={name}"] = leak_nc
                    out[f"other/slice={value}/class={name}"] = other_nc
        return out
