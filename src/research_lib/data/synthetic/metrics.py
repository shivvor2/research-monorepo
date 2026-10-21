"""Answer-position metrics for the synthetic recall tasks.

Accuracy/leakage are computed at **answer positions only**, under the repo
shift convention: the prediction for answer position ``p`` is
``argmax(logits[p - 1])`` and the target is ``labels[p]``
(``DualOptimizerModule.compute_loss`` shifts the same way with
``ignore_index=-100``; ``labels[:, 0]`` is never an answer position for
either task).

``RecallMetrics`` accumulates over evaluation batches:

- ``'acc'``: overall answer accuracy, the mean over sequences of each
  sequence's accuracy (the convention of Arora et al., 2023); with equal
  numbers of sequences per slice this equals the unweighted mean of the
  per-slice accuracies;
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

Predictions can arrive by three paths, all feeding the same counting
body (:meth:`update_predictions`):

- :meth:`update` with full ``(B, T, V)`` logits (the classic path);
- :meth:`update_answers` with ``(N, V)`` answer-row logits, as produced
  by :func:`research_lib.training.losses.answer_position_logits` (the
  answer-position head of the pilot);
- :meth:`update_predictions` directly with ``(B, T - 1)`` predicted
  tokens.

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
import math
from collections.abc import Sequence

import torch

from .mqar import IGNORE_INDEX
from .overwrite import OverwriteClass

logger = logging.getLogger(__name__)

CLASS_NAMES: tuple[str, ...] = tuple(c.name.lower() for c in OverwriteClass)
"""Metric key fragments per class code: ('same', 'adjacent', 'distant', 'single')."""

_NO_CLASS_BUCKET = 4
"""Column of the count tensor's class axis used when class metrics are off."""

DEFAULT_DISTANCE_BINS: tuple[float, ...] = (0, 16, 32, 64, 128, 256, math.inf)
"""Default bin edges for the write-distance metrics: the token distance
between a key's two writes, binned half-open ``[lo, hi)`` (the last bin
runs to infinity). Only overwritten keys carry a distance."""


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
        distance_bins: bin edges for the write-distance metrics (the token
            distance between a key's two writes, from ``batch[
            "write_distance"]``, only overwritten keys), as half-open
            ``[lo, hi)`` bins with the last running to infinity; a
            distance below the first edge is in no bin.
            ``DEFAULT_DISTANCE_BINS`` = ``(0, 16, 32, 64, 128, 256, inf)``
            by default; ``None`` disables the distance metrics. Batches
            without ``write_distance`` (MQAR) never accumulate them.

    Raises:
        ValueError: empty ``slice_values``, negative or duplicate entries;
            invalid ``distance_bins`` (fewer than two edges, non-increasing
            or starting below zero).

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
        distance_bins: Sequence[float] | None = DEFAULT_DISTANCE_BINS,
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
        if distance_bins is None:
            self._distance_bins = None
        else:
            bins = tuple(float(b) for b in distance_bins)
            if len(bins) < 2:
                raise ValueError(f"distance_bins needs >= 2 edges, got {bins}")
            if any(bins[i] >= bins[i + 1] for i in range(len(bins) - 1)):
                raise ValueError(f"distance_bins must be increasing, got {bins}")
            if bins[0] < 0:
                raise ValueError(f"distance_bins must start >= 0, got {bins}")
            self._distance_bins = bins
        self.reset()

    def reset(self) -> None:
        """Zero all counters."""
        # One long tensor of shape (num_slices, 5, 3): axis 1 = the four
        # placement classes + the "no class" bucket (class metrics off),
        # axis 2 = (answers, correct, leaked). Created lazily on the device
        # of the first update; the unknown-slice counter is a scalar.
        self._counts: torch.Tensor | None = None
        self._unknown: torch.Tensor | None = None
        # Per-sequence accuracy accumulator: sum of per-sequence accuracies
        # (float64 to avoid any rounding question in tie-breaks), count of
        # sequences with at least one answer, and count of sequences with no
        # answers (reported by compute).
        self._seq_acc_sum: torch.Tensor | None = None
        self._seq_count: torch.Tensor | None = None
        self._seq_zero_count: torch.Tensor | None = None
        # Write-distance counts, one more axis: (num_slices, num_bins, 3)
        # for (answers, correct, leaked) per (slice, distance bin). Only
        # accumulated for batches that carry ``write_distance``.
        self._distance_counts: torch.Tensor | None = None
        self._has_class_batches = False
        self._has_distance_batches = False

    # ------------------------------------------------------------------ #
    # Update (no host synchronization)
    # ------------------------------------------------------------------ #

    def update(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> None:
        """Accumulate one evaluation batch from full logits (free of host
        synchronization).

        Args:
            logits: ``(B, T, V)`` model logits.
            batch: batch dict with ``labels`` and ``slice_id``; for
                overwrite also ``old_labels`` and ``key_class`` (class
                metrics are accumulated iff ``key_class`` is present, or
                always when ``with_classes=True``). Answers whose
                ``slice_id`` is not among ``slice_values`` are counted in
                the unknown-slice counter (reported by :meth:`compute`).
        """
        preds, _, _ = answer_predictions(logits, batch["labels"])
        self.update_predictions(preds, batch)

    def update_predictions(
        self, preds: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> None:
        """Accumulate one evaluation batch from predicted next tokens.

        The counting body behind :meth:`update` and
        :meth:`update_answers` (free of host synchronization).

        Args:
            preds: ``(B, T - 1)`` long, the predicted next token at every
                position (``argmax`` of the shifted logits); values at
                non-answer positions are ignored.
            batch: As in :meth:`update`.
        """
        targets = batch["labels"][:, 1:]
        mask = targets != IGNORE_INDEX
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

        # Per-sequence accuracy accumulator (float64 sum, long count).
        seq_correct = (correct & known).sum(dim=1).long()
        seq_total = (mask & known).sum(dim=1).long()
        seq_valid = seq_total > 0
        seq_zero = ~seq_valid
        seq_acc = torch.where(
            seq_valid,
            seq_correct.double() / seq_total.double(),
            torch.zeros_like(seq_correct, dtype=torch.double),
        )
        seq_rows = row_pos[:, 0]
        if self._seq_acc_sum is None or self._seq_count is None:
            self._seq_acc_sum = torch.zeros(
                self._num_slices, dtype=torch.double, device=device
            )
            self._seq_count = torch.zeros(
                self._num_slices, dtype=torch.long, device=device
            )
            self._seq_zero_count = torch.zeros(
                self._num_slices, dtype=torch.long, device=device
            )
        self._seq_acc_sum.index_add_(0, seq_rows, seq_acc)
        self._seq_count.index_add_(0, seq_rows, seq_valid.long())
        self._seq_zero_count.index_add_(0, seq_rows, seq_zero.long())

        # Write-distance bins, one more axis: (num_slices, num_bins, 3),
        # the same index_add_ mechanism over a combined flat index. Only
        # batches that carry ``write_distance`` (overwrite) and only
        # overwritten keys (single-class answers carry NO_DISTANCE = -1).
        # The bin index counts the satisfied inner edges (comparisons
        # against Python scalars), so update() creates no tensors and
        # stays free of host synchronization (torch.tensor(..., device)
        # or torch.bucketize with device-created boundaries would sync).
        if self._distance_bins is not None and "write_distance" in batch:
            bins = self._distance_bins
            num_bins = len(bins) - 1
            if self._distance_counts is None:
                self._distance_counts = torch.zeros(
                    (self._num_slices, num_bins, 3), dtype=torch.long, device=device
                )
                self._has_distance_batches = True
            distances = batch["write_distance"].to(device)[:, 1:]
            # A distance below the first edge is in no bin ([lo, hi) bins,
            # the last running to infinity). bins[0] >= 0 is validated at
            # construction, so this subsumes the >= 0 check; it compares
            # against a Python scalar, so update() stays free of host
            # synchronization.
            valid = (distances >= bins[0]) & mask & known
            bin_idx = torch.zeros_like(distances)
            for edge in bins[1:-1]:
                bin_idx += (distances >= edge).to(bin_idx.dtype)
            dist_flat_idx = (row_pos * num_bins + bin_idx).reshape(-1) * 3
            dist_flat = self._distance_counts.view(-1)
            dist_flat.index_add_(0, dist_flat_idx + 0, valid.reshape(-1).long())
            dist_flat.index_add_(
                0, dist_flat_idx + 1, (correct & valid).reshape(-1).long()
            )
            dist_flat.index_add_(
                0, dist_flat_idx + 2, (leak & valid).reshape(-1).long()
            )

    def update_answers(
        self, answer_logits: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> None:
        """Accumulate one evaluation batch from answer-row logits.

        ``answer_logits`` is ``(N, V)``, the rows of the answer positions
        in row-major order, as produced by
        :func:`research_lib.training.losses.answer_position_logits`.
        The mask is recomputed from ``batch["labels"][:, 1:] != -100``,
        ``preds = full((B, T - 1), -1)`` is filled with
        ``preds.masked_scatter_(mask, answer_logits.argmax(-1))`` (no
        host synchronization, keeping :meth:`update`'s property), and
        :meth:`update_predictions` does the counting.

        Args:
            answer_logits: ``(N, V)`` logits at the answer positions, in
                row-major order.
            batch: As in :meth:`update`.
        """
        labels = batch["labels"]
        mask = (labels[:, 1:] != IGNORE_INDEX).to(answer_logits.device)
        preds = torch.full(
            (labels.shape[0], labels.shape[1] - 1),
            -1,
            dtype=torch.long,
            device=answer_logits.device,
        )
        preds.masked_scatter_(mask, answer_logits.argmax(-1))
        self.update_predictions(preds, batch)

    # ------------------------------------------------------------------ #
    # Compute
    # ------------------------------------------------------------------ #

    def _global_counts(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """(counts, unknown, distance_counts, seq_acc_sum, seq_count,
        seq_zero_count) with a SUM all-reduce across ranks when a distributed
        process group with world size > 1 is initialized.

        Reduces clones, so the accumulator itself is unchanged and further
        updates / repeated compute() calls stay correct.
        """
        assert self._counts is not None and self._unknown is not None
        counts, unknown = self._counts, self._unknown
        distance_counts = self._distance_counts
        seq_acc_sum = self._seq_acc_sum
        seq_count = self._seq_count
        seq_zero_count = self._seq_zero_count
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            counts = counts.clone()
            unknown = unknown.clone()
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(unknown, op=torch.distributed.ReduceOp.SUM)
            if distance_counts is not None:
                distance_counts = distance_counts.clone()
                torch.distributed.all_reduce(
                    distance_counts, op=torch.distributed.ReduceOp.SUM
                )
            if seq_acc_sum is not None:
                seq_acc_sum = seq_acc_sum.clone()
                seq_count = seq_count.clone()
                seq_zero_count = seq_zero_count.clone()
                torch.distributed.all_reduce(
                    seq_acc_sum, op=torch.distributed.ReduceOp.SUM
                )
                torch.distributed.all_reduce(
                    seq_count, op=torch.distributed.ReduceOp.SUM
                )
                torch.distributed.all_reduce(
                    seq_zero_count, op=torch.distributed.ReduceOp.SUM
                )
        return counts, unknown, distance_counts, seq_acc_sum, seq_count, seq_zero_count

    def compute(self) -> dict[str, float]:
        """Return the accumulated metrics as Python floats.

        Under a distributed process group with world size > 1 the counts
        (and the unknown-slice counter) are SUM-all-reduced across ranks
        first, so the metrics are global.

        Keys: ``'acc'`` (the mean over sequences of each sequence's accuracy,
        i.e. the convention of Arora et al., 2023); ``'acc/slice={n}'`` per
        observed slice. If class metrics were accumulated:
        ``'acc/class={c}'``, ``'leak/class={c}'``, ``'other/class={c}'`` per
        observed class and ``'acc/slice={n}/class={c}'``,
        ``'leak/slice={n}/class={c}'``, ``'other/slice={n}/class={c}'`` per
        observed (slice, class) cell. If write-distance batches were
        accumulated: ``'acc/dist={lo}-{hi}'`` and ``'leak/dist={lo}-{hi}'``
        per observed distance bin (the token distance between the two
        writes, binned ``[lo, hi)``, the last bin to infinity), and
        ``'acc/slice={n}/dist={lo}-{hi}'``,
        ``'leak/slice={n}/dist={lo}-{hi}'`` per observed (slice, bin) cell.

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
            distance_counts = self._distance_counts
            seq_acc_sum = self._seq_acc_sum
            seq_count = self._seq_count
            seq_zero_count = self._seq_zero_count
        else:
            counts, unknown, distance_counts, seq_acc_sum, seq_count, seq_zero_count = (
                self._global_counts()
            )
        num_unknown = int(unknown)
        if num_unknown != 0:
            raise ValueError(
                f"{num_unknown} answers carried slice values outside "
                f"slice_values={self._slice_values}"
            )
        if seq_zero_count is not None and int(seq_zero_count.sum()) != 0:
            raise ValueError(
                f"{int(seq_zero_count.sum())} sequences had zero answer positions"
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
        if seq_count is not None and seq_count.sum() > 0:
            out["acc"] = float(seq_acc_sum.sum() / seq_count.sum().double())
        else:
            out["acc"] = 0.0
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

        # Write-distance bins (one more axis on the counts): accuracy and
        # old-value leakage per bin, overall and per (slice, bin), for
        # overwritten keys only (single-class answers carry no distance).
        # Empty bins and (slice, bin) cells are omitted, never emitted 0.0.
        if distance_counts is not None:
            bins = self._distance_bins
            labels = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                hi_label = "inf" if math.isinf(hi) else f"{hi:g}"
                labels.append(f"{lo:g}-{hi_label}")
            for b, label in enumerate(labels):
                stats = distance_counts[:, b, :].sum(dim=0)
                if int(stats[0]) == 0:
                    continue
                acc_b, leak_b, _ = _ratio(stats)
                out[f"acc/dist={label}"] = acc_b
                out[f"leak/dist={label}"] = leak_b
                for row, value in enumerate(self._slice_values):
                    stats = distance_counts[row, b, :]
                    if int(stats[0]) == 0:
                        continue
                    acc_nb, leak_nb, _ = _ratio(stats)
                    out[f"acc/slice={value}/dist={label}"] = acc_nb
                    out[f"leak/slice={value}/dist={label}"] = leak_nb
        return out
