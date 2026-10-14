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
(slice, class) cells that never occur are absent from the dict. Computation
stays on the logits' device; :meth:`RecallMetrics.compute` returns Python
floats.
"""

from __future__ import annotations

import logging

import torch

from .mqar import IGNORE_INDEX
from .overwrite import OverwriteClass

logger = logging.getLogger(__name__)

CLASS_NAMES: tuple[str, ...] = tuple(c.name.lower() for c in OverwriteClass)
"""Metric key fragments per class code: ('same', 'adjacent', 'distant', 'single')."""


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
    accumulated too.

    Args:
        with_classes: accumulate class-conditioned metrics. ``None``
            (default) auto-detects per batch from the ``key_class`` key.

    Example:
        >>> metrics = RecallMetrics()
        >>> metrics.update(logits, batch)  # doctest: +SKIP
        >>> metrics.compute()["acc"]  # doctest: +SKIP
        0.83
    """

    CLASS_NAMES = CLASS_NAMES

    def __init__(self, with_classes: bool | None = None) -> None:
        self._with_classes = with_classes
        self.reset()

    def reset(self) -> None:
        """Zero all counters."""
        # Scalar counters (long tensors, created lazily on the device of the
        # first update): (answers, correct, leaked).
        self._total = None
        self._slice_totals: dict[int, torch.Tensor] = {}
        self._class_totals: dict[int, torch.Tensor] = {}
        self._slice_class_totals: dict[tuple[int, int], torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> None:
        """Accumulate one evaluation batch.

        Args:
            logits: ``(B, T, V)`` model logits.
            batch: batch dict with ``labels`` and ``slice_id``; for
                overwrite also ``old_labels`` and ``key_class`` (class
                metrics are accumulated iff ``key_class`` is present, or
                always when ``with_classes=True``).
        """
        preds, targets, mask = answer_predictions(logits, batch["labels"])
        if self._total is None:
            self._total = torch.zeros(3, dtype=torch.long, device=preds.device)
        correct = (preds == targets) & mask
        self._total[0] += int(mask.sum())
        self._total[1] += int(correct.sum())

        slice_ids = batch["slice_id"]
        slice_pos = slice_ids.to(preds.device)[:, None].expand_as(targets)

        with_classes = self._with_classes
        if with_classes is None:
            with_classes = "key_class" in batch
        if with_classes:
            if "key_class" not in batch:
                raise KeyError("with_classes=True requires 'key_class' in the batch")
            if "old_labels" not in batch:
                raise KeyError("class metrics require 'old_labels' in the batch")
            # IGNORE_INDEX / NO_CLASS can never equal a vocab-id prediction.
            old_targets = batch["old_labels"].to(preds.device)[:, 1:]
            leak = (preds == old_targets) & mask
            key_pos = batch["key_class"].to(preds.device)[:, 1:]
            self._total[2] += int(leak.sum())
        else:
            leak = None
            key_pos = None

        for n in slice_ids.unique().tolist():
            m = mask & (slice_pos == int(n))
            stats = self._slice_totals.setdefault(
                int(n), torch.zeros(3, dtype=torch.long, device=preds.device)
            )
            stats[0] += int(m.sum())
            stats[1] += int((correct & m).sum())
            if leak is not None:
                stats[2] += int((leak & m).sum())

        if with_classes:
            for c in range(len(CLASS_NAMES)):
                cm = mask & (key_pos == c)
                stats = self._class_totals.setdefault(
                    c, torch.zeros(3, dtype=torch.long, device=preds.device)
                )
                stats[0] += int(cm.sum())
                stats[1] += int((correct & cm).sum())
                stats[2] += int((leak & cm).sum())
                for n in slice_ids.unique().tolist():
                    sm = mask & (slice_pos == int(n))
                    scm = sm & (key_pos == c)
                    key = (int(n), int(c))
                    stats = self._slice_class_totals.setdefault(
                        key, torch.zeros(3, dtype=torch.long, device=preds.device)
                    )
                    stats[0] += int(scm.sum())
                    stats[1] += int((correct & scm).sum())
                    stats[2] += int((leak & scm).sum())

    # ------------------------------------------------------------------ #
    # Compute
    # ------------------------------------------------------------------ #

    def compute(self) -> dict[str, float]:
        """Return the accumulated metrics as Python floats.

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
        """
        out: dict[str, float] = {}

        def _ratio(stats: torch.Tensor) -> tuple[float, float, float]:
            answers = int(stats[0])
            if answers == 0:
                return 0.0, 0.0, 0.0
            acc = int(stats[1]) / answers
            leak = int(stats[2]) / answers
            return acc, leak, 1.0 - acc - leak

        total = (
            self._total if self._total is not None else torch.zeros(3, dtype=torch.long)
        )
        acc, _, _ = _ratio(total)
        out["acc"] = acc
        for n, stats in sorted(self._slice_totals.items()):
            if int(stats[0]) == 0:
                # Empty slice cell: omit (cannot occur for valid batches).
                continue
            acc_n, _, _ = _ratio(stats)
            out[f"acc/slice={n}"] = acc_n
        for c, stats in sorted(self._class_totals.items()):
            if int(stats[0]) == 0:
                # Class never observed: omit.
                continue
            name = CLASS_NAMES[c]
            acc_c, leak_c, other_c = _ratio(stats)
            out[f"acc/class={name}"] = acc_c
            out[f"leak/class={name}"] = leak_c
            out[f"other/class={name}"] = other_c
        for (n, c), stats in sorted(self._slice_class_totals.items()):
            if int(stats[0]) == 0:
                # Empty (slice, class) cell: omit.
                continue
            name = CLASS_NAMES[c]
            acc_nc, leak_nc, other_nc = _ratio(stats)
            out[f"acc/slice={n}/class={name}"] = acc_nc
            out[f"leak/slice={n}/class={name}"] = leak_nc
            out[f"other/slice={n}/class={name}"] = other_nc
        return out
