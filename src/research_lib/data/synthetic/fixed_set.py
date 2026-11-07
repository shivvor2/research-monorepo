"""Lightning DataModule for a task trained on a fixed set for many epochs.

Contrast with :class:`~research_lib.data.synthetic.SyntheticRecallDataModule`
(streams fresh batches): this module materialises a **fixed training set**
once and presents it as a new random permutation every epoch, which is the
training regime for the MAD-style harness.

- **Train:** the ``num_train`` instances are generated once in
  :meth:`setup` from ``np.random.default_rng(train_seed)`` (train-style).
  Each epoch ``e`` draws a permutation of the set from
  ``np.random.SeedSequence([train_seed, e])`` and yields whole batches in
  that order; the last partial batch is kept (``drop_last=False``).  The
  stream comes from an :class:`~torch.utils.data.IterableDataset` keyed by
  the GLOBAL batch index ``g = epoch * batches_per_epoch + position``, so
  it is independent of ``num_workers`` and resumable from a batch index,
  exactly as in :class:`~research_lib.data.synthetic.SyntheticRecallDataModule`.
- **Validation / test:** ``num_val`` / ``num_test`` test-style instances
  from their own fixed seeds, loaded in a fixed order.
- **Leakage check:** :meth:`setup` counts validation and test sequences
  that also occur in the training set and logs the fraction at WARNING
  level when it exceeds 0.001.

The mapping to the repo batch format (unshifted ``input_ids`` /
``labels``, ``labels[p]`` predicted from position ``p - 1`` with the
pad positions and untouched non-score positions set to ``-100``, then
right-padded with the pad token / ``-100`` to ``pad_to``) is done by the
per-row transform ``to_row``, so any task that produces aligned
``(inputs, targets)`` arrays can use this module.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

logger = logging.getLogger(__name__)

DatasetFn = Callable[..., tuple[np.ndarray, np.ndarray]]
"""``(config, num_examples, is_training, rng) -> (inputs, targets)`` each
stacked over instances as int arrays of shape ``(N, S)``."""

RowFn = Callable[[np.ndarray, np.ndarray, bool], dict[str, torch.Tensor]]
"""``(inputs, targets, is_training) -> one batch row dict`` (``input_ids``,
``labels``, ``slice_id``, …).  Training rows may map ``labels`` differently
(e.g. every next token is a target)."""

IGNORE_INDEX = -100
"""Label reserved for non-score positions (repo convention)."""


def _batch_seed(train_seed: int, index: int) -> int:
    """Deterministic per-index seed: ``SeedSequence([train_seed, index])``
    (same convention as the streaming datamodule)."""
    return int(
        np.random.SeedSequence([int(train_seed), int(index)]).generate_state(
            1, dtype=np.uint64
        )[0]
    )


def _distributed_world_size_and_rank() -> tuple[int, int]:
    """World size and rank: ``torch.distributed`` when a process group is
    initialized, else ``(1, 0)`` (read fresh at each use)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    return 1, 0


def _permutation(train_seed: int, epoch: int, n: int) -> np.ndarray:
    """The training-set permutation for ``epoch`` (deterministic)."""
    rng = np.random.default_rng(_batch_seed(train_seed, epoch))
    return rng.permutation(n)


class _FixedEpochTrainDataset(IterableDataset):
    """The per-epoch-permuted train stream, keyed by global batch index.

    Batch ``g`` belongs to epoch ``e = g // batches_per_epoch`` (its
    permutation) and holds the rows at positions
    ``[pos * batch_size, min((pos + 1) * batch_size, n))`` of that
    permutation.  Rows are materialised tensors already mapped to the repo
    batch format by ``to_row``.
    """

    def __init__(
        self,
        rows: list[dict[str, torch.Tensor]],
        batch_size: int,
        train_seed: int,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.batch_size = batch_size
        self.train_seed = train_seed
        self._batch_size = int(batch_size)
        self._n = len(rows)
        self._batches_per_epoch = math.ceil(self._n / self._batch_size)
        self.next_local_idx = 0

    def _make_batch(self, index: int) -> dict[str, torch.Tensor]:
        """Batch ``index`` (global) from its epoch's permutation."""
        epoch, position = divmod(int(index), self._batches_per_epoch)
        perm = _permutation(self.train_seed, epoch, self._n)
        start = position * self._batch_size
        end = min(start + self._batch_size, self._n)
        idx = perm[start:end]
        batch: dict[str, torch.Tensor] = {}
        for key in self.rows[0]:
            batch[key] = torch.stack([self.rows[int(i)][key] for i in idx])
        batch["batch_index"] = torch.tensor(int(index), dtype=torch.int64)
        batch["epoch"] = torch.tensor(int(epoch), dtype=torch.int64)
        return batch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        world_size, rank = _distributed_world_size_and_rank()
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        local = self.next_local_idx + worker_id
        while True:
            batch = self._make_batch(local * world_size + rank)
            if info is None:
                self.next_local_idx = local + 1
            yield batch
            local += num_workers


class _FixedRowDataset(Dataset):
    """Map-style dataset over fixed per-example dict rows."""

    def __init__(self, rows: list[dict[str, torch.Tensor]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def _generate_rows(
    dataset_fn: DatasetFn,
    config: Any,
    num_examples: int,
    is_training: bool,
    seed: int,
    to_row: RowFn,
) -> list[dict[str, torch.Tensor]]:
    """Generate ``num_examples`` instances and map each to a batch row."""
    rng = np.random.default_rng(seed)
    inputs, targets = dataset_fn(config, num_examples, is_training, rng)
    return [to_row(inputs[i], targets[i], is_training) for i in range(num_examples)]


class FixedSetRecallDataModule(L.LightningDataModule):
    """A fixed-training-set DataModule for a task with a dataset function.

    Args:
        dataset_fn: ``(config, num_examples, is_training, rng)`` returning
            ``(inputs, targets)`` stacked over examples (un-shifted,
            aligned position by position).
        config: Forwarded to ``dataset_fn``.
        to_row: Maps one ``(inputs, targets)`` row to one batch-row dict
            (``input_ids``, ``labels``, ``slice_id``, …); the caller owns
            the repo's unshifted-label / right-padding conventions.
        num_train: Number of training instances (fixed, generated once).
        num_val: Number of validation instances.
        num_test: Number of test instances.
        batch_size: Sequences per training batch (the last partial batch is
            kept).
        train_seed: Seed of the fixed training set (its base seed; the
            per-epoch permutations use ``SeedSequence([train_seed, e])``).
        val_seed: Seed of the fixed validation set.
        test_seed: Seed of the fixed test set.

    Raises:
        ValueError: Non-positive sizes / batch size, or colliding
            train/val/test seeds.
    """

    def __init__(
        self,
        *,
        dataset_fn: DatasetFn,
        config: Any,
        to_row: RowFn,
        num_train: int,
        num_val: int,
        num_test: int,
        batch_size: int,
        train_seed: int = 0,
        val_seed: int = 1000_003,
        test_seed: int = 2000_003,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        if num_train < 1 or num_val < 1 or num_test < 1:
            raise ValueError("num_train/num_val/num_test must be >= 1")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        seeds = (train_seed, val_seed, test_seed)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"train/val/test seeds must be distinct, got {seeds}")
        self.dataset_fn = dataset_fn
        self.config = config
        self.to_row = to_row
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.batch_size = batch_size
        self.train_seed = train_seed
        self.val_seed = val_seed
        self.test_seed = test_seed
        self.num_workers = num_workers

        self._train_rows: list[dict[str, torch.Tensor]] | None = None
        self.train_dataset: _FixedEpochTrainDataset | None = None
        self._val_rows: list[dict[str, torch.Tensor]] | None = None
        self._test_rows: list[dict[str, torch.Tensor]] | None = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _ensure_train_rows(self) -> list[dict[str, torch.Tensor]]:
        if self._train_rows is None:
            rows = _generate_rows(
                self.dataset_fn,
                self.config,
                self.num_train,
                True,
                self.train_seed,
                self.to_row,
            )
            self._train_rows = rows
            self.train_dataset = _FixedEpochTrainDataset(
                rows, self.batch_size, self.train_seed
            )
        return self._train_rows

    def _val_rows_(self) -> list[dict[str, torch.Tensor]]:
        if self._val_rows is None:
            self._val_rows = _generate_rows(
                self.dataset_fn,
                self.config,
                self.num_val,
                False,
                self.val_seed,
                self.to_row,
            )
        return self._val_rows

    def _test_rows_(self) -> list[dict[str, torch.Tensor]]:
        if self._test_rows is None:
            self._test_rows = _generate_rows(
                self.dataset_fn,
                self.config,
                self.num_test,
                False,
                self.test_seed,
                self.to_row,
            )
        return self._test_rows

    def _leakage_fraction(self) -> float:
        """Fraction of validation/test rows that also occur in the training
        set (compared on the raw input token rows)."""
        train_rows = {
            tuple(int(v) for v in row["input_ids"].tolist())
            for row in self._ensure_train_rows()
        }
        total = 0
        dup = 0
        for rows in (self._val_rows_(), self._test_rows_()):
            for row in rows:
                total += 1
                if tuple(int(v) for v in row["input_ids"].tolist()) in train_rows:
                    dup += 1
        return (dup / total) if total else 0.0

    def setup(self, stage: str | None = None) -> None:
        """Generate the fixed training/validation/test sets (idempotent).

        Logs the configuration and set sizes, and logs the val/test-
        into-train leakage fraction at WARNING level when it exceeds
        0.001.
        """
        self._ensure_train_rows()
        self._val_rows_()
        self._test_rows_()
        leakage = self._leakage_fraction()
        logger.info(
            "FixedSetRecallDataModule: config=%s num_train=%d num_val=%d "
            "num_test=%d batch_size=%d train_seed=%d val_seed=%d test_seed=%d",
            self.config,
            self.num_train,
            self.num_val,
            self.num_test,
            self.batch_size,
            self.train_seed,
            self.val_seed,
            self.test_seed,
        )
        if leakage > 0.001:
            logger.warning(
                "validation/test leakage into the training set: %.4f (> 0.001)",
                leakage,
            )

    # ------------------------------------------------------------------ #
    # Dataloaders
    # ------------------------------------------------------------------ #

    def train_dataloader(self) -> DataLoader:
        """The permuted-epoch stream of whole training batches."""
        return DataLoader(
            self._ensure_train_dataset(),
            batch_size=None,
            num_workers=self.num_workers,
        )

    def _ensure_train_dataset(self) -> _FixedEpochTrainDataset:
        self._ensure_train_rows()
        assert self.train_dataset is not None
        return self.train_dataset

    def _fixed_loader(self, rows: list[dict[str, torch.Tensor]]) -> DataLoader:
        """Loader over fixed in-memory rows (``shuffle=False``)."""
        return DataLoader(
            _FixedRowDataset(rows),
            batch_size=self.batch_size,
            shuffle=False,
        )

    @property
    def val_rows(self) -> list[dict[str, torch.Tensor]]:
        """The fixed validation rows."""
        return self._val_rows_()

    @property
    def test_rows(self) -> list[dict[str, torch.Tensor]]:
        """The fixed test rows."""
        return self._test_rows_()

    def val_dataloader(self) -> DataLoader:
        """Loader over the fixed validation set (``shuffle=False``)."""
        return self._fixed_loader(self._val_rows_())

    def test_dataloader(self) -> DataLoader:
        """Loader over the fixed test set (``shuffle=False``)."""
        return self._fixed_loader(self._test_rows_())

    # ------------------------------------------------------------------ #
    # Resumption
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        """Serialize the train stream position (the next LOCAL batch index,
        equal to the global batch index at world size 1)."""
        return {"next_batch_idx": int(self._ensure_train_dataset().next_local_idx)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Resume the train stream from a serialized batch index."""
        self._ensure_train_dataset().next_local_idx = int(state_dict["next_batch_idx"])

    def on_before_batch_transfer(
        self, batch: dict[str, torch.Tensor], dataloader_idx: int = 0
    ) -> dict[str, torch.Tensor]:
        """Advance the train stream position from the batch's global index
        (see :class:`~research_lib.data.synthetic.SyntheticRecallDataModule`).
        Nontrain batches are returned unchanged.
        """
        trainer = getattr(self, "trainer", None)
        if (
            isinstance(batch, dict)
            and "batch_index" in batch
            and trainer is not None
            and bool(getattr(trainer, "training", False))
        ):
            world_size, _ = _distributed_world_size_and_rank()
            index = int(batch["batch_index"])
            self._ensure_train_dataset().next_local_idx = index // world_size + 1
        return batch
