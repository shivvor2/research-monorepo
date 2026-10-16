"""Lightning DataModule for the synthetic recall tasks (MQAR, overwrite).

Streams fresh training batches from a seeded, infinite generator and holds
fixed validation/test sets, so training never reuses examples (protocol
``06`` §2: "data are generated, not stored"; the generators are pure
functions of a seed).

- **Train:** an :class:`~torch.utils.data.IterableDataset` yielding whole
  batches (dicts of stacked int64 tensors). Batch ``i`` is generated from
  ``np.random.SeedSequence([train_seed, i])``, so the stream is
  deterministic per ``(train_seed, batch index)`` and independent of the
  worker count: worker ``w`` produces the batches ``w, w + num_workers, ...``
  (each batch index exactly once, never duplicated).
- **Validation / test:** generated once in :meth:`setup`:
  ``num_val_per_slice`` / ``num_test_per_slice`` examples per slice value
  (``D`` for MQAR, ``K`` for overwrite), concatenated into one fixed set,
  loaded with ``shuffle=False`` and the default dict collate. Val/test seeds
  must differ from each other and from any train seed (defaults do).

Resumption: :meth:`state_dict` / :meth:`load_state_dict` carry the next
batch index, so a resumed run continues the same stream. **Exact resumption
requires ``num_workers=0``**: only in-process iteration advances the stream
position — DataLoader worker processes iterate pickled copies of the
dataset, so the parent's position does not advance. With
``num_workers > 0`` the caller must instead set the next batch index on
resume from the trainer's consumed batch count (its global step, absent
gradient accumulation), e.g.
``dm.load_state_dict({"next_batch_idx": trainer.global_step})``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .mqar import MQARConfig, generate_mqar
from .overwrite import OverwriteConfig, generate_overwrite

logger = logging.getLogger(__name__)

TaskConfig = MQARConfig | OverwriteConfig

_GENERATORS = {
    "mqar": generate_mqar,
    "overwrite": generate_overwrite,
}
CONFIG_TYPES = {"mqar": MQARConfig, "overwrite": OverwriteConfig}
TASKS = ("mqar", "overwrite")


def _batch_seed(train_seed: int, index: int) -> int:
    """Deterministic per-batch seed: ``SeedSequence([train_seed, index])``.

    Args:
        train_seed: base seed of the train stream.
        index: global batch index.

    Returns:
        A usable int seed for ``np.random.default_rng`` (64-bit draw).
    """
    return int(
        np.random.SeedSequence([int(train_seed), int(index)]).generate_state(
            1, dtype=np.uint64
        )[0]
    )


class _SyntheticTrainDataset(IterableDataset):
    """Infinite train stream yielding whole batches.

    Batch ``i`` is generated from ``np.random.SeedSequence([train_seed, i])``
    (via :func:`_batch_seed`). Worker ``w`` of ``num_workers`` produces the
    batch indices ``w', w' + num_workers, ...`` where ``w'`` is the smallest
    index ``>= next_batch_idx`` congruent to ``w (mod num_workers)``, so no
    batch index is ever produced twice and every batch's content depends
    only on its index.
    """

    def __init__(
        self,
        task: str,
        config: TaskConfig,
        batch_size: int,
        train_seed: int,
    ) -> None:
        """Args follow :class:`SyntheticRecallDataModule`."""
        super().__init__()
        self.task = task
        self.config = config
        self.batch_size = batch_size
        self.train_seed = train_seed
        self.next_batch_idx = 0

    def generate_batch(self, index: int) -> dict[str, torch.Tensor]:
        """Generate batch ``index`` (pure; independent of call order)."""
        return _GENERATORS[self.task](
            self.config,  # type: ignore[arg-type]
            num_examples=self.batch_size,
            seed=_batch_seed(self.train_seed, index),
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        # Worker w owns indices ≡ w (mod num_workers); start at the smallest
        # owned index >= next_batch_idx.
        index = self.next_batch_idx + ((worker_id - self.next_batch_idx) % num_workers)
        while True:
            batch = self.generate_batch(index)
            if info is None:
                # In-process iteration advances the stream position (exact
                # resumption for the default num_workers=0; worker processes
                # iterate copies and advance their own).
                self.next_batch_idx = index + num_workers
            yield batch
            index += num_workers


class _FixedRowDataset(Dataset):
    """Map-style dataset over a fixed set of per-example dict rows."""

    def __init__(self, rows: list[dict[str, torch.Tensor]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def _generate_fixed_set(
    task: str,
    config: TaskConfig,
    num_per_slice: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Generate a fixed evaluation set: ``num_per_slice`` examples per slice
    value, concatenated along the batch dimension.

    Each slice is seeded by its own VALUE — ``SeedSequence([seed,
    slice_value])`` — not by its position in the config, so a slice's examples
    are identical across configs that share the slice value (e.g.
    ``num_keys=(8, 16, 32)`` and ``(16, 32, 64)`` share the ``K=16`` rows)
    and runs / later re-evaluations stay comparable. Base seeds of the
    val/test/train streams must differ (enforced by the DataModule).
    """
    generator = _GENERATORS[task]
    parts = []
    for slice_value in config.slices:  # type: ignore[attr-defined]
        part_seed = int(
            np.random.SeedSequence([int(seed), int(slice_value)]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        if task == "mqar":
            part = generator(config, num_per_slice, part_seed, num_kv_pairs=slice_value)  # type: ignore[arg-type, call-arg]
        else:
            part = generator(config, num_per_slice, part_seed, num_keys=slice_value)  # type: ignore[arg-type, call-arg]
        parts.append(part)
    return {key: torch.cat([part[key] for part in parts], dim=0) for key in parts[0]}


class SyntheticRecallDataModule(L.LightningDataModule):
    """DataModule for the synthetic recall tasks.

    Training streams freshly generated batches (infinite; ``Trainer``'s
    ``max_steps`` ends training). Validation and test are fixed sets,
    generated once per :meth:`setup` call from held-out seeds.

    Args:
        task: ``"mqar"`` or ``"overwrite"``.
        task_config: the generator config (:class:`MQARConfig` or
            :class:`OverwriteConfig`); its slice values define the fixed
            validation/test slices.
        batch_size: sequences per batch (train stream and eval loaders).
        num_val_per_slice: validation examples per slice value.
        num_test_per_slice: test examples per slice value.
        train_seed: base seed of the train stream.
        val_seed: seed of the fixed validation set.
        test_seed: seed of the fixed test set.
        num_workers: DataLoader workers for the train stream.

    Raises:
        ValueError: invalid ``task``, sizes, or colliding seeds.
        TypeError: ``task_config`` type does not match ``task``.

    Note:
        Exact resumption via :meth:`state_dict` / :meth:`load_state_dict`
        requires ``num_workers=0``: only in-process iteration advances the
        stream position (worker processes iterate pickled copies of the
        dataset, so the parent's counter does not move). With
        ``num_workers > 0``, set the batch index on resume from the
        trainer's consumed batch count (its global step, absent gradient
        accumulation), e.g.
        ``dm.load_state_dict({"next_batch_idx": trainer.global_step})``.
        Batch *contents* are per-index deterministic for any worker count.

    Example:
        >>> dm = SyntheticRecallDataModule(
        ...     task="mqar",
        ...     task_config=MQARConfig(vocab_size=512, seq_len=64, num_kv_pairs=(4, 8)),
        ...     batch_size=64,
        ... )  # doctest: +SKIP
        >>> dm.setup("validate")  # doctest: +SKIP
    """

    def __init__(
        self,
        task: str = "mqar",
        task_config: TaskConfig = MQARConfig(),
        batch_size: int = 256,
        num_val_per_slice: int = 1000,
        num_test_per_slice: int = 1000,
        train_seed: int = 0,
        val_seed: int = 1_000_003,
        test_seed: int = 2_000_003,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        expected_type = CONFIG_TYPES[task]
        if not isinstance(task_config, expected_type):
            raise TypeError(
                f"task {task!r} requires a {expected_type.__name__}, "
                f"got {type(task_config).__name__}"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_val_per_slice < 1 or num_test_per_slice < 1:
            raise ValueError("num_val_per_slice and num_test_per_slice must be >= 1")
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        seeds = (train_seed, val_seed, test_seed)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"train/val/test seeds must be distinct, got {seeds}")

        self.task = task
        self.task_config = task_config
        self.batch_size = batch_size
        self.num_val_per_slice = num_val_per_slice
        self.num_test_per_slice = num_test_per_slice
        self.train_seed = train_seed
        self.val_seed = val_seed
        self.test_seed = test_seed
        self.num_workers = num_workers

        self.train_dataset: _SyntheticTrainDataset | None = None
        self._val_set: dict[str, torch.Tensor] | None = None
        self._test_set: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _ensure_train_dataset(self) -> _SyntheticTrainDataset:
        if self.train_dataset is None:
            self.train_dataset = _SyntheticTrainDataset(
                task=self.task,
                config=self.task_config,
                batch_size=self.batch_size,
                train_seed=self.train_seed,
            )
        return self.train_dataset

    def _build_val_set(self) -> dict[str, torch.Tensor]:
        return _generate_fixed_set(
            self.task, self.task_config, self.num_val_per_slice, self.val_seed
        )

    def _build_test_set(self) -> dict[str, torch.Tensor]:
        return _generate_fixed_set(
            self.task, self.task_config, self.num_test_per_slice, self.test_seed
        )

    def setup(self, stage: str | None = None) -> None:
        """Build the train stream and the fixed val/test sets (idempotent).

        Logs (via ``logging``, never ``print``): task, config, slice values
        and set sizes.

        Args:
            stage: Lightning stage; unused (all artifacts are built lazily
                and cached).
        """
        self._ensure_train_dataset()
        if self._val_set is None:
            self._val_set = self._build_val_set()
        if self._test_set is None:
            self._test_set = self._build_test_set()
        logger.info(
            "SyntheticRecallDataModule: task=%s config=%s slices=%s "
            "num_val_per_slice=%d num_test_per_slice=%d "
            "train_seed=%d val_seed=%d test_seed=%d",
            self.task,
            self.task_config,
            tuple(self.task_config.slices),  # type: ignore[attr-defined]
            self.num_val_per_slice,
            self.num_test_per_slice,
            self.train_seed,
            self.val_seed,
            self.test_seed,
        )

    # ------------------------------------------------------------------ #
    # Dataloaders
    # ------------------------------------------------------------------ #

    def train_dataloader(self) -> DataLoader:
        """Infinite stream of whole generated batches (``batch_size=None``)."""
        return DataLoader(
            self._ensure_train_dataset(),
            batch_size=None,
            num_workers=self.num_workers,
        )

    def _fixed_loader(self, tensors: dict[str, torch.Tensor]) -> DataLoader:
        """Loader over a fixed in-memory set of example dicts (``shuffle=False``)."""
        rows = [
            {key: value[i] for key, value in tensors.items()}
            for i in range(tensors["input_ids"].shape[0])
        ]
        return DataLoader(
            _FixedRowDataset(rows),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def _val_tensors(self) -> dict[str, torch.Tensor]:
        """The fixed validation set (built lazily on first use)."""
        if self._val_set is None:
            self._val_set = self._build_val_set()
        return self._val_set

    def _test_tensors(self) -> dict[str, torch.Tensor]:
        """The fixed test set (built lazily on first use)."""
        if self._test_set is None:
            self._test_set = self._build_test_set()
        return self._test_set

    @property
    def val_set(self) -> dict[str, torch.Tensor]:
        """The fixed validation set: ``num_val_per_slice`` examples per slice
        value, concatenated along the batch dimension."""
        return self._val_tensors()

    @property
    def test_set(self) -> dict[str, torch.Tensor]:
        """The fixed test set: ``num_test_per_slice`` examples per slice
        value, concatenated along the batch dimension."""
        return self._test_tensors()

    def val_dataloader(self) -> DataLoader:
        """Loader over the fixed validation set (``shuffle=False``)."""
        return self._fixed_loader(self._val_tensors())

    def test_dataloader(self) -> DataLoader:
        """Loader over the fixed test set (``shuffle=False``)."""
        return self._fixed_loader(self._test_tensors())

    # ------------------------------------------------------------------ #
    # Resumption
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        """Serialize the train stream position (next global batch index)."""
        return {"next_batch_idx": int(self._ensure_train_dataset().next_batch_idx)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Resume the train stream from a serialized position."""
        self._ensure_train_dataset().next_batch_idx = int(state_dict["next_batch_idx"])
