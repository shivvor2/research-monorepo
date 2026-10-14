"""
Test suite for SyntheticRecallDataModule (spec PR 3, test D08).

Covers:
  * Construction validation (task/config-type/seed checks)
  * Train stream determinism per (seed, batch index)
  * Worker-count independence of the train stream (residue sharding)
  * Resumption via state_dict / load_state_dict (in-process and sharded)
  * Fixed validation/test sets: slice counts, seed disjointness, keys/dtypes

Run with: pytest src/research_lib/data/synthetic/tests/test_datamodule.py -v
"""

import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from ..datamodule import (
    SyntheticRecallDataModule,
    _batch_seed,
    _generate_fixed_set,
    _SyntheticTrainDataset,
)
from ..mqar import MQARConfig, generate_mqar
from ..overwrite import OverwriteConfig

MQAR_CFG = MQARConfig(vocab_size=128, seq_len=32, num_kv_pairs=(2, 4))
# Feasible: for K=8, R = 128 - 16 = 112 spans (112-2)//16 + 1 = 7 blocks
# >= distant_min_blocks + 1 = 5, and 6 * K = 48 <= 128.
OW_CFG = OverwriteConfig(
    vocab_size=512,
    seq_len=128,
    num_keys=(4, 8),
    block_size=16,
    distant_min_blocks=4,
)


def _take(loader_or_iter, n):
    """Consume the first n items of a loader/iterator into a list."""
    return list(itertools.islice(iter(loader_or_iter), n))


def _batch_content(batch):
    """Hashable content key of a whole-batch dict (order-independent per key)."""
    return tuple((key, tuple(batch[key].flatten().tolist())) for key in sorted(batch))


def _dm(**kwargs):
    """Fresh module with small sizes; extra kwargs pass through."""
    task = kwargs.pop("task", "mqar")
    config = kwargs.pop("task_config", MQAR_CFG)
    return SyntheticRecallDataModule(
        task=task,
        task_config=config,
        batch_size=kwargs.pop("batch_size", 4),
        num_val_per_slice=kwargs.pop("num_val_per_slice", 3),
        num_test_per_slice=kwargs.pop("num_test_per_slice", 5),
        **kwargs,
    )


# =============================================================================
# Tests: construction validation
# =============================================================================


def test_datamodule_construction_validation():
    with pytest.raises(ValueError):
        SyntheticRecallDataModule(task="nope", task_config=MQAR_CFG)
    with pytest.raises(TypeError):
        SyntheticRecallDataModule(task="mqar", task_config=OW_CFG)
    with pytest.raises(TypeError):
        SyntheticRecallDataModule(task="overwrite", task_config=MQAR_CFG)
    with pytest.raises(ValueError):
        # val_seed == test_seed
        SyntheticRecallDataModule(
            task="mqar", task_config=MQAR_CFG, val_seed=7, test_seed=7
        )
    with pytest.raises(ValueError):
        # train_seed == val_seed
        SyntheticRecallDataModule(
            task="mqar", task_config=MQAR_CFG, train_seed=7, val_seed=7
        )


# =============================================================================
# Tests: D08 train stream
# =============================================================================


def test_D08_train_stream_deterministic_per_seed_and_batch_index():
    dm = _dm()
    batches = _take(dm.train_dataloader(), 3)

    # Same (train_seed, batch index) -> identical batches (fresh module).
    dm2 = _dm()
    batches2 = _take(dm2.train_dataloader(), 3)
    for first, second in zip(batches, batches2, strict=True):
        assert _batch_content(first) == _batch_content(second)

    # Batch i equals a direct generation from SeedSequence([train_seed, i]).
    for i, batch in enumerate(batches):
        expected = generate_mqar(
            MQAR_CFG, num_examples=dm.batch_size, seed=_batch_seed(dm.train_seed, i)
        )
        assert _batch_content(batch) == _batch_content(expected)

    # Different train seed -> different stream.
    dm3 = _dm(train_seed=123)
    batches3 = _take(dm3.train_dataloader(), 3)
    for first, second in zip(batches, batches3, strict=True):
        assert _batch_content(first) != _batch_content(second)


def test_D08_train_stream_worker_sharding():
    """Worker w of num_workers produces the batches w, w + num_workers, ...;
    the union over workers covers every stream index exactly once, for any
    worker count."""
    reference = [
        generate_mqar(MQAR_CFG, num_examples=4, seed=_batch_seed(0, i))
        for i in range(9)
    ]

    for num_workers in (1, 2, 3):
        produced = {}
        for worker_id in range(num_workers):
            dataset = _SyntheticTrainDataset(
                task="mqar", config=MQAR_CFG, batch_size=4, train_seed=0
            )
            with patch(
                "torch.utils.data.get_worker_info",
                return_value=SimpleNamespace(id=worker_id, num_workers=num_workers),
            ):
                worker_batches = _take(iter(dataset), 3)
            indices = [worker_id + num_workers * j for j in range(3)]
            for index, batch in zip(indices, worker_batches, strict=True):
                assert index not in produced, "batch index produced twice"
                produced[index] = batch
        # Every stream index covered exactly once (3 per worker), identical
        # content per index, for any worker count.
        assert sorted(produced) == list(range(3 * num_workers))
        for index, batch in produced.items():
            assert _batch_content(batch) == _batch_content(reference[index])


def test_D08_train_stream_worker_count_independent_via_dataloader():
    """Through a real DataLoader with num_workers=2: the collected stream has
    no duplicate batches and its contents are exactly the per-index batches
    of the num_workers=0 stream (arrival order across workers is not
    guaranteed; contents are)."""
    dm2 = _dm(num_workers=2)
    dm2.setup()
    loader = DataLoader(dm2.train_dataset, batch_size=None, num_workers=2)
    collected = _take(loader, 6)
    del loader

    reference = {
        _batch_content(generate_mqar(MQAR_CFG, 4, seed=_batch_seed(0, i)))
        for i in range(6)
    }
    assert len({_batch_content(b) for b in collected}) == 6  # no duplicates
    assert {_batch_content(b) for b in collected} == reference


def test_D08_resume_via_state_dict():
    dm = _dm()
    _take(dm.train_dataloader(), 3)
    state = dm.state_dict()
    assert state == {"next_batch_idx": 3}

    # A fresh module resumes the same stream at index 3.
    fresh = _dm()
    fresh.load_state_dict(state)
    resumed = _take(fresh.train_dataloader(), 2)
    for offset, batch in enumerate(resumed):
        expected = generate_mqar(
            MQAR_CFG,
            num_examples=dm.batch_size,
            seed=_batch_seed(dm.train_seed, 3 + offset),
        )
        assert _batch_content(batch) == _batch_content(expected)


def test_D08_resume_sharded_with_workers():
    """Resuming mid-stream with workers: worker w starts at the smallest owned
    index >= the saved position; the union over workers covers exactly the
    indices >= the saved position, each once."""
    num_workers = 2
    saved = 5
    produced = {}
    for worker_id in range(num_workers):
        dataset = _SyntheticTrainDataset(
            task="mqar", config=MQAR_CFG, batch_size=4, train_seed=0
        )
        dataset.next_batch_idx = saved
        with patch(
            "torch.utils.data.get_worker_info",
            return_value=SimpleNamespace(id=worker_id, num_workers=num_workers),
        ):
            worker_batches = _take(iter(dataset), 3)
        # Worker w owns indices ≡ w (mod num_workers), starting at the
        # smallest owned index >= saved.
        start = saved + ((worker_id - saved) % num_workers)
        indices = [start + num_workers * j for j in range(3)]
        for index, batch in zip(indices, worker_batches, strict=True):
            assert index not in produced, "batch index produced twice"
            assert index >= saved
            produced[index] = batch
    assert sorted(produced) == [5, 6, 7, 8, 9, 10]
    for index, batch in produced.items():
        expected = generate_mqar(MQAR_CFG, num_examples=4, seed=_batch_seed(0, index))
        assert _batch_content(batch) == _batch_content(expected)


# =============================================================================
# Tests: D08 validation / test sets
# =============================================================================


def test_D08_val_test_fixed_and_disjoint():
    dm = _dm(num_val_per_slice=7, num_test_per_slice=5)
    assert len({dm.train_seed, dm.val_seed, dm.test_seed}) == 3
    dm.setup()
    dm.setup()  # idempotent: same fixed sets

    # Keys and dtypes as in the batch format (MQAR subset).
    assert set(dm.val_set) == {"input_ids", "labels", "slice_id"}
    assert all(t.dtype == torch.int64 for t in dm.val_set.values())
    assert dm.val_set["input_ids"].shape == (2 * 7, MQAR_CFG.seq_len)
    assert dm.test_set["input_ids"].shape == (2 * 5, MQAR_CFG.seq_len)

    # Slice counts: num_*_per_slice examples per slice value.
    for value in MQAR_CFG.slices:
        assert int((dm.val_set["slice_id"] == value).sum()) == 7
        assert int((dm.test_set["slice_id"] == value).sum()) == 5

    # Fixed: regenerating with the same seeds reproduces the sets.
    fresh = _dm(num_val_per_slice=7, num_test_per_slice=5)
    fresh.setup()
    for key in dm.val_set:
        assert torch.equal(dm.val_set[key], fresh.val_set[key])
        assert torch.equal(dm.test_set[key], fresh.test_set[key])
    # Val and test differ (disjoint seeds).
    assert not torch.equal(dm.val_set["input_ids"], dm.test_set["input_ids"])
    # And both differ from train batches (disjoint seed bases).
    train_batch = _take(dm.train_dataloader(), 1)[0]
    assert not torch.equal(dm.val_set["input_ids"][:4], train_batch["input_ids"])


def test_D08_val_test_loader_shapes():
    dm = _dm(num_val_per_slice=3, num_test_per_slice=5)
    dm.setup()
    val_batches = list(dm.val_dataloader())
    assert sum(b["input_ids"].shape[0] for b in val_batches) == 2 * 3
    assert val_batches[0]["input_ids"].shape == (dm.batch_size, MQAR_CFG.seq_len)
    assert val_batches[0]["slice_id"].shape == (dm.batch_size,)
    assert val_batches[0]["labels"].dtype == torch.int64
    test_batches = list(dm.test_dataloader())
    assert sum(b["input_ids"].shape[0] for b in test_batches) == 2 * 5


def test_D08_overwrite_task_batches():
    dm = _dm(task="overwrite", task_config=OW_CFG, num_val_per_slice=3)
    dm.setup()
    train_batch = _take(dm.train_dataloader(), 1)[0]
    assert set(train_batch) == {
        "input_ids",
        "labels",
        "slice_id",
        "old_labels",
        "key_class",
        "write_distance",
    }
    assert all(t.dtype == torch.int64 for t in train_batch.values())
    assert train_batch["input_ids"].shape == (dm.batch_size, OW_CFG.seq_len)
    val_batch = list(dm.val_dataloader())[0]
    assert set(val_batch) == set(train_batch)


def test_D08_fixed_set_generator_matches_dm():
    dm = _dm(num_val_per_slice=6)
    dm.setup()
    direct = _generate_fixed_set("mqar", MQAR_CFG, 6, dm.val_seed)
    for key in direct:
        assert torch.equal(dm.val_set[key], direct[key])
