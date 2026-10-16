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

    # Batch i equals a direct generation from SeedSequence([train_seed, i]),
    # plus its global batch index in the "batch_index" key.
    for i, batch in enumerate(batches):
        expected = generate_mqar(
            MQAR_CFG, num_examples=dm.batch_size, seed=_batch_seed(dm.train_seed, i)
        )
        expected["batch_index"] = torch.tensor(i, dtype=torch.int64)
        assert _batch_content(batch) == _batch_content(expected)
        assert int(batch["batch_index"]) == i
        assert batch["batch_index"].dtype == torch.int64
        assert batch["batch_index"].shape == ()

    # Different train seed -> different stream.
    dm3 = _dm(train_seed=123)
    batches3 = _take(dm3.train_dataloader(), 3)
    for first, second in zip(batches, batches3, strict=True):
        assert _batch_content(first) != _batch_content(second)


def test_D08_train_stream_worker_sharding():
    """Worker w of num_workers produces the batches w, w + num_workers, ...;
    the union over workers covers every stream index exactly once, for any
    worker count."""
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
                assert int(batch["batch_index"]) == index
                produced[index] = batch
        # Every stream index covered exactly once (3 per worker), identical
        # content per index (including the batch_index key), for any worker
        # count.
        assert sorted(produced) == list(range(3 * num_workers))
        for index, batch in produced.items():
            expected = generate_mqar(
                MQAR_CFG, num_examples=4, seed=_batch_seed(0, index)
            )
            expected["batch_index"] = torch.tensor(index, dtype=torch.int64)
            assert _batch_content(batch) == _batch_content(expected)


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

    reference = set()
    for i in range(6):
        expected = generate_mqar(MQAR_CFG, 4, seed=_batch_seed(0, i))
        expected["batch_index"] = torch.tensor(i, dtype=torch.int64)
        reference.add(_batch_content(expected))
    assert len({_batch_content(b) for b in collected}) == 6  # no duplicates
    assert {_batch_content(b) for b in collected} == reference


def test_D08_resume_via_state_dict():
    dm = _dm()
    consumed = _take(dm.train_dataloader(), 3)
    assert [int(b["batch_index"]) for b in consumed] == [0, 1, 2]
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
        expected["batch_index"] = torch.tensor(3 + offset, dtype=torch.int64)
        assert _batch_content(batch) == _batch_content(expected)
        assert int(batch["batch_index"]) == 3 + offset


@pytest.mark.parametrize("num_workers", [2, 3])
def test_D08_stream_position_exact_for_any_worker_count(num_workers):
    """The stream position advanced by on_before_batch_transfer makes
    state_dict resumption exact for any worker count: batches fed through
    the hook in DataLoader (round-robin) order, checkpointed mid-stream,
    restore into a fresh module that continues the uninterrupted stream
    with no replay and no skip."""
    # The uninterrupted reference stream, batch by batch.
    reference = []
    for i in range(10):
        expected = generate_mqar(MQAR_CFG, num_examples=4, seed=_batch_seed(0, i))
        expected["batch_index"] = torch.tensor(i, dtype=torch.int64)
        reference.append(expected)

    dm = _dm()
    # The hook requires an attached trainer in the training stage.
    dm.trainer = SimpleNamespace(training=True)

    # Simulated workers (patched get_worker_info). The generator body reads
    # the (patched) worker info lazily on first next(), so each worker's
    # batches are collected inside its patch context; they are then fed
    # through the hook in DataLoader round-robin order: step s takes worker
    # s % num_workers' next batch, which has global index s.
    worker_batches = {}
    for worker_id in range(num_workers):
        dataset = _SyntheticTrainDataset(
            task="mqar", config=MQAR_CFG, batch_size=4, train_seed=0
        )
        with patch(
            "torch.utils.data.get_worker_info",
            return_value=SimpleNamespace(id=worker_id, num_workers=num_workers),
        ):
            worker_batches[worker_id] = _take(
                iter(dataset), len(range(worker_id, 10, num_workers))
            )
    mid = 7
    state = None
    consumed = []
    for step in range(10):
        batch = worker_batches[step % num_workers][step // num_workers]
        dm.on_before_batch_transfer(batch, 0)
        consumed.append(batch)
        if step == mid - 1:
            state = dm.state_dict()

    assert [int(b["batch_index"]) for b in consumed] == list(range(10))
    assert state is not None and state == {"next_batch_idx": mid}

    # Fresh module: the continuation equals the uninterrupted stream from
    # `mid` on (no replay of 0..mid-1, no skip).
    fresh = _dm()
    fresh.load_state_dict(state)
    resumed = _take(fresh.train_dataloader(), 3)
    for offset, batch in enumerate(resumed):
        assert _batch_content(batch) == _batch_content(reference[mid + offset])

    # Without an attached trainer the hook is a no-op.
    quiet = _dm()
    quiet.on_before_batch_transfer(consumed[0], 0)
    assert quiet.state_dict() == {"next_batch_idx": 0}


def test_D08_batch_index_ignored_by_loss():
    """The extra "batch_index" key does not affect the training loss:
    DualOptimizerModule.compute_loss reads only `labels` (and falls back to
    `input_ids`), so extra batch keys are ignored."""
    import torch.nn.functional as F
    from torch import nn

    from ....training.modules.dual_optimizer import DualOptimizerModule
    from ....training.presets import default_adamw_config

    dm = _dm()
    batch = _take(dm.train_dataloader(), 1)[0]
    assert "batch_index" in batch

    matrix_opt, matrix_sched = default_adamw_config()
    vector_opt, vector_sched = default_adamw_config()
    module = DualOptimizerModule(
        model=nn.Identity(),
        matrix_optimizer_config=matrix_opt,
        vector_optimizer_config=vector_opt,
        matrix_schedule_config=matrix_sched,
        vector_schedule_config=vector_sched,
    )
    torch.manual_seed(0)
    logits = torch.randn(
        dm.batch_size, MQAR_CFG.seq_len, MQAR_CFG.vocab_size, dtype=torch.float64
    )
    loss = module.compute_loss(logits, batch)
    labels = batch["labels"]
    ans_b, ans_p = (labels != -100).nonzero(as_tuple=True)
    manual = F.cross_entropy(logits[ans_b, ans_p - 1], labels[ans_b, ans_p])
    assert torch.allclose(loss, manual, atol=1e-10, rtol=1e-10)

    # Removing the key changes nothing.
    without_index = {k: v for k, v in batch.items() if k != "batch_index"}
    assert torch.allclose(module.compute_loss(logits, without_index), loss)


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
        expected["batch_index"] = torch.tensor(index, dtype=torch.int64)
        assert _batch_content(batch) == _batch_content(expected)
        assert int(batch["batch_index"]) == index


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
        "batch_index",
    }
    assert all(t.dtype == torch.int64 for t in train_batch.values())
    assert train_batch["input_ids"].shape == (dm.batch_size, OW_CFG.seq_len)
    val_batch = list(dm.val_dataloader())[0]
    # Eval batches carry the task keys only (no "batch_index").
    assert set(val_batch) == set(train_batch) - {"batch_index"}
    assert "batch_index" not in val_batch


def test_D08_fixed_set_generator_matches_dm():
    dm = _dm(num_val_per_slice=6)
    dm.setup()
    direct = _generate_fixed_set("mqar", MQAR_CFG, 6, dm.val_seed)
    for key in direct:
        assert torch.equal(dm.val_set[key], direct[key])


def test_D08_val_test_slices_seeded_by_slice_value():
    """Slices are seeded by slice VALUE, not position: configs that share a
    slice value (and base seed) share that slice's rows bit-exactly, so
    e.g. re-evaluations under a different slice grid stay comparable."""
    # Overwrite: K in {16, 32} shared between (8, 16, 32) and (16, 32, 64).
    ow_a = OverwriteConfig(vocab_size=4096, seq_len=512, num_keys=(8, 16, 32))
    ow_b = OverwriteConfig(vocab_size=4096, seq_len=512, num_keys=(16, 32, 64))
    seed = 777
    set_a = _generate_fixed_set("overwrite", ow_a, 3, seed)
    set_b = _generate_fixed_set("overwrite", ow_b, 3, seed)
    for key in set_a:
        for slice_value in (16, 32):
            rows_a = set_a[key][set_a["slice_id"] == slice_value]
            rows_b = set_b[key][set_b["slice_id"] == slice_value]
            assert rows_a.shape[0] == 3 and rows_b.shape[0] == 3
            assert torch.equal(rows_a, rows_b), f"{key} K={slice_value} differs"
    # Slices exclusive to one config are absent from the other.
    assert int((set_a["slice_id"] == 8).sum()) == 3
    assert int((set_b["slice_id"] == 8).sum()) == 0
    assert int((set_b["slice_id"] == 64).sum()) == 3
    assert int((set_a["slice_id"] == 64).sum()) == 0
    # A different base seed changes the shared slice's rows.
    set_c = _generate_fixed_set("overwrite", ow_a, 3, seed + 1)
    rows_c = set_c["input_ids"][set_c["slice_id"] == 16]
    rows_a = set_a["input_ids"][set_a["slice_id"] == 16]
    assert not torch.equal(rows_a, rows_c)

    # MQAR: D in {4, 8} shared between (2, 4, 8) and (4, 8, 16).
    m_a = MQARConfig(vocab_size=256, seq_len=64, num_kv_pairs=(2, 4, 8))
    m_b = MQARConfig(vocab_size=256, seq_len=64, num_kv_pairs=(4, 8, 16))
    s_a = _generate_fixed_set("mqar", m_a, 5, seed=555)
    s_b = _generate_fixed_set("mqar", m_b, 5, seed=555)
    for key in s_a:
        for slice_value in (4, 8):
            assert torch.equal(
                s_a[key][s_a["slice_id"] == slice_value],
                s_b[key][s_b["slice_id"] == slice_value],
            )

    # DataModule level: two modules with the same val_seed share val rows.
    dm_a = _dm(
        task="overwrite",
        task_config=ow_a,
        num_val_per_slice=3,
        val_seed=777,
        test_seed=888,
    )
    dm_b = _dm(
        task="overwrite",
        task_config=ow_b,
        num_val_per_slice=3,
        val_seed=777,
        test_seed=999,
    )
    dm_a.setup()
    dm_b.setup()
    mask_a = dm_a.val_set["slice_id"] == 16
    mask_b = dm_b.val_set["slice_id"] == 16
    assert torch.equal(
        dm_a.val_set["input_ids"][mask_a], dm_b.val_set["input_ids"][mask_b]
    )
