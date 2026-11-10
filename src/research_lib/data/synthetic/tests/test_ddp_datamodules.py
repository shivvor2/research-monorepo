"""
Test suite for the multi-device behaviour of the synthetic datamodules.

Covers the single-device invariants this change must preserve and the
multi-device semantics it adds:

* single-device train streams and val/test sets unchanged (frozen
  fixtures generated before the change);
* global batches: every rank iterates the same global batch indices and
  keeps the contiguous rows ``[r * B / W, (r + 1) * B / W)`` of each
  batch, for ``num_workers`` 0 and 2, in both datamodules;
* ``batch_size`` divisibility by the world size is enforced at setup;
* the rank identity lives on the dataset (captured at setup in the rank
  process), so DataLoader workers under the ``spawn`` start method
  produce the right rows even without a process group;
* the checkpointed cursor is the shared GLOBAL batch index and resuming
  at an unaligned cursor continues the uninterrupted stream on every
  rank.

Run with: pytest src/research_lib/data/synthetic/tests/test_ddp_datamodules.py -v
"""

import itertools
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from .. import datamodule as datamodule_module
from ..datamodule import SyntheticRecallDataModule, _batch_seed, _SyntheticTrainDataset
from ..entity_field import EntityFieldConfig
from ..fixed_set import FixedSetRecallDataModule, _FixedEpochTrainDataset
from ..mad_fuzzy import MadFuzzyConfig
from ..mqar import MQARConfig, generate_mqar
from ..overwrite import OverwriteConfig
from ..registry import require_task

_HERE = Path(__file__).resolve().parent
_FIXTURES = _HERE / "fixtures"

BATCH_SIZE = 8
NUM_VAL = 6  # per slice for the streaming tasks; total for the fixed set
NUM_TEST = 6

MQAR_CFG = MQARConfig(vocab_size=128, seq_len=32, num_kv_pairs=(2, 4))
OW_CFG = OverwriteConfig(
    vocab_size=512,
    seq_len=128,
    num_keys=(4, 8),
    block_size=16,
    distant_min_blocks=4,
)
EF_CFG = EntityFieldConfig(
    vocab_size=2048,
    seq_len=256,
    num_fields=4,
    num_entities=(2, 4, 8),
    block_size=16,
    distant_min_blocks=4,
    reserved_filler=512,
)
MAD_CFG = MadFuzzyConfig(
    vocab_size=16,
    seq_len=32,
    k_motif_size=3,
    v_motif_size=3,
    multi_query=True,
    noise_vocab_size=0,
    frac_noise=0.0,
)
MAD_PAD_TO = 48  # padded_generated_length(32, 16)

STREAMING_CASES = (
    ("mqar", MQAR_CFG, 2),
    ("overwrite", OW_CFG, 2),
    ("entity_field", EF_CFG, 3),
)

_TASK_SKIP = frozenset({"batch_index", "epoch"})


def _tensors(batch_or_set: dict) -> dict[str, np.ndarray]:
    return {
        k: v.numpy() for k, v in batch_or_set.items() if isinstance(v, torch.Tensor)
    }


def _streaming_dm(name, config):
    return SyntheticRecallDataModule(
        task=name,
        task_config=config,
        batch_size=BATCH_SIZE,
        num_val_per_slice=NUM_VAL,
        num_test_per_slice=NUM_TEST,
        train_seed=0,
        val_seed=1_000_003,
        test_seed=2_000_003,
        num_workers=0,
    )


def _fixed_dm():
    spec = require_task("mad_fuzzy")
    return FixedSetRecallDataModule(
        dataset_fn=spec.dataset_fn,
        config=MAD_CFG,
        to_row=spec.row_fn(MAD_CFG, MAD_PAD_TO),
        num_train=64,
        num_val=NUM_VAL,
        num_test=NUM_TEST,
        batch_size=BATCH_SIZE,
        train_seed=0,
        val_seed=1_000_003,
        test_seed=2_000_003,
        num_workers=0,
    )


def _fixed_batches(dm, n):
    return list(itertools.islice(dm.train_dataloader(), n))


def _content(batch: dict) -> tuple:
    """Order-independent content key of a whole-batch dict."""
    return tuple((key, tuple(v.flatten().tolist())) for key, v in sorted(batch.items()))


def _slice_rows(batch: dict, start: int, per: int) -> dict[str, torch.Tensor]:
    return {
        key: (value[start : start + per] if value.dim() >= 1 else value)
        for key, value in batch.items()
    }


# =============================================================================
# MD01 · single device unchanged (frozen fixtures from before this change)
# =============================================================================


def test_single_device_batches_and_sets_unchanged_from_fixtures():
    """The first three train batches and the fixed val/test sets of every
    task (streaming and fixed-set) equal the frozen fixtures byte for
    byte, so a change to the shared datamodules cannot silently alter what
    a single-device run consumes."""
    for name, config in ((n, c) for n, c, _ in STREAMING_CASES):
        dm = _streaming_dm(name, config)
        dm.setup()
        batches = list(itertools.islice(dm.train_dataloader(), 3))
        assert [int(b["batch_index"]) for b in batches] == [0, 1, 2]
        with np.load(_FIXTURES / f"ddp_unchanged_{name}.npz") as fixture:
            for i, batch in enumerate(batches):
                for key, value in _tensors(batch).items():
                    assert np.array_equal(value, fixture[f"train{i}_{key}"])
            for prefix, tensors in (("val", dm.val_set), ("test", dm.test_set)):
                for key, value in _tensors(tensors).items():
                    assert np.array_equal(value, fixture[f"{prefix}_{key}"])

    dm = _fixed_dm()
    dm.setup()
    batches = _fixed_batches(dm, 3)
    assert [int(b["batch_index"]) for b in batches] == [0, 1, 2]
    val = {k: torch.stack([r[k] for r in dm.val_rows]) for k in dm.val_rows[0]}
    test = {k: torch.stack([r[k] for r in dm.test_rows]) for k in dm.test_rows[0]}
    with np.load(_FIXTURES / "ddp_unchanged_mad_fuzzy.npz") as fixture:
        for i, batch in enumerate(batches):
            for key, value in _tensors(batch).items():
                assert np.array_equal(value, fixture[f"train{i}_{key}"])
        for prefix, tensors in (("val", val), ("test", test)):
            for key, value in _tensors(tensors).items():
                assert np.array_equal(value, fixture[f"{prefix}_{key}"])


# =============================================================================
# MD02 · the ranks' rows are contiguous slices of the global batch
# =============================================================================


def test_streaming_rank_rows_are_contiguous_slices():
    """W = 2 (explicit identity), ``num_workers`` 0 and 2: every rank
    iterates the same global batch indices, each batch carries the rank's
    contiguous rows ``[r * B / 2, (r + 1) * B / 2)``, and the two ranks'
    rows are disjoint and concatenate to the single-device batch."""
    single = _streaming_dm("mqar", MQAR_CFG)
    single.setup()
    whole = list(itertools.islice(single.train_dataloader(), 4))

    for num_workers, count in ((0, 2), (2, 4)):
        ranks = []
        for rank in (0, 1):
            with patch.object(
                datamodule_module,
                "_distributed_world_size_and_rank",
                return_value=(2, rank),
            ):
                dm = _streaming_dm("mqar", MQAR_CFG)
                dm.setup()
                dm.num_workers = num_workers
                ranks.append(list(itertools.islice(dm.train_dataloader(), count)))
        for index in range(count):
            r0, r1 = ranks[0][index], ranks[1][index]
            expected = whole[index]
            assert int(r0["batch_index"]) == int(expected["batch_index"])
            per = BATCH_SIZE // 2
            assert _content(r0) == _content(_slice_rows(expected, 0, per))
            assert _content(r1) == _content(_slice_rows(expected, per, per))


def test_fixed_set_rank_rows_are_contiguous_slices():
    """The same rank-row slicing for the fixed-set datamodule (a global
    batch is its epoch's permutation slice; a rank keeps its rows)."""
    single = _fixed_dm()
    single.setup()
    whole = _fixed_batches(single, 2)

    for rank in (0, 1):
        with patch(
            "research_lib.data.synthetic.fixed_set.distributed_world_size_and_rank",
            return_value=(2, rank),
        ):
            dm = _fixed_dm()
            dm.setup()
            batches = _fixed_batches(dm, 2)
        per = BATCH_SIZE // 2
        for index, batch in enumerate(batches):
            assert int(batch["batch_index"]) == index
            assert _content(batch) == _content(
                _slice_rows(whole[index], rank * per, per)
            )


# =============================================================================
# MD03 · batch_size divisibility by the world size
# =============================================================================


def test_batch_size_not_divisible_by_world_size_raises_at_setup():
    """``batch_size % world_size != 0`` raises at ``setup()``, naming both
    values, in both datamodules (a global batch must split evenly)."""
    odd = SyntheticRecallDataModule(
        task="mqar",
        task_config=MQARConfig(vocab_size=128, seq_len=32, num_kv_pairs=(2, 4)),
        batch_size=7,  # not divisible by 2
        num_val_per_slice=2,
        num_test_per_slice=2,
    )
    with patch.object(
        datamodule_module, "_distributed_world_size_and_rank", return_value=(2, 0)
    ):
        with pytest.raises(ValueError, match="7.*2"):
            odd.setup()

    spec = require_task("mad_fuzzy")
    fixed = FixedSetRecallDataModule(
        dataset_fn=spec.dataset_fn,
        config=MAD_CFG,
        to_row=spec.row_fn(MAD_CFG, MAD_PAD_TO),
        num_train=64,
        num_val=6,
        num_test=6,
        batch_size=7,
    )
    with patch(
        "research_lib.data.synthetic.fixed_set.distributed_world_size_and_rank",
        return_value=(2, 0),
    ):
        with pytest.raises(ValueError, match="7.*2"):
            fixed.setup()

    # Divisible sizes pass at W = 2 and an odd size is fine at W = 1.
    with patch.object(
        datamodule_module, "_distributed_world_size_and_rank", return_value=(2, 0)
    ):
        _streaming_dm("mqar", MQAR_CFG).setup()
    _streaming_dm("mqar", MQAR_CFG).setup()


# =============================================================================
# MD04 · the stored identity is what DataLoader workers use (spawn)
# =============================================================================


def test_spawn_workers_use_the_captured_rank_identity():
    """A rank-1 dataset iterated through real DataLoader workers under the
    ``spawn`` start method yields rank 1's rows: the identity was captured
    at setup (stored on the dataset), and the iterator never queries
    :mod:`torch.distributed`, which a worker under ``spawn`` has no process
    group for (it would look like rank 0)."""
    ctx = multiprocessing.get_context("spawn")
    dataset = _SyntheticTrainDataset(
        task="mqar", config=MQAR_CFG, batch_size=BATCH_SIZE, train_seed=0
    )
    dataset.set_identity(2, 1)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=2, multiprocessing_context=ctx
    )
    batches = list(itertools.islice(loader, 3))
    for index, batch in enumerate(batches):
        assert int(batch["batch_index"]) == index
        expected = generate_mqar(
            MQAR_CFG, num_examples=BATCH_SIZE, seed=_batch_seed(0, index)
        )
        expected["batch_index"] = torch.tensor(index, dtype=torch.int64)
        assert _content(batch) == _content(
            _slice_rows(expected, BATCH_SIZE // 2, BATCH_SIZE // 2)
        )


# =============================================================================
# MD05 · the cursor is the shared global batch index (both datamodules)
# =============================================================================


def test_streaming_cursor_is_global_and_resume_unaligned():
    """The checkpointed cursor is the global batch index (identical on
    both ranks), and resuming at an unaligned cursor continues the
    uninterrupted stream on each rank (no replay, no skip)."""
    dm = _streaming_dm("mqar", MQAR_CFG)
    dm.trainer = SimpleNamespace(training=True)
    with patch.object(
        datamodule_module, "_distributed_world_size_and_rank", return_value=(2, 1)
    ):
        dm.setup()
        rank1 = list(itertools.islice(dm.train_dataloader(), 3))  # globals 0..2
        for batch in rank1:
            dm.on_before_batch_transfer(batch, 0)
        state = dm.state_dict()
    assert state == {"next_batch_idx": 3}

    dm0 = _streaming_dm("mqar", MQAR_CFG)
    dm0.trainer = SimpleNamespace(training=True)
    with patch.object(
        datamodule_module, "_distributed_world_size_and_rank", return_value=(2, 0)
    ):
        dm0.setup()
        for batch in itertools.islice(dm0.train_dataloader(), 3):
            dm0.on_before_batch_transfer(batch, 0)
        assert dm0.state_dict() == {"next_batch_idx": 3}

    # Resume at an unaligned cursor (3) on the other rank's stream: the
    # batch sequence continues 3, 4, ... with that rank's rows.
    fresh = _streaming_dm("mqar", MQAR_CFG)
    fresh.load_state_dict(state)
    with patch.object(
        datamodule_module, "_distributed_world_size_and_rank", return_value=(2, 0)
    ):
        fresh.setup()
        resumed = list(itertools.islice(fresh.train_dataloader(), 2))
    assert [int(b["batch_index"]) for b in resumed] == [3, 4]
    per = BATCH_SIZE // 2
    for offset, batch in enumerate(resumed):
        expected = generate_mqar(
            MQAR_CFG, num_examples=BATCH_SIZE, seed=_batch_seed(0, 3 + offset)
        )
        expected["batch_index"] = torch.tensor(3 + offset, dtype=torch.int64)
        assert _content(batch) == _content(_slice_rows(expected, 0, per))


def test_fixed_set_cursor_is_global_and_resume_unaligned():
    """The fixed-set cursor is the shared global batch index; resuming at
    an unaligned cursor continues the per-epoch permuted stream on every
    rank."""
    dm = _fixed_dm()
    dm.trainer = SimpleNamespace(training=True)
    with patch(
        "research_lib.data.synthetic.fixed_set.distributed_world_size_and_rank",
        return_value=(2, 1),
    ):
        dm.setup()
        for batch in _fixed_batches(dm, 3):  # globals 0, 1, 2 (epoch 0)
            dm.on_before_batch_transfer(batch, 0)
        state = dm.state_dict()
    assert state == {"next_batch_idx": 3}

    fresh = _fixed_dm()
    fresh.load_state_dict(state)
    with patch(
        "research_lib.data.synthetic.fixed_set.distributed_world_size_and_rank",
        return_value=(2, 1),
    ):
        fresh.setup()
        resumed = _fixed_batches(fresh, 2)
    assert [int(b["batch_index"]) for b in resumed] == [3, 4]
    assert [int(b["epoch"]) for b in resumed] == [0, 0]
    # The uninterrupted stream (never resumed) yields the same globals 3, 4.
    ref = _fixed_dm()
    with patch(
        "research_lib.data.synthetic.fixed_set.distributed_world_size_and_rank",
        return_value=(2, 1),
    ):
        ref.setup()
        ref_batches = _fixed_batches(ref, 5)
    for resumed_batch, ref_batch in zip(resumed, ref_batches[3:5], strict=True):
        assert _content(resumed_batch) == _content(ref_batch)


# =============================================================================
# Direct-dataset fallback
# =============================================================================


def test_dataset_without_captured_identity_is_single_device():
    """A dataset iterated directly (outside a datamodule) falls back to
    world size 1 / rank 0 and logs that it did, once: its batches are the
    whole global batches (no row slicing)."""
    dataset = _SyntheticTrainDataset(
        task="mqar", config=MQAR_CFG, batch_size=BATCH_SIZE, train_seed=0
    )
    batches = list(itertools.islice(iter(dataset), 2))
    for index, batch in enumerate(batches):
        expected = generate_mqar(
            MQAR_CFG, num_examples=BATCH_SIZE, seed=_batch_seed(0, index)
        )
        expected["batch_index"] = torch.tensor(index, dtype=torch.int64)
        assert _content(batch) == _content(expected)

    fixed = _FixedEpochTrainDataset(rows=[{}], batch_size=BATCH_SIZE, train_seed=0)
    assert fixed.world_size == 1 and fixed.rank == 0
