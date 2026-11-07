"""
Tests for the fixed-training-set recall datamodule.

Covers: the fixed training set is identical for the same seed, each epoch
is a fresh permutation, the training stream contents are independent of
the worker count and resume from a saved batch index, the validation and
test sets are fixed and distinct from the training set, each epoch's last
partial batch is kept, and the leakage fraction is computed.

Run with: pytest src/research_lib/data/synthetic/tests/test_fixed_set.py -v
"""

import torch

from ..fixed_set import FixedSetRecallDataModule
from ..mad_fuzzy import MadFuzzyConfig, fuzzy_recall_dataset, mad_batch_row

CFG = MadFuzzyConfig(vocab_size=16, seq_len=32)


def _to_row(inputs, targets, is_training):
    return mad_batch_row(CFG, inputs, targets, pad_to=48, is_training=is_training)


def _make(train_seed=0, val_seed=1003, test_seed=2003, num_workers=0):
    return FixedSetRecallDataModule(
        dataset_fn=fuzzy_recall_dataset,
        config=CFG,
        to_row=_to_row,
        num_train=10,
        num_val=4,
        num_test=4,
        batch_size=4,
        train_seed=train_seed,
        val_seed=val_seed,
        test_seed=test_seed,
        num_workers=num_workers,
    )


def _first_batches(train_loader, n):
    out = []
    for batch in train_loader:
        out.append(batch)
        if len(out) >= n:
            break
    return out


class TestFixedTrainingSet:
    def test_same_seed_is_identical(self):
        a = _make(train_seed=5)
        b = _make(train_seed=5)
        a.setup()
        b.setup()
        ba = _first_batches(a.train_dataloader(), 4)
        bb = _first_batches(b.train_dataloader(), 4)
        for x, y in zip(ba, bb):
            assert torch.equal(x["input_ids"], y["input_ids"])
            assert torch.equal(x["labels"], y["labels"])
            assert int(x["batch_index"]) == int(y["batch_index"])

    def test_different_seeds_differ(self):
        a = _make(train_seed=5)
        b = _make(train_seed=6)
        a.setup()
        b.setup()
        ba = _first_batches(a.train_dataloader(), 1)[0]
        bb = _first_batches(b.train_dataloader(), 1)[0]
        assert not torch.equal(ba["input_ids"], bb["input_ids"])

    def test_epochs_are_permutations(self):
        from ..fixed_set import _permutation

        n = 10
        p0 = _permutation(3, 0, n)
        p1 = _permutation(3, 1, n)
        assert sorted(p0.tolist()) == list(range(n))
        assert sorted(p1.tolist()) == list(range(n))
        assert not torch.equal(torch.tensor(p0.tolist()), torch.tensor(p1.tolist()))
        # The epoch identity travels with the batch: the first epoch's
        # batches carry epoch 0.
        m = _make(train_seed=3)
        m.setup()
        bpe = m._ensure_train_dataset()._batches_per_epoch
        batches = _first_batches(m.train_dataloader(), bpe)
        assert {int(b["epoch"]) for b in batches} == {0}

    def test_last_partial_batch_kept(self):
        # 10 rows at batch size 4 -> batches of 4, 4, 2 (the tail kept).
        m = _make(train_seed=0)
        m.setup()
        sizes = []
        for batch in m.train_dataloader():
            sizes.append(batch["input_ids"].shape[0])
            if len(sizes) >= 3:
                break
        bpe = m._ensure_train_dataset()._batches_per_epoch
        assert bpe == 3
        assert sizes == [4, 4, 2]


class TestWorkerIndependence:
    def test_stream_identical_across_workers(self):
        # Each instance is iterated exactly once; the global batch-index
        # sequence and the batch contents are the same for 0 and 2 workers.
        m0 = _make(num_workers=0)
        m1 = _make(num_workers=2)
        m0.setup()
        m1.setup()
        c0 = _first_batches(m0.train_dataloader(), 6)
        c1 = _first_batches(m1.train_dataloader(), 6)
        seq0 = [int(b["batch_index"]) for b in c0]
        seq1 = [int(b["batch_index"]) for b in c1]
        assert seq0 == list(range(6))
        assert seq1 == seq0
        for x, y in zip(c0, c1):
            assert torch.equal(x["input_ids"], y["input_ids"])

    def test_resume_from_batch_index(self):
        m0 = _make(num_workers=0)
        m0.setup()
        ref = _first_batches(m0.train_dataloader(), 5)
        resumed = _make(num_workers=0)
        resumed.setup()
        resumed.load_state_dict({"next_batch_idx": 3})
        first = next(iter(resumed.train_dataloader()))
        assert int(first["batch_index"]) == 3
        assert torch.equal(first["input_ids"], ref[3]["input_ids"])


class TestFixedEvalSets:
    def test_evaluation_sets_fixed_and_distinct(self):
        a = _make(val_seed=10, test_seed=20)
        b = _make(val_seed=10, test_seed=20)
        a.setup()
        b.setup()
        va = next(iter(a.val_dataloader()))
        vb = next(iter(b.val_dataloader()))
        assert torch.equal(va["input_ids"], vb["input_ids"])
        ta = next(iter(a.test_dataloader()))
        # Validation and test use distinct fixed seeds.
        assert not torch.equal(va["input_ids"], ta["input_ids"])

    def test_leakage_fraction_reported(self):
        m = _make()
        m.setup()
        frac = m._leakage_fraction()
        assert isinstance(frac, float) and 0.0 <= frac <= 1.0
        # Override a couple of validation rows to training rows: the
        # fraction rises accordingly (2 of the (4+4) eval rows).)
        train_rows = m._ensure_train_rows()
        m._val_rows = [dict(r) for r in train_rows[:2]]
        m._test_rows = [m._test_rows_()[0]]
        frac2 = m._leakage_fraction()
        assert frac2 == 2 / 3

    def test_batch_keys_present(self):
        m = _make()
        m.setup()
        train_b = next(iter(m.train_dataloader()))
        for key in ("input_ids", "labels", "slice_id", "batch_index", "epoch"):
            assert key in train_b
        val_b = next(iter(m.val_dataloader()))
        assert "input_ids" in val_b and "labels" in val_b and "slice_id" in val_b
        # Eval batditions have no stream keys.
        assert "batch_index" not in val_b
