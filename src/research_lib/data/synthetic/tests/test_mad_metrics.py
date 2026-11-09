"""
Tests for the MAD-style accuracy metrics.

Covers: the macro token accuracy equals ``torchmetrics``
``MulticlassAccuracy`` (default, macro averaging, ``ignore_index=-100``)
on hand-made predictions, the plain token accuracy counts scored tokens,
and the two genuinely differ in a constructed imbalanced case.

Run with: pytest src/research_lib/data/synthetic/tests/test_mad_metrics.py -v
"""

import torch
from torchmetrics.classification import MulticlassAccuracy

from ..mad_metrics import MacroTokenAccuracy, TokenAccuracy


def _batch(labels_row):
    """A single-row batch (B=1) with the given scatter of scored tokens."""
    return {"labels": torch.tensor([labels_row], dtype=torch.long)}


class TestMacroTokenAccuracy:
    def test_matches_torchmetrics_macro(self):
        # 4 scored positions: class 0 all correct, class 1 wrong.
        macro = MacroTokenAccuracy(num_classes=5)
        labels = _batch([-100, 0, 0, 0, 1])  # T=5 -> 4 shifted positions
        preds = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        macro.update(preds, labels)
        expected = MulticlassAccuracy(num_classes=5, ignore_index=-100)
        targets = labels["labels"][:, 1:]
        mask = targets != -100
        expected.update(preds[mask], targets[mask])
        got = macro.compute()["acc_mad"]
        assert got == pytest_float(expected.compute())
        # class-0 recall 1.0, class-1 recall 0.0 -> macro 0.5.
        assert got == 0.5

    def test_differs_from_token_accuracy(self):
        macro = MacroTokenAccuracy(num_classes=5)
        token = TokenAccuracy()
        labels = _batch([-100, 0, 0, 0, 1])
        preds = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        for m in (macro, token):
            m.update(preds, labels)
        assert macro.compute()["acc_mad"] == 0.5
        assert token.compute()["acc_token"] == 0.75
        assert macro.compute()["acc_mad"] != token.compute()["acc_token"]


class TestTokenAccuracy:
    def test_fraction_of_scored(self):
        t = TokenAccuracy()
        labels = _batch([-100, 0, -100, 1, 2])  # 3 scored shifted positions
        # predictions: one wrong
        preds = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        t.update(preds, labels)
        # scored targets: 0, 1, 2 ; predictions 0,0,0 -> only first correct
        assert t.compute()["acc_token"] == 1 / 3

    def test_reset(self):
        t = TokenAccuracy()
        labels = _batch([-100, 0, 0])
        preds = torch.tensor([[0, 0]], dtype=torch.long)
        t.update(preds, labels)
        assert t.compute()["acc_token"] == 1.0
        t.reset()
        assert t.compute()["acc_token"] == 0.0


def pytest_float(x):
    return float(x)


class TestSyncFreeUpdate:
    def test_update_methods_have_no_host_sync(self):
        import inspect

        import research_lib.data.synthetic.mad_metrics as mm

        src_tok = inspect.getsource(mm.TokenAccuracy.update)
        src_macro = inspect.getsource(mm.MacroTokenAccuracy.update)
        for src in (src_tok, src_macro):
            assert ".item()" not in src
            assert "int(" not in src
            assert ".tolist()" not in src

    def test_values_match_on_cpu(self):
        macro = MacroTokenAccuracy(num_classes=5)
        token = TokenAccuracy()
        preds = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        batch = {"labels": torch.tensor([[-100, 0, 1, 2, 3]], dtype=torch.long)}
        macro.update(preds, batch)
        token.update(preds, batch)
        assert macro.compute()["acc_mad"] == 1.0
        assert token.compute()["acc_token"] == 1.0


class TestDistributedCompute:
    def test_compute_twice_does_not_double_count(self):
        # No process group: compute reads copies, so two calls agree.
        t = TokenAccuracy()
        preds = torch.tensor([[0, 0, 0]], dtype=torch.long)
        batch = {"labels": torch.tensor([[-100, 0, 1, 2]], dtype=torch.long)}
        t.update(preds, batch)
        first = t.compute()["acc_token"]
        second = t.compute()["acc_token"]
        assert first == pytest_float(second)
        assert first == pytest_float(1 / 3)

    def _mock_gloo_glue(self, monkeypatch):
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(
            torch.distributed,
            "get_backend",
            lambda: torch.distributed.Backend.GLOO,
        )

    def test_no_update_returns_zero_and_still_calls_the_collective(self, monkeypatch):
        # A rank that saw no batch still joins the all-reduce (its counters
        # exist from reset) instead of skipping it and stalling the others.
        calls = []
        self._mock_gloo_glue(monkeypatch)
        monkeypatch.setattr(
            torch.distributed,
            "all_reduce",
            lambda tensor, op=None: calls.append(1),
        )
        t = TokenAccuracy()
        out = t.compute()
        assert out["acc_token"] == 0.0
        assert len(calls) == 2  # correct and total

    def test_all_reduce_reads_copies(self, monkeypatch):
        # The reduced tensors are copies, so a second compute (a logged value
        # and a record collector) does not double the accumulated counts.
        self._mock_gloo_glue(monkeypatch)
        monkeypatch.setattr(
            torch.distributed, "all_reduce", lambda tensor, op=None: tensor
        )
        t = TokenAccuracy()
        preds = torch.tensor([[0, 0]], dtype=torch.long)
        batch = {"labels": torch.tensor([[-100, 0, 1]], dtype=torch.long)}
        t.update(preds, batch)
        assert t.compute()["acc_token"] == pytest_float(0.5)
        assert t.compute()["acc_token"] == pytest_float(0.5)

    def test_nccl_moves_the_counters_to_cuda_before_reducing(self, monkeypatch):
        # Under an NCCL backend the all-reduce rejects CPU tensors, so a rank
        # that saw no batch (its counters still on CPU) must have its copies
        # moved to the CUDA device first. The mocked reduction adds non-zero
        # "remote" counts and the empty rank returns the global accuracy.
        self._mock_gloo_glue(monkeypatch)
        monkeypatch.setattr(
            torch.distributed,
            "get_backend",
            lambda: torch.distributed.Backend.NCCL,
        )

        class _FakeCuda:
            def current_device(self):
                return 0

        monkeypatch.setattr(torch, "cuda", _FakeCuda())
        requested = []

        def fake_to(self, device, *args, **kwargs):
            requested.append(device)
            return self  # no real move (no CUDA); the mock reduction sums after

        monkeypatch.setattr(torch.Tensor, "to", fake_to)
        state = {"calls": 0}

        def fake_reduce(tensor, op=None):
            state["calls"] += 1
            tensor.add_(4 if state["calls"] == 1 else 5)  # remote correct / total

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_reduce)
        t = TokenAccuracy()
        out = t.compute()
        assert out["acc_token"] == pytest_float(4 / 5)
        assert state["calls"] == 2
        assert all(str(d).startswith("cuda") for d in requested)
