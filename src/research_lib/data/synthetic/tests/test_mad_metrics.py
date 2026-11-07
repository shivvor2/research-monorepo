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
