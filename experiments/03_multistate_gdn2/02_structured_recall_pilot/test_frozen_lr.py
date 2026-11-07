"""
Tests for the frozen-protocol LR lookup: a null-safe walk treats a
missing or empty table node as absent, and attention has its own row.

Target: experiments/03_multistate_gdn2/02_structured_recall_pilot/01_train.py
Run with: pytest experiments/03_multistate_gdn2/02_structured_recall_pilot/test_frozen_lr.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_PILOT = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("pilot_01_train", _PILOT / "01_train.py")
_01 = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_01)
_frozen_lr = _01._frozen_lr


def _cfg(task="mqar", layout="ws", gate="full", mixer=None):
    cfg = OmegaConf.create(
        {
            "task": {"name": task},
            "model": {"layout": layout, "gate_mode": gate},
        }
    )
    if mixer is not None:
        cfg.model.mixer = mixer
    return cfg


def _frozen(entries):
    return OmegaConf.create({"lr_table": entries or {}})


class TestFrozenLR:
    def test_empty_table_raises(self):
        with pytest.raises(RuntimeError, match="fill conf/frozen_protocol"):
            _frozen_lr(_cfg(), _frozen({}))

    def test_null_node_raises_instead_of_attributexrr(self):
        # A hand-edited node present but null: the walk treats it as absent.
        frozen = OmegaConf.create(
            {"lr_table": {"mqar": {"gated_delta": {"ws": {"full": None}}}}}
        )
        with pytest.raises(RuntimeError, match="fill conf/frozen_protocol"):
            _frozen_lr(_cfg(), frozen)

    def test_missing_gate_raises(self):
        # The frozen table holds only ``kda``; a ``full`` run has no entry.
        frozen = _frozen({"mqar": {"gated_delta": {"ws": {"kda": 1e-3}}}})
        with pytest.raises(RuntimeError, match="fill conf/frozen_protocol"):
            _frozen_lr(_cfg(gate="full"), frozen)
        # The present gate reads fine.
        assert _frozen_lr(_cfg(gate="kda"), frozen) == 1e-3

    def test_gated_delta_lookup(self):
        frozen = _frozen({"mqar": {"gated_delta": {"ws": {"full": 3e-3, "kda": 1e-3}}}})
        assert _frozen_lr(_cfg(mixer="gated_delta"), frozen) == 3e-3

    def test_attention_uses_its_own_row(self):
        frozen = _frozen({"mqar": {"attention": 6e-3}})
        # The attention run has no layout/gate row.
        assert _frozen_lr(_cfg(mixer="attention"), frozen) == 6e-3
        with pytest.raises(RuntimeError):
            _frozen_lr(_cfg(mixer="attention"), _frozen({"mqar": {}}))
