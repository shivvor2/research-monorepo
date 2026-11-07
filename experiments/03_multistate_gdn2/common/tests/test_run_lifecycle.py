"""
Tests for the shared one-run lifecycle: the monitor key drives both early
stopping and the run summary's best-value track, and the curves reset
builds callbacks before deleting stale curves.

Target: experiments/03_multistate_gdn2/common/run_lifecycle.py
Run with: pytest experiments/03_multistate_gdn2/common/tests/test_run_lifecycle.py -v
"""

import json
from pathlib import Path

import pytest
from common.run_lifecycle import create_callbacks, create_callbacks_reset_curves
from omegaconf import OmegaConf

from research_lib.training.callbacks import RunSummaryCallback


def cfg(runs_dir, curves_dir):
    return OmegaConf.create(
        {
            "seed": 0,
            "output": {"runs_dir": str(runs_dir)},
            "training": {"early_stop_threshold": 0.99, "collect_mixer_stats": False},
            "logging": {"csv": {"enabled": True, "save_dir": str(curves_dir)}},
            "optimizer": {"decay_scheme": "grouped"},
            "experiment": {"name": "t", "tier": "tier_x"},
        }
    )


class TestMonitorKey:
    def test_custom_key_drives_early_stopping_and_the_run_summary(self, tmp_path):
        runs = tmp_path / "runs"
        curves = tmp_path / "curves"
        c = cfg(runs, curves)
        callbacks = create_callbacks(
            c, "run-1", {"a": False, "b": False}, lr=0.001, monitor_key="val/f1"
        )
        from lightning.pytorch.callbacks import EarlyStopping

        early = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
        summary = next(cb for cb in callbacks if isinstance(cb, RunSummaryCallback))
        assert early.monitor == "val/f1"
        assert summary.monitor == "val/f1"

    def test_default_key_is_val_acc(self, tmp_path):
        runs = tmp_path / "runs"
        curves = tmp_path / "curves"
        c = cfg(runs, curves)
        callbacks = create_callbacks(c, "run-1", {"a": False, "b": False}, lr=0.001)
        from lightning.pytorch.callbacks import EarlyStopping

        early = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
        assert early.monitor == "val/acc"


class TestCurvesReset:
    def test_stale_curves_are_removed_after_the_identity_check(self, tmp_path):
        runs = tmp_path / "runs"
        curves = tmp_path / "curves"
        runs.mkdir(parents=True)
        curves_dir = curves / "run-1"
        curves_dir.mkdir(parents=True)
        (curves_dir / "metrics.csv").write_text("step,val/acc\n0,0.5\n")

        c = cfg(runs, curves)
        from common.run_id import HASH_EXCLUDE

        from research_lib.training.callbacks import config_hash

        identity = (
            config_hash(OmegaConf.to_container(c, resolve=True), HASH_EXCLUDE),
            c.seed,
        )
        (runs / "run-1.json").write_text(
            json.dumps({"config_hash": identity[0], "seed": identity[1]})
        )

        callbacks = create_callbacks_reset_curves(c, "run-1", {}, lr=0.001)
        assert not (curves / "run-1").exists()
        assert any(isinstance(cb, RunSummaryCallback) for cb in callbacks)

    def test_refused_relaunch_keeps_the_curves(self, tmp_path):
        runs = tmp_path / "runs"
        curves = tmp_path / "curves"
        runs.mkdir(parents=True)
        curves_dir = curves / "run-1"
        curves_dir.mkdir(parents=True)
        (curves_dir / "metrics.csv").write_text("step,val/acc\n0,0.5\n")
        (runs / "run-1.json").write_text(
            json.dumps({"config_hash": "other", "seed": 0})
        )
        c = cfg(runs, curves)
        with pytest.raises(FileExistsError):
            create_callbacks_reset_curves(c, "run-1", {}, lr=0.001)
        assert (curves / "run-1" / "metrics.csv").exists()
