"""
Tests for the run-summary callback.

Target: src/research_lib/training/callbacks/run_summary.py
Run with: pytest src/research_lib/training/callbacks/tests/test_run_summary.py -v
"""

import json
from pathlib import Path
from typing import Any, cast

import lightning as L
import pytest
import torch

from research_lib.architectures.config import ZoologyLMConfig
from research_lib.architectures.zoology_lm import ZoologyLM
from research_lib.training.callbacks.run_summary import (
    BUILTIN_SECTIONS,
    DEFAULT_COLLECTORS,
    Collector,
    RunSummaryCallback,
    Stage,
    _check_stage_hooks,
    _missing_stage_hooks,
    config_hash,
    metrics_by_prefix,
    param_counts,
)
from research_lib.training.configs import ScheduleConfig
from research_lib.training.modules.synthetic_recall import SyntheticRecallModule
from research_lib.training.scheduling import WarmupStableDecaySchedule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tiny_model() -> ZoologyLM:
    return ZoologyLM(
        ZoologyLMConfig(
            vocab_size=64, d_model=16, num_heads=1, t_max=32, layout="single"
        )
    )


def tiny_config_dict(seed: int = 0) -> dict:
    return {"seed": seed, "model": {"d_model": 16, "layout": "single"}, "task": "mqar"}


def tiny_module() -> SyntheticRecallModule:
    schedule = ScheduleConfig(
        global_schedules=[
            WarmupStableDecaySchedule(
                param_name="lr",
                warmup_frac=0.0,
                cooldown_frac=1.0,
                min_value=0.0,
                max_value=1e-3,
                decay_type="cosine",
            )
        ]
    )
    return SyntheticRecallModule(
        model=tiny_model(),
        slice_values=(4, 8),
        collect_mixer_stats=False,
        matrix_optimizer_config=type(
            "C",
            (),
            {"optimizer_class": torch.optim.AdamW, "optimizer_kwargs": {"lr": 0.0}},
        )(),
        vector_optimizer_config=type(
            "C",
            (),
            {"optimizer_class": torch.optim.AdamW, "optimizer_kwargs": {"lr": 0.0}},
        )(),
        matrix_schedule_config=schedule,
        vector_schedule_config=schedule,
        matrix_target_modules=["lm_head"],
    )


def make_loader():
    from torch.utils.data import Dataset

    class _DictDataset(Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, index):
            return {
                "input_ids": torch.zeros(8, 16, dtype=torch.long),
                "labels": torch.full((8, 16), -100, dtype=torch.long),
                "slice_id": torch.zeros(8, dtype=torch.long),
            }

    return torch.utils.data.DataLoader(_DictDataset(), batch_size=None)


def make_trainer(callbacks, max_steps=2):
    return L.Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        val_check_interval=max_steps,
    )


@pytest.fixture
def fitted(tmp_path):
    """A 2-step fit + test with the callback attached (strict for tests)."""
    runs_dir = tmp_path / "runs"
    cb = RunSummaryCallback(
        runs_dir,
        "test-run",
        tiny_config_dict(),
        strict=True,
    )
    trainer = make_trainer([cb])
    module = tiny_module()
    loader = make_loader()
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    trainer.test(module, dataloaders=loader)
    return cb, runs_dir, trainer, module


def read(runs_dir, run_id="test-run"):
    return json.loads((runs_dir / f"{run_id}.json").read_text())


# ---------------------------------------------------------------------------
# Stage hooks
# ---------------------------------------------------------------------------


class TestStageHooks:
    def test_every_stage_has_its_hook(self):
        assert _missing_stage_hooks(RunSummaryCallback) == []

    def test_missing_hook_raises_type_error(self):
        class Broken(L.Callback):
            pass

        with pytest.raises(TypeError, match="on_fit_start"):
            _check_stage_hooks(Broken)


# ---------------------------------------------------------------------------
# Filename
# ---------------------------------------------------------------------------


class TestFilename:
    def make(self, tmp_path, filename):
        return RunSummaryCallback(tmp_path, "my-run", {"seed": 7}, filename=filename)

    def test_default_and_fields(self, tmp_path):
        assert self.make(tmp_path, "{run_id}.json").filename == "my-run.json"
        cb = self.make(tmp_path, "{seed}_{hash8}_{date}_{datetime}.json")
        assert cb.filename.startswith("7_")
        parts = cb.filename.split("_")
        assert parts[1] == config_hash({"seed": 7})[:8]
        assert len(parts[2]) == 10  # YYYY-MM-DD
        assert len(parts[3].split(".")[0]) == 15  # YYYYMMDD-HHMMSS
        assert cb.path == tmp_path / cb.filename

    def test_unknown_field_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="allowed fields"):
            self.make(tmp_path, "{nope}.json")

    def test_must_end_in_json(self, tmp_path):
        with pytest.raises(ValueError, match="must end in .json"):
            self.make(tmp_path, "{run_id}.txt")

    def test_absolute_path_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="relative"):
            self.make(tmp_path, "/tmp/{run_id}.json")

    def test_parent_component_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"\.\."):
            self.make(tmp_path, "../{run_id}.json")

    def test_subdirectory_resolved_and_written(self, tmp_path):
        cb = RunSummaryCallback(
            tmp_path, "sub-run", tiny_config_dict(), filename="nested/{run_id}.json"
        )
        assert cb.filename == "nested/sub-run.json"
        trainer = make_trainer([cb])
        module = tiny_module()
        trainer.fit(
            module, train_dataloaders=make_loader(), val_dataloaders=make_loader()
        )
        assert (tmp_path / "nested" / "sub-run.json").exists()


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_duplicate_sections_rejected(self, tmp_path):
        collector = Collector("x", lambda t, m: {})
        with pytest.raises(ValueError, match="duplicate"):
            RunSummaryCallback(
                tmp_path, "r", {}, collectors=(collector, collector), strict=True
            )

    def test_builtin_collision_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="built-in"):
            RunSummaryCallback(
                tmp_path, "r", {}, collectors=(Collector("config", lambda t, m: {}),)
            )
        assert "config" in BUILTIN_SECTIONS

    def test_extra_collision_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="extra"):
            RunSummaryCallback(
                tmp_path,
                "r",
                {},
                collectors=(Collector("mine", lambda t, m: {}),),
                extra={"mine": 1},
            )

    def test_invalid_mode_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="mode"):
            RunSummaryCallback(tmp_path, "r", {}, mode=cast(Any, "up"))

    def test_missing_package_recorded_as_none(self, tmp_path):
        cb = RunSummaryCallback(
            tmp_path, "r", {}, packages=("torch", "not-a-real-package-xyz")
        )
        versions = cb._versions()
        # Distribution names: the recorded version is importlib.metadata's
        # (which may differ from torch.__version__'s local-version suffix).
        assert isinstance(versions["torch"], str) and versions["torch"]
        assert versions["not-a-real-package-xyz"] is None
        assert "python" in versions
        assert "cuda" in versions


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


class TestCollectors:
    def test_collectors_run_at_their_stage(self, tmp_path):
        calls = []

        def starter(trainer, pl_module):
            calls.append(("fit_start", pl_module is not None))
            return {"stage": "fit_start"}

        def ender(trainer, pl_module):
            value = float(trainer.callback_metrics["val/loss"])
            calls.append(("fit_end", value))
            return {"val_loss": value}

        def tester(trainer, pl_module):
            value = float(trainer.callback_metrics["test/loss"])
            calls.append(("test_end", value))
            return {"test_loss": value}

        cb = RunSummaryCallback(
            tmp_path,
            "stages",
            tiny_config_dict(),
            strict=True,
            collectors=(
                Collector("start", starter, Stage.FIT_START),
                Collector("end", ender, Stage.FIT_END),
                Collector("testsec", tester, Stage.TEST_END),
            ),
        )
        trainer = make_trainer([cb])
        module = tiny_module()
        loader = make_loader()
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        trainer.test(module, dataloaders=loader)
        assert [c[0] for c in calls] == ["fit_start", "fit_end", "test_end"]
        summary = read(tmp_path, "stages")
        assert summary["status"] == "test_done"

    def test_defaults_provide_params_val_test(self, fitted):
        cb, runs_dir, trainer, module = fitted
        summary = read(runs_dir)
        assert set(summary["params"]) == {"total", "trainable", "groups"}
        assert summary["params"]["total"] == sum(
            p.numel() for p in module.model.parameters()
        )
        assert "val/loss" not in summary["val_final"]
        assert "loss" in summary["val_final"]
        assert "loss" in summary["test"]

    def test_collector_error_recorded_when_not_strict(self, tmp_path):
        def boom(trainer, pl_module):
            raise RuntimeError("collector boom")

        cb = RunSummaryCallback(
            tmp_path,
            "boom",
            tiny_config_dict(),
            collectors=(Collector("bad", boom, Stage.FIT_END),),
        )
        trainer = make_trainer([cb])
        module = tiny_module()
        loader = make_loader()
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        summary = read(tmp_path, "boom")
        assert "collector boom" in summary["bad"]["error"]
        assert summary["status"] == "fit_done"  # the run completed

    def test_collector_error_raised_when_strict(self, tmp_path):
        def boom(trainer, pl_module):
            raise RuntimeError("strict boom")

        cb = RunSummaryCallback(
            tmp_path,
            "strict",
            tiny_config_dict(),
            collectors=(Collector("bad", boom, Stage.FIT_END),),
            strict=True,
        )
        trainer = make_trainer([cb])
        module = tiny_module()
        loader = make_loader()
        with pytest.raises(RuntimeError, match="strict boom"):
            trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    def test_param_counts_groups(self, tmp_path):
        cb = RunSummaryCallback(
            tmp_path,
            "params",
            tiny_config_dict(),
            strict=True,
            collectors=(
                Collector(
                    "params",
                    param_counts({"head": ["lm_head"], "embedding": ["embedding"]}),
                    Stage.FIT_START,
                ),
            ),
        )
        trainer = make_trainer([cb])
        module = tiny_module()
        loader = make_loader()
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=make_loader())
        summary = read(tmp_path, "params")
        assert (
            summary["params"]["groups"]["head"] == module.model.lm_head.weight.numel()
        )
        assert (
            summary["params"]["groups"]["embedding"]
            == module.model.embedding.weight.numel()
        )

    def test_non_numeric_metrics_skipped(self, tmp_path):
        cb = RunSummaryCallback(
            tmp_path,
            "skip",
            tiny_config_dict(),
            strict=True,
            collectors=(
                Collector("val_final", metrics_by_prefix("val"), Stage.FIT_END),
            ),
        )
        trainer = make_trainer([cb])
        module = tiny_module()
        loader = make_loader()
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        assert all(
            isinstance(v, float) for v in read(tmp_path, "skip")["val_final"].values()
        )


# ---------------------------------------------------------------------------
# Status lifecycle and monitor
# ---------------------------------------------------------------------------


class TestStatus:
    def test_lifecycle_running_to_test_done(self, tmp_path):
        seen = []

        class _Spy(L.Callback):
            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                seen.append(read(tmp_path, "life").get("status"))

        cb = RunSummaryCallback(tmp_path, "life", tiny_config_dict(), strict=True)
        trainer = make_trainer([cb, _Spy()])
        module = tiny_module()
        loader = make_loader()
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        assert read(tmp_path, "life")["status"] == "fit_done"
        trainer.test(module, dataloaders=loader)
        assert read(tmp_path, "life")["status"] == "test_done"
        assert seen and all(s == "running" for s in seen)

    def test_failed_status_from_training_step_exception(self, tmp_path):
        class _Failing(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 2)

            def training_step(self, batch, batch_idx):
                raise RuntimeError("boom in training")

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.0)

        cb = RunSummaryCallback(tmp_path, "fail", tiny_config_dict(), strict=True)
        trainer = L.Trainer(
            max_steps=1,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            callbacks=[cb],
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )
        with pytest.raises(RuntimeError, match="boom in training"):
            trainer.fit(_Failing(), train_dataloaders=make_loader())
        summary = read(tmp_path, "fail")
        assert summary["status"] == "failed"
        assert "boom in training" in summary["error"]

    def test_interrupted_status_from_on_exception(self, tmp_path):
        cb = RunSummaryCallback(tmp_path, "int", tiny_config_dict(), strict=True)
        trainer = make_trainer([cb])
        module = tiny_module()
        loader = make_loader()
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        cb.on_exception(trainer, module, KeyboardInterrupt())
        summary = read(tmp_path, "int")
        assert summary["status"] == "interrupted"
        assert "KeyboardInterrupt" in summary["error"]

    def test_no_monitor_no_tracking(self, fitted):
        cb, runs_dir, trainer, module = fitted
        summary = read(runs_dir)
        # The fitted fixture passes no monitor: no tracking.
        assert summary["stopping"]["monitor"] is None
        assert summary["stopping"]["best_value"] is None
        assert summary["stopping"]["best_step"] is None
        assert summary["stopping"]["early_stopped"] is False

    def test_monitor_passes(self, tmp_path):
        cb = RunSummaryCallback(
            tmp_path,
            "mon",
            tiny_config_dict(),
            monitor="val/acc",
            mode="max",
            strict=True,
        )
        trainer = make_trainer([cb])
        module = tiny_module()
        loader = make_loader()
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        summary = read(tmp_path, "mon")
        assert summary["stopping"]["monitor"] == "val/acc"
        assert summary["stopping"]["mode"] == "max"
        assert summary["stopping"]["best_value"] is not None
        assert summary["stopping"]["best_step"] is not None

    def test_test_only_launch_updates_existing_file(self, tmp_path):
        cb = RunSummaryCallback(tmp_path, "resume", tiny_config_dict(), strict=True)
        trainer = make_trainer([cb])
        module = tiny_module()
        loader = make_loader()
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        fit_summary = read(tmp_path, "resume")

        # A fresh callback + trainer, test only: the existing file is
        # loaded and updated, not replaced by a test-only file.
        cb2 = RunSummaryCallback(tmp_path, "resume", tiny_config_dict(), strict=True)
        trainer2 = L.Trainer(
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            callbacks=[cb2],
            limit_val_batches=0,
        )
        trainer2.test(module, dataloaders=loader)
        summary = read(tmp_path, "resume")
        assert summary["status"] == "test_done"
        assert "loss" in summary["test"]
        assert summary["params"] == fit_summary["params"]  # fit sections retained
        assert summary["timing"] == fit_summary["timing"]
        assert summary["val_final"] == fit_summary["val_final"]


# ---------------------------------------------------------------------------
# Config hash (unchanged behaviour)
# ---------------------------------------------------------------------------


class TestConfigHash:
    def test_same_config_two_seeds_share_hash(self):
        assert config_hash(tiny_config_dict(seed=0)) == config_hash(
            tiny_config_dict(seed=1)
        )

    def test_different_configs_differ(self):
        other = tiny_config_dict(seed=0)
        other["model"] = {"d_model": 32, "layout": "single"}
        assert config_hash(tiny_config_dict()) != config_hash(other)

    def test_extra_merged_at_every_write(self, tmp_path):
        cb = RunSummaryCallback(
            tmp_path, "extra", tiny_config_dict(), extra={"my_key": 42}, strict=True
        )
        trainer = make_trainer([cb])
        module = tiny_module()
        loader = make_loader()
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        assert read(tmp_path, "extra")["my_key"] == 42
        trainer.test(module, dataloaders=loader)
        assert read(tmp_path, "extra")["my_key"] == 42
