"""
Tests for the run-summary callback.

Target: src/research_lib/training/callbacks/run_summary.py
Run with: pytest src/research_lib/training/callbacks/tests/test_run_summary.py -v
"""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import EarlyStopping

from research_lib.architectures.config import ZoologyLMConfig
from research_lib.architectures.zoology_lm import ZoologyLM, zoology_mixers
from research_lib.layers.attention import MultistateGatedDelta
from research_lib.training.callbacks.run_summary import (
    BUILTIN_SECTIONS,
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
from research_lib.training.modules.mgdn2_synthetic_recall import (
    MGDN2SyntheticRecallModule,
)
from research_lib.training.scheduling import WarmupStableDecaySchedule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tiny_model() -> ZoologyLM:
    """Zoology harness with a BaseConv block and one gated-delta mixer."""
    mixers = zoology_mixers(
        2,
        16,
        make_mixer=lambda: MultistateGatedDelta(
            hidden_size=16, num_heads=1, head_dim=16, layout="single", t_max=32
        ),
    )
    return ZoologyLM(ZoologyLMConfig(vocab_size=64, d_model=16), mixers)


def tiny_config_dict(seed: int = 0) -> dict:
    return {"seed": seed, "model": {"d_model": 16, "layout": "single"}, "task": "mqar"}


def tiny_module() -> MGDN2SyntheticRecallModule:
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
    return MGDN2SyntheticRecallModule(
        model=tiny_model(),
        slice_values=(4, 8),
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


class DecreasingVal(L.LightningModule):
    """Logs ``val/acc = 10 - global_step``: the best value is always in
    the FIRST fit segment, so a resumed fit must restore it to see the
    maximum over both segments."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)

    def training_step(self, batch, batch_idx):
        return self.layer.weight.sum()

    def validation_step(self, batch, batch_idx):
        self.log("val/acc", 10.0 - float(self.global_step))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def make_loader():
    from torch.utils.data import Dataset

    class _DictDataset(Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, index):
            # One answer per sequence (a valid token id for the tiny
            # model) and a valid slice id: RecallMetrics raises on
            # sequences with no answer positions.
            labels = torch.full((8, 16), -100, dtype=torch.long)
            labels[:, -1] = 1
            return {
                "input_ids": torch.zeros(8, 16, dtype=torch.long),
                "labels": labels,
                "slice_id": torch.full((8,), 4, dtype=torch.long),
            }

    return torch.utils.data.DataLoader(_DictDataset(), batch_size=None)


def make_trainer(callbacks, max_steps=2, val_check_interval=None):
    return L.Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        val_check_interval=val_check_interval or max_steps,
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
        assert summary["failed_stage"] == "fit"

    def test_test_only_failure_keeps_fit_sections(self, tmp_path):
        """A failing test-only launch keeps the fit sections, and records
        that the run failed during testing (not during fitting)."""

        class _FailingTest(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 2)

            def test_step(self, batch, batch_idx):
                raise RuntimeError("boom in test")

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.0)

        cb = RunSummaryCallback(tmp_path, "fail-test", tiny_config_dict(), strict=True)
        trainer = make_trainer([cb])
        module = tiny_module()
        loader = make_loader()
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        fit_summary = read(tmp_path, "fail-test")

        cb2 = RunSummaryCallback(tmp_path, "fail-test", tiny_config_dict(), strict=True)
        trainer2 = L.Trainer(
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            callbacks=[cb2],
            limit_val_batches=0,
        )
        with pytest.raises(RuntimeError, match="boom in test"):
            trainer2.test(_FailingTest(), dataloaders=loader)
        summary = read(tmp_path, "fail-test")
        assert summary["status"] == "failed"
        assert summary["failed_stage"] == "test"
        assert "boom in test" in summary["error"]
        # The fit sections are retained, not replaced.
        assert summary["params"] == fit_summary["params"]
        assert summary["timing"] == fit_summary["timing"]
        assert summary["stopping"] == fit_summary["stopping"]

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

    def test_sigterm_is_interrupted_not_failed(self, tmp_path):
        """Lightning raises SIGTERMException at the next batch after
        SIGTERM (kill, Slurm cancel/preemption, cloud stop): the run is
        interrupted, not failed, so a relaunch treats it as incomplete."""
        from lightning.pytorch.utilities.exceptions import SIGTERMException

        cb = RunSummaryCallback(tmp_path, "sig", tiny_config_dict(), strict=True)
        trainer = make_trainer([cb])
        module = tiny_module()
        loader = make_loader()
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        cb.on_exception(trainer, module, SIGTERMException())
        summary = read(tmp_path, "sig")
        assert summary["status"] == "interrupted"
        assert "SIGTERMException" in summary["error"]
        assert summary["failed_stage"] == "fit"

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

    def test_test_stage_keeps_fit_sections(self, tmp_path):
        """Fit + test in one process: ``timing`` / ``stopping`` keep their
        fit-end values (the test stage measures neither)."""
        cb = RunSummaryCallback(
            tmp_path,
            "keep-fit",
            tiny_config_dict(),
            monitor="val/acc",
            strict=True,
        )
        trainer = make_trainer([cb])
        module = tiny_module()
        loader = make_loader()
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        fit_summary = read(tmp_path, "keep-fit")
        trainer.test(module, dataloaders=loader)
        after = read(tmp_path, "keep-fit")
        assert after["status"] == "test_done"
        assert after["timing"] == fit_summary["timing"]
        assert after["stopping"] == fit_summary["stopping"]

    def test_test_only_launch_updates_existing_file(self, tmp_path):
        cb = RunSummaryCallback(
            tmp_path,
            "resume",
            tiny_config_dict(),
            monitor="val/acc",
            strict=True,
        )
        trainer = make_trainer([cb])
        module = tiny_module()
        loader = make_loader()
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        fit_summary = read(tmp_path, "resume")
        assert fit_summary["stopping"]["best_value"] is not None

        # A fresh callback + trainer, test only: the existing file is
        # loaded and updated, not replaced by a test-only file.
        cb2 = RunSummaryCallback(
            tmp_path,
            "resume",
            tiny_config_dict(),
            monitor="val/acc",
            strict=True,
        )
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
        # The fresh process tracked nothing: stopping must be the loaded
        # fit-end section, not rewritten with best_value=None.
        assert summary["stopping"] == fit_summary["stopping"]

    def test_best_value_survives_a_resumed_fit(self, tmp_path):
        """A resumed fit with a fresh callback keeps the best value /
        step seen before the interruption (restored from the checkpoint,
        the maximum over both segments)."""
        module = DecreasingVal()
        ckpt = tmp_path / "seg1.ckpt"
        loader = make_loader()

        cb1 = RunSummaryCallback(
            tmp_path,
            "resume-best",
            tiny_config_dict(),
            monitor="val/acc",
            mode="max",
            strict=True,
        )
        trainer1 = make_trainer([cb1], max_steps=2, val_check_interval=1)
        trainer1.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        trainer1.save_checkpoint(ckpt)
        seg1 = read(tmp_path, "resume-best")
        # val/acc = 9 at step 1, 8 at step 2: the best is 9 @ 1.
        assert seg1["stopping"]["best_value"] == 9.0
        assert seg1["stopping"]["best_step"] == 1

        # Fresh callback + trainer, resumed from the checkpoint. The
        # resumed segment sees only val/acc = 7 @ 3.
        cb2 = RunSummaryCallback(
            tmp_path,
            "resume-best",
            tiny_config_dict(),
            monitor="val/acc",
            mode="max",
            strict=True,
        )
        trainer2 = make_trainer([cb2], max_steps=3, val_check_interval=1)
        trainer2.fit(
            module,
            train_dataloaders=loader,
            val_dataloaders=loader,
            ckpt_path=str(ckpt),
        )
        seg2 = read(tmp_path, "resume-best")
        assert seg2["stopping"]["best_value"] == 9.0  # max over both segments
        assert seg2["stopping"]["best_step"] == 1


# ---------------------------------------------------------------------------
# Stale failure fields are cleared on success
# ---------------------------------------------------------------------------


class TestClearsStaleFailureFields:
    """A successful relaunch that loaded a previously failed record must
    not keep the old error next to the success status."""

    def _failed_record(self, tmp_path, run_id, module, loader):
        cb = RunSummaryCallback(tmp_path, run_id, tiny_config_dict(), strict=True)
        trainer = make_trainer([cb])
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        cb.on_exception(trainer, module, RuntimeError("old boom"))
        failed = read(tmp_path, run_id)
        assert failed["status"] == "failed"
        assert "old boom" in failed["error"]
        assert failed["failed_stage"] is not None

    def test_test_only_launch_clears_stale_failure_fields(self, tmp_path):
        module = tiny_module()
        loader = make_loader()
        self._failed_record(tmp_path, "retry", module, loader)

        cb2 = RunSummaryCallback(tmp_path, "retry", tiny_config_dict(), strict=True)
        trainer2 = L.Trainer(
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            callbacks=[cb2],
            limit_val_batches=0,
        )
        trainer2.test(module, dataloaders=loader)
        summary = read(tmp_path, "retry")
        assert summary["status"] == "test_done"
        assert "error" not in summary
        assert "failed_stage" not in summary

    def test_fit_relauunch_clears_stale_failure_fields(self, tmp_path):
        module = tiny_module()
        loader = make_loader()
        self._failed_record(tmp_path, "retry", module, loader)

        cb2 = RunSummaryCallback(tmp_path, "retry", tiny_config_dict(), strict=True)
        trainer2 = make_trainer([cb2])
        trainer2.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        summary = read(tmp_path, "retry")
        assert summary["status"] == "fit_done"
        assert "error" not in summary
        assert "failed_stage" not in summary


# ---------------------------------------------------------------------------
# Writer-unique temporary files (no *.tmp left behind)
# ---------------------------------------------------------------------------


class TestTempFiles:
    """The write is atomic and never leaves a temporary file behind,
    whatever else is writing the same path (two launches, two machines)."""

    def test_no_temp_files_left_behind(self, tmp_path, monkeypatch):
        cb = RunSummaryCallback(tmp_path, "tmp", tiny_config_dict(), strict=True)
        zero = SimpleNamespace(is_global_zero=True)
        cb._write(zero)
        assert not list(tmp_path.glob("*.tmp"))

        def boom(*args, **kwargs):
            raise RuntimeError("dump boom")

        monkeypatch.setattr(json, "dump", boom)
        with pytest.raises(RuntimeError, match="dump boom"):
            cb._write(zero)
        assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Run identity: a record belongs to (config_hash, seed)
# ---------------------------------------------------------------------------


class TestRunIdentity:
    """The identity of a record is ``(config_hash, seed)``: constructing
    against another run's record, or loading one, must raise instead of
    clobbering or merging it; the same run may overwrite its own record."""

    def _fit(self, tmp_path, run_id, cb, module=None):
        trainer = make_trainer([cb])
        trainer.fit(
            module or tiny_module(),
            train_dataloaders=make_loader(),
            val_dataloaders=make_loader(),
        )
        return read(tmp_path, run_id)

    def test_different_config_at_same_path_raises(self, tmp_path):
        self._fit(
            tmp_path, "id", RunSummaryCallback(tmp_path, "id", {"seed": 0, "a": 1})
        )
        with pytest.raises(FileExistsError, match="config_hash"):
            RunSummaryCallback(tmp_path, "id", {"seed": 0, "a": 2})

    def test_same_config_and_seed_relaunch_overwrites(self, tmp_path):
        first = self._fit(
            tmp_path, "id", RunSummaryCallback(tmp_path, "id", {"seed": 0, "a": 1})
        )
        second = self._fit(
            tmp_path, "id", RunSummaryCallback(tmp_path, "id", {"seed": 0, "a": 1})
        )
        assert second["config_hash"] == first["config_hash"]
        assert second["status"] == "fit_done"  # its own record, overwritten

    def test_test_only_launch_against_another_runs_file_raises(self, tmp_path):
        # The fresh callback is built while the path is still empty (so
        # construction passes); another run's record appears before its
        # test-only launch: the load must refuse to merge it.
        cb2 = RunSummaryCallback(tmp_path, "id", {"seed": 0, "a": 1})
        other = self._fit(
            tmp_path, "id", RunSummaryCallback(tmp_path, "id", {"seed": 0, "a": 2})
        )
        assert other["status"] == "fit_done"
        trainer2 = L.Trainer(
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            callbacks=[cb2],
            limit_val_batches=0,
        )
        with pytest.raises(FileExistsError, match="config_hash"):
            trainer2.test(tiny_module(), dataloaders=make_loader())

    def test_corrupt_json_file_raises(self, tmp_path):
        (tmp_path / "id.json").write_text("{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            RunSummaryCallback(tmp_path, "id", {"seed": 0})


# ---------------------------------------------------------------------------
# Early-stopping flag (current Lightning initialises stopping_reason to
# EarlyStoppingReason.NOT_STOPPED, so a None check marks every run stopped)
# ---------------------------------------------------------------------------


class TestEarlyStoppingFlag:
    def _fit(self, tmp_path, run_id, early_stopping, module=None):
        cb = RunSummaryCallback(tmp_path, run_id, tiny_config_dict(), strict=True)
        trainer = make_trainer([cb, early_stopping], val_check_interval=1)
        trainer.fit(
            module or DecreasingVal(),
            train_dataloaders=make_loader(),
            val_dataloaders=make_loader(),
        )
        return read(tmp_path, run_id)

    def test_unreachable_threshold_records_not_stopped(self, tmp_path):
        es = EarlyStopping(
            monitor="val/acc", mode="max", stopping_threshold=99.0, patience=10**9
        )
        summary = self._fit(tmp_path, "es-no", es)
        assert summary["stopping"]["early_stopped"] is False
        assert summary["stopping"]["early_stop_reason"] == "NOT_STOPPED"

    def test_threshold_reached_records_stopping_threshold(self, tmp_path):
        # DecreasingVal logs val/acc = 10 - step >= 9: the threshold is
        # reached on the first validation and the fit stops there.
        es = EarlyStopping(
            monitor="val/acc", mode="max", stopping_threshold=1.0, patience=10**9
        )
        summary = self._fit(tmp_path, "es-yes", es)
        assert summary["stopping"]["early_stopped"] is True
        assert summary["stopping"]["early_stop_reason"] == "STOPPING_THRESHOLD"

    def test_callback_without_stopping_reason_records_unknown(self, tmp_path):
        cb = RunSummaryCallback(tmp_path, "es-old", tiny_config_dict(), strict=True)
        es = EarlyStopping(
            monitor="val/acc", mode="max", stopping_threshold=99.0, patience=10**9
        )
        del es.stopping_reason  # a Lightning without the attribute
        cb._update_stopping(cast(Any, SimpleNamespace(callbacks=[es])))
        assert cb.summary["stopping"]["early_stopped"] is None
        assert cb.summary["stopping"]["early_stop_reason"] is None


# ---------------------------------------------------------------------------
# DDP safety: only the global rank zero writes
# ---------------------------------------------------------------------------


class TestRankZero:
    def test_non_zero_rank_writes_nothing(self, tmp_path):
        cb = RunSummaryCallback(tmp_path, "rank", tiny_config_dict(), strict=True)
        cb.summary["note"] = "pending"
        non_zero = SimpleNamespace(is_global_zero=False)
        cb._write(non_zero)
        assert not (tmp_path / "rank.json").exists()
        zero = SimpleNamespace(is_global_zero=True)
        cb._write(zero)
        assert (tmp_path / "rank.json").exists()


class TestFsync:
    """One write fsyncs the record file and the parent directory, so a
    power cut cannot truncate the JSON the next relaunch reads."""

    def test_write_fsyncs_the_file_and_the_directory(self, tmp_path, monkeypatch):
        cb = RunSummaryCallback(tmp_path, "fsync", tiny_config_dict(), strict=True)
        zero = SimpleNamespace(is_global_zero=True)
        fsynced = []
        monkeypatch.setattr(os, "fsync", lambda fd: fsynced.append(fd))
        real_open = os.open
        opened = {}

        def recording_open(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            opened[fd] = Path(path)
            return fd

        monkeypatch.setattr(os, "open", recording_open)
        cb._write(zero)
        assert len(fsynced) == 2
        # The directory fsynced after the rename is the record's parent.
        assert opened[fsynced[1]] == (tmp_path / "fsync.json").parent


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


class TestConfigHashExclude:
    """Configs differing only in an excluded key share the hash;
    differing in any other key do not; the seed is still excluded."""

    def test_excluded_keys_share_the_hash(self):
        a = {
            "seed": 0,
            "output": {"runs_dir": "/a"},
            "data": {"num_workers": 2},
            "model": {"d_model": 16},
        }
        b = {
            "seed": 1,
            "output": {"runs_dir": "/b"},
            "data": {"num_workers": 4},
            "model": {"d_model": 16},
        }
        assert config_hash(a, ("output", "data.num_workers")) == config_hash(
            b, ("output", "data.num_workers")
        )

    def test_any_other_key_still_differs(self):
        a = {"seed": 0, "output": {"runs_dir": "/a"}, "model": {"d_model": 16}}
        b = {"seed": 0, "output": {"runs_dir": "/b"}, "model": {"d_model": 32}}
        assert config_hash(a, ("output",)) != config_hash(b, ("output",))

    def test_seed_is_still_excluded(self):
        a = {"seed": 0, "output": {"runs_dir": "/a"}}
        b = {"seed": 9, "output": {"runs_dir": "/a"}}
        assert config_hash(a, ("output",)) == config_hash(b, ("output",))

    def test_missing_excluded_keys_are_ignored(self):
        a = {"seed": 0, "a": 1}
        assert config_hash(a, ("nope", "a.b.c")) == config_hash(a)

    def test_excluding_a_subtree_covers_its_leaves(self):
        a = {"seed": 0, "output": {"runs_dir": "/a", "other": 1}}
        assert config_hash(a, ("output",)) == config_hash({"seed": 0}, ("output",))

    def test_default_exclude_keeps_today_behaviour(self):
        a = {"seed": 0, "output": {"runs_dir": "/a"}}
        b = {"seed": 0, "output": {"runs_dir": "/b"}}
        assert config_hash(a) != config_hash(b)


class TestHashExcludeIdentity:
    """The callback's identity uses the excluded hash, and the
    record stores the sorted excluded keys."""

    def _fit(self, tmp_path, run_id, cb):
        trainer = make_trainer([cb])
        trainer.fit(
            tiny_module(),
            train_dataloaders=make_loader(),
            val_dataloaders=make_loader(),
        )
        return read(tmp_path, run_id)

    def test_record_contains_hash_excluded_keys(self, tmp_path):
        """The record stores the sorted excluded keys."""
        cb = RunSummaryCallback(
            tmp_path,
            "hx",
            tiny_config_dict(),
            hash_exclude=("output", "data.num_workers"),
            strict=True,
        )
        summary = self._fit(tmp_path, "hx", cb)
        assert summary["hash_excluded_keys"] == ["data.num_workers", "output"]

    def test_different_runs_dir_is_the_same_identity(self, tmp_path):
        """A record written with runs_dir=A is accepted as the same
        identity by a callback for the same run whose config has
        output.runs_dir=B."""
        config_a = {"seed": 0, "output": {"runs_dir": "/a"}, "model": {"d": 16}}
        config_b = {"seed": 0, "output": {"runs_dir": "/b"}, "model": {"d": 16}}
        self._fit(
            tmp_path,
            "id",
            RunSummaryCallback(
                tmp_path, "id", config_a, hash_exclude=("output",), strict=True
            ),
        )
        # A fresh callback for the same run with a different runs_dir is
        # accepted (same identity), and overwrites its own record.
        cb2 = RunSummaryCallback(
            tmp_path, "id", config_b, hash_exclude=("output",), strict=True
        )
        summary = self._fit(tmp_path, "id", cb2)
        assert summary["status"] == "fit_done"
        assert summary["config_hash"] == config_hash(config_a, ("output",))

    def test_without_exclude_the_difference_is_refused(self, tmp_path):
        """The default (no exclude) keeps today's behaviour: a different
        runs_dir is a different identity and is refused."""
        config_a = {"seed": 0, "output": {"runs_dir": "/a"}}
        config_b = {"seed": 0, "output": {"runs_dir": "/b"}}
        self._fit(
            tmp_path, "id", RunSummaryCallback(tmp_path, "id", config_a, strict=True)
        )
        with pytest.raises(FileExistsError, match="config_hash"):
            RunSummaryCallback(tmp_path, "id", config_b, strict=True)


class TestProvenance:
    """Provenance hashes files relative to the repository root; a
    missing file gives null."""

    def test_provenance_hashes_files_and_records_them(self, tmp_path, monkeypatch):
        from research_lib.training.callbacks import run_summary as rs

        monkeypatch.setattr(rs, "_repo_root", lambda: tmp_path)
        lock = tmp_path / "uv.lock"
        lock.write_bytes(b"lock content")
        cb = RunSummaryCallback(
            tmp_path,
            "prov",
            tiny_config_dict(),
            provenance_files=("uv.lock",),
            strict=True,
        )
        expected = hashlib.sha256(b"lock content").hexdigest()
        assert cb._provenance == {"uv.lock": expected}
        trainer = make_trainer([cb])
        trainer.fit(
            tiny_module(),
            train_dataloaders=make_loader(),
            val_dataloaders=make_loader(),
        )
        assert read(tmp_path, "prov")["provenance"] == {"uv.lock": expected}

    def test_missing_file_gives_null(self, tmp_path, monkeypatch):
        from research_lib.training.callbacks import run_summary as rs

        monkeypatch.setattr(rs, "_repo_root", lambda: tmp_path)
        cb = RunSummaryCallback(
            tmp_path,
            "prov2",
            tiny_config_dict(),
            provenance_files=("missing.lock",),
        )
        assert cb._provenance == {"missing.lock": None}

    def test_no_repo_root_gives_null(self, tmp_path, monkeypatch):
        from research_lib.training.callbacks import run_summary as rs

        monkeypatch.setattr(rs, "_repo_root", lambda: None)
        cb = RunSummaryCallback(
            tmp_path, "prov3", tiny_config_dict(), provenance_files=("uv.lock",)
        )
        assert cb._provenance == {"uv.lock": None}

    def test_default_records_no_provenance(self, tmp_path):
        cb = RunSummaryCallback(tmp_path, "prov4", tiny_config_dict())
        assert cb._provenance == {}
        assert "provenance" in BUILTIN_SECTIONS
