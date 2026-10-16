"""
Tests for the run-summary callback.

Target: src/research_lib/training/callbacks/run_summary.py
Run with: pytest src/research_lib/training/callbacks/tests/test_run_summary.py -v
"""

import json

import pytest
import torch

from research_lib.architectures.config import ZoologyLMConfig
from research_lib.architectures.zoology_lm import ZoologyLM
from research_lib.training.callbacks.run_summary import RunSummaryCallback, config_hash


def tiny_model() -> ZoologyLM:
    return ZoologyLM(
        ZoologyLMConfig(
            vocab_size=64,
            d_model=16,
            num_heads=1,
            t_max=32,
            layout="single",
        )
    )


def tiny_config_dict(seed: int = 0) -> dict:
    return {"seed": seed, "model": {"d_model": 16, "layout": "single"}, "task": "mqar"}


class TestConfigHash:
    """Tests for the seed-independent config hash."""

    def test_same_config_two_seeds_share_hash(self):
        assert config_hash(tiny_config_dict(seed=0)) == config_hash(
            tiny_config_dict(seed=1)
        )

    def test_different_configs_differ(self):
        other = tiny_config_dict(seed=0)
        other["model"] = {"d_model": 32, "layout": "single"}
        assert config_hash(tiny_config_dict()) != config_hash(other)


class TestRunSummaryCallback:
    """Tests for RunSummaryCallback."""

    @pytest.fixture
    def trained(self, tmp_path):
        """A 2-step fit on the tiny config with the callback attached.

        Returns (callback, runs_dir, trainer).
        """
        import lightning as L

        runs_dir = tmp_path / "runs"
        cb = RunSummaryCallback(
            runs_dir=runs_dir, run_id="test-run", config=tiny_config_dict()
        )
        trainer = L.Trainer(
            max_steps=2,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            callbacks=[cb],
            num_sanity_val_steps=0,
            val_check_interval=2,
        )
        model = tiny_model()
        from research_lib.training.configs import OptimizerConfig, ScheduleConfig
        from research_lib.training.modules.synthetic_recall import SyntheticRecallModule
        from research_lib.training.scheduling import WarmupStableDecaySchedule

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
        module = SyntheticRecallModule(
            model=model,
            slice_values=(4, 8),
            matrix_optimizer_config=OptimizerConfig(
                torch.optim.AdamW, {"lr": 0.0, "weight_decay": 0.1}
            ),
            vector_optimizer_config=OptimizerConfig(
                torch.optim.AdamW, {"lr": 0.0, "weight_decay": 0.0}
            ),
            matrix_schedule_config=schedule,
            vector_schedule_config=schedule,
            matrix_target_modules=["lm_head"],
        )

        def batch_dict(_):
            return {
                "input_ids": torch.zeros(8, 16, dtype=torch.long),
                "labels": torch.full((8, 16), -100, dtype=torch.long),
                "slice_id": torch.zeros(8, dtype=torch.long),
            }

        from torch.utils.data import Dataset

        class _DictDataset(Dataset):
            def __len__(self):
                return 64

            def __getitem__(self, index):
                return batch_dict(index)

        loader = torch.utils.data.DataLoader(_DictDataset(), batch_size=None)
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        return cb, runs_dir, trainer, module

    def test_summary_json_written_with_expected_keys(self, trained):
        cb, runs_dir, trainer, module = trained
        path = runs_dir / "test-run.json"
        assert path.exists()
        summary = json.loads(path.read_text())
        for key in (
            "run_id",
            "config",
            "config_hash",
            "seed",
            "git",
            "versions",
            "device",
            "params",
            "timing",
            "stopping",
            "val_final",
            "mixer_stats_final",
            "test",
        ):
            assert key in summary, key
        assert summary["run_id"] == "test-run"
        assert summary["seed"] == 0
        assert summary["timing"]["steps_completed"] == 2
        assert summary["params"]["total"] == sum(
            p.numel() for p in module.model.parameters()
        )
        assert summary["params"]["state_bytes_per_mixer"] == 1 * 1 * 16 * 16 * 4
        assert summary["stopping"]["stopped"] is False
        assert summary["stopping"]["best_val_acc"] is not None
        # The file is complete even though the test phase never ran:
        # "test" is present (explicitly null).
        assert summary["test"] is None

    def test_two_seeds_share_config_hash_in_json(self, trained, tmp_path):
        cb, runs_dir, trainer, module = trained
        summary = json.loads((runs_dir / "test-run.json").read_text())
        assert summary["config_hash"] == config_hash(tiny_config_dict())
        assert summary["config_hash"] == config_hash(tiny_config_dict(seed=7))

    def test_test_phase_updates_json(self, trained):
        cb, runs_dir, trainer, module = trained
        loader = torch.utils.data.DataLoader(
            [
                {
                    "input_ids": torch.zeros(8, 16, dtype=torch.long),
                    "labels": torch.full((8, 16), -100, dtype=torch.long),
                    "slice_id": torch.zeros(8, dtype=torch.long),
                }
            ],
            batch_size=None,
        )
        trainer.test(module, dataloaders=loader)
        summary = json.loads((runs_dir / "test-run.json").read_text())
        assert summary["test"] is not None
        assert "acc" in summary["test"]
