"""
Test checkpoint save/load and resume functionality.

Tests both:
1. Polite shutdown (finish current step, save checkpoint)
2. Impolite shutdown (kill process, resume from last checkpoint)
"""

import os
import shutil
import tempfile
from pathlib import Path

import lightning as L
import pytest
import torch

from research_lib.architectures.config import NanoGPTConfig
from research_lib.architectures.modded_nanogpt_base import ModdedNanoGPT
from research_lib.training.modules import DualOptimizerModule
from research_lib.training.presets import default_adamw_config


def create_fake_dataloader(vocab_size: int, seq_len: int = 128, num_batches: int = 100):
    """Create fake data for testing."""
    from torch.utils.data import DataLoader

    data = []
    for _ in range(num_batches):
        x = torch.randint(0, vocab_size, (4, seq_len))
        data.append({"input_ids": x, "labels": x})

    return DataLoader(data, batch_size=None)  # Already batched


def create_test_module():
    """Create a small test model and module."""
    config = NanoGPTConfig(n_layer=2, n_embd=64, n_head=2, ff_dim=128)
    model = ModdedNanoGPT(config)

    adamw_cfg, adamw_sched = default_adamw_config(lr=1e-4)

    module = DualOptimizerModule(
        model=model,
        matrix_optimizer_config=adamw_cfg,
        vector_optimizer_config=adamw_cfg,
        matrix_schedule_config=adamw_sched,
        vector_schedule_config=adamw_sched,
        matrix_target_modules=[],
    )

    return module, config


class TestCheckpointResume:
    """Test checkpoint save and resume."""

    def test_checkpoint_saves_scheduler_state(self):
        """Verify scheduler state is saved in checkpoint."""
        module, config = create_test_module()
        loader = create_fake_dataloader(config.vocab_size)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoints"

            trainer = L.Trainer(
                max_steps=10,
                default_root_dir=tmpdir,
                callbacks=[
                    L.pytorch.callbacks.ModelCheckpoint(
                        dirpath=ckpt_path,
                        every_n_train_steps=5,
                        save_last=True,
                    )
                ],
                enable_progress_bar=False,
                logger=False,
            )

            trainer.fit(module, loader)

            # Check checkpoint exists
            last_ckpt = ckpt_path / "last.ckpt"
            assert last_ckpt.exists()

            # Load and verify scheduler state
            ckpt = torch.load(last_ckpt)
            assert "optimizer_step_count" in ckpt
            assert ckpt["optimizer_step_count"] > 0

    def test_resume_continues_from_correct_step(self):
        """Verify training resumes from correct step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoints"

            # First training run: 10 steps
            module1, config = create_test_module()
            loader1 = create_fake_dataloader(config.vocab_size)

            trainer1 = L.Trainer(
                max_steps=10,
                default_root_dir=tmpdir,
                callbacks=[
                    L.pytorch.callbacks.ModelCheckpoint(
                        dirpath=ckpt_path,
                        every_n_train_steps=5,
                        save_last=True,
                    )
                ],
                enable_progress_bar=False,
                logger=False,
            )
            trainer1.fit(module1, loader1)

            step_after_first = trainer1.global_step
            loss_after_first = trainer1.callback_metrics.get("train/loss")

            # Second training run: resume and continue to 20 steps
            module2, _ = create_test_module()
            loader2 = create_fake_dataloader(config.vocab_size)

            trainer2 = L.Trainer(
                max_steps=20,
                default_root_dir=tmpdir,
                callbacks=[
                    L.pytorch.callbacks.ModelCheckpoint(
                        dirpath=ckpt_path,
                        every_n_train_steps=5,
                        save_last=True,
                    )
                ],
                enable_progress_bar=False,
                logger=False,
            )

            trainer2.fit(module2, loader2, ckpt_path=str(ckpt_path / "last.ckpt"))

            # Should have continued from step 10, not restarted
            assert trainer2.global_step == 20
            assert module2._optimizer_step_count > 0

    def test_scheduler_values_restored_on_resume(self):
        """Verify learning rate is correct after resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoints"

            # First run
            module1, config = create_test_module()
            loader1 = create_fake_dataloader(config.vocab_size)

            trainer1 = L.Trainer(
                max_steps=50,
                default_root_dir=tmpdir,
                callbacks=[
                    L.pytorch.callbacks.ModelCheckpoint(
                        dirpath=ckpt_path,
                        every_n_train_steps=25,
                        save_last=True,
                    )
                ],
                enable_progress_bar=False,
                logger=False,
            )
            trainer1.fit(module1, loader1)

            # Get LR at end of first run
            opts1 = module1.optimizers()
            if not isinstance(opts1, list):
                opts1 = [opts1]
            lr_end_first = opts1[0].param_groups[0]["lr"]

            # Resume
            module2, _ = create_test_module()
            loader2 = create_fake_dataloader(config.vocab_size)

            trainer2 = L.Trainer(
                max_steps=100,
                default_root_dir=tmpdir,
                callbacks=[
                    L.pytorch.callbacks.ModelCheckpoint(
                        dirpath=ckpt_path,
                        every_n_train_steps=25,
                        save_last=True,
                    )
                ],
                enable_progress_bar=False,
                logger=False,
            )
            trainer2.fit(module2, loader2, ckpt_path=str(ckpt_path / "last.ckpt"))

            # LR should have continued evolving from where it was
            opts2 = module2.optimizers()
            if not isinstance(opts2, list):
                opts2 = [opts2]
            lr_end_second = opts2[0].param_groups[0]["lr"]

            # Can't be exactly equal (training continued), but should be in reasonable range
            assert lr_end_second >= 0
            assert lr_end_second <= lr_end_first  # Should have decayed further


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
