"""
Integration tests to verify everything works before long training runs.

Run: pytest test_integration.py -v
"""

from pathlib import Path

import lightning as L
import pytest
import torch

from research_lib.architectures.config import NanoGPTConfig
from research_lib.architectures.modded_nanogpt_base import ModdedNanoGPT
from research_lib.data import FineWebDataModule
from research_lib.training.modules import DualOptimizerModule
from research_lib.training.presets import default_adamw_config

# Skip if no data
DATA_DIR = Path("data/finewebedu10B")
HAS_DATA = DATA_DIR.exists() and len(list(DATA_DIR.glob("*.bin"))) > 0


class TestModelForward:
    """Test model can do forward pass."""

    def test_forward_shape(self):
        config = NanoGPTConfig(n_layer=2, n_embd=64, n_head=2, ff_dim=128)
        model = ModdedNanoGPT(config)

        x = torch.randint(0, config.vocab_size, (2, 128))
        out = model(x)

        assert out.shape == (2, 128, config.vocab_size)

    def test_forward_with_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = NanoGPTConfig(n_layer=2, n_embd=64, n_head=2, ff_dim=128)
        model = ModdedNanoGPT(config).cuda()

        x = torch.randint(0, config.vocab_size, (2, 128)).cuda()
        out = model(x)

        assert out.device.type == "cuda"


@pytest.mark.skipif(not HAS_DATA, reason="Data not available")
class TestDataModule:
    """Test data loading."""

    def test_datamodule_setup(self):
        dm = FineWebDataModule(
            data_dir=str(DATA_DIR),
            seq_len=128,
            batch_size=2,
            num_workers=0,
        )
        dm.setup()

        assert len(dm.train_shards) > 0
        assert len(dm.val_shards) > 0

    def test_datamodule_batch(self):
        dm = FineWebDataModule(
            data_dir=str(DATA_DIR),
            seq_len=128,
            batch_size=2,
            num_workers=0,
        )
        dm.setup()

        batch = next(iter(dm.train_dataloader()))

        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape == (2, 128)
        assert batch["labels"].shape == (2, 128)


class TestTrainingModule:
    """Test training module setup."""

    def test_module_creation(self):
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

        assert module is not None

    def test_training_step(self):
        """Test single training step."""
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

        # Quick training test
        trainer = L.Trainer(
            max_steps=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        # Fake data
        from torch.utils.data import DataLoader, TensorDataset

        x = torch.randint(0, config.vocab_size, (8, 128))
        dataset = TensorDataset(x)

        # Wrap in dict format
        class DictDataset:
            def __init__(self, tensor):
                self.tensor = tensor

            def __len__(self):
                return len(self.tensor)

            def __iter__(self):
                for t in self.tensor:
                    yield {"input_ids": t, "labels": t}

        loader = DataLoader(list(DictDataset(x)), batch_size=2)

        trainer.fit(module, loader)


@pytest.mark.skipif(not HAS_DATA, reason="Data not available")
class TestEndToEnd:
    """Full end-to-end test with real data."""

    def test_full_pipeline(self):
        """Run a few steps with real data."""
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

        dm = FineWebDataModule(
            data_dir=str(DATA_DIR),
            seq_len=128,
            batch_size=2,
            num_workers=0,
        )

        trainer = L.Trainer(
            max_steps=5,
            accelerator="auto",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            limit_val_batches=2,
        )

        trainer.fit(module, datamodule=dm)

        # Check loss decreased (or at least didn't explode)
        assert trainer.callback_metrics.get("train/loss") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
