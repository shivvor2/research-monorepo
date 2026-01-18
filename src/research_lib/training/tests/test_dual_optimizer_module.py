"""Tests for DualOptimizerModule."""

import lightning as L
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from research_lib.training import (
    DualOptimizerModule,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    default_adam_config,
)


class SimpleModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, vocab_size=100, dim=32, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.Linear(dim, dim, bias=False),
                        "mlp": nn.Linear(dim, dim, bias=False),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer["attn"](x)
            x = x + layer["mlp"](x)
        return self.lm_head(x)


def create_dummy_dataloader(batch_size=4, seq_len=16, num_batches=10, vocab_size=100):
    """Create a dummy dataloader for testing."""
    input_ids = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len))
    dataset = TensorDataset(input_ids)

    def collate_fn(batch):
        input_ids = torch.stack([b[0] for b in batch])
        return {"input_ids": input_ids}

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


class TestDualOptimizerModuleInit:
    """Tests for DualOptimizerModule initialization."""

    def test_init_with_configs(self):
        """Test basic initialization."""
        model = SimpleModel()
        training_config = TrainingConfig(total_steps=100)

        # Use AdamW for both since Muon requires CUDA
        matrix_config = default_adam_config(lr=0.01)
        vector_config = default_adam_config(lr=0.001)

        module = DualOptimizerModule(
            model=model,
            training_config=training_config,
            matrix_optimizer_config=matrix_config,
            vector_optimizer_config=vector_config,
            target_modules=["attn", "mlp"],
        )

        assert module.automatic_optimization is False
        assert module.training_config == training_config
        assert module.target_modules == ["attn", "mlp"]

    def test_init_with_empty_targets(self):
        """Test initialization with no target modules (single optimizer)."""
        model = SimpleModel()
        training_config = TrainingConfig(total_steps=100)
        adam_config = default_adam_config()

        module = DualOptimizerModule(
            model=model,
            training_config=training_config,
            matrix_optimizer_config=adam_config,
            vector_optimizer_config=adam_config,
            target_modules=[],  # No targets = all params to vector optimizer
        )

        assert module.target_modules == []


class TestDualOptimizerModuleConfigureOptimizers:
    """Tests for configure_optimizers method."""

    def test_configure_optimizers_dual(self):
        """Test configuring two optimizers."""
        model = SimpleModel()
        training_config = TrainingConfig(total_steps=100)
        matrix_config = default_adam_config(lr=0.01)
        vector_config = default_adam_config(lr=0.001)

        module = DualOptimizerModule(
            model=model,
            training_config=training_config,
            matrix_optimizer_config=matrix_config,
            vector_optimizer_config=vector_config,
            target_modules=["attn", "mlp"],
        )

        # Mock trainer for is_global_zero
        module._trainer = type("MockTrainer", (), {"is_global_zero": True})()

        optimizers, schedulers = module.configure_optimizers()

        assert len(optimizers) == 2
        assert len(schedulers) == 2

        # Check base LRs are set correctly (initial_lr is stored by LambdaLR)
        # Note: Current LR may be 0 due to warmup, so we check initial_lr
        assert optimizers[0].param_groups[0]["initial_lr"] == 0.01
        assert optimizers[1].param_groups[0]["initial_lr"] == 0.001

    def test_configure_optimizers_single(self):
        """Test configuring single optimizer (empty targets)."""
        model = SimpleModel()
        training_config = TrainingConfig(total_steps=100)
        adam_config = default_adam_config()

        module = DualOptimizerModule(
            model=model,
            training_config=training_config,
            matrix_optimizer_config=adam_config,
            vector_optimizer_config=adam_config,
            target_modules=[],
        )

        module._trainer = type("MockTrainer", (), {"is_global_zero": True})()

        optimizers, schedulers = module.configure_optimizers()

        # Only vector optimizer should exist (no matrix params)
        assert len(optimizers) == 1
        assert len(schedulers) == 1


class TestDualOptimizerModuleForward:
    """Tests for forward pass."""

    def test_forward(self):
        """Test forward pass returns correct shape."""
        model = SimpleModel(vocab_size=100, dim=32)
        module = DualOptimizerModule(
            model=model,
            training_config=TrainingConfig(total_steps=100),
            matrix_optimizer_config=default_adam_config(),
            vector_optimizer_config=default_adam_config(),
            target_modules=["attn"],
        )

        input_ids = torch.randint(0, 100, (2, 16))
        logits = module.forward(input_ids)

        assert logits.shape == (2, 16, 100)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDualOptimizerModuleTraining:
    """Integration tests for training (requires CUDA for full test)."""

    def test_training_step_runs(self):
        """Test that training_step executes without error."""
        model = SimpleModel().cuda()
        training_config = TrainingConfig(total_steps=10, grad_accum_steps=2)
        adam_config = default_adam_config()

        module = DualOptimizerModule(
            model=model,
            training_config=training_config,
            matrix_optimizer_config=adam_config,
            vector_optimizer_config=adam_config,
            target_modules=["attn", "mlp"],
        ).cuda()

        # Create trainer for short run
        trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            max_steps=4,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        dataloader = create_dummy_dataloader(batch_size=2, num_batches=10)

        # Should complete without error
        trainer.fit(module, dataloader)

        # Check that steps were taken
        assert module._optimizer_step_count == 2  # 4 batches / 2 accum = 2 steps


class TestDualOptimizerModuleCPU:
    """Tests that can run on CPU."""

    def test_training_step_cpu(self):
        """Test training step on CPU."""
        model = SimpleModel()
        training_config = TrainingConfig(total_steps=10, grad_accum_steps=2)
        adam_config = default_adam_config()

        module = DualOptimizerModule(
            model=model,
            training_config=training_config,
            matrix_optimizer_config=adam_config,
            vector_optimizer_config=adam_config,
            target_modules=["attn", "mlp"],
        )

        trainer = L.Trainer(
            accelerator="cpu",
            max_steps=4,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        dataloader = create_dummy_dataloader(batch_size=2, num_batches=10)

        # Should complete without error
        trainer.fit(module, dataloader)

        assert module._optimizer_step_count == 2
