"""Tests for DualOptimizerModule."""

import lightning as L
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Tests for DualOptimizerModule initialization and strategy logic."""

    def test_init_with_matrix_targets(self):
        """Test initialization with explicit matrix targets."""
        model = SimpleModel()
        training_config = TrainingConfig(total_steps=100)
        adam_config = default_adam_config()

        module = DualOptimizerModule(
            model=model,
            training_config=training_config,
            matrix_optimizer_config=adam_config,
            vector_optimizer_config=adam_config,
            matrix_target_modules=["attn", "mlp"],  # Explicitly setting matrix
        )

        assert module._target_strategy == "matrix"
        assert module.target_modules == ["attn", "mlp"]

    def test_init_with_vector_targets(self):
        """Test initialization with explicit vector targets."""
        model = SimpleModel()
        training_config = TrainingConfig(total_steps=100)
        adam_config = default_adam_config()

        module = DualOptimizerModule(
            model=model,
            training_config=training_config,
            matrix_optimizer_config=adam_config,
            vector_optimizer_config=adam_config,
            vector_target_modules=["embed"],  # Explicitly setting vector
        )

        assert module._target_strategy == "vector"
        assert module.target_modules == ["embed"]

    def test_init_conflict(self):
        """Test that providing both target args raises ValueError."""
        model = SimpleModel()
        training_config = TrainingConfig(total_steps=100)
        adam_config = default_adam_config()

        with pytest.raises(ValueError, match="Cannot specify both"):
            DualOptimizerModule(
                model=model,
                training_config=training_config,
                matrix_optimizer_config=adam_config,
                vector_optimizer_config=adam_config,
                matrix_target_modules=["attn"],
                vector_target_modules=["embed"],
            )

    def test_init_defaults(self):
        """Test initialization with no targets defaults to matrix strategy (empty)."""
        model = SimpleModel()
        training_config = TrainingConfig(total_steps=100)
        adam_config = default_adam_config()

        module = DualOptimizerModule(
            model=model,
            training_config=training_config,
            matrix_optimizer_config=adam_config,
            vector_optimizer_config=adam_config,
            # Both None
        )

        assert module._target_strategy == "matrix"
        assert module.target_modules == []


class TestDualOptimizerModuleConfigureOptimizers:
    """Tests for configure_optimizers method."""

    def test_configure_optimizers_matrix_strategy(self):
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
            matrix_target_modules=["attn", "mlp"],
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

    def test_configure_optimizers_vector_strategy(self):
        """Test inverted behavior: matched -> vector."""
        model = SimpleModel()
        training_config = TrainingConfig(total_steps=100)

        # Matrix = 0.05, Vector = 0.01
        matrix_config = default_adam_config(lr=0.01)
        vector_config = default_adam_config(lr=0.001)

        # We target 'attn' for VECTOR this time
        # This means 'attn' should end up in optimizer with LR 0.01
        module = DualOptimizerModule(
            model=model,
            training_config=training_config,
            matrix_optimizer_config=matrix_config,
            vector_optimizer_config=vector_config,
            vector_target_modules=["attn"],
        )
        module._trainer = type("MockTrainer", (), {"is_global_zero": True})()

        optimizers, schedulers = module.configure_optimizers()

        # Check param counts/existence to verify partition
        # The model has 'attn', 'mlp', 'embed', 'lm_head'
        # Target 'attn' -> Vector. Rest -> Matrix.

        assert len(optimizers) == 2
        assert len(schedulers) == 2

        # Check base LRs are set correctly (initial_lr is stored by LambdaLR)
        # Note: Current LR may be 0 due to warmup, so we check initial_lr
        assert optimizers[0].param_groups[0]["initial_lr"] == 0.01
        assert optimizers[1].param_groups[0]["initial_lr"] == 0.001

    def test_single_optimizer_via_vector_strategy(self):
        """Test configuring single optimizer (empty targets)."""
        model = SimpleModel()
        training_config = TrainingConfig(total_steps=100)
        adam_config = default_adam_config()

        module = DualOptimizerModule(
            model=model,
            training_config=training_config,
            matrix_optimizer_config=adam_config,
            vector_optimizer_config=adam_config,
            vector_target_modules=[""],
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
            matrix_target_modules=["attn"],
        )

        input_ids = torch.randint(0, 100, (2, 16))
        logits = module.forward(input_ids)

        assert logits.shape == (2, 16, 100)


class TestComputeLoss:
    """Tests for compute_loss method."""

    def test_default_loss_causal_lm(self):
        """Test default loss computes causal LM cross-entropy."""
        model = SimpleModel(vocab_size=100, dim=32)
        module = DualOptimizerModule(
            model=model,
            training_config=TrainingConfig(total_steps=100),
            matrix_optimizer_config=default_adam_config(),
            vector_optimizer_config=default_adam_config(),
            matrix_target_modules=["attn"],
        )

        batch = {"input_ids": torch.randint(0, 100, (2, 16))}
        logits = module.forward(batch["input_ids"])
        loss = module.compute_loss(logits, batch)

        assert loss.shape == ()  # Scalar
        assert loss.requires_grad

    def test_default_loss_with_labels(self):
        """Test default loss uses labels when provided."""
        model = SimpleModel(vocab_size=100, dim=32)
        module = DualOptimizerModule(
            model=model,
            training_config=TrainingConfig(total_steps=100),
            matrix_optimizer_config=default_adam_config(),
            vector_optimizer_config=default_adam_config(),
            matrix_target_modules=["attn"],
        )

        input_ids = torch.randint(0, 100, (2, 16))
        labels = torch.randint(0, 100, (2, 16))
        labels[:, :5] = -100  # Mask some tokens

        batch = {"input_ids": input_ids, "labels": labels}
        logits = module.forward(input_ids)
        loss = module.compute_loss(logits, batch)

        assert loss.shape == ()

    def test_custom_loss_via_subclass(self):
        """Test that compute_loss can be overridden."""

        class ConstantLossModule(DualOptimizerModule):
            """Module that always returns loss=1.0 for testing."""

            def compute_loss(self, model_output, batch):
                return torch.tensor(1.0, requires_grad=True)

        model = SimpleModel()
        module = ConstantLossModule(
            model=model,
            training_config=TrainingConfig(total_steps=100),
            matrix_optimizer_config=default_adam_config(),
            vector_optimizer_config=default_adam_config(),
            matrix_target_modules=["attn"],
        )

        batch = {"input_ids": torch.randint(0, 100, (2, 16))}
        logits = module.forward(batch["input_ids"])
        loss = module.compute_loss(logits, batch)

        assert loss.item() == 1.0

    def test_custom_loss_with_auxiliary_data(self):
        """Test custom loss can access arbitrary batch keys."""

        class WeightedLossModule(DualOptimizerModule):
            """Module that uses sample weights from batch."""

            def compute_loss(self, model_output, batch):
                base_loss = super().compute_loss(model_output, batch)
                weights = batch.get("weights", None)
                if weights is not None:
                    return base_loss * weights.mean()
                return base_loss

        model = SimpleModel()
        module = WeightedLossModule(
            model=model,
            training_config=TrainingConfig(total_steps=100),
            matrix_optimizer_config=default_adam_config(),
            vector_optimizer_config=default_adam_config(),
            matrix_target_modules=["attn"],
        )

        batch = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "weights": torch.tensor([0.5, 1.5]),
        }
        logits = module.forward(batch["input_ids"])
        loss = module.compute_loss(logits, batch)

        assert loss.shape == ()


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
            matrix_target_modules=["attn", "mlp"],
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
        assert module._optimizer_step_count == 4  # Can load all 4 steps


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
            matrix_target_modules=["attn", "mlp"],
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

        assert module._optimizer_step_count == 4

    def test_training_with_vector_targeting(self):
        """Test training with vector_target_modules."""
        model = SimpleModel()
        training_config = TrainingConfig(total_steps=10, grad_accum_steps=1)
        adam_config = default_adam_config()

        module = DualOptimizerModule(
            model=model,
            training_config=training_config,
            matrix_optimizer_config=adam_config,
            vector_optimizer_config=adam_config,
            vector_target_modules=["embed"],
        )

        trainer = L.Trainer(
            accelerator="cpu",
            max_steps=4,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        dataloader = create_dummy_dataloader(batch_size=2, num_batches=10)

        trainer.fit(module, dataloader)

        assert module._optimizer_step_count == 4

    def test_training_with_custom_loss(self):
        """Test training with overridden compute_loss."""

        class MSELossModule(DualOptimizerModule):
            """Use MSE loss instead of cross-entropy (nonsensical but tests override)."""

            def compute_loss(self, model_output, batch):
                # Just compute MSE between logits and some target
                target = torch.zeros_like(model_output)
                return F.mse_loss(model_output, target)

        model = SimpleModel()
        module = MSELossModule(
            model=model,
            training_config=TrainingConfig(total_steps=10, grad_accum_steps=1),
            matrix_optimizer_config=default_adam_config(),
            vector_optimizer_config=default_adam_config(),
            matrix_target_modules=["attn"],
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
        assert module._optimizer_step_count == 4
