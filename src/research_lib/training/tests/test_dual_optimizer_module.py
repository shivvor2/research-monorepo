"""Tests for DualOptimizerModule."""

import lightning as L
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from research_lib.training.configs import (
    GradAccumSchedule,
    OptimizerConfig,
    ScheduleConfig,
)
from research_lib.training.modules import DualOptimizerModule
from research_lib.training.presets import default_adamw_config
from research_lib.training.scheduling import WarmupStableDecaySchedule

# =============================================================================
# Test Fixtures
# =============================================================================


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


def create_simple_configs():
    """Create simple configs for testing."""
    opt_config = OptimizerConfig(
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3},
    )
    schedule_config = ScheduleConfig(
        global_schedules=[
            WarmupStableDecaySchedule(param_name="lr", max_value=1e-3),
        ],
    )
    return opt_config, schedule_config


# =============================================================================
# Initialization Tests
# =============================================================================


class TestDualOptimizerModuleInit:
    """Tests for DualOptimizerModule initialization."""

    def test_init_with_matrix_targets(self):
        """Test initialization with explicit matrix targets."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
            matrix_target_modules=["attn", "mlp"],
        )

        assert module._target_strategy == "matrix"
        assert module.target_modules == ["attn", "mlp"]

    def test_init_with_vector_targets(self):
        """Test initialization with explicit vector targets."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
            vector_target_modules=["embed"],
        )

        assert module._target_strategy == "vector"
        assert module.target_modules == ["embed"]

    def test_init_conflict_raises(self):
        """Test that providing both target args raises ValueError."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()

        with pytest.raises(ValueError, match="Cannot specify both"):
            DualOptimizerModule(
                model=model,
                matrix_optimizer_config=opt_config,
                vector_optimizer_config=opt_config,
                matrix_schedule_config=schedule_config,
                vector_schedule_config=schedule_config,
                matrix_target_modules=["attn"],
                vector_target_modules=["embed"],
            )

    def test_init_defaults(self):
        """Test initialization with no targets defaults to matrix strategy."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
        )

        assert module._target_strategy == "matrix"
        assert module.target_modules == []

    def test_training_with_grad_accum_constant(self):
        """Test training with constant gradient accumulation via grad_accum."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
            matrix_target_modules=["attn"],
            grad_accum=2,  # Constant accumulation
        )

        trainer = L.Trainer(
            accelerator="cpu",
            max_steps=4,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        dataloader = create_dummy_dataloader(batch_size=2, num_batches=20)

        trainer.fit(module, dataloader)

        # With accum=2, we need 8 batches for 4 optimizer steps
        assert module._optimizer_step_count == 4

    def test_training_with_grad_accum_schedule(self):
        """Test training with step-based gradient accumulation schedule."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()
        grad_accum = GradAccumSchedule({0: 2})

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
            matrix_target_modules=["attn"],
            grad_accum_schedule=grad_accum,
        )

        trainer = L.Trainer(
            accelerator="cpu",
            max_steps=4,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        dataloader = create_dummy_dataloader(batch_size=2, num_batches=20)

        trainer.fit(module, dataloader)

        # With accum=2, we need 8 batches for 4 optimizer steps
        assert module._optimizer_step_count == 4

    def test_grad_accum_mutual_exclusivity(self):
        """Test that providing both grad_accum and grad_accum_schedule raises."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()

        with pytest.raises(ValueError, match="Cannot specify both"):
            DualOptimizerModule(
                model=model,
                matrix_optimizer_config=opt_config,
                vector_optimizer_config=opt_config,
                matrix_schedule_config=schedule_config,
                vector_schedule_config=schedule_config,
                grad_accum=2,
                grad_accum_schedule=GradAccumSchedule({0: 4}),
            )

    def test_grad_accum_validation(self):
        """Test that invalid grad_accum raises."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()

        with pytest.raises(ValueError, match="grad_accum must be >= 1"):
            DualOptimizerModule(
                model=model,
                matrix_optimizer_config=opt_config,
                vector_optimizer_config=opt_config,
                matrix_schedule_config=schedule_config,
                vector_schedule_config=schedule_config,
                grad_accum=0,
            )

    def test_grad_accum_default(self):
        """Test that default grad_accum is 1 (no accumulation)."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
            matrix_target_modules=["attn"],
            # No grad_accum or grad_accum_schedule specified
        )

        # Should default to 1
        assert module._get_current_grad_accum() == 1


# =============================================================================
# configure_optimizers Tests
# =============================================================================


class TestDualOptimizerModuleConfigureOptimizers:
    """Tests for configure_optimizers method."""

    def test_configure_optimizers_matrix_strategy(self):
        """Test configuring two optimizers with matrix strategy."""
        model = SimpleModel()
        matrix_config = OptimizerConfig(
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={"lr": 0.01},
        )
        vector_config = OptimizerConfig(
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={"lr": 0.001},
        )
        schedule_config = ScheduleConfig()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=matrix_config,
            vector_optimizer_config=vector_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
            matrix_target_modules=["attn", "mlp"],
        )

        # Mock trainer for is_global_zero
        module._trainer = type("MockTrainer", (), {"is_global_zero": True})()

        optimizers = module.configure_optimizers()

        assert len(optimizers) == 2
        assert optimizers[0].defaults["lr"] == 0.01
        assert optimizers[1].defaults["lr"] == 0.001

    def test_configure_optimizers_vector_strategy(self):
        """Test configuring optimizers with vector strategy."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
            vector_target_modules=["attn"],
        )
        module._trainer = type("MockTrainer", (), {"is_global_zero": True})()

        optimizers = module.configure_optimizers()

        assert len(optimizers) == 2
        # Both optimizers should have params
        assert len(optimizers[0].param_groups[0]["params"]) > 0
        assert len(optimizers[1].param_groups[0]["params"]) > 0

    def test_single_optimizer_when_no_targets(self):
        """Test single optimizer when target matches nothing."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
            matrix_target_modules=["nonexistent"],
        )
        module._trainer = type("MockTrainer", (), {"is_global_zero": True})()

        optimizers = module.configure_optimizers()

        # Only vector optimizer should exist (no matrix params)
        assert len(optimizers) == 1


# =============================================================================
# Forward and Loss Tests
# =============================================================================


class TestDualOptimizerModuleForward:
    """Tests for forward pass."""

    def test_forward(self):
        """Test forward pass returns correct shape."""
        model = SimpleModel(vocab_size=100, dim=32)
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
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
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
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
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
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
        opt_config, schedule_config = create_simple_configs()

        module = ConstantLossModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
            matrix_target_modules=["attn"],
        )

        batch = {"input_ids": torch.randint(0, 100, (2, 16))}
        logits = module.forward(batch["input_ids"])
        loss = module.compute_loss(logits, batch)

        assert loss.item() == 1.0


# =============================================================================
# Training Tests (CPU)
# =============================================================================


class TestDualOptimizerModuleTrainingCPU:
    """Tests that run on CPU."""

    def test_training_step_cpu(self):
        """Test training step on CPU."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
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
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
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

    def test_training_with_grad_accum_schedule_override(self):
        """Test training with GradAccumSchedule override."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()
        grad_accum = GradAccumSchedule({0: 2})

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
            matrix_target_modules=["attn"],
            grad_accum_schedule=grad_accum,
        )

        trainer = L.Trainer(
            accelerator="cpu",
            max_steps=4,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        dataloader = create_dummy_dataloader(batch_size=2, num_batches=20)

        trainer.fit(module, dataloader)

        # With accum=2, we need 8 batches for 4 optimizer steps
        assert module._optimizer_step_count == 4

    def test_training_with_custom_loss(self):
        """Test training with overridden compute_loss."""

        class MSELossModule(DualOptimizerModule):
            """Use MSE loss instead of cross-entropy."""

            def compute_loss(self, model_output, batch):
                target = torch.zeros_like(model_output)
                return F.mse_loss(model_output, target)

        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()

        module = MSELossModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
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

        trainer.fit(module, dataloader)
        assert module._optimizer_step_count == 4


# =============================================================================
# Checkpointing Tests
# =============================================================================


class TestDualOptimizerModuleCheckpointing:
    """Tests for checkpoint save/load."""

    def test_checkpoint_roundtrip(self, tmp_path):
        """Test that checkpoint save/load works correctly."""
        model = SimpleModel()
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
            matrix_target_modules=["attn"],
        )

        # Train for a few steps
        trainer = L.Trainer(
            accelerator="cpu",
            max_steps=4,
            default_root_dir=str(tmp_path),
            enable_checkpointing=True,
            logger=False,
            enable_progress_bar=False,
        )

        dataloader = create_dummy_dataloader(batch_size=2, num_batches=10)
        trainer.fit(module, dataloader)

        # Save checkpoint
        checkpoint = {
            "optimizer_step_count": module._optimizer_step_count,
            "matrix_scheduler_state": (
                module._matrix_scheduler.state_dict()
                if module._matrix_scheduler
                else None
            ),
            "vector_scheduler_state": (
                module._vector_scheduler.state_dict()
                if module._vector_scheduler
                else None
            ),
        }

        # Create new module and load checkpoint
        new_module = DualOptimizerModule(
            model=SimpleModel(),
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
            matrix_target_modules=["attn"],
        )

        new_module.on_load_checkpoint(checkpoint)

        assert new_module._optimizer_step_count == 4
        assert new_module._pending_scheduler_states is not None


# =============================================================================
# GPU Tests (Optional)
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDualOptimizerModuleGPU:
    """Integration tests requiring CUDA."""

    def test_training_step_gpu(self):
        """Test that training_step executes on GPU without error."""
        model = SimpleModel().cuda()
        opt_config, schedule_config = create_simple_configs()

        module = DualOptimizerModule(
            model=model,
            matrix_optimizer_config=opt_config,
            vector_optimizer_config=opt_config,
            matrix_schedule_config=schedule_config,
            vector_schedule_config=schedule_config,
            matrix_target_modules=["attn", "mlp"],
        ).cuda()

        trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            max_steps=4,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        dataloader = create_dummy_dataloader(batch_size=2, num_batches=10)

        trainer.fit(module, dataloader)

        assert module._optimizer_step_count == 4
