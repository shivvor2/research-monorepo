"""Tests for training configuration dataclasses."""

import math

import pytest
import torch
from torch.optim import AdamW

from research_lib.training.configs import (
    CustomScheduleConfig,
    MomentumScheduleConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    default_adam_config,
    default_muon_config,
)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_valid_config(self):
        """Test creating valid TrainingConfig."""
        config = TrainingConfig(
            total_steps=1000,
            grad_accum_steps=4,
            gradient_clip_val=1.0,
        )
        assert config.total_steps == 1000
        assert config.grad_accum_steps == 4
        assert config.gradient_clip_val == 1.0

    def test_defaults(self):
        """Test TrainingConfig defaults."""
        config = TrainingConfig(total_steps=1000)
        assert config.grad_accum_steps == 1
        assert config.gradient_clip_val == 1.0

    def test_invalid_total_steps(self):
        """Test that zero/negative total_steps raises ValueError."""
        with pytest.raises(ValueError, match="total_steps must be positive"):
            TrainingConfig(total_steps=0)

        with pytest.raises(ValueError, match="total_steps must be positive"):
            TrainingConfig(total_steps=-10)

    def test_invalid_grad_accum(self):
        """Test that zero/negative grad_accum raises ValueError."""
        with pytest.raises(ValueError, match="grad_accum_steps must be positive"):
            TrainingConfig(total_steps=1000, grad_accum_steps=0)

    def test_invalid_gradient_clip(self):
        """Test that negative gradient_clip raises ValueError."""
        with pytest.raises(ValueError, match="gradient_clip_val must be non-negative"):
            TrainingConfig(total_steps=1000, gradient_clip_val=-0.5)


class TestCustomScheduleConfig:
    """Tests for CustomScheduleConfig dataclass."""

    def test_get_value_warmup_phase(self):
        """Test value computation during warmup."""
        schedule = CustomScheduleConfig(
            param_name="momentum",
            warmup_steps=100,
            cooldown_steps=50,
            min_value=0.85,
            max_value=0.95,
        )

        # At step 0
        assert schedule.get_value(0, 1000) == 0.85

        # At step 50 (midway through warmup)
        assert schedule.get_value(50, 1000) == pytest.approx(0.90, rel=1e-6)

        # At step 100 (end of warmup)
        assert schedule.get_value(100, 1000) == pytest.approx(0.95, rel=1e-6)

    def test_get_value_stable_phase(self):
        """Test value during stable phase."""
        schedule = CustomScheduleConfig(
            param_name="momentum",
            warmup_steps=100,
            cooldown_steps=50,
            min_value=0.85,
            max_value=0.95,
        )

        # During stable phase
        assert schedule.get_value(500, 1000) == 0.95
        assert schedule.get_value(900, 1000) == 0.95

    def test_get_value_cooldown_phase(self):
        """Test value during cooldown."""
        schedule = CustomScheduleConfig(
            param_name="momentum",
            warmup_steps=100,
            cooldown_steps=50,
            min_value=0.85,
            max_value=0.95,
        )

        # At start of cooldown (step 950)
        assert schedule.get_value(950, 1000) == pytest.approx(0.95, rel=1e-6)

        # Midway through cooldown (step 975)
        assert schedule.get_value(975, 1000) == pytest.approx(0.90, rel=1e-6)

        # At end of cooldown (step 999)
        assert schedule.get_value(999, 1000) == pytest.approx(0.852, rel=1e-2)

    def test_no_warmup(self):
        """Test schedule with no warmup."""
        schedule = CustomScheduleConfig(
            param_name="momentum",
            warmup_steps=0,
            cooldown_steps=50,
            min_value=0.85,
            max_value=0.95,
        )

        # Should start at max_value immediately
        assert schedule.get_value(0, 1000) == 0.95

    def test_no_cooldown(self):
        """Test schedule with no cooldown."""
        schedule = CustomScheduleConfig(
            param_name="momentum",
            warmup_steps=100,
            cooldown_steps=0,
            min_value=0.85,
            max_value=0.95,
        )

        # Should stay at max_value until the end
        assert schedule.get_value(999, 1000) == 0.95


class TestMomentumScheduleConfig:
    """Tests for MomentumScheduleConfig dataclass."""

    def test_to_custom_schedule(self):
        """Test conversion to CustomScheduleConfig."""
        momentum = MomentumScheduleConfig(
            warmup_steps=300,
            cooldown_steps=50,
            min_value=0.85,
            max_value=0.95,
        )

        custom = momentum.to_custom_schedule()

        assert custom.param_name == "momentum"
        assert custom.warmup_steps == 300
        assert custom.cooldown_steps == 50
        assert custom.min_value == 0.85
        assert custom.max_value == 0.95
        assert custom.param_index is None


class TestSchedulerConfig:
    """Tests for SchedulerConfig dataclass."""

    def test_valid_config(self):
        """Test creating valid SchedulerConfig."""
        config = SchedulerConfig(
            warmup_steps=100,
            cooldown_frac=0.5,
            min_lr_ratio=0.1,
        )
        assert config.warmup_steps == 100
        assert config.cooldown_frac == 0.5
        assert config.min_lr_ratio == 0.1

    def test_invalid_cooldown_frac(self):
        """Test that out-of-range cooldown_frac raises ValueError."""
        with pytest.raises(ValueError, match="cooldown_frac must be in"):
            SchedulerConfig(cooldown_frac=1.5)

        with pytest.raises(ValueError, match="cooldown_frac must be in"):
            SchedulerConfig(cooldown_frac=-0.1)

    def test_build_lr_lambda_warmup(self):
        """Test LR lambda during warmup phase."""
        config = SchedulerConfig(warmup_steps=100, cooldown_frac=0.5, min_lr_ratio=0.1)
        lr_lambda = config.build_lr_lambda(total_steps=1000)

        # At step 0
        assert lr_lambda(0) == 0.0

        # At step 50
        assert lr_lambda(50) == pytest.approx(0.5, rel=1e-6)

        # At step 100
        assert lr_lambda(100) == pytest.approx(1.0, rel=1e-6)

    def test_build_lr_lambda_stable(self):
        """Test LR lambda during stable phase."""
        config = SchedulerConfig(warmup_steps=100, cooldown_frac=0.5, min_lr_ratio=0.1)
        lr_lambda = config.build_lr_lambda(total_steps=1000)

        # During stable phase
        assert lr_lambda(200) == 1.0
        assert lr_lambda(400) == 1.0

    def test_build_lr_lambda_cooldown(self):
        """Test LR lambda during cooldown (cosine decay)."""
        config = SchedulerConfig(warmup_steps=100, cooldown_frac=0.5, min_lr_ratio=0.1)
        lr_lambda = config.build_lr_lambda(total_steps=1000)

        # Cooldown starts at step 500
        assert lr_lambda(500) == pytest.approx(1.0, rel=1e-6)

        # At end
        assert lr_lambda(1000) == pytest.approx(0.1, rel=1e-6)

    def test_get_all_custom_schedules(self):
        """Test aggregation of all custom schedules."""
        custom1 = CustomScheduleConfig(
            param_name="eps", min_value=1e-10, max_value=1e-8
        )
        custom2 = CustomScheduleConfig(
            param_name="weight_decay", min_value=0.0, max_value=0.1
        )
        momentum = MomentumScheduleConfig()

        config = SchedulerConfig(
            momentum_schedule=momentum,
            custom_schedules=[custom1, custom2],
        )

        all_schedules = config.get_all_custom_schedules()

        # Should have momentum + 2 custom = 3 schedules
        assert len(all_schedules) == 3
        assert all_schedules[0].param_name == "momentum"  # From momentum_schedule
        assert all_schedules[1].param_name == "eps"
        assert all_schedules[2].param_name == "weight_decay"


class TestOptimizerConfig:
    """Tests for OptimizerConfig dataclass."""

    def test_missing_lr_raises(self):
        """Test that missing lr in kwargs raises ValueError."""
        with pytest.raises(ValueError, match="must include 'lr'"):
            OptimizerConfig(
                optimizer_class=AdamW,
                optimizer_kwargs={"weight_decay": 0.1},  # No lr!
            )

    def test_build_optimizer(self):
        """Test building an optimizer from config."""
        config = OptimizerConfig(
            optimizer_class=AdamW,
            optimizer_kwargs={"lr": 0.001, "weight_decay": 0.1},
        )

        model = torch.nn.Linear(10, 10)
        opt = config.build_optimizer(model.parameters())

        assert isinstance(opt, AdamW)
        assert opt.param_groups[0]["lr"] == 0.001
        assert opt.param_groups[0]["weight_decay"] == 0.1

    def test_build_scheduler(self):
        """Test building a scheduler from config."""
        config = OptimizerConfig(
            optimizer_class=AdamW,
            optimizer_kwargs={"lr": 0.001},
            scheduler_config=SchedulerConfig(warmup_steps=10),
        )
        training_config = TrainingConfig(total_steps=100)

        model = torch.nn.Linear(10, 10)
        opt = config.build_optimizer(model.parameters())
        sch = config.build_scheduler(opt, training_config)

        # Step through warmup
        for _ in range(10):
            sch.step()

        # LR should be at base value after warmup
        assert opt.param_groups[0]["lr"] == pytest.approx(0.001, rel=1e-6)


class TestFactoryFunctions:
    """Tests for default config factory functions."""

    def test_default_muon_config(self):
        """Test default_muon_config creates valid config."""
        config = default_muon_config()

        assert config.optimizer_class == torch.optim.Muon
        assert config.optimizer_kwargs["lr"] == 0.02
        assert config.optimizer_kwargs["momentum"] == 0.95
        assert config.scheduler_config.momentum_schedule is not None

    def test_default_muon_config_custom_values(self):
        """Test default_muon_config with custom values."""
        config = default_muon_config(lr=0.05, momentum=0.9)

        assert config.optimizer_kwargs["lr"] == 0.05
        assert config.optimizer_kwargs["momentum"] == 0.9

    def test_default_adam_config(self):
        """Test default_adam_config creates valid config."""
        config = default_adam_config()

        assert config.optimizer_class == torch.optim.AdamW
        assert config.optimizer_kwargs["lr"] == 0.001
        assert config.optimizer_kwargs["betas"] == (0.9, 0.95)
        assert config.scheduler_config.momentum_schedule is None

    def test_default_adam_config_custom_values(self):
        """Test default_adam_config with custom values."""
        config = default_adam_config(lr=0.0005, betas=(0.9, 0.999))

        assert config.optimizer_kwargs["lr"] == 0.0005
        assert config.optimizer_kwargs["betas"] == (0.9, 0.999)
