"""Tests for training configuration dataclasses."""

import pytest
import torch
from torch.optim import SGD, AdamW

from research_lib.training.configs import (
    OptimizerConfig,
    ParamGroupConfig,
    TrainingConfig,
    default_adamw_config,
    default_muon_config,
)
from research_lib.training.scheduling import WarmupStableDecaySchedule


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


class TestParamGroupConfig:
    """Tests for ParamGroupConfig dataclass."""

    def test_valid_config(self):
        """Test creating valid ParamGroupConfig."""
        schedule = WarmupStableDecaySchedule(param_name="lr", warmup_steps=100)
        config = ParamGroupConfig(
            group_index=0,
            schedules=[schedule],
            param_group_kwargs={"lr": 0.001},
        )
        assert config.group_index == 0
        assert len(config.schedules) == 1
        assert config.param_group_kwargs == {"lr": 0.001}

    def test_defaults(self):
        """Test ParamGroupConfig defaults."""
        config = ParamGroupConfig(group_index=0)
        assert config.schedules == []
        assert config.param_group_kwargs is None

    def test_invalid_group_index(self):
        """Test that negative group_index raises ValueError."""
        with pytest.raises(ValueError, match="group_index must be non-negative"):
            ParamGroupConfig(group_index=-1)


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

    def test_build_optimizer_with_param_group_config(self):
        """Test building optimizer with ParamGroupConfig overrides."""
        config = OptimizerConfig(
            optimizer_class=SGD,
            optimizer_kwargs={"lr": 0.1},
            param_group_configs=[
                ParamGroupConfig(
                    group_index=0,
                    param_group_kwargs={"lr": 0.01},  # Override
                ),
            ],
        )

        model = torch.nn.Linear(10, 10)
        opt = config.build_optimizer(model.parameters())

        # lr should be overridden
        assert opt.param_groups[0]["lr"] == 0.01

    def test_build_optimizer_invalid_group_index(self):
        """Test that invalid group_index in ParamGroupConfig raises IndexError."""
        config = OptimizerConfig(
            optimizer_class=SGD,
            optimizer_kwargs={"lr": 0.1},
            param_group_configs=[
                ParamGroupConfig(group_index=5),  # Invalid
            ],
        )

        model = torch.nn.Linear(10, 10)
        with pytest.raises(IndexError, match="exceeds optimizer param group count"):
            config.build_optimizer(model.parameters())

    def test_get_schedules_for_group_global(self):
        """Test get_schedules_for_group returns global schedules by default."""
        global_schedule = WarmupStableDecaySchedule(param_name="lr")
        config = OptimizerConfig(
            optimizer_class=SGD,
            optimizer_kwargs={"lr": 0.1},
            schedules=[global_schedule],
        )

        schedules = config.get_schedules_for_group(0)
        assert len(schedules) == 1
        assert schedules[0] is global_schedule

    def test_get_schedules_for_group_override(self):
        """Test get_schedules_for_group returns per-group schedules when present."""
        global_schedule = WarmupStableDecaySchedule(param_name="lr", warmup_steps=100)
        group_schedule = WarmupStableDecaySchedule(param_name="lr", warmup_steps=50)

        config = OptimizerConfig(
            optimizer_class=SGD,
            optimizer_kwargs={"lr": 0.1},
            schedules=[global_schedule],
            param_group_configs=[
                ParamGroupConfig(group_index=0, schedules=[group_schedule]),
            ],
        )

        schedules = config.get_schedules_for_group(0)
        assert len(schedules) == 1
        assert schedules[0] is group_schedule

        # Group 1 should return global schedules
        schedules = config.get_schedules_for_group(1)
        assert len(schedules) == 1
        assert schedules[0] is global_schedule


class TestFactoryFunctions:
    """Tests for default config factory functions."""

    def test_default_muon_config(self):
        """Test default_muon_config creates valid config."""
        config = default_muon_config()

        assert config.optimizer_class == torch.optim.Muon
        assert config.optimizer_kwargs["lr"] == 0.02
        assert config.optimizer_kwargs["momentum"] == 0.95
        assert len(config.schedules) == 2  # lr and momentum

    def test_default_muon_config_custom_values(self):
        """Test default_muon_config with custom values."""
        config = default_muon_config(lr=0.05, momentum=0.9)

        assert config.optimizer_kwargs["lr"] == 0.05
        assert config.optimizer_kwargs["momentum"] == 0.9

    def test_default_adam_config(self):
        """Test default_adam_config creates valid config."""
        config = default_adamw_config()

        assert config.optimizer_class == torch.optim.AdamW
        assert config.optimizer_kwargs["lr"] == 0.001
        assert config.optimizer_kwargs["betas"] == (0.9, 0.95)
        assert len(config.schedules) == 1  # lr only

    def test_default_adam_config_custom_values(self):
        """Test default_adam_config with custom values."""
        config = default_adamw_config(lr=0.0005, betas=(0.9, 0.999))

        assert config.optimizer_kwargs["lr"] == 0.0005
        assert config.optimizer_kwargs["betas"] == (0.9, 0.999)
