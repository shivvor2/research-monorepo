"""Tests for configuration dataclasses."""

import pickle

import pytest
import torch

from research_lib.training.configs import (
    GradAccumSchedule,
    OptimizerConfig,
    ScheduleConfig,
    build_optimizer,
)
from research_lib.training.scheduling import ParamSchedule, WarmupStableDecaySchedule

# =============================================================================
# Helper Functions for Tests
# =============================================================================


def constant_schedule_fn(step: int, total_steps: int) -> float:
    """Simple constant schedule for testing."""
    return 0.1


def linear_decay_fn(step: int, total_steps: int) -> float:
    """Linear decay schedule for testing."""
    if total_steps <= 0:
        return 1.0
    return 1.0 - step / total_steps


# =============================================================================
# OptimizerConfig Tests
# =============================================================================


class TestOptimizerConfig:
    """Tests for OptimizerConfig dataclass."""

    def test_basic_creation(self):
        """Should create config with required fields."""
        config = OptimizerConfig(
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-3},
        )
        assert config.optimizer_class == torch.optim.AdamW
        assert config.optimizer_kwargs == {"lr": 1e-3}

    def test_default_kwargs(self):
        """Should default kwargs to empty dict."""
        config = OptimizerConfig(optimizer_class=torch.optim.SGD)
        assert config.optimizer_kwargs == {}

    def test_picklable(self):
        """Should be picklable for distributed training."""
        config = OptimizerConfig(
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-3, "betas": (0.9, 0.95)},
        )
        restored = pickle.loads(pickle.dumps(config))
        assert restored.optimizer_class == config.optimizer_class
        assert restored.optimizer_kwargs == config.optimizer_kwargs


# =============================================================================
# ScheduleConfig Tests
# =============================================================================


class TestScheduleConfig:
    """Tests for ScheduleConfig dataclass."""

    def test_basic_creation(self):
        """Should create config with schedules."""
        schedule = ParamSchedule("lr", constant_schedule_fn)
        config = ScheduleConfig(global_schedules=[schedule])
        assert len(config.global_schedules) == 1
        assert config.group_overrides == {}

    def test_default_empty(self):
        """Should default to empty schedules."""
        config = ScheduleConfig()
        assert config.global_schedules == []
        assert config.group_overrides == {}

    def test_with_group_overrides(self):
        """Should store group overrides."""
        schedule1 = ParamSchedule("lr", constant_schedule_fn)
        schedule2 = WarmupStableDecaySchedule(param_name="lr", max_value=0.01)

        config = ScheduleConfig(
            global_schedules=[schedule1],
            group_overrides={1: [schedule2]},
        )

        assert len(config.global_schedules) == 1
        assert 1 in config.group_overrides
        assert len(config.group_overrides[1]) == 1

    def test_multiple_schedules(self):
        """Should handle multiple schedules per group."""
        lr_schedule = WarmupStableDecaySchedule(param_name="lr", max_value=0.1)
        momentum_schedule = WarmupStableDecaySchedule(
            param_name="momentum", max_value=0.95
        )

        config = ScheduleConfig(
            global_schedules=[lr_schedule, momentum_schedule],
        )

        assert len(config.global_schedules) == 2
        param_names = [s.param_name for s in config.global_schedules]
        assert "lr" in param_names
        assert "momentum" in param_names


# =============================================================================
# GradAccumSchedule Tests
# =============================================================================


class TestGradAccumSchedule:
    """Tests for GradAccumSchedule dataclass."""

    def test_basic_creation(self):
        """Should create schedule with dict."""
        schedule = GradAccumSchedule({0: 1, 1000: 2, 5000: 4})
        assert schedule.get_accum(0) == 1
        assert schedule.get_accum(999) == 1
        assert schedule.get_accum(1000) == 2
        assert schedule.get_accum(4999) == 2
        assert schedule.get_accum(5000) == 4
        assert schedule.get_accum(10000) == 4

    def test_constant_schedule(self):
        """Should work for constant accumulation."""
        schedule = GradAccumSchedule({0: 4})
        assert schedule.get_accum(0) == 4
        assert schedule.get_accum(100) == 4
        assert schedule.get_accum(10000) == 4

    def test_auto_adds_step_zero(self):
        """Should add step 0 with accum=1 if not specified."""
        schedule = GradAccumSchedule({1000: 4})
        assert schedule.get_accum(0) == 1
        assert schedule.get_accum(999) == 1
        assert schedule.get_accum(1000) == 4

    def test_empty_schedule_raises(self):
        """Should raise on empty schedule."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GradAccumSchedule({})

    def test_negative_step_raises(self):
        """Should raise on negative step."""
        with pytest.raises(ValueError, match="non-negative int"):
            GradAccumSchedule({-1: 4})

    def test_zero_accum_raises(self):
        """Should raise on accumulation < 1."""
        with pytest.raises(ValueError, match=">= 1"):
            GradAccumSchedule({0: 0})

    def test_negative_accum_raises(self):
        """Should raise on negative accumulation."""
        with pytest.raises(ValueError, match=">= 1"):
            GradAccumSchedule({0: -1})

    def test_picklable(self):
        """Should be picklable for checkpointing."""
        schedule = GradAccumSchedule({0: 1, 1000: 4})
        restored = pickle.loads(pickle.dumps(schedule))
        assert restored.get_accum(0) == 1
        assert restored.get_accum(1000) == 4

    def test_boundary_values(self):
        """Should handle boundary values correctly."""
        schedule = GradAccumSchedule({0: 1, 100: 2, 200: 4})

        # At exact boundaries
        assert schedule.get_accum(100) == 2
        assert schedule.get_accum(200) == 4

        # Just before boundaries
        assert schedule.get_accum(99) == 1
        assert schedule.get_accum(199) == 2

        # Just after boundaries
        assert schedule.get_accum(101) == 2
        assert schedule.get_accum(201) == 4


# =============================================================================
# build_optimizer Tests
# =============================================================================


class TestBuildOptimizer:
    """Tests for build_optimizer function."""

    def test_builds_adamw(self):
        """Should build AdamW optimizer."""
        config = OptimizerConfig(
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-3},
        )
        model = torch.nn.Linear(10, 10)
        optimizer = build_optimizer(config, model.parameters())

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 1e-3

    def test_builds_sgd(self):
        """Should build SGD optimizer."""
        config = OptimizerConfig(
            optimizer_class=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1, "momentum": 0.9},
        )
        model = torch.nn.Linear(10, 10)
        optimizer = build_optimizer(config, model.parameters())

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults["lr"] == 0.1
        assert optimizer.defaults["momentum"] == 0.9

    def test_with_param_groups(self):
        """Should work with param groups."""
        config = OptimizerConfig(
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-3},
        )
        model = torch.nn.Linear(10, 10)
        param_groups = [
            {"params": [model.weight], "lr": 1e-2},
            {"params": [model.bias], "lr": 1e-4},
        ]
        optimizer = build_optimizer(config, param_groups)

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 1e-2
        assert optimizer.param_groups[1]["lr"] == 1e-4

    def test_builds_with_weight_decay(self):
        """Should pass weight_decay to optimizer."""
        config = OptimizerConfig(
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.1},
        )
        model = torch.nn.Linear(10, 10)
        optimizer = build_optimizer(config, model.parameters())

        assert optimizer.defaults["weight_decay"] == 0.1
