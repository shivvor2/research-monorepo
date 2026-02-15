"""Tests for preset configurations."""

import pytest
import torch

from research_lib.training.configs import (
    OptimizerConfig,
    ScheduleConfig,
    build_optimizer,
)
from research_lib.training.presets import (
    default_adamw_config,
    default_muon_config,
    default_sgd_config,
)
from research_lib.training.scheduling import ParamScheduler

# =============================================================================
# default_adamw_config Tests
# =============================================================================


class TestDefaultAdamWConfig:
    """Tests for default_adamw_config preset."""

    def test_returns_tuple(self):
        """Should return (OptimizerConfig, ScheduleConfig) tuple."""
        result = default_adamw_config()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], OptimizerConfig)
        assert isinstance(result[1], ScheduleConfig)

    def test_optimizer_config_correct(self):
        """Should configure AdamW with correct defaults."""
        opt_config, _ = default_adamw_config()
        assert opt_config.optimizer_class == torch.optim.AdamW
        assert opt_config.optimizer_kwargs["lr"] == 3e-4
        assert opt_config.optimizer_kwargs["betas"] == (0.9, 0.95)
        assert opt_config.optimizer_kwargs["weight_decay"] == 0.1

    def test_schedule_config_has_lr_schedule(self):
        """Should include LR schedule."""
        _, schedule_config = default_adamw_config()
        assert len(schedule_config.global_schedules) == 1
        assert schedule_config.global_schedules[0].param_name == "lr"

    def test_custom_lr(self):
        """Should accept custom learning rate."""
        opt_config, schedule_config = default_adamw_config(lr=1e-4)
        assert opt_config.optimizer_kwargs["lr"] == 1e-4
        # Schedule should also use the custom lr as max_value
        schedule = schedule_config.global_schedules[0]
        assert schedule.max_value == 1e-4

    def test_custom_betas(self):
        """Should accept custom betas."""
        opt_config, _ = default_adamw_config(betas=(0.9, 0.999))
        assert opt_config.optimizer_kwargs["betas"] == (0.9, 0.999)

    def test_custom_weight_decay(self):
        """Should accept custom weight decay."""
        opt_config, _ = default_adamw_config(weight_decay=0.05)
        assert opt_config.optimizer_kwargs["weight_decay"] == 0.05

    def test_custom_warmup_frac(self):
        """Should accept custom warmup fraction."""
        _, schedule_config = default_adamw_config(warmup_frac=0.2)
        schedule = schedule_config.global_schedules[0]
        assert schedule.warmup_frac == 0.2

    def test_builds_working_optimizer(self):
        """Should build a working optimizer."""
        opt_config, _ = default_adamw_config()
        model = torch.nn.Linear(10, 10)
        optimizer = build_optimizer(opt_config, model.parameters())
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_builds_working_scheduler(self):
        """Should build a working scheduler."""
        opt_config, schedule_config = default_adamw_config()
        model = torch.nn.Linear(10, 10)
        optimizer = build_optimizer(opt_config, model.parameters())

        # Use the new ParamScheduler API directly
        scheduler = ParamScheduler(
            optimizer=optimizer,
            global_schedules=schedule_config.global_schedules,
            total_steps=1000,
            group_overrides=schedule_config.group_overrides,
        )

        # Should not raise
        scheduler.step()
        assert scheduler.get_current_step() == 1


# =============================================================================
# default_muon_config Tests
# =============================================================================


class TestDefaultMuonConfig:
    """Tests for default_muon_config preset."""

    @pytest.fixture
    def has_muon(self):
        """Check if Muon is available."""
        return hasattr(torch.optim, "Muon")

    def test_returns_tuple(self, has_muon):
        """Should return (OptimizerConfig, ScheduleConfig) tuple."""
        if not has_muon:
            pytest.skip("torch.optim.Muon not available")

        result = default_muon_config()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], OptimizerConfig)
        assert isinstance(result[1], ScheduleConfig)

    def test_raises_without_muon(self, has_muon):
        """Should raise ImportError if Muon not available."""
        if has_muon:
            pytest.skip("Muon is available, cannot test import error")

        with pytest.raises(ImportError, match="Muon"):
            default_muon_config()

    def test_has_lr_and_momentum_schedules(self, has_muon):
        """Should include both LR and momentum schedules."""
        if not has_muon:
            pytest.skip("torch.optim.Muon not available")

        _, schedule_config = default_muon_config()
        param_names = [s.param_name for s in schedule_config.global_schedules]
        assert "lr" in param_names
        assert "momentum" in param_names

    def test_optimizer_config_correct(self, has_muon):
        """Should configure Muon with correct defaults."""
        if not has_muon:
            pytest.skip("torch.optim.Muon not available")

        opt_config, _ = default_muon_config()
        assert opt_config.optimizer_class == torch.optim.Muon
        assert opt_config.optimizer_kwargs["lr"] == 0.02
        assert opt_config.optimizer_kwargs["momentum"] == 0.95

    def test_custom_values(self, has_muon):
        """Should accept custom values."""
        if not has_muon:
            pytest.skip("torch.optim.Muon not available")

        opt_config, schedule_config = default_muon_config(lr=0.01, momentum=0.9)
        assert opt_config.optimizer_kwargs["lr"] == 0.01
        assert opt_config.optimizer_kwargs["momentum"] == 0.9


# =============================================================================
# default_sgd_config Tests
# =============================================================================


class TestDefaultSGDConfig:
    """Tests for default_sgd_config preset."""

    def test_returns_tuple(self):
        """Should return (OptimizerConfig, ScheduleConfig) tuple."""
        result = default_sgd_config()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], OptimizerConfig)
        assert isinstance(result[1], ScheduleConfig)

    def test_optimizer_config_correct(self):
        """Should configure SGD with correct defaults."""
        opt_config, _ = default_sgd_config()
        assert opt_config.optimizer_class == torch.optim.SGD
        assert opt_config.optimizer_kwargs["lr"] == 0.1
        assert opt_config.optimizer_kwargs["momentum"] == 0.9
        assert opt_config.optimizer_kwargs["weight_decay"] == 1e-4

    def test_schedule_config_has_lr_schedule(self):
        """Should include LR schedule."""
        _, schedule_config = default_sgd_config()
        assert len(schedule_config.global_schedules) == 1
        assert schedule_config.global_schedules[0].param_name == "lr"

    def test_default_no_stable_phase(self):
        """Should default to no stable phase (warmup then decay).

        With warmup_frac=0.05 and stable_frac=0.0, cooldown_frac should be 0.95.
        This means the schedule goes: 5% warmup, 0% stable, 95% cooldown.
        """
        _, schedule_config = default_sgd_config()
        schedule = schedule_config.global_schedules[0]
        # Check warmup_frac is set correctly
        assert schedule.warmup_frac == 0.05
        # Check cooldown_frac is the remainder (1 - 0.05 - 0 = 0.95)
        assert schedule.cooldown_frac == 0.95

    def test_custom_values(self):
        """Should accept custom values."""
        opt_config, _ = default_sgd_config(lr=0.01, momentum=0.95, weight_decay=1e-3)
        assert opt_config.optimizer_kwargs["lr"] == 0.01
        assert opt_config.optimizer_kwargs["momentum"] == 0.95
        assert opt_config.optimizer_kwargs["weight_decay"] == 1e-3

    def test_builds_working_optimizer(self):
        """Should build a working optimizer."""
        opt_config, _ = default_sgd_config()
        model = torch.nn.Linear(10, 10)
        optimizer = build_optimizer(opt_config, model.parameters())
        assert isinstance(optimizer, torch.optim.SGD)
