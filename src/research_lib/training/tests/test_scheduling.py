"""Tests for scheduling utilities."""

import pytest
import torch
from torch.optim import SGD, AdamW

from research_lib.training.configs import (
    CustomScheduleConfig,
    MomentumScheduleConfig,
    SchedulerConfig,
)
from research_lib.training.scheduling import (
    get_current_lr,
    get_param_group_value,
    update_param_group_schedules,
    update_single_param_schedule,
)


class TestUpdateSingleParamSchedule:
    """Tests for update_single_param_schedule function."""

    def test_update_scalar_param(self):
        """Test updating a scalar param_group value."""
        model = torch.nn.Linear(10, 10)
        opt = SGD(model.parameters(), lr=0.01, momentum=0.9)

        schedule = CustomScheduleConfig(
            param_name="momentum",
            warmup_steps=10,
            cooldown_steps=10,
            min_value=0.8,
            max_value=0.95,
        )

        # Update at step 0 (start of warmup)
        update_single_param_schedule(opt, schedule, step=0, total_steps=100)
        assert opt.param_groups[0]["momentum"] == 0.8

        # Update at step 10 (end of warmup)
        update_single_param_schedule(opt, schedule, step=10, total_steps=100)
        assert opt.param_groups[0]["momentum"] == pytest.approx(0.95, rel=1e-6)

        # Update at step 95 (during cooldown)
        update_single_param_schedule(opt, schedule, step=95, total_steps=100)
        assert opt.param_groups[0]["momentum"] == pytest.approx(0.875, rel=1e-2)

    def test_update_tuple_param(self):
        """Test updating an indexed element of a tuple param."""
        model = torch.nn.Linear(10, 10)
        opt = AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999))

        # Schedule beta1 (index 0)
        schedule = CustomScheduleConfig(
            param_name="betas",
            param_index=0,
            warmup_steps=10,
            cooldown_steps=0,
            min_value=0.8,
            max_value=0.95,
        )

        # Update at step 0
        update_single_param_schedule(opt, schedule, step=0, total_steps=100)

        # beta1 should be updated, beta2 unchanged
        assert opt.param_groups[0]["betas"][0] == 0.8
        assert opt.param_groups[0]["betas"][1] == 0.999

        # Update at step 10
        update_single_param_schedule(opt, schedule, step=10, total_steps=100)
        assert opt.param_groups[0]["betas"][0] == pytest.approx(0.95, rel=1e-6)
        assert opt.param_groups[0]["betas"][1] == 0.999

    def test_nonexistent_param_is_added(self):
        """Test that scheduling a nonexistent param adds it silently."""
        model = torch.nn.Linear(10, 10)
        opt = AdamW(model.parameters(), lr=0.01)

        schedule = CustomScheduleConfig(
            param_name="custom_param",
            min_value=0.0,
            max_value=1.0,
        )

        # Should not raise
        update_single_param_schedule(opt, schedule, step=50, total_steps=100)

        # The param should now exist
        assert "custom_param" in opt.param_groups[0]
        assert opt.param_groups[0]["custom_param"] == 1.0


class TestUpdateParamGroupSchedules:
    """Tests for update_param_group_schedules function."""

    def test_updates_all_schedules(self):
        """Test that all custom schedules are updated."""
        model = torch.nn.Linear(10, 10)
        opt = SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Add a fake param
        opt.param_groups[0]["custom"] = 0.0

        scheduler_config = SchedulerConfig(
            momentum_schedule=MomentumScheduleConfig(
                warmup_steps=10,
                cooldown_steps=0,
                min_value=0.8,
                max_value=0.95,
            ),
            custom_schedules=[
                CustomScheduleConfig(
                    param_name="custom",
                    warmup_steps=10,
                    cooldown_steps=0,
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
        )

        # Update at step 5 (midway through warmup)
        update_param_group_schedules(opt, scheduler_config, step=5, total_steps=100)

        assert opt.param_groups[0]["momentum"] == pytest.approx(0.875, rel=1e-2)
        assert opt.param_groups[0]["custom"] == pytest.approx(0.5, rel=1e-2)


class TestGetterFunctions:
    """Tests for getter utility functions."""

    def test_get_current_lr(self):
        """Test getting current LR from optimizer."""
        model = torch.nn.Linear(10, 10)
        opt = AdamW(model.parameters(), lr=0.001)

        assert get_current_lr(opt) == 0.001

    def test_get_param_group_value_scalar(self):
        """Test getting a scalar param_group value."""
        model = torch.nn.Linear(10, 10)
        opt = SGD(model.parameters(), lr=0.01, momentum=0.9)

        assert get_param_group_value(opt, "momentum") == 0.9
        assert get_param_group_value(opt, "lr") == 0.01

    def test_get_param_group_value_indexed(self):
        """Test getting an indexed element from a tuple param."""
        model = torch.nn.Linear(10, 10)
        opt = AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999))

        assert get_param_group_value(opt, "betas", param_index=0) == 0.9
        assert get_param_group_value(opt, "betas", param_index=1) == 0.999

    def test_get_param_group_value_nonexistent(self):
        """Test getting a nonexistent param returns None."""
        model = torch.nn.Linear(10, 10)
        opt = AdamW(model.parameters(), lr=0.01)

        assert get_param_group_value(opt, "nonexistent") is None
