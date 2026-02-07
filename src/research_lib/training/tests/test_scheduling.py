"""Tests for the new scheduling system."""

import math
import pickle

import pytest
import torch
from torch.optim import SGD, AdamW

from research_lib.training.configs import (
    OptimizerConfig,
    ParamGroupConfig,
    update_optimizer_schedules,
)
from research_lib.training.scheduling import ParamSchedule, WarmupStableDecaySchedule
from research_lib.training.scheduling import wrappers as sw
from research_lib.training.scheduling.utils import (
    apply_schedule_to_all_groups,
    apply_schedule_to_param_group,
    get_current_lr,
    get_param_value,
    get_param_values,
)
from research_lib.training.scheduling.validation import (
    check_finite,
    check_in_range,
    check_in_set,
    check_monotonic_decreasing,
    check_monotonic_increasing,
    check_monotonic_non_decreasing,
    check_monotonic_non_increasing,
    check_non_negative,
    check_positive,
    validate_schedule,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def simple_linear_decay(step: int, total_steps: int) -> float:
    """Simple linear decay from 1.0 to 0.0."""
    return 1.0 - (step / total_steps)


class ParameterizedDecay:
    """Picklable callable class for testing."""

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, step: int, total_steps: int) -> float:
        progress = step / total_steps
        return self.max_val - (self.max_val - self.min_val) * progress


# =============================================================================
# ParamSchedule Tests
# =============================================================================


class TestParamSchedule:
    """Tests for ParamSchedule primitive."""

    def test_basic_construction(self):
        """Test basic ParamSchedule construction."""
        schedule = ParamSchedule(param_name="lr", schedule_fn=simple_linear_decay)
        assert schedule.param_name == "lr"
        assert callable(schedule.schedule_fn)

    def test_call(self):
        """Test calling the schedule."""
        schedule = ParamSchedule(param_name="lr", schedule_fn=simple_linear_decay)

        assert schedule(0, 100) == 1.0
        assert schedule(50, 100) == pytest.approx(0.5)
        assert schedule(100, 100) == pytest.approx(0.0)

    def test_callable_class(self):
        """Test using a callable class as schedule_fn."""
        schedule = ParamSchedule(
            param_name="lr",
            schedule_fn=ParameterizedDecay(0.1, 1.0),
        )

        assert schedule(0, 100) == 1.0
        assert schedule(100, 100) == pytest.approx(0.1)

    def test_not_callable_raises(self):
        """Test that non-callable schedule_fn raises TypeError."""
        with pytest.raises(TypeError, match="must be callable"):
            ParamSchedule(param_name="lr", schedule_fn="not a function")

    def test_wrong_signature_raises(self):
        """Test that wrong signature raises ValueError."""

        def wrong_sig(x):
            return x

        with pytest.raises(ValueError, match="at least 2 positional arguments"):
            ParamSchedule(param_name="lr", schedule_fn=wrong_sig)

    def test_lambda_not_picklable_raises(self):
        """Test that lambda (not picklable) raises ValueError."""
        with pytest.raises(ValueError, match="picklable"):
            ParamSchedule(param_name="lr", schedule_fn=lambda s, t: s / t)

    def test_picklable(self):
        """Test that ParamSchedule with valid fn is picklable."""
        schedule = ParamSchedule(param_name="lr", schedule_fn=simple_linear_decay)
        pickled = pickle.dumps(schedule)
        unpickled = pickle.loads(pickled)

        assert unpickled(50, 100) == pytest.approx(0.5)


# =============================================================================
# WarmupStableDecaySchedule Tests
# =============================================================================


class TestWarmupStableDecaySchedule:
    """Tests for WarmupStableDecaySchedule preset."""

    def test_default_values(self):
        """Test default construction."""
        schedule = WarmupStableDecaySchedule()
        assert schedule.param_name == "lr"
        assert schedule.warmup_steps == 0
        assert schedule.cooldown_steps == 0
        assert schedule.min_value == 0.0
        assert schedule.max_value == 1.0

    def test_warmup_phase(self):
        """Test warmup phase."""
        schedule = WarmupStableDecaySchedule(
            warmup_steps=100,
            min_value=0.0,
            max_value=1.0,
        )

        # Start of warmup
        assert schedule(0, 1000) == 0.0
        # Mid warmup
        assert schedule(50, 1000) == pytest.approx(0.5)
        # End of warmup
        assert schedule(100, 1000) == pytest.approx(1.0)

    def test_stable_phase(self):
        """Test stable phase."""
        schedule = WarmupStableDecaySchedule(
            warmup_steps=100,
            cooldown_steps=100,
            min_value=0.0,
            max_value=1.0,
        )

        # During stable
        assert schedule(200, 1000) == 1.0
        assert schedule(500, 1000) == 1.0
        assert schedule(899, 1000) == 1.0

    def test_cooldown_phase_linear(self):
        """Test cooldown phase with linear decay."""
        schedule = WarmupStableDecaySchedule(
            warmup_steps=0,
            cooldown_steps=100,
            min_value=0.0,
            max_value=1.0,
            decay_type="linear",
        )

        # Start of cooldown (step 900)
        assert schedule(900, 1000) == pytest.approx(1.0)
        # Mid cooldown
        assert schedule(950, 1000) == pytest.approx(0.5)
        # End of cooldown
        assert schedule(999, 1000) == pytest.approx(0.01, abs=0.02)

    def test_cooldown_phase_cosine(self):
        """Test cooldown phase with cosine decay."""
        schedule = WarmupStableDecaySchedule(
            warmup_steps=0,
            cooldown_frac=0.5,
            min_value=0.0,
            max_value=1.0,
            decay_type="cosine",
        )

        # Start of cooldown (step 500)
        assert schedule(500, 1000) == pytest.approx(1.0)
        # End of cooldown
        assert schedule(1000, 1000) == pytest.approx(0.0, abs=0.01)

    def test_warmup_frac(self):
        """Test warmup_frac parameter."""
        schedule = WarmupStableDecaySchedule(
            warmup_frac=0.1,
            min_value=0.0,
            max_value=1.0,
        )

        # 10% of 1000 = 100 warmup steps
        assert schedule(0, 1000) == 0.0
        assert schedule(50, 1000) == pytest.approx(0.5)
        assert schedule(100, 1000) == pytest.approx(1.0)

    def test_cooldown_frac(self):
        """Test cooldown_frac parameter."""
        schedule = WarmupStableDecaySchedule(
            cooldown_frac=0.5,
            min_value=0.0,
            max_value=1.0,
            decay_type="linear",
        )

        # 50% of 1000 = 500 cooldown steps starting at 500
        assert schedule(500, 1000) == pytest.approx(1.0)
        assert schedule(750, 1000) == pytest.approx(0.5)
        assert schedule(999, 1000) == pytest.approx(0.002, abs=0.01)

    def test_mutual_exclusivity_warmup(self):
        """Test warmup_steps and warmup_frac are mutually exclusive."""
        with pytest.raises(ValueError, match="warmup_steps OR warmup_frac"):
            WarmupStableDecaySchedule(warmup_steps=100, warmup_frac=0.1)

    def test_mutual_exclusivity_cooldown(self):
        """Test cooldown_steps and cooldown_frac are mutually exclusive."""
        with pytest.raises(ValueError, match="cooldown_steps OR cooldown_frac"):
            WarmupStableDecaySchedule(cooldown_steps=100, cooldown_frac=0.1)

    def test_invalid_warmup_type(self):
        """Test invalid warmup_type raises."""
        with pytest.raises(ValueError, match="warmup_type must be one of"):
            WarmupStableDecaySchedule(warmup_type="invalid")

    def test_invalid_decay_type(self):
        """Test invalid decay_type raises."""
        with pytest.raises(ValueError, match="decay_type must be one of"):
            WarmupStableDecaySchedule(decay_type="exponential")

    def test_picklable(self):
        """Test that WarmupStableDecaySchedule is picklable."""
        schedule = WarmupStableDecaySchedule(
            warmup_steps=100,
            cooldown_frac=0.5,
            min_value=0.1,
            max_value=1.0,
        )

        pickled = pickle.dumps(schedule)
        unpickled = pickle.loads(pickled)

        assert unpickled(50, 1000) == schedule(50, 1000)


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestApplyScheduleToParamGroup:
    """Tests for apply_schedule_to_param_group."""

    def test_apply_to_single_group(self):
        """Test applying schedule to a single param group."""
        model = torch.nn.Linear(10, 10)
        opt = SGD(model.parameters(), lr=0.1)
        schedule = ParamSchedule(param_name="lr", schedule_fn=simple_linear_decay)

        apply_schedule_to_param_group(
            opt, schedule, group_idx=0, step=50, total_steps=100
        )

        assert opt.param_groups[0]["lr"] == pytest.approx(0.5)

    def test_apply_momentum(self):
        """Test applying schedule to momentum."""
        model = torch.nn.Linear(10, 10)
        opt = SGD(model.parameters(), lr=0.1, momentum=0.9)
        schedule = WarmupStableDecaySchedule(
            param_name="momentum",
            warmup_steps=10,
            min_value=0.8,
            max_value=0.95,
        )

        apply_schedule_to_param_group(
            opt, schedule, group_idx=0, step=0, total_steps=100
        )
        assert opt.param_groups[0]["momentum"] == 0.8

        apply_schedule_to_param_group(
            opt, schedule, group_idx=0, step=10, total_steps=100
        )
        assert opt.param_groups[0]["momentum"] == pytest.approx(0.95)

    def test_invalid_group_idx(self):
        """Test that invalid group_idx raises IndexError."""
        model = torch.nn.Linear(10, 10)
        opt = SGD(model.parameters(), lr=0.1)
        schedule = ParamSchedule(param_name="lr", schedule_fn=simple_linear_decay)

        with pytest.raises(IndexError, match="out of range"):
            apply_schedule_to_param_group(
                opt, schedule, group_idx=5, step=0, total_steps=100
            )


class TestApplyScheduleToAllGroups:
    """Tests for apply_schedule_to_all_groups."""

    def test_apply_to_all(self):
        """Test applying schedule to all param groups."""
        model = torch.nn.Linear(10, 10)
        # Create optimizer with 2 param groups
        opt = SGD(
            [
                {"params": [model.weight], "lr": 0.1},
                {"params": [model.bias], "lr": 0.2},
            ]
        )
        schedule = ParamSchedule(param_name="lr", schedule_fn=simple_linear_decay)

        apply_schedule_to_all_groups(opt, schedule, step=50, total_steps=100)

        assert opt.param_groups[0]["lr"] == pytest.approx(0.5)
        assert opt.param_groups[1]["lr"] == pytest.approx(0.5)


class TestUpdateOptimizerSchedules:
    """Tests for update_optimizer_schedules with OptimizerConfig."""

    def test_global_schedules(self):
        """Test global schedules apply to all groups."""
        model = torch.nn.Linear(10, 10)
        config = OptimizerConfig(
            optimizer_class=SGD,
            optimizer_kwargs={"lr": 0.1, "momentum": 0.9},
            schedules=[
                WarmupStableDecaySchedule(
                    param_name="momentum",
                    warmup_steps=10,
                    min_value=0.8,
                    max_value=0.95,
                ),
            ],
        )
        opt = config.build_optimizer(model.parameters())

        update_optimizer_schedules(opt, config, step=0, total_steps=100)
        assert opt.param_groups[0]["momentum"] == 0.8

    def test_per_group_overrides(self):
        """Test that ParamGroupConfig overrides global schedules."""
        model = torch.nn.Linear(10, 10)

        lr_schedule_fast = WarmupStableDecaySchedule(
            param_name="lr",
            warmup_steps=5,
            min_value=0.0,
            max_value=1.0,
        )
        lr_schedule_slow = WarmupStableDecaySchedule(
            param_name="lr",
            warmup_steps=20,
            min_value=0.0,
            max_value=1.0,
        )

        config = OptimizerConfig(
            optimizer_class=SGD,
            optimizer_kwargs={"lr": 0.1},
            schedules=[lr_schedule_slow],  # Global: slow warmup
            param_group_configs=[
                ParamGroupConfig(
                    group_index=0,
                    schedules=[lr_schedule_fast],  # Override: fast warmup
                ),
            ],
        )
        opt = config.build_optimizer(model.parameters())

        # At step 5, fast warmup should be complete (value=1.0)
        # slow warmup would only be at 0.25
        update_optimizer_schedules(opt, config, step=5, total_steps=100)
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0)


class TestGetterFunctions:
    """Tests for getter utility functions."""

    def test_get_param_value(self):
        """Test get_param_value."""
        model = torch.nn.Linear(10, 10)
        opt = SGD(model.parameters(), lr=0.1, momentum=0.9)

        assert get_param_value(opt, "lr") == 0.1
        assert get_param_value(opt, "momentum") == 0.9
        assert get_param_value(opt, "nonexistent") is None

    def test_get_param_value_invalid_index(self):
        """Test get_param_value with invalid group index."""
        model = torch.nn.Linear(10, 10)
        opt = SGD(model.parameters(), lr=0.1)

        with pytest.raises(IndexError):
            get_param_value(opt, "lr", group_idx=5)

    def test_get_param_values(self):
        """Test get_param_values for multiple groups."""
        model = torch.nn.Linear(10, 10)
        opt = SGD(
            [
                {"params": [model.weight], "lr": 0.1},
                {"params": [model.bias], "lr": 0.2},
            ]
        )

        values = get_param_values(opt, "lr", [0, 1])
        assert values == [0.1, 0.2]

    def test_get_current_lr(self):
        """Test get_current_lr."""
        model = torch.nn.Linear(10, 10)
        opt = AdamW(model.parameters(), lr=0.001)

        assert get_current_lr(opt) == 0.001


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateSchedule:
    """Tests for validate_schedule function."""

    def test_basic_validation(self):
        """Test basic validation passes for valid schedule."""
        schedule = WarmupStableDecaySchedule(
            warmup_steps=10,
            cooldown_steps=10,
            min_value=0.0,
            max_value=1.0,
        )

        # Should not raise
        validate_schedule(schedule, total_steps=100)

    def test_with_single_checks(self):
        """Test validation with single-value checks."""
        schedule = WarmupStableDecaySchedule(
            min_value=0.0,
            max_value=1.0,
        )

        # Should pass
        validate_schedule(
            schedule,
            total_steps=100,
            single_checks=[check_finite, check_non_negative],
        )

    def test_check_in_range(self):
        """Test check_in_range factory."""
        schedule = WarmupStableDecaySchedule(
            min_value=0.0,
            max_value=1.0,
        )

        # Should pass - values are in [0, 1]
        validate_schedule(
            schedule,
            total_steps=100,
            single_checks=[check_in_range(0.0, 1.0)],
        )

    def test_check_fails(self):
        """Test that check failures raise ValueError."""
        schedule = WarmupStableDecaySchedule(
            warmup_steps=10,  # Add warmup so min_value is actually used
            min_value=-0.5,  # Negative min
            max_value=1.0,
        )

        with pytest.raises(ValueError, match="Negative value"):
            validate_schedule(
                schedule,
                total_steps=100,
                single_checks=[check_non_negative],
            )

    def test_sequence_check_monotonic(self):
        """Test monotonicity checks."""
        # Schedule that's non-increasing (warmup then decay)
        schedule = WarmupStableDecaySchedule(
            warmup_steps=0,
            cooldown_frac=1.0,
            min_value=0.0,
            max_value=1.0,
            decay_type="linear",
        )

        # Should pass non-increasing check
        validate_schedule(
            schedule,
            total_steps=100,
            sequence_checks=[check_monotonic_non_increasing],
        )

        # Should fail strictly decreasing (stable phase has equal values)
        # Actually, this schedule decays every step, so it might pass...
        # Let's use a schedule with stable phase
        schedule_with_stable = WarmupStableDecaySchedule(
            warmup_steps=10,
            cooldown_steps=10,
            min_value=0.0,
            max_value=1.0,
        )

        with pytest.raises(ValueError, match="Not monotonic"):
            validate_schedule(
                schedule_with_stable,
                total_steps=100,
                sequence_checks=[check_monotonic_decreasing],
            )


# =============================================================================
# Wrapper Tests
# =============================================================================


class TestCyclicWrapper:
    """Tests for Cyclic wrapper."""

    def test_basic_cycling(self):
        """Test basic cycling behavior."""
        base_schedule = WarmupStableDecaySchedule(
            warmup_steps=0,
            cooldown_frac=1.0,
            min_value=0.0,
            max_value=1.0,
            decay_type="linear",
        )

        cyclic = sw.Cyclic(
            base_schedule_fn=base_schedule.schedule_fn,
            cycle_steps=100,
        )

        # First cycle
        assert cyclic(0, 1000) == pytest.approx(1.0)
        assert cyclic(99, 1000) == pytest.approx(0.01, abs=0.02)

        # Second cycle should restart
        assert cyclic(100, 1000) == pytest.approx(1.0)

    def test_skip_on_restart(self):
        """Test skip_on_restart parameter."""
        # FIXED: Use a schedule where step 10 is still in stable phase
        # With warmup_steps=10 and cooldown_steps=10 (not frac),
        # steps 10-89 are stable at max_value
        base_schedule = WarmupStableDecaySchedule(
            warmup_steps=10,
            cooldown_steps=10,  # Use absolute steps, not frac
            min_value=0.0,
            max_value=1.0,
        )

        cyclic = sw.Cyclic(
            base_schedule_fn=base_schedule.schedule_fn,
            cycle_steps=100,
            skip_on_restart=10,  # Skip warmup on restart
        )

        # First cycle includes warmup
        assert cyclic(0, 1000) == 0.0

        # Second cycle skips warmup (starts at step 10 of base)
        # At step 10 with cooldown_steps=10 (not frac), we're in stable phase
        assert cyclic(100, 1000) == pytest.approx(1.0)

    def test_invalid_params(self):
        """Test invalid parameter validation."""
        with pytest.raises(ValueError, match="cycle_steps must be positive"):
            sw.Cyclic(base_schedule_fn=simple_linear_decay, cycle_steps=0)

        with pytest.raises(ValueError, match="skip_on_restart must be non-negative"):
            sw.Cyclic(
                base_schedule_fn=simple_linear_decay,
                cycle_steps=100,
                skip_on_restart=-1,
            )


class TestDecayingCyclic:
    """Tests for DecayingCyclic wrapper."""

    def test_decay_envelope(self):
        """Test decaying envelope."""

        def constant_one(step, total_steps):
            return 1.0

        decaying = sw.DecayingCyclic(
            base_schedule_fn=constant_one,
            cycle_steps=100,
            decay_factor=0.5,
        )

        # First cycle: 1.0 * 0.5^0 = 1.0
        assert decaying(0, 1000) == 1.0
        assert decaying(99, 1000) == 1.0

        # Second cycle: 1.0 * 0.5^1 = 0.5
        assert decaying(100, 1000) == 0.5

        # Third cycle: 1.0 * 0.5^2 = 0.25
        assert decaying(200, 1000) == 0.25


class TestWarmRestarts:
    """Tests for WarmRestarts wrapper."""

    def test_growing_cycles(self):
        """Test that cycles grow geometrically."""

        def constant_one(step, total_steps):
            return 1.0

        warm_restarts = sw.WarmRestarts(
            base_schedule_fn=constant_one,
            initial_cycle_steps=100,
            cycle_mult=2.0,
        )

        # First cycle: 0-99 (length 100)
        cycle, step_in_cycle, cycle_length = warm_restarts._find_cycle(0)
        assert cycle == 0
        assert cycle_length == 100

        # Second cycle: 100-299 (length 200)
        cycle, step_in_cycle, cycle_length = warm_restarts._find_cycle(100)
        assert cycle == 1
        assert cycle_length == 200

        # Third cycle: 300-699 (length 400)
        cycle, step_in_cycle, cycle_length = warm_restarts._find_cycle(300)
        assert cycle == 2
        assert cycle_length == 400

    def test_invalid_params(self):
        """Test invalid parameter validation."""
        with pytest.raises(ValueError, match="initial_cycle_steps must be positive"):
            sw.WarmRestarts(base_schedule_fn=simple_linear_decay, initial_cycle_steps=0)

        with pytest.raises(ValueError, match="cycle_mult must be >= 1.0"):
            sw.WarmRestarts(
                base_schedule_fn=simple_linear_decay,
                initial_cycle_steps=100,
                cycle_mult=0.5,
            )
