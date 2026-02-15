"""Tests for the scheduling system."""

import pickle

import pytest
import torch
from torch.optim import SGD, AdamW

from research_lib.training.scheduling import (
    ParamSchedule,
    ParamScheduler,
    WarmupStableDecaySchedule,
)
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
# Test Fixtures / Helpers
# =============================================================================


def simple_linear_decay(step: int, total_steps: int) -> float:
    """Simple linear decay from 1.0 to 0.0."""
    if total_steps <= 0:
        return 1.0
    return 1.0 - (step / total_steps)


def constant_one(step: int, total_steps: int) -> float:
    """Returns constant 1.0."""
    return 1.0


def constant_half(step: int, total_steps: int) -> float:
    """Returns constant 0.5."""
    return 0.5


class ParameterizedDecay:
    """Picklable callable class for testing."""

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            return self.max_val
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
        """Test default construction values."""
        schedule = WarmupStableDecaySchedule()
        assert schedule.param_name == "lr"
        assert schedule.max_value == 1.0
        assert schedule.min_value == 0.0
        assert schedule.warmup_steps == 0
        assert schedule.cooldown_steps == 0

    def test_warmup_phase(self):
        """Test warmup phase using fraction."""
        schedule = WarmupStableDecaySchedule(
            max_value=1.0,
            min_value=0.0,
            warmup_frac=0.1,
            cooldown_frac=0.1,
        )

        # Start of warmup
        assert schedule(0, 1000) == 0.0
        # Mid warmup (step 50 of 100)
        assert schedule(50, 1000) == pytest.approx(0.5)
        # End of warmup
        assert schedule(100, 1000) == pytest.approx(1.0)

    def test_stable_phase(self):
        """Test stable phase."""
        schedule = WarmupStableDecaySchedule(
            max_value=1.0,
            warmup_frac=0.1,
            cooldown_frac=0.1,
        )

        # During stable phase (10%-90% of training)
        assert schedule(200, 1000) == 1.0
        assert schedule(800, 1000) == 1.0

    def test_cooldown_phase_linear(self):
        """Test cooldown phase with linear decay."""
        schedule = WarmupStableDecaySchedule(
            max_value=1.0,
            min_value=0.0,
            warmup_frac=0.0,
            cooldown_frac=0.5,
            decay_type="linear",
        )

        # Start of cooldown (step 500 of 1000)
        assert schedule(500, 1000) == pytest.approx(1.0)
        # Mid cooldown (step 750)
        assert schedule(750, 1000) == pytest.approx(0.5)
        # End of cooldown
        assert schedule(1000, 1000) == pytest.approx(0.0)

    def test_mutual_exclusivity(self):
        """Test arguments mutual exclusivity checks."""
        with pytest.raises(ValueError, match="Specify warmup_steps OR warmup_frac"):
            WarmupStableDecaySchedule(warmup_steps=100, warmup_frac=0.1)

        with pytest.raises(ValueError, match="Specify cooldown_steps OR cooldown_frac"):
            WarmupStableDecaySchedule(cooldown_steps=100, cooldown_frac=0.1)


# =============================================================================
# ParamScheduler Tests
# =============================================================================


class TestParamScheduler:
    """Tests for ParamScheduler runtime class."""

    @pytest.fixture
    def simple_model(self):
        """Simple model for testing."""
        return torch.nn.Linear(10, 10)

    @pytest.fixture
    def simple_optimizer(self, simple_model):
        """Simple optimizer for testing."""
        return torch.optim.SGD(simple_model.parameters(), lr=0.1)

    @pytest.fixture
    def constant_schedule(self):
        """Schedule that returns constant value."""
        return ParamSchedule(param_name="lr", schedule_fn=constant_one)

    @pytest.fixture
    def linear_schedule(self):
        """Schedule that returns linear decay."""
        return ParamSchedule(param_name="lr", schedule_fn=simple_linear_decay)

    def test_basic_init(self, simple_optimizer, constant_schedule):
        """Should initialize without error."""
        scheduler = ParamScheduler(
            optimizer=simple_optimizer,
            global_schedules=[constant_schedule],
            total_steps=100,
        )
        assert scheduler.get_current_step() == 0

    def test_validates_group_overrides(self, simple_optimizer, constant_schedule):
        """Should raise on invalid group index."""
        with pytest.raises(IndexError, match="group_overrides contains index 5"):
            ParamScheduler(
                optimizer=simple_optimizer,
                global_schedules=[],
                total_steps=100,
                group_overrides={5: [constant_schedule]},
            )

    def test_applies_schedule(self, simple_optimizer, constant_schedule):
        """Should apply schedule value to param group."""
        scheduler = ParamScheduler(
            optimizer=simple_optimizer,
            global_schedules=[constant_schedule],
            total_steps=100,
        )

        # Initial LR is 0.1
        assert simple_optimizer.param_groups[0]["lr"] == 0.1

        scheduler.step()

        # After step, LR should be 1.0 (from constant_one)
        assert simple_optimizer.param_groups[0]["lr"] == 1.0

    def test_increments_step(self, simple_optimizer, constant_schedule):
        """Should increment step count."""
        scheduler = ParamScheduler(
            optimizer=simple_optimizer,
            global_schedules=[constant_schedule],
            total_steps=100,
        )

        assert scheduler.get_current_step() == 0
        scheduler.step()
        assert scheduler.get_current_step() == 1
        scheduler.step()
        assert scheduler.get_current_step() == 2

    def test_linear_decay(self, simple_optimizer, linear_schedule):
        """Should apply changing values over steps."""
        scheduler = ParamScheduler(
            optimizer=simple_optimizer,
            global_schedules=[linear_schedule],
            total_steps=100,
        )

        # Step 0: applies value for step 0, then increments
        # simple_linear_decay(0, 100) = 1.0
        scheduler.step()
        assert simple_optimizer.param_groups[0]["lr"] == pytest.approx(1.0)

        # Advance to step 50 (we're at step 1, need 49 more steps)
        for _ in range(49):
            scheduler.step()

        # Now at step 50, last applied was step 49
        # simple_linear_decay(49, 100) = 1.0 - 49/100 = 0.51
        assert simple_optimizer.param_groups[0]["lr"] == pytest.approx(0.51)

    def test_group_overrides(self, simple_model):
        """Should apply different schedules to different groups."""
        # Create optimizer with two param groups
        optimizer = torch.optim.SGD(
            [
                {"params": [simple_model.weight], "lr": 0.1},
                {"params": [simple_model.bias], "lr": 0.01},
            ]
        )

        schedule_group0 = ParamSchedule("lr", constant_one)
        schedule_group1 = ParamSchedule("lr", constant_half)

        scheduler = ParamScheduler(
            optimizer=optimizer,
            global_schedules=[schedule_group0],
            total_steps=100,
            group_overrides={1: [schedule_group1]},
        )

        scheduler.step()

        assert optimizer.param_groups[0]["lr"] == 1.0
        assert optimizer.param_groups[1]["lr"] == 0.5

    def test_empty_global_schedules(self, simple_optimizer):
        """Should work with empty global schedules."""
        scheduler = ParamScheduler(
            optimizer=simple_optimizer,
            global_schedules=[],
            total_steps=100,
        )

        # Should not raise
        scheduler.step()
        assert scheduler.get_current_step() == 1

        # LR should be unchanged
        assert simple_optimizer.param_groups[0]["lr"] == 0.1

    def test_multiple_schedules_same_group(self, simple_optimizer):
        """Should apply multiple schedules to the same group."""
        simple_optimizer.param_groups[0]["momentum"] = 0.9

        lr_schedule = ParamSchedule("lr", constant_one)
        momentum_schedule = ParamSchedule("momentum", constant_half)

        scheduler = ParamScheduler(
            optimizer=simple_optimizer,
            global_schedules=[lr_schedule, momentum_schedule],
            total_steps=100,
        )

        scheduler.step()

        assert simple_optimizer.param_groups[0]["lr"] == 1.0
        assert simple_optimizer.param_groups[0]["momentum"] == 0.5

    def test_state_dict(self, simple_optimizer, constant_schedule):
        """Should return current step count."""
        scheduler = ParamScheduler(
            optimizer=simple_optimizer,
            global_schedules=[constant_schedule],
            total_steps=100,
        )

        scheduler.step()
        scheduler.step()
        scheduler.step()

        state = scheduler.state_dict()
        assert state == {"step_count": 3}

    def test_load_state_dict(self, simple_optimizer, constant_schedule):
        """Should restore step count."""
        scheduler = ParamScheduler(
            optimizer=simple_optimizer,
            global_schedules=[constant_schedule],
            total_steps=100,
        )

        scheduler.load_state_dict({"step_count": 50})

        assert scheduler.get_current_step() == 50

    def test_checkpoint_roundtrip(self, simple_optimizer, linear_schedule):
        """Should restore to correct state after save/load."""
        scheduler1 = ParamScheduler(
            optimizer=simple_optimizer,
            global_schedules=[linear_schedule],
            total_steps=100,
        )

        # Advance scheduler1
        for _ in range(50):
            scheduler1.step()

        # Save state
        state = scheduler1.state_dict()

        # Create new scheduler and restore
        scheduler2 = ParamScheduler(
            optimizer=simple_optimizer,
            global_schedules=[linear_schedule],
            total_steps=100,
        )
        scheduler2.load_state_dict(state)

        # Should be at same step
        assert scheduler2.get_current_step() == 50

        # Next step should apply value for step 50
        # simple_linear_decay(50, 100) = 0.5
        scheduler2.step()
        assert simple_optimizer.param_groups[0]["lr"] == pytest.approx(0.5)


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
            max_value=0.95,
            min_value=0.8,
            warmup_frac=0.1,
            cooldown_frac=0.1,
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
            max_value=1.0,
            warmup_frac=0.1,
            cooldown_frac=0.3,
        )

        # Should not raise
        validate_schedule(schedule, total_steps=100)

    def test_with_single_checks(self):
        """Test validation with single-value checks."""
        schedule = WarmupStableDecaySchedule(
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
        # Create schedule with negative start value
        schedule = WarmupStableDecaySchedule(
            max_value=1.0,
            min_value=-0.5,  # Negative start
            warmup_frac=0.1,
        )

        with pytest.raises(ValueError, match="Negative value"):
            validate_schedule(
                schedule,
                total_steps=100,
                single_checks=[check_non_negative],
            )


# =============================================================================
# Wrapper Tests
# =============================================================================


class TestCyclicWrapper:
    """Tests for Cyclic wrapper."""

    def test_basic_cycling(self):
        """Test basic cycling behavior."""
        # Create a schedule that decays from 1.0 to 0.0 over the cycle
        # Using cooldown_frac=1.0 means the entire cycle is decay
        base_schedule = WarmupStableDecaySchedule(
            max_value=1.0,
            min_value=0.0,
            warmup_frac=0.0,
            cooldown_frac=1.0,  # 100% of cycle is decay
            decay_type="linear",
        )

        cyclic = sw.Cyclic(
            base_schedule_fn=base_schedule.schedule_fn,
            cycle_steps=100,
        )

        # First cycle - step 0 should be at max (start of decay)
        assert cyclic(0, 1000) == pytest.approx(1.0)
        # Step 99 should be near min (end of decay)
        # linear decay: 1.0 - (99/100) = 0.01
        assert cyclic(99, 1000) == pytest.approx(0.01)

        # Second cycle should restart - step 100 should be at max again
        assert cyclic(100, 1000) == pytest.approx(1.0)

    def test_skip_on_restart(self):
        """Test skip_on_restart parameter."""
        # Schedule with warmup in first 10% then stable
        base_schedule = WarmupStableDecaySchedule(
            max_value=1.0,
            min_value=0.0,
            warmup_frac=0.1,  # 10 steps warmup in 100 step cycle
            cooldown_frac=0.1,
        )

        cyclic = sw.Cyclic(
            base_schedule_fn=base_schedule.schedule_fn,
            cycle_steps=100,
            skip_on_restart=10,  # Skip warmup on restart
        )

        # First cycle includes warmup - step 0 is at min_value
        assert cyclic(0, 1000) == 0.0

        # Second cycle skips warmup (starts at step 10 of base which is end of warmup)
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
