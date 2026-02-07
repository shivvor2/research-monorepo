"""
Validation utilities for schedule functions.

This module provides tools to validate that schedule functions behave correctly
across all training steps. Validation is opt-in and O(total_steps).

The validation system is composable:
    - Single-value checks run during the iteration (O(1) per step)
    - Sequence checks run after collecting all values (O(total_steps) memory)

Example:
    Basic validation (just check it doesn't crash)::

        from research_lib.training.scheduling import validate_schedule

        validate_schedule(my_schedule, total_steps=10000)

    With numeric checks::

        from research_lib.training.scheduling import (
            validate_schedule,
            check_finite,
            check_non_negative,
        )

        validate_schedule(
            my_schedule,
            total_steps=10000,
            single_checks=[check_finite, check_non_negative],
        )

    With monotonicity check::

        from research_lib.training.scheduling import (
            validate_schedule,
            check_monotonic_decreasing,
        )

        validate_schedule(
            my_schedule,
            total_steps=10000,
            sequence_checks=[check_monotonic_decreasing],
        )

See Also:
    - :class:`research_lib.training.scheduling.ParamSchedule` for schedule definition
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Set, Tuple

from .schedules import ParamSchedule

# Type aliases for check functions
SingleValueCheck = Callable[[Any, int, int], None]
"""Signature: (value, step, total_steps) -> None. Raises ValueError if check fails."""

SequenceCheck = Callable[[List[Tuple[int, Any]], int], None]
"""Signature: ([(step, value), ...], total_steps) -> None. Raises ValueError if check fails."""


def validate_schedule(
    schedule: ParamSchedule,
    total_steps: int,
    single_checks: Optional[List[SingleValueCheck]] = None,
    sequence_checks: Optional[List[SequenceCheck]] = None,
) -> None:
    """Validate a schedule for all steps.

    By default (no checks provided), only verifies the schedule doesn't crash
    for any step in [0, total_steps). This is O(total_steps).

    Args:
        schedule: The schedule to validate.
        total_steps: Total training steps.
        single_checks: Functions called for each (value, step, total_steps).
            Should raise ValueError if check fails. Run during the single pass.
        sequence_checks: Functions called once with all [(step, value), ...].
            Should raise ValueError if check fails. Run after collecting all values.

    Raises:
        ValueError: If schedule raises for any step, or if any check fails.

    Example:
        Just check it doesn't crash::

            validate_schedule(my_schedule, total_steps=10000)

        With numeric checks::

            validate_schedule(
                my_schedule,
                total_steps=10000,
                single_checks=[check_finite, check_non_negative],
            )

        With monotonicity check::

            validate_schedule(
                my_schedule,
                total_steps=10000,
                sequence_checks=[check_monotonic_decreasing],
            )

        Custom check::

            def check_below_threshold(value, step, total_steps):
                if step > 5000 and value > 0.5:
                    raise ValueError(f"Value should be <= 0.5 after step 5000")

            validate_schedule(
                my_schedule,
                total_steps=10000,
                single_checks=[check_below_threshold],
            )
    """
    single_checks = single_checks or []
    sequence_checks = sequence_checks or []

    all_values: List[Tuple[int, Any]] = []

    for step in range(total_steps):
        try:
            value = schedule(step, total_steps)
        except Exception as e:
            raise ValueError(f"schedule raised at step {step}: {e}") from e

        all_values.append((step, value))

        for check in single_checks:
            check(value, step, total_steps)

    for check in sequence_checks:
        check(all_values, total_steps)


# =============================================================================
# Pre-Built Single-Value Checks
# =============================================================================


def check_finite(value: Any, step: int, total_steps: int) -> None:
    """Check that value is finite (not NaN or Inf).

    Skips non-numeric values silently.

    Args:
        value: The value to check.
        step: Current step (for error messages).
        total_steps: Total steps (unused, for signature compatibility).

    Raises:
        ValueError: If value is NaN or Inf.
    """
    if not isinstance(value, (int, float)):
        return
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value {value} at step {step}")


def check_non_negative(value: Any, step: int, total_steps: int) -> None:
    """Check that value is non-negative.

    Skips non-numeric values silently.

    Args:
        value: The value to check.
        step: Current step (for error messages).
        total_steps: Total steps (unused, for signature compatibility).

    Raises:
        ValueError: If value is negative.
    """
    if not isinstance(value, (int, float)):
        return
    if value < 0:
        raise ValueError(f"Negative value {value} at step {step}")


def check_positive(value: Any, step: int, total_steps: int) -> None:
    """Check that value is strictly positive.

    Skips non-numeric values silently.

    Args:
        value: The value to check.
        step: Current step (for error messages).
        total_steps: Total steps (unused, for signature compatibility).

    Raises:
        ValueError: If value is zero or negative.
    """
    if not isinstance(value, (int, float)):
        return
    if value <= 0:
        raise ValueError(f"Non-positive value {value} at step {step}")


def check_in_range(min_val: float, max_val: float) -> SingleValueCheck:
    """Factory for range check.

    Args:
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        A check function that verifies value is in [min_val, max_val].

    Example:
        >>> check = check_in_range(0.0, 1.0)
        >>> check(0.5, step=0, total_steps=100)  # OK
        >>> check(1.5, step=0, total_steps=100)  # Raises ValueError
    """

    def _check(value: Any, step: int, total_steps: int) -> None:
        if not isinstance(value, (int, float)):
            return
        if not (min_val <= value <= max_val):
            raise ValueError(
                f"Value {value} outside [{min_val}, {max_val}] at step {step}"
            )

    return _check


def check_in_set(valid_values: Set[Any]) -> SingleValueCheck:
    """Factory for discrete value check.

    Args:
        valid_values: Set of allowed values.

    Returns:
        A check function that verifies value is in the set.

    Example:
        >>> check = check_in_set({0, 1, 2})
        >>> check(1, step=0, total_steps=100)  # OK
        >>> check(5, step=0, total_steps=100)  # Raises ValueError
    """

    def _check(value: Any, step: int, total_steps: int) -> None:
        if value not in valid_values:
            raise ValueError(
                f"Value {value} not in valid set {valid_values} at step {step}"
            )

    return _check


# =============================================================================
# Pre-Built Sequence Checks
# =============================================================================


def check_monotonic_increasing(
    values: List[Tuple[int, Any]],
    total_steps: int,
) -> None:
    """Check values are strictly monotonically increasing.

    Args:
        values: List of (step, value) tuples.
        total_steps: Total steps (unused, for signature compatibility).

    Raises:
        ValueError: If any value is not greater than the previous.
    """
    for i in range(1, len(values)):
        prev_step, prev_val = values[i - 1]
        curr_step, curr_val = values[i]
        if curr_val <= prev_val:
            raise ValueError(
                f"Not monotonic increasing: {prev_val} (step {prev_step}) "
                f"-> {curr_val} (step {curr_step})"
            )


def check_monotonic_decreasing(
    values: List[Tuple[int, Any]],
    total_steps: int,
) -> None:
    """Check values are strictly monotonically decreasing.

    Args:
        values: List of (step, value) tuples.
        total_steps: Total steps (unused, for signature compatibility).

    Raises:
        ValueError: If any value is not less than the previous.
    """
    for i in range(1, len(values)):
        prev_step, prev_val = values[i - 1]
        curr_step, curr_val = values[i]
        if curr_val >= prev_val:
            raise ValueError(
                f"Not monotonic decreasing: {prev_val} (step {prev_step}) "
                f"-> {curr_val} (step {curr_step})"
            )


def check_monotonic_non_increasing(
    values: List[Tuple[int, Any]],
    total_steps: int,
) -> None:
    """Check values are monotonically non-increasing (allows equal).

    Args:
        values: List of (step, value) tuples.
        total_steps: Total steps (unused, for signature compatibility).

    Raises:
        ValueError: If any value is greater than the previous.
    """
    for i in range(1, len(values)):
        prev_step, prev_val = values[i - 1]
        curr_step, curr_val = values[i]
        if curr_val > prev_val:
            raise ValueError(
                f"Not monotonic non-increasing: {prev_val} (step {prev_step}) "
                f"-> {curr_val} (step {curr_step})"
            )


def check_monotonic_non_decreasing(
    values: List[Tuple[int, Any]],
    total_steps: int,
) -> None:
    """Check values are monotonically non-decreasing (allows equal).

    Args:
        values: List of (step, value) tuples.
        total_steps: Total steps (unused, for signature compatibility).

    Raises:
        ValueError: If any value is less than the previous.
    """
    for i in range(1, len(values)):
        prev_step, prev_val = values[i - 1]
        curr_step, curr_val = values[i]
        if curr_val < prev_val:
            raise ValueError(
                f"Not monotonic non-decreasing: {prev_val} (step {prev_step}) "
                f"-> {curr_val} (step {curr_step})"
            )
