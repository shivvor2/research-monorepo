"""
Core schedule classes for optimizer parameter scheduling.

This module provides the fundamental schedule primitives:
    - :class:`ParamSchedule`: The base primitive wrapping any schedule function
    - :class:`WarmupStableDecaySchedule`: A preset for warmup → stable → decay patterns

The fundamental primitive is a function::

    f: (step: int, total_steps: int) -> float

Everything else is built on this primitive.

Example:
    Using ParamSchedule with a custom function::

        def cosine_decay(step: int, total_steps: int) -> float:
            progress = step / total_steps
            return 0.5 * (1 + math.cos(math.pi * progress))

        schedule = ParamSchedule(param_name="lr", schedule_fn=cosine_decay)

    Using a callable class for parameterization::

        class ScaledCosine:
            def __init__(self, min_val: float, max_val: float):
                self.min_val = min_val
                self.max_val = max_val

            def __call__(self, step: int, total_steps: int) -> float:
                progress = step / total_steps
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                return self.min_val + (self.max_val - self.min_val) * cosine

        schedule = ParamSchedule(
            param_name="lr",
            schedule_fn=ScaledCosine(0.001, 0.1),
        )

See Also:
    - :mod:`research_lib.training.scheduling.wrappers` for cyclic schedule wrappers
    - :mod:`research_lib.training.scheduling.utils` for applying schedules to optimizers
"""

from __future__ import annotations

import inspect
import logging
import math
import pickle
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParamSchedule:
    """Schedule for an optimizer parameter.

    This is the primitive for parameter scheduling. It wraps a function that
    maps (step, total_steps) to a parameter value, along with metadata about
    which parameter to update.

    The schedule function signature is::

        f(step: int, total_steps: int) -> float

    Where:
        - step: Current optimizer step (0-indexed)
        - total_steps: Total number of training steps
        - Returns: The parameter value at this step

    Attributes:
        param_name: The key in optimizer.param_groups[i] to update.
            Examples: 'lr', 'momentum', 'weight_decay', 'betas', 'eps'.
        schedule_fn: The scheduling function.

    Example:
        Simple linear decay::

            def linear_decay(step, total_steps):
                return 1.0 - (step / total_steps) * 0.9  # 1.0 -> 0.1

            schedule = ParamSchedule(
                param_name="lr",
                schedule_fn=linear_decay,
            )

        Using a callable class for parameterization::

            class CosineDecay:
                def __init__(self, min_value: float, max_value: float):
                    self.min_value = min_value
                    self.max_value = max_value

                def __call__(self, step: int, total_steps: int) -> float:
                    progress = step / total_steps
                    cosine = 0.5 * (1 + math.cos(math.pi * progress))
                    return self.min_value + (self.max_value - self.min_value) * cosine

            schedule = ParamSchedule(
                param_name="lr",
                schedule_fn=CosineDecay(min_value=0.001, max_value=0.1),
            )

    Note:
        For tuple parameters like ``betas``, the schedule_fn should return the
        complete tuple. See Future Extensions in the design doc for helper utilities.

    Warning:
        The schedule_fn must be picklable for checkpointing. Use module-level
        functions, functools.partial, or callable classes. Lambdas and closures
        are NOT picklable and will raise ValueError at construction.
    """

    param_name: str
    schedule_fn: Callable[[int, int], float]

    def __call__(self, step: int, total_steps: int) -> float:
        """Compute the scheduled value for a given step.

        Args:
            step: Current optimizer step (0-indexed).
            total_steps: Total number of training steps.

        Returns:
            The parameter value at this step.
        """
        return self.schedule_fn(step, total_steps)

    def __post_init__(self) -> None:
        """Validate the schedule configuration.

        Checks (all O(1), run automatically):
            1. schedule_fn is callable
            2. schedule_fn accepts at least 2 positional arguments
            3. schedule_fn is picklable (required for checkpointing)

        Raises:
            TypeError: If schedule_fn is not callable.
            ValueError: If schedule_fn has wrong signature or is not picklable.
        """
        self._validate_callable()
        self._validate_signature()
        self._validate_picklable()

    def _validate_callable(self) -> None:
        """Check that schedule_fn is callable."""
        if not callable(self.schedule_fn):
            raise TypeError(
                f"schedule_fn must be callable, got {type(self.schedule_fn).__name__}"
            )

    def _validate_signature(self) -> None:
        """Check that schedule_fn accepts (step, total_steps) arguments."""
        try:
            sig = inspect.signature(self.schedule_fn)
        except (ValueError, TypeError):
            # Can't inspect (e.g., built-in), skip validation
            return

        # Count required positional parameters
        positional_params = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is p.empty
        ]

        if len(positional_params) < 2:
            raise ValueError(
                f"schedule_fn must accept at least 2 positional arguments "
                f"(step, total_steps), got signature {sig}"
            )

    def _validate_picklable(self) -> None:
        """Check that schedule_fn can be pickled for checkpointing."""
        try:
            pickle.dumps(self.schedule_fn)
        except (pickle.PicklingError, AttributeError, TypeError) as e:
            raise ValueError(
                f"schedule_fn must be picklable for checkpointing. "
                f"Use a module-level function, functools.partial, or callable class. "
                f"Lambdas and closures are not picklable. "
                f"Error: {e}"
            ) from e


def _apply_curve(progress: float, curve_type: str) -> float:
    """Apply curve transformation to progress value.

    Args:
        progress: Linear progress in [0, 1].
        curve_type: One of "linear", "cosine".

    Returns:
        Transformed progress in [0, 1].

    Raises:
        ValueError: If curve_type is not recognized.
    """
    if curve_type == "linear":
        return progress
    elif curve_type == "cosine":
        return 0.5 * (1 - math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unknown curve_type: {curve_type}")


@dataclass(kw_only=True)
class WarmupStableDecaySchedule(ParamSchedule):
    """Preset schedule with warmup → stable → decay pattern.

    The schedule follows three phases:
        1. **Warmup** (steps 0 to warmup_steps): min_value → max_value
        2. **Stable** (warmup_steps to cooldown_start): constant at max_value
        3. **Decay** (cooldown_start to total_steps): max_value → min_value

    Both warmup and decay phases support different curve types:
        - "linear": Linear interpolation
        - "cosine": Cosine curve (smooth start/end)

    Attributes:
        param_name: The optimizer parameter to schedule. Default: "lr".
        warmup_steps: Absolute number of warmup steps. Mutually exclusive with warmup_frac.
        warmup_frac: Warmup as fraction of total_steps. Mutually exclusive with warmup_steps.
        cooldown_steps: Absolute number of decay steps. Mutually exclusive with cooldown_frac.
        cooldown_frac: Decay as fraction of total_steps. Mutually exclusive with cooldown_steps.
        min_value: Value at start of warmup and end of decay.
        max_value: Value during stable phase.
        warmup_type: Curve type for warmup. One of: "linear", "cosine".
        decay_type: Curve type for decay. One of: "linear", "cosine".

    Example:
        LR schedule with linear warmup and cosine decay::

            lr_schedule = WarmupStableDecaySchedule(
                param_name="lr",
                warmup_steps=100,
                cooldown_frac=0.5,
                min_value=0.0,
                max_value=1.0,  # This is a multiplier; actual LR = base_lr * value
                warmup_type="linear",
                decay_type="cosine",
            )

        Momentum schedule (ramps up and stays high)::

            momentum_schedule = WarmupStableDecaySchedule(
                param_name="momentum",
                warmup_steps=300,
                cooldown_steps=50,
                min_value=0.85,
                max_value=0.95,
            )

    Note:
        - If both warmup_steps and warmup_frac are None, warmup_steps defaults to 0.
        - If both cooldown_steps and cooldown_frac are None, cooldown_steps defaults to 0.
        - When using frac, the step count is computed via round() and logged if rounding occurs.
    """

    # Override parent field with default
    param_name: str = "lr"

    # Warmup configuration (mutually exclusive)
    warmup_steps: Optional[int] = None
    warmup_frac: Optional[float] = None

    # Cooldown/decay configuration (mutually exclusive)
    cooldown_steps: Optional[int] = None
    cooldown_frac: Optional[float] = None

    # Value range
    min_value: float = 0.0
    max_value: float = 1.0

    # Curve types
    warmup_type: str = "linear"
    decay_type: str = "linear"

    # Internal: schedule_fn is built from fields, not user-provided
    schedule_fn: Callable[[int, int], float] = field(init=False)

    def __post_init__(self) -> None:
        """Validate fields and build schedule_fn."""
        self._validate_fields()
        self.schedule_fn = self._build_schedule_fn()
        super().__post_init__()

    def _validate_fields(self) -> None:
        """Validate field values and mutual exclusivity."""
        # Mutual exclusivity
        if self.warmup_steps is not None and self.warmup_frac is not None:
            raise ValueError("Specify warmup_steps OR warmup_frac, not both")
        if self.cooldown_steps is not None and self.cooldown_frac is not None:
            raise ValueError("Specify cooldown_steps OR cooldown_frac, not both")

        # Defaults - use object.__setattr__ for frozen-like behavior in post_init
        if self.warmup_steps is None and self.warmup_frac is None:
            object.__setattr__(self, "warmup_steps", 0)
        if self.cooldown_steps is None and self.cooldown_frac is None:
            object.__setattr__(self, "cooldown_steps", 0)

        # Value validation
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        if self.warmup_frac is not None and not (0.0 <= self.warmup_frac <= 1.0):
            raise ValueError(f"warmup_frac must be in [0, 1], got {self.warmup_frac}")
        if self.cooldown_steps is not None and self.cooldown_steps < 0:
            raise ValueError(
                f"cooldown_steps must be non-negative, got {self.cooldown_steps}"
            )
        if self.cooldown_frac is not None and not (0.0 <= self.cooldown_frac <= 1.0):
            raise ValueError(
                f"cooldown_frac must be in [0, 1], got {self.cooldown_frac}"
            )

        # Curve type validation
        valid_types = {"linear", "cosine"}
        if self.warmup_type not in valid_types:
            raise ValueError(
                f"warmup_type must be one of {valid_types}, got {self.warmup_type}"
            )
        if self.decay_type not in valid_types:
            raise ValueError(
                f"decay_type must be one of {valid_types}, got {self.decay_type}"
            )

    def _resolve_steps(self, total_steps: int) -> tuple[int, int]:
        """Resolve warmup and cooldown to absolute step counts.

        Args:
            total_steps: Total number of training steps.

        Returns:
            Tuple of (warmup_steps, cooldown_steps).

        Note:
            Uses round() for fractional values. Logs info if rounding occurs.
        """
        if self.warmup_steps is not None:
            warmup = self.warmup_steps
        else:
            raw = self.warmup_frac * total_steps
            warmup = round(raw)
            if warmup != raw:
                logger.info(
                    f"warmup_frac={self.warmup_frac} * total_steps={total_steps} = {raw}, "
                    f"rounded to {warmup}"
                )

        if self.cooldown_steps is not None:
            cooldown = self.cooldown_steps
        else:
            raw = self.cooldown_frac * total_steps
            cooldown = round(raw)
            if cooldown != raw:
                logger.info(
                    f"cooldown_frac={self.cooldown_frac} * total_steps={total_steps} = {raw}, "
                    f"rounded to {cooldown}"
                )

        return warmup, cooldown

    def _build_schedule_fn(self) -> Callable[[int, int], float]:
        """Build the schedule function from fields.

        Returns a callable class instance that captures all necessary state.
        This is picklable unlike a closure.
        """
        return _WarmupStableDecayFn(
            warmup_steps=self.warmup_steps,
            warmup_frac=self.warmup_frac,
            cooldown_steps=self.cooldown_steps,
            cooldown_frac=self.cooldown_frac,
            min_value=self.min_value,
            max_value=self.max_value,
            warmup_type=self.warmup_type,
            decay_type=self.decay_type,
        )


@dataclass
class _WarmupStableDecayFn:
    """Picklable callable for WarmupStableDecaySchedule.

    This is a helper class that stores all schedule parameters and implements
    the schedule logic. It's picklable because it's a dataclass with only
    basic types as fields.
    """

    warmup_steps: Optional[int]
    warmup_frac: Optional[float]
    cooldown_steps: Optional[int]
    cooldown_frac: Optional[float]
    min_value: float
    max_value: float
    warmup_type: str
    decay_type: str

    def _resolve_steps(self, total_steps: int) -> tuple[int, int]:
        """Resolve warmup and cooldown to absolute step counts."""
        if self.warmup_steps is not None:
            warmup = self.warmup_steps
        else:
            raw = self.warmup_frac * total_steps
            warmup = round(raw)
            if warmup != raw:
                logger.info(
                    f"warmup_frac={self.warmup_frac} * total_steps={total_steps} = {raw}, "
                    f"rounded to {warmup}"
                )

        if self.cooldown_steps is not None:
            cooldown = self.cooldown_steps
        else:
            raw = self.cooldown_frac * total_steps
            cooldown = round(raw)
            if cooldown != raw:
                logger.info(
                    f"cooldown_frac={self.cooldown_frac} * total_steps={total_steps} = {raw}, "
                    f"rounded to {cooldown}"
                )

        return warmup, cooldown

    def __call__(self, step: int, total_steps: int) -> float:
        """Compute the scheduled value for a given step."""
        warmup, cooldown = self._resolve_steps(total_steps)
        cooldown_start = total_steps - cooldown

        if step < warmup:
            # Warmup phase
            progress = step / warmup if warmup > 0 else 1.0
            multiplier = _apply_curve(progress, self.warmup_type)
            return self.min_value + (self.max_value - self.min_value) * multiplier

        elif step >= cooldown_start:
            # Decay phase
            progress = (step - cooldown_start) / cooldown if cooldown > 0 else 1.0
            progress = min(1.0, progress)  # Clamp
            multiplier = 1.0 - _apply_curve(progress, self.decay_type)
            return self.min_value + (self.max_value - self.min_value) * multiplier

        else:
            # Stable phase
            return self.max_value
