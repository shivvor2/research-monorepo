"""
Schedule function wrappers for cyclic and other complex patterns.

These are callable classes that transform a base schedule function.
Use them as the schedule_fn argument to ParamSchedule.

All wrappers are picklable (required for checkpointing and distributed training).

Example:
    Basic cyclic schedule::

        from research_lib.training.scheduling import ParamSchedule, WarmupStableDecaySchedule
        from research_lib.training.scheduling import wrappers as sw

        base_schedule = WarmupStableDecaySchedule(
            peak=1.0,
            warmup_frac=0.0,
            stable_frac=0.0,  # Pure decay
        )

        schedule = ParamSchedule(
            param_name="lr",
            schedule_fn=sw.Cyclic(base_schedule.schedule_fn, cycle_steps=1000),
        )

    Decaying cycles (SGDR-style)::

        schedule = ParamSchedule(
            param_name="lr",
            schedule_fn=sw.DecayingCyclic(
                base_schedule_fn=base_fn,
                cycle_steps=1000,
                decay_factor=0.8,
            ),
        )

    Warm restarts with growing periods::

        schedule = ParamSchedule(
            param_name="lr",
            schedule_fn=sw.WarmRestarts(
                base_schedule_fn=base_fn,
                initial_cycle_steps=1000,
                cycle_mult=2.0,
            ),
        )

See Also:
    - :class:`research_lib.training.scheduling.ParamSchedule` for using wrappers
    - :class:`research_lib.training.scheduling.WarmupStableDecaySchedule` for base schedules
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Cyclic:
    """Repeats a schedule cyclically.

    The base schedule is repeated every cycle_steps steps.
    On the first cycle, the full schedule runs. On subsequent cycles,
    skip_on_restart steps are skipped (useful for skipping warmup).

    Attributes:
        base_schedule_fn: The schedule to repeat.
        cycle_steps: Length of each cycle.
        skip_on_restart: Steps to skip at the start of cycles after the first.

    Example:
        Cosine annealing with restarts (skip warmup after first cycle)::

            base_fn = WarmupStableDecaySchedule(
                peak=1.0,
                warmup_frac=0.1,
                stable_frac=0.0,  # Warmup then decay
            )

            cyclic = Cyclic(
                base_schedule_fn=base_fn.schedule_fn,
                cycle_steps=1000,
                skip_on_restart=100,  # Skip warmup on restart
            )

    Note:
        The total_steps argument to __call__ is IGNORED. The wrapper uses
        cycle_steps as the effective total for the base schedule.
    """

    base_schedule_fn: Callable[[int, int], float]
    cycle_steps: int
    skip_on_restart: int = 0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.cycle_steps <= 0:
            raise ValueError(f"cycle_steps must be positive, got {self.cycle_steps}")
        if self.skip_on_restart < 0:
            raise ValueError(
                f"skip_on_restart must be non-negative, got {self.skip_on_restart}"
            )
        if self.skip_on_restart >= self.cycle_steps:
            raise ValueError(
                f"skip_on_restart ({self.skip_on_restart}) must be less than "
                f"cycle_steps ({self.cycle_steps})"
            )

    def __call__(self, step: int, total_steps: int) -> float:
        """Compute the cyclic schedule value.

        Args:
            step: Current step.
            total_steps: Ignored (uses cycle_steps internally).

        Returns:
            The scheduled value.
        """
        if step < self.cycle_steps:
            # First cycle: use full schedule
            return self.base_schedule_fn(step, self.cycle_steps)
        else:
            # Subsequent cycles: skip initial steps
            effective_cycle_length = self.cycle_steps - self.skip_on_restart
            steps_after_first = step - self.cycle_steps
            step_in_cycle = steps_after_first % effective_cycle_length
            effective_step = self.skip_on_restart + step_in_cycle
            return self.base_schedule_fn(effective_step, self.cycle_steps)


@dataclass
class DecayingCyclic:
    """Repeats a schedule with a decaying envelope.

    Each cycle, the output is multiplied by decay_factor^cycle_number.

    Attributes:
        base_schedule_fn: The schedule to repeat.
        cycle_steps: Length of each cycle.
        decay_factor: Multiplier applied each cycle (e.g., 0.5 halves each cycle).
        skip_on_restart: Steps to skip at the start of cycles after the first.

    Example:
        SGDR-style schedule with decay::

            decaying = DecayingCyclic(
                base_schedule_fn=cosine_fn,
                cycle_steps=1000,
                decay_factor=0.8,
            )
    """

    base_schedule_fn: Callable[[int, int], float]
    cycle_steps: int
    decay_factor: float = 1.0
    skip_on_restart: int = 0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.cycle_steps <= 0:
            raise ValueError(f"cycle_steps must be positive, got {self.cycle_steps}")
        if self.decay_factor <= 0:
            raise ValueError(f"decay_factor must be positive, got {self.decay_factor}")

    def __call__(self, step: int, total_steps: int) -> float:
        """Compute the decaying cyclic schedule value.

        Args:
            step: Current step.
            total_steps: Ignored (uses cycle_steps internally).

        Returns:
            The scheduled value multiplied by the decay envelope.
        """
        cycle_num = step // self.cycle_steps
        step_in_cycle = step % self.cycle_steps

        # Apply skip on restart for cycles after the first
        if cycle_num > 0 and step_in_cycle < self.skip_on_restart:
            step_in_cycle = self.skip_on_restart

        base_value = self.base_schedule_fn(step_in_cycle, self.cycle_steps)
        return base_value * (self.decay_factor**cycle_num)


@dataclass
class WarmRestarts:
    """SGDR-style schedule with growing cycle periods.

    Cycle i has length: initial_cycle_steps * (cycle_mult ^ i)

    Attributes:
        base_schedule_fn: The schedule to repeat.
        initial_cycle_steps: Length of the first cycle.
        cycle_mult: Multiplier for cycle length each restart.

    Example:
        Cycles of 1000, 2000, 4000, ...::

            warm_restarts = WarmRestarts(
                base_schedule_fn=cosine_fn,
                initial_cycle_steps=1000,
                cycle_mult=2.0,
            )
    """

    base_schedule_fn: Callable[[int, int], float]
    initial_cycle_steps: int
    cycle_mult: float = 2.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.initial_cycle_steps <= 0:
            raise ValueError(
                f"initial_cycle_steps must be positive, got {self.initial_cycle_steps}"
            )
        if self.cycle_mult < 1.0:
            raise ValueError(f"cycle_mult must be >= 1.0, got {self.cycle_mult}")

    def __call__(self, step: int, total_steps: int) -> float:
        """Compute the warm restarts schedule value.

        Args:
            step: Current step.
            total_steps: Ignored (cycle lengths determined by initial_cycle_steps and cycle_mult).

        Returns:
            The scheduled value.
        """
        cycle, step_in_cycle, cycle_length = self._find_cycle(step)
        return self.base_schedule_fn(step_in_cycle, cycle_length)

    def _find_cycle(self, step: int) -> tuple[int, int, int]:
        """Find which cycle a step is in and position within it.

        Args:
            step: The global step.

        Returns:
            Tuple of (cycle_number, step_in_cycle, cycle_length).
        """
        if self.cycle_mult == 1.0:
            # Simple case: all cycles same length
            cycle = step // self.initial_cycle_steps
            step_in_cycle = step % self.initial_cycle_steps
            return cycle, step_in_cycle, self.initial_cycle_steps

        # Geometric series: find cycle containing step
        # Total steps after n cycles = T_0 * (mult^n - 1) / (mult - 1)
        cycle = 0
        cumulative = 0

        while True:
            cycle_length = int(self.initial_cycle_steps * (self.cycle_mult**cycle))
            if cumulative + cycle_length > step:
                step_in_cycle = step - cumulative
                return cycle, step_in_cycle, cycle_length
            cumulative += cycle_length
            cycle += 1
