"""
Parameter scheduling system for optimizer parameter values.

This module provides a flexible, layered system for scheduling arbitrary
optimizer parameters (lr, momentum, betas, etc.) during training.

Architecture:
    The system follows a layered design where users can enter at any level:

    Runtime Layer:         ParamScheduler (binds schedules to optimizer)
                                  ↓
    Schedule Layer:        ParamSchedule, WarmupStableDecaySchedule
                                  ↓
    PyTorch Layer:         optimizer.param_groups[i][param_name] = value

Design Principles:
    1. **No Restricting Action Space**: The abstraction should not reduce
       capability compared to using PyTorch directly.
    2. **User-Facing Primitives**: Primitive classes are designed for direct
       use by end users, not just as internal building blocks.
    3. **Illegal States Should Be Unrepresentable**: Required configurations
       are not Optional. Validation happens at construction time where possible.
    4. **Minimal Coupling**: Schedules describe curves, configs describe where
       they apply, optimizer instance only required at application time.
    5. **Stateful Schedulers**: ParamScheduler tracks step count for checkpointing.

Example:
    Basic usage with ParamSchedule primitive::

        from research_lib.training.scheduling import ParamSchedule

        def linear_decay(step: int, total_steps: int) -> float:
            return 1.0 - (step / total_steps) * 0.9

        schedule = ParamSchedule(param_name="lr", schedule_fn=linear_decay)
        value = schedule(step=500, total_steps=1000)

    Using the WarmupStableDecaySchedule preset::

        from research_lib.training.scheduling import WarmupStableDecaySchedule

        lr_schedule = WarmupStableDecaySchedule(
            param_name="lr",
            warmup_steps=100,
            cooldown_frac=0.5,
            min_value=0.0,
            max_value=1.0,
            decay_type="cosine",
        )

    Using ParamScheduler at runtime::

        from research_lib.training.scheduling import ParamScheduler

        scheduler = ParamScheduler(
            optimizer=optimizer,
            global_schedules=[lr_schedule, momentum_schedule],
            total_steps=10000,
        )

        for step in range(total_steps):
            loss.backward()
            optimizer.step()
            scheduler.step()

    With per-group overrides::

        scheduler = ParamScheduler(
            optimizer=optimizer,
            global_schedules=[lr_schedule],
            group_overrides={
                1: [slower_lr_schedule],  # Different schedule for group 1
            },
            total_steps=10000,
        )

Submodules:
    scheduler: Runtime ParamScheduler class
    schedules: ParamSchedule Primitive and common presets
    utils: Low-level utilities (get_param_value, apply_schedule_to_param_group, etc.)
    validation: Schedule validation (validate_schedule, check_* functions)
    wrappers: Schedule function wrappers (Cyclic, DecayingCyclic, WarmRestarts)

See Also:
    - :mod:`research_lib.training.scheduling.wrappers` for cyclic schedule wrappers
    - :mod:`research_lib.training.scheduling.validation` for schedule validation utilities
"""

from . import utils, validation, wrappers
from .scheduler import ParamScheduler
from .schedules import ParamSchedule, WarmupStableDecaySchedule

__all__ = [
    # Core classes
    "ParamSchedule",
    "WarmupStableDecaySchedule",
    "ParamScheduler",
    # Submodules
    "utils",
    "validation",
    "wrappers",
]
