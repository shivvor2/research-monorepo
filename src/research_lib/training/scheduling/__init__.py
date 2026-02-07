"""
Parameter scheduling system for optimizer parameter values.

This module provides a flexible, layered system for scheduling arbitrary
optimizer parameters (lr, momentum, betas, etc.) during training.

Architecture:
    The system follows a layered design where users can enter at any level:

    User-Facing Layer:     OptimizerConfig, default_muon_config(), etc.
                                  ↓
    Config Layer:          ParamGroupConfig, schedule precedence logic
                                  ↓
    Schedule Layer:        ParamSchedule, WarmupStableDecaySchedule
                                  ↓
    Primitive Layer:       update_optimizer_schedules(), apply_schedule_to_param_group()
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

    Applying schedules during training::

        from research_lib.training.scheduling import update_optimizer_schedules

        for step in range(total_steps):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            update_optimizer_schedules(optimizer, config, step, total_steps)

Submodules:
    utils: Low-level utilities (get_param_value, apply_schedule_to_param_group, etc.)
    validation: Schedule validation (validate_schedule, check_* functions)
    wrappers: Schedule function wrappers (Cyclic, DecayingCyclic, WarmRestarts)

See Also:
    - :mod:`research_lib.training.configs` for OptimizerConfig and ParamGroupConfig
    - :mod:`research_lib.training.scheduling.wrappers` for cyclic schedule wrappers
    - :mod:`research_lib.training.scheduling.validation` for schedule validation utilities
"""

from . import utils, validation, wrappers
from .schedules import ParamSchedule, WarmupStableDecaySchedule

__all__ = [
    # Core classes
    "ParamSchedule",
    "WarmupStableDecaySchedule",
    # Submodules
    "utils",
    "validation",
    "wrappers",
]
