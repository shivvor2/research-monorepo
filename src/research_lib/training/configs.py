"""
Configuration dataclasses for training with multiple optimizers.

This module provides pure data configuration classes:
    - :class:`OptimizerConfig`: Optimizer construction configuration
    - :class:`ScheduleConfig`: Parameter schedule configuration
    - :class:`GradAccumSchedule`: Step-based gradient accumulation schedule

Design Principles:
    1. **Configs are pure data**: No methods requiring runtime objects
    2. **Trainer is the config surface**: Training loop params belong to Trainer
    3. **Late binding**: Schedules bound to optimizers at runtime via ParamScheduler

Example:
    Basic configuration::

        from research_lib.training.configs import (
            OptimizerConfig,
            ScheduleConfig,
            build_optimizer,
        )
        from research_lib.training.scheduling import WarmupStableDecaySchedule

        opt_config = OptimizerConfig(
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={"lr": 3e-4, "weight_decay": 0.1},
        )

        schedule_config = ScheduleConfig(
            global_schedules=[
                WarmupStableDecaySchedule(param_name="lr", peak=3e-4),
            ],
        )

        optimizer = build_optimizer(opt_config, model.parameters())

See Also:
    - :mod:`research_lib.training.presets` for factory functions
    - :mod:`research_lib.training.scheduling` for schedule classes
    - :mod:`research_lib.training.scheduling.scheduler` for ParamScheduler
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

from torch.optim import Optimizer

from .scheduling import ParamSchedule

# =============================================================================
# OptimizerConfig: Pure Optimizer Construction Config
# =============================================================================


@dataclass
class OptimizerConfig:
    """Configuration for optimizer construction.

    This is pure data—no methods that require runtime objects.
    Use :func:`build_optimizer` to construct the actual optimizer.

    Attributes:
        optimizer_class: The optimizer class (e.g., torch.optim.AdamW).
        optimizer_kwargs: Keyword arguments passed to optimizer constructor.

    Example:
        >>> config = OptimizerConfig(
        ...     optimizer_class=torch.optim.AdamW,
        ...     optimizer_kwargs={"lr": 3e-4, "betas": (0.9, 0.95), "weight_decay": 0.1},
        ... )
        >>> optimizer = build_optimizer(config, model.parameters())

    Note:
        This class intentionally does NOT include schedules. Schedules are
        specified separately via :class:`ScheduleConfig` to maintain separation
        of concerns.
    """

    optimizer_class: Type[Optimizer]
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ScheduleConfig: Parameter Schedule Configuration
# =============================================================================


@dataclass
class ScheduleConfig:
    """Configuration for parameter schedules applied to an optimizer.

    Defines which :class:`ParamSchedule` instances apply to an optimizer.
    Schedules are bound at runtime via :class:`ParamScheduler`.

    Attributes:
        global_schedules: Schedules applied to all param groups by default.
        group_overrides: Per-group schedule overrides. Keys are group indices.
            If a group index is present, those schedules REPLACE (not extend)
            the global_schedules for that group.

    Example:
        Single param group (common case)::

            config = ScheduleConfig(
                global_schedules=[
                    WarmupStableDecaySchedule("lr", peak=0.02, warmup_frac=0.1),
                ],
            )

        Multiple param groups with different schedules::

            config = ScheduleConfig(
                global_schedules=[
                    WarmupStableDecaySchedule("lr", peak=0.01),
                ],
                group_overrides={
                    1: [WarmupStableDecaySchedule("lr", peak=0.001, warmup_frac=0.2)],
                },
            )

    Note:
        Override semantics: group_overrides[i] completely replaces global_schedules
        for group i. They do not merge.
    """

    global_schedules: List[ParamSchedule] = field(default_factory=list)
    group_overrides: Dict[int, List[ParamSchedule]] = field(default_factory=dict)


# =============================================================================
# GradAccumSchedule: Step-Based Gradient Accumulation
# =============================================================================


@dataclass
class GradAccumSchedule:
    """Step-based gradient accumulation schedule.

    When provided to a LightningModule, overrides trainer.accumulate_grad_batches.
    Keys are optimizer steps (not batches, not epochs).

    Attributes:
        schedule: Mapping from optimizer step to accumulation value.
            If step 0 is not specified, defaults to 1.

    Example:
        Ramp up accumulation over training::

            GradAccumSchedule({0: 1, 1000: 2, 5000: 4})

        Constant accumulation::

            GradAccumSchedule({0: 4})

    Note:
        Step 0 is automatically added with value 1 if not specified.
        This ensures there's always a valid accumulation value.
    """

    schedule: Dict[int, int]
    _sorted_steps: List[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate schedule and precompute sorted steps."""
        if not self.schedule:
            raise ValueError("Schedule cannot be empty")

        for step, accum in self.schedule.items():
            if not isinstance(step, int) or step < 0:
                raise ValueError(
                    f"Step must be non-negative int, got {step} (type: {type(step).__name__})"
                )
            if not isinstance(accum, int) or accum < 1:
                raise ValueError(
                    f"Accumulation must be >= 1, got {accum} (type: {type(accum).__name__})"
                )

        # Add default step 0 if not specified
        if 0 not in self.schedule:
            self.schedule = {0: 1, **self.schedule}

        self._sorted_steps = sorted(self.schedule.keys())

    def get_accum(self, step: int) -> int:
        """Get accumulation value for given optimizer step.

        Args:
            step: Current optimizer step (0-indexed).

        Returns:
            The gradient accumulation factor for this step.

        Example:
            >>> schedule = GradAccumSchedule({0: 1, 1000: 4})
            >>> schedule.get_accum(500)   # Returns 1
            >>> schedule.get_accum(1500)  # Returns 4
        """
        accum = 1
        for s in self._sorted_steps:
            if step >= s:
                accum = self.schedule[s]
            else:
                break
        return accum


# =============================================================================
# build_optimizer: Standalone Factory Function
# =============================================================================


def build_optimizer(config: OptimizerConfig, params) -> Optimizer:
    """Construct an optimizer from config.

    This is a standalone factory function, keeping OptimizerConfig as pure data.

    Args:
        config: Optimizer configuration.
        params: Parameters to optimize. Can be:
            - An iterable of Tensors (model.parameters())
            - A list of param group dicts (each with 'params' key)

    Returns:
        Configured optimizer instance.

    Example:
        Basic usage::

            >>> config = OptimizerConfig(torch.optim.AdamW, {"lr": 3e-4})
            >>> optimizer = build_optimizer(config, model.parameters())

        With param groups::

            >>> param_groups = [
            ...     {"params": model.encoder.parameters(), "lr": 1e-4},
            ...     {"params": model.decoder.parameters(), "lr": 3e-4},
            ... ]
            >>> optimizer = build_optimizer(config, param_groups)
    """
    return config.optimizer_class(params, **config.optimizer_kwargs)
