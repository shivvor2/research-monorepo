"""
Configuration dataclasses for training with multiple optimizers.

This module provides a flexible configuration system for training neural networks
with different optimizers for different parameter groups (e.g., Muon for weight
matrices, AdamW for embeddings).

Design Principles:
    1. **Illegal states should be unrepresentable**: Required configs are not Optional.
    2. **Minimal coupling enforcement**: TrainingConfig only contains parameters that
       MUST be consistent across optimizers (e.g., total_steps). Per-optimizer
       schedule parameters (warmup, cooldown) are in SchedulerConfig.
    3. **Extensible scheduling**: Custom param_group values can be scheduled via
       CustomScheduleConfig, enabling support for arbitrary optimizer parameters.

Example:
    Basic usage with Muon + AdamW::

        from research_lib.training.configs import (
            TrainingConfig,
            OptimizerConfig,
            SchedulerConfig,
            MomentumScheduleConfig,
            default_muon_config,
            default_adam_config,
        )

        training_config = TrainingConfig(total_steps=10000)
        muon_config = default_muon_config()
        adam_config = default_adam_config()

    Custom optimizer configuration::

        muon_config = OptimizerConfig(
            optimizer_class=torch.optim.Muon,
            optimizer_kwargs={"lr": 0.03, "momentum": 0.9, "weight_decay": 0.5},
            scheduler_config=SchedulerConfig(
                warmup_steps=200,
                cooldown_frac=0.4,
                min_lr_ratio=0.05,
                momentum_schedule=MomentumScheduleConfig(
                    warmup_steps=500,
                    cooldown_steps=100,
                    min_value=0.8,
                    max_value=0.95,
                ),
            ),
        )

    Custom param_group value scheduling (arbitrary optimizer params)::

        from research_lib.training.configs import CustomScheduleConfig

        # Schedule beta1 for an Adam-like optimizer
        beta1_schedule = CustomScheduleConfig(
            param_name="betas",  # The key in param_group
            param_index=0,       # betas is a tuple, schedule the first element
            warmup_steps=100,
            cooldown_steps=50,
            min_value=0.85,
            max_value=0.95,
        )

        scheduler_config = SchedulerConfig(
            warmup_steps=100,
            custom_schedules=[beta1_schedule],
        )

Possible Extensions:
    - ``decay_type`` in SchedulerConfig: Support 'cosine', 'linear', 'exponential' LR decay
    - ``warmup_type``: Support 'linear', 'exponential', 'polynomial' warmup curves
    - ``cycle_length`` for cyclical schedules (SGDR-style restarts)
    - Integration with ``torch.optim.lr_scheduler.SequentialLR`` for complex multi-phase schedules

See Also:
    - :mod:`research_lib.training.scheduling` for schedule computation functions
    - :mod:`research_lib.training.lightning_module` for the LightningModule implementation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


@dataclass
class TrainingConfig:
    """Global training parameters that must be consistent across all optimizers.

    This config contains ONLY parameters where mismatches between optimizers
    would break training semantically. Parameters that can legitimately differ
    between optimizers (warmup_steps, cooldown_frac) belong in SchedulerConfig.

    Attributes:
        total_steps: Total number of optimizer steps for training. Both optimizers
            will use this to compute their schedules. This is the number of
            actual parameter updates, not the number of batches (which may differ
            due to gradient accumulation).
        grad_accum_steps: Number of batches to accumulate before stepping the
            optimizer. Effective batch size = physical_batch_size * grad_accum_steps.
        gradient_clip_val: Maximum gradient norm for gradient clipping. Set to 0.0
            to disable gradient clipping.

    Example:
        >>> config = TrainingConfig(
        ...     total_steps=10000,
        ...     grad_accum_steps=8,
        ...     gradient_clip_val=1.0,
        ... )
    """

    total_steps: int
    grad_accum_steps: int = 1
    gradient_clip_val: float = 1.0

    def __post_init__(self):
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {self.total_steps}")
        if self.grad_accum_steps <= 0:
            raise ValueError(
                f"grad_accum_steps must be positive, got {self.grad_accum_steps}"
            )
        if self.gradient_clip_val < 0:
            raise ValueError(
                f"gradient_clip_val must be non-negative, got {self.gradient_clip_val}"
            )


@dataclass
class CustomScheduleConfig:
    """Configuration for scheduling arbitrary param_group values.

    This enables scheduling any numeric value in an optimizer's param_groups,
    supporting optimizers with custom parameters beyond lr and momentum.

    The schedule follows a warmup-stable-cooldown pattern:
        1. Warmup: Linear interpolation from min_value to max_value
        2. Stable: Constant at max_value
        3. Cooldown: Linear interpolation from max_value back to min_value

    Attributes:
        param_name: The key in optimizer.param_groups[i] to schedule.
            Examples: 'momentum', 'betas', 'weight_decay', 'eps'.
        param_index: If the param is a tuple/list (e.g., betas=(0.9, 0.999)),
            this specifies which index to schedule. None for scalar params.
        warmup_steps: Number of steps for warmup phase (min → max).
        cooldown_steps: Number of steps for cooldown phase (max → min).
            Cooldown starts at (total_steps - cooldown_steps).
        min_value: Value at start of warmup and end of cooldown.
        max_value: Value during stable phase.

    Example:
        Schedule momentum from 0.85 to 0.95 and back::

            momentum_schedule = CustomScheduleConfig(
                param_name="momentum",
                warmup_steps=300,
                cooldown_steps=50,
                min_value=0.85,
                max_value=0.95,
            )

        Schedule the first element of Adam's betas tuple::

            beta1_schedule = CustomScheduleConfig(
                param_name="betas",
                param_index=0,  # Schedule beta1, leave beta2 unchanged
                warmup_steps=100,
                cooldown_steps=50,
                min_value=0.85,
                max_value=0.95,
            )

    Warning:
        Setting a param that the optimizer doesn't use will silently add the key
        to param_groups but have no effect. This is by design to avoid crashes
        with unknown optimizers, but be careful to verify param names.
    """

    param_name: str
    warmup_steps: int = 0
    cooldown_steps: int = 0
    min_value: float = 0.0
    max_value: float = 1.0
    param_index: Optional[int] = None

    def __post_init__(self):
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        if self.cooldown_steps < 0:
            raise ValueError(
                f"cooldown_steps must be non-negative, got {self.cooldown_steps}"
            )

    def get_value(self, step: int, total_steps: int) -> float:
        """Compute the scheduled value for a given step.

        Args:
            step: Current optimizer step (0-indexed).
            total_steps: Total number of training steps.

        Returns:
            The scheduled value for this step.
        """
        if step < self.warmup_steps:
            # Warmup: linear from min to max
            progress = step / self.warmup_steps if self.warmup_steps > 0 else 1.0
            return self.min_value + (self.max_value - self.min_value) * progress
        elif step >= total_steps - self.cooldown_steps:
            # Cooldown: linear from max to min
            if self.cooldown_steps > 0:
                progress = (
                    step - (total_steps - self.cooldown_steps)
                ) / self.cooldown_steps
                return self.max_value - (self.max_value - self.min_value) * progress
            else:
                return self.max_value
        else:
            # Stable phase
            return self.max_value


@dataclass
class MomentumScheduleConfig:
    """Convenience config for momentum scheduling (common for Muon-style optimizers).

    This is a specialized version of CustomScheduleConfig for the 'momentum'
    parameter, which is commonly scheduled in Muon and similar optimizers.

    The schedule follows:
        1. Warmup (steps 0 to warmup_steps): min_value → max_value
        2. Stable (middle of training): constant at max_value
        3. Cooldown (last cooldown_steps): max_value → min_value

    Attributes:
        warmup_steps: Steps for momentum warmup. Default: 300.
        cooldown_steps: Steps for momentum cooldown at end of training. Default: 50.
        min_value: Momentum at start/end. Lower momentum = more exploration. Default: 0.85.
        max_value: Momentum during stable phase. Higher momentum = smoother updates. Default: 0.95.

    Note:
        This config is converted to a CustomScheduleConfig internally. It exists
        for convenience and discoverability since momentum scheduling is common.

    Example:
        >>> momentum = MomentumScheduleConfig(
        ...     warmup_steps=300,
        ...     cooldown_steps=50,
        ...     min_value=0.85,
        ...     max_value=0.95,
        ... )
    """

    warmup_steps: int = 300
    cooldown_steps: int = 50
    min_value: float = 0.85
    max_value: float = 0.95

    def to_custom_schedule(self) -> CustomScheduleConfig:
        """Convert to a generic CustomScheduleConfig."""
        return CustomScheduleConfig(
            param_name="momentum",
            warmup_steps=self.warmup_steps,
            cooldown_steps=self.cooldown_steps,
            min_value=self.min_value,
            max_value=self.max_value,
        )


@dataclass
class SchedulerConfig:
    """Per-optimizer learning rate and param_group scheduling configuration.

    This config defines how an optimizer's learning rate (and optionally other
    param_group values) evolve during training. Different optimizers can have
    different schedules.

    The LR schedule follows a warmup-stable-cooldown pattern:
        1. Warmup (steps 0 to warmup_steps): Linear 0 → base_lr
        2. Stable (warmup_steps to cooldown_start): Constant at base_lr
        3. Cooldown (cooldown_start to total_steps): Cosine decay to min_lr_ratio * base_lr

    Attributes:
        warmup_steps: Number of steps for LR warmup. Default: 100.
        cooldown_frac: Fraction of total_steps for cooldown phase. Cooldown starts
            at step (total_steps * (1 - cooldown_frac)). Default: 0.5.
        min_lr_ratio: Minimum LR as fraction of base LR at end of cooldown.
            Final LR = base_lr * min_lr_ratio. Default: 0.1.
        momentum_schedule: Optional momentum scheduling config for Muon-style
            optimizers. Set to None for optimizers that don't use momentum
            (e.g., Adam). Default: None.
        custom_schedules: List of custom param_group schedules for arbitrary
            optimizer parameters. Default: empty list.

    Example:
        Basic LR schedule::

            config = SchedulerConfig(
                warmup_steps=100,
                cooldown_frac=0.5,
                min_lr_ratio=0.1,
            )

        With momentum scheduling for Muon::

            config = SchedulerConfig(
                warmup_steps=100,
                cooldown_frac=0.5,
                momentum_schedule=MomentumScheduleConfig(
                    warmup_steps=300,
                    min_value=0.85,
                    max_value=0.95,
                ),
            )
    """

    warmup_steps: int = 100
    cooldown_frac: float = 0.5
    min_lr_ratio: float = 0.1
    momentum_schedule: Optional[MomentumScheduleConfig] = None
    custom_schedules: List[CustomScheduleConfig] = field(default_factory=list)

    def __post_init__(self):
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        if not 0.0 <= self.cooldown_frac <= 1.0:
            raise ValueError(
                f"cooldown_frac must be in [0, 1], got {self.cooldown_frac}"
            )
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")

    def get_all_custom_schedules(self) -> List[CustomScheduleConfig]:
        """Get all custom schedules including momentum schedule if present.

        Returns:
            List of all CustomScheduleConfig objects, with momentum_schedule
            converted and prepended if it exists.
        """
        schedules = []
        if self.momentum_schedule is not None:
            schedules.append(self.momentum_schedule.to_custom_schedule())
        schedules.extend(self.custom_schedules)
        return schedules

    def build_lr_lambda(self, total_steps: int):
        """Create the LR lambda function for this schedule.

        Args:
            total_steps: Total number of training steps.

        Returns:
            A callable that takes step and returns LR multiplier.
        """
        warmup_steps = self.warmup_steps
        cooldown_frac = self.cooldown_frac
        min_lr_ratio = self.min_lr_ratio
        cooldown_start = int(total_steps * (1 - cooldown_frac))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps if warmup_steps > 0 else 1.0
            elif step < cooldown_start:
                # Constant
                return 1.0
            else:
                # Cosine decay
                progress = (step - cooldown_start) / (total_steps - cooldown_start)
                progress = min(1.0, progress)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine

        return lr_lambda


@dataclass
class OptimizerConfig:
    """Complete configuration for an optimizer and its scheduling.

    This bundles together:
        1. The optimizer class and its initialization kwargs
        2. The LR schedule configuration
        3. Optional param_group value schedules (momentum, etc.)

    Attributes:
        optimizer_class: The optimizer class (e.g., torch.optim.AdamW, torch.optim.Muon).
        optimizer_kwargs: Keyword arguments passed to the optimizer constructor.
            Must include 'lr'. May include 'weight_decay', 'momentum', 'betas', etc.
        scheduler_config: Configuration for LR and param_group scheduling.

    Example:
        Muon optimizer with momentum scheduling::

            config = OptimizerConfig(
                optimizer_class=torch.optim.Muon,
                optimizer_kwargs={
                    "lr": 0.02,
                    "momentum": 0.95,
                    "weight_decay": 1.0,
                },
                scheduler_config=SchedulerConfig(
                    warmup_steps=100,
                    cooldown_frac=0.5,
                    momentum_schedule=MomentumScheduleConfig(),
                ),
            )
    """

    optimizer_class: Type[Optimizer]
    optimizer_kwargs: Dict[str, Any]
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self):
        if "lr" not in self.optimizer_kwargs:
            raise ValueError("optimizer_kwargs must include 'lr'")

    def build_optimizer(self, params) -> Optimizer:
        """Construct the optimizer with the given parameters.

        Args:
            params: Iterable of parameters or param_groups to optimize.

        Returns:
            Configured optimizer instance.
        """
        return self.optimizer_class(params, **self.optimizer_kwargs)

    def build_scheduler(
        self, optimizer: Optimizer, training_config: TrainingConfig
    ) -> LRScheduler:
        """Construct the LR scheduler for this optimizer.

        Args:
            optimizer: The optimizer to schedule.
            training_config: Global training config (provides total_steps).

        Returns:
            Configured LR scheduler.
        """
        lr_lambda = self.scheduler_config.build_lr_lambda(training_config.total_steps)
        return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def default_muon_config(
    lr: float = 0.02,
    momentum: float = 0.95,
    weight_decay: float = 1.0,
    warmup_steps: int = 100,
    cooldown_frac: float = 0.5,
    min_lr_ratio: float = 0.1,
    momentum_warmup_steps: int = 300,
    momentum_cooldown_steps: int = 50,
    momentum_min: float = 0.85,
    momentum_max: float = 0.95,
) -> OptimizerConfig:
    """Create default OptimizerConfig for Muon optimizer.

    This provides sensible defaults based on the modded-nanogpt record runs.
    All parameters can be overridden.

    Args:
        lr: Learning rate. Muon typically uses higher LR than Adam. Default: 0.02.
        momentum: Initial momentum value. Default: 0.95.
        weight_decay: Weight decay coefficient. Default: 1.0.
        warmup_steps: LR warmup steps. Default: 100.
        cooldown_frac: Fraction of training for LR cooldown. Default: 0.5.
        min_lr_ratio: Minimum LR as fraction of base. Default: 0.1.
        momentum_warmup_steps: Steps for momentum warmup. Default: 300.
        momentum_cooldown_steps: Steps for momentum cooldown. Default: 50.
        momentum_min: Minimum momentum value. Default: 0.85.
        momentum_max: Maximum momentum value. Default: 0.95.

    Returns:
        Configured OptimizerConfig for Muon.

    Example:
        >>> config = default_muon_config(lr=0.03, weight_decay=0.5)
    """
    return OptimizerConfig(
        optimizer_class=torch.optim.Muon,
        optimizer_kwargs={
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        },
        scheduler_config=SchedulerConfig(
            warmup_steps=warmup_steps,
            cooldown_frac=cooldown_frac,
            min_lr_ratio=min_lr_ratio,
            momentum_schedule=MomentumScheduleConfig(
                warmup_steps=momentum_warmup_steps,
                cooldown_steps=momentum_cooldown_steps,
                min_value=momentum_min,
                max_value=momentum_max,
            ),
        ),
    )


def default_adam_config(
    lr: float = 0.001,
    betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    eps: float = 1e-8,
    warmup_steps: int = 100,
    cooldown_frac: float = 0.5,
    min_lr_ratio: float = 0.1,
) -> OptimizerConfig:
    """Create default OptimizerConfig for AdamW optimizer.

    This provides sensible defaults for the vector/embedding optimizer.
    All parameters can be overridden.

    Args:
        lr: Learning rate. Default: 0.001.
        betas: Adam beta coefficients. Default: (0.9, 0.95).
        weight_decay: Weight decay coefficient. Default: 0.1.
        eps: Epsilon for numerical stability. Default: 1e-8.
        warmup_steps: LR warmup steps. Default: 100.
        cooldown_frac: Fraction of training for LR cooldown. Default: 0.5.
        min_lr_ratio: Minimum LR as fraction of base. Default: 0.1.

    Returns:
        Configured OptimizerConfig for AdamW.

    Example:
        >>> config = default_adam_config(lr=0.0005, weight_decay=0.05)
    """
    return OptimizerConfig(
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        },
        scheduler_config=SchedulerConfig(
            warmup_steps=warmup_steps,
            cooldown_frac=cooldown_frac,
            min_lr_ratio=min_lr_ratio,
            momentum_schedule=None,  # Adam doesn't use momentum scheduling
        ),
    )
