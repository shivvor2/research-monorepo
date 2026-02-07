"""
Configuration dataclasses for training with multiple optimizers.

This module provides a flexible configuration system for training neural networks
with different optimizers for different parameter groups (e.g., Muon for weight
matrices, AdamW for embeddings).

Design Principles:
    1. **Illegal states should be unrepresentable**: Required configs are not Optional.
    2. **Minimal coupling**: Schedules describe curves, configs describe where
       they apply, optimizer instance only required at application time.
    3. **Layered architecture**: Users can enter at any layer of abstraction.
    4. **LambdaLR abandonment**: LR is handled via ParamSchedule like any other param.

Example:
    Basic usage with schedules::

        from research_lib.training.configs import (
            TrainingConfig,
            OptimizerConfig,
            ParamGroupConfig,
            update_optimizer_schedules,
            default_muon_config,
            default_adam_config,
        )
        from research_lib.training.scheduling import WarmupStableDecaySchedule

        training_config = TrainingConfig(total_steps=10000)
        muon_config = default_muon_config()
        adam_config = default_adam_config()

    Custom optimizer configuration with per-group schedules::

        config = OptimizerConfig(
            optimizer_class=torch.optim.Muon,
            optimizer_kwargs={"lr": 0.03, "momentum": 0.9, "weight_decay": 0.5},
            schedules=[
                WarmupStableDecaySchedule(param_name="lr", warmup_steps=100),
                WarmupStableDecaySchedule(param_name="momentum", warmup_steps=300),
            ],
            param_group_configs=[
                ParamGroupConfig(
                    group_index=1,
                    schedules=[
                        WarmupStableDecaySchedule(param_name="lr", warmup_steps=50),
                    ],
                    param_group_kwargs={"lr": 0.001},
                ),
            ],
        )

See Also:
    - :mod:`research_lib.training.scheduling` for schedule classes and utilities
    - :mod:`research_lib.training.modules` for the LightningModule implementation
    - :mod:`research_lib.training.param_utils` for parameter partitioning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch
from torch.optim import Optimizer

from .scheduling import ParamSchedule, WarmupStableDecaySchedule
from .scheduling.utils import apply_schedule_to_param_group


@dataclass
class TrainingConfig:
    """Global training parameters that must be consistent across all optimizers.

    This config contains ONLY parameters where mismatches between optimizers
    would break training semantically. Parameters that can legitimately differ
    between optimizers (warmup_steps, cooldown_frac) belong in schedule configs.

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

    def __post_init__(self) -> None:
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
class ParamGroupConfig:
    """Configuration for a specific optimizer param group.

    Used to specify schedules and initial values that differ from the
    optimizer-level defaults for a specific param group.

    When used in OptimizerConfig:
        - Schedules in this config REPLACE (not merge with) global schedules
        - param_group_kwargs override optimizer_kwargs for this group

    Attributes:
        group_index: The index of the param group this config applies to.
            Must be in range [0, num_param_groups) at application time.
        schedules: List of schedules to apply to this param group.
            These REPLACE any global schedules from OptimizerConfig.
        param_group_kwargs: Initial parameter values that override optimizer_kwargs.
            Example: {"lr": 0.001} to give this group a different initial LR.

    Example:
        Different LR for classifier head::

            ParamGroupConfig(
                group_index=1,  # Classifier head
                schedules=[WarmupStableDecaySchedule(
                    param_name="lr",
                    warmup_steps=50,
                    cooldown_frac=0.3,
                    min_value=0.0,
                    max_value=1.0,
                )],
                param_group_kwargs={"lr": 0.001},  # Lower initial LR
            )
    """

    group_index: int
    schedules: List["ParamSchedule"] = field(default_factory=list)
    param_group_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.group_index < 0:
            raise ValueError(
                f"group_index must be non-negative, got {self.group_index}"
            )


@dataclass
class OptimizerConfig:
    """Complete configuration for an optimizer and its parameter schedules.

    This bundles together:
        1. The optimizer class and initialization kwargs
        2. Global schedules that apply to all param groups
        3. Per-group schedule overrides via ParamGroupConfig

    Schedule Precedence:
        - Global schedules (self.schedules) apply to ALL param groups by default
        - ParamGroupConfig.schedules REPLACE global schedules for that specific group
        - If a group has no ParamGroupConfig, it uses global schedules

    Attributes:
        optimizer_class: The optimizer class (e.g., torch.optim.AdamW).
        optimizer_kwargs: Kwargs passed to optimizer constructor. Must include 'lr'.
        schedules: Global schedules applied to all param groups (unless overridden).
        param_group_configs: Per-group configurations that override global schedules.

    Example:
        Single param group (common case)::

            config = OptimizerConfig(
                optimizer_class=torch.optim.AdamW,
                optimizer_kwargs={"lr": 0.001, "weight_decay": 0.1},
                schedules=[
                    WarmupStableDecaySchedule(param_name="lr", warmup_steps=100),
                ],
            )

        Multiple param groups with different schedules::

            config = OptimizerConfig(
                optimizer_class=torch.optim.SGD,
                optimizer_kwargs={"lr": 0.01, "momentum": 0.9},
                schedules=[
                    WarmupStableDecaySchedule(param_name="momentum", warmup_steps=300),
                ],
                param_group_configs=[
                    ParamGroupConfig(
                        group_index=0,  # Backbone
                        schedules=[
                            WarmupStableDecaySchedule(param_name="lr", cooldown_frac=0.5),
                            WarmupStableDecaySchedule(param_name="momentum", warmup_steps=300),
                        ],
                    ),
                    ParamGroupConfig(
                        group_index=1,  # Head
                        schedules=[
                            WarmupStableDecaySchedule(param_name="lr", warmup_steps=50),
                        ],
                        param_group_kwargs={"lr": 0.001},
                    ),
                ],
            )

    Note:
        Param group index validation happens at application time (when optimizer
        exists), not at config construction time.
    """

    optimizer_class: Type[Optimizer]
    optimizer_kwargs: Dict[str, Any]
    schedules: List["ParamSchedule"] = field(default_factory=list)
    param_group_configs: List[ParamGroupConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if "lr" not in self.optimizer_kwargs:
            raise ValueError("optimizer_kwargs must include 'lr'")

    def build_optimizer(self, params) -> Optimizer:
        """Construct the optimizer with the given parameters.

        Args:
            params: Iterable of parameters or list of param group dicts.
                If list of dicts, each dict should have 'params' key.

        Returns:
            Configured optimizer instance.

        Note:
            This method validates that ParamGroupConfig.group_index values
            are valid for the created optimizer.
        """
        optimizer = self.optimizer_class(params, **self.optimizer_kwargs)

        # Validate param group configs
        num_groups = len(optimizer.param_groups)
        for pgc in self.param_group_configs:
            if pgc.group_index >= num_groups:
                raise IndexError(
                    f"ParamGroupConfig.group_index={pgc.group_index} exceeds "
                    f"optimizer param group count ({num_groups})"
                )

            # Apply param_group_kwargs overrides
            if pgc.param_group_kwargs:
                optimizer.param_groups[pgc.group_index].update(pgc.param_group_kwargs)

        return optimizer

    def get_schedules_for_group(self, group_index: int) -> List["ParamSchedule"]:
        """Get the schedules that apply to a specific param group.

        Args:
            group_index: The param group index.

        Returns:
            List of schedules. Returns ParamGroupConfig.schedules if one exists
            for this group, otherwise returns global self.schedules.
        """
        for pgc in self.param_group_configs:
            if pgc.group_index == group_index:
                return pgc.schedules
        return self.schedules


# =============================================================================
# Schedule Update Utility
# =============================================================================


def update_optimizer_schedules(
    optimizer: Optimizer,
    config: OptimizerConfig,
    step: int,
    total_steps: int,
) -> None:
    """Update all scheduled parameters for an optimizer.

    This is the main entry point for applying schedules during training.
    It handles the precedence logic between global and per-group schedules.

    Precedence:
        1. Global schedules (config.schedules) apply to all param groups
        2. ParamGroupConfig.schedules REPLACE global schedules for that group

    Args:
        optimizer: The optimizer to update.
        config: The OptimizerConfig containing schedules.
        step: Current training step (0-indexed).
        total_steps: Total number of training steps.

    Raises:
        IndexError: If any ParamGroupConfig.group_index exceeds param group count.

    Example:
        In a training loop::

            for step in range(total_steps):
                loss = compute_loss(model, batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Apply schedules
                update_optimizer_schedules(optimizer, config, step, total_steps)
    """
    num_groups = len(optimizer.param_groups)

    # Validate param group indices
    for pgc in config.param_group_configs:
        if pgc.group_index >= num_groups:
            raise IndexError(
                f"ParamGroupConfig.group_index={pgc.group_index} exceeds "
                f"optimizer param group count ({num_groups})"
            )

    # Build group_index -> schedules map
    group_schedules: Dict[int, List[ParamSchedule]] = {
        i: list(config.schedules) for i in range(num_groups)
    }

    # Override with per-group configs (REPLACE, not merge)
    for pgc in config.param_group_configs:
        group_schedules[pgc.group_index] = list(pgc.schedules)

    # Apply schedules
    for group_idx, schedules in group_schedules.items():
        for schedule in schedules:
            apply_schedule_to_param_group(
                optimizer, schedule, group_idx, step, total_steps
            )


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
    # Import here to avoid circular imports
    from .scheduling import WarmupStableDecaySchedule

    lr_schedule = WarmupStableDecaySchedule(
        param_name="lr",
        warmup_steps=warmup_steps,
        cooldown_frac=cooldown_frac,
        min_value=min_lr_ratio,
        max_value=1.0,
        decay_type="cosine",
    )

    momentum_schedule = WarmupStableDecaySchedule(
        param_name="momentum",
        warmup_steps=momentum_warmup_steps,
        cooldown_steps=momentum_cooldown_steps,
        min_value=momentum_min,
        max_value=momentum_max,
    )

    return OptimizerConfig(
        optimizer_class=torch.optim.Muon,
        optimizer_kwargs={
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        },
        schedules=[lr_schedule, momentum_schedule],
    )


def default_adamw_config(
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
    # Import here to avoid circular imports
    from .scheduling import WarmupStableDecaySchedule

    lr_schedule = WarmupStableDecaySchedule(
        param_name="lr",
        warmup_steps=warmup_steps,
        cooldown_frac=cooldown_frac,
        min_value=min_lr_ratio,
        max_value=1.0,
        decay_type="cosine",
    )

    return OptimizerConfig(
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        },
        schedules=[lr_schedule],
    )
