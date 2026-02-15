"""
Low-level utilities for applying schedules and querying optimizer state.

This module provides functions for direct optimizer manipulation:
    - :func:`apply_schedule_to_param_group` for single schedule application
    - :func:`get_param_value` for reading optimizer parameters
    - :func:`get_current_lr` for convenience LR access

Note:
    For most use cases, prefer :class:`ParamScheduler` over these utilities.
    These functions are primarily for custom scheduling logic or debugging.

Example:
    Reading optimizer state::

        from research_lib.training.scheduling.utils import get_param_value, get_current_lr

        current_lr = get_current_lr(optimizer)
        momentum = get_param_value(optimizer, "momentum")

See Also:
    - :class:`research_lib.training.scheduling.ParamScheduler` for runtime scheduling
    - :class:`research_lib.training.scheduling.ParamSchedule` for schedule definition
"""

from __future__ import annotations

from typing import Any, List, Sequence

from torch.optim import Optimizer

from .schedules import ParamSchedule


def apply_schedule_to_param_group(
    optimizer: Optimizer,
    schedule: ParamSchedule,
    group_idx: int,
    step: int,
    total_steps: int,
) -> None:
    """Apply a schedule to a specific param group.

    This is the low-level function that actually modifies the optimizer's
    param_groups. Use this when you need fine-grained control.

    Args:
        optimizer: The optimizer to update.
        schedule: The schedule to apply.
        group_idx: Index of the param group to update.
        step: Current training step.
        total_steps: Total training steps.

    Raises:
        IndexError: If group_idx is out of range.

    Example:
        >>> apply_schedule_to_param_group(
        ...     optimizer, lr_schedule, group_idx=0, step=100, total_steps=1000
        ... )
    """
    if group_idx < 0 or group_idx >= len(optimizer.param_groups):
        raise IndexError(
            f"group_idx={group_idx} out of range [0, {len(optimizer.param_groups)})"
        )

    new_value = schedule(step, total_steps)
    optimizer.param_groups[group_idx][schedule.param_name] = new_value


def apply_schedule_to_all_groups(
    optimizer: Optimizer,
    schedule: ParamSchedule,
    step: int,
    total_steps: int,
) -> None:
    """Apply a schedule to all param groups.

    Convenience function when you want the same schedule for all groups.

    Args:
        optimizer: The optimizer to update.
        schedule: The schedule to apply.
        step: Current training step.
        total_steps: Total training steps.

    Example:
        >>> apply_schedule_to_all_groups(optimizer, lr_schedule, step=100, total_steps=1000)
    """
    for group_idx in range(len(optimizer.param_groups)):
        apply_schedule_to_param_group(optimizer, schedule, group_idx, step, total_steps)


def get_param_value(
    optimizer: Optimizer,
    param_name: str,
    group_idx: int = 0,
) -> Any:
    """Get a parameter value from a specific param group.

    This function returns the raw value and performs no further processing.
    For tuple params like 'betas', it returns the full tuple.

    Args:
        optimizer: The optimizer to query.
        param_name: The parameter key (e.g., 'lr', 'momentum', 'betas').
        group_idx: Index of the param group. Default: 0.

    Returns:
        The value from param_groups[group_idx][param_name].
        Returns None if the key doesn't exist.

    Raises:
        IndexError: If group_idx is out of range.

    Example:
        >>> lr = get_param_value(optimizer, "lr")
        >>> betas = get_param_value(optimizer, "betas")
        >>> beta1 = betas[0] if betas else None
    """
    if group_idx < 0 or group_idx >= len(optimizer.param_groups):
        raise IndexError(
            f"group_idx={group_idx} out of range [0, {len(optimizer.param_groups)})"
        )
    return optimizer.param_groups[group_idx].get(param_name)


def get_param_values(
    optimizer: Optimizer,
    param_name: str,
    group_indices: Sequence[int],
) -> List[Any]:
    """Get a parameter value from multiple param groups.

    Args:
        optimizer: The optimizer to query.
        param_name: The parameter key.
        group_indices: Indices of param groups to query.

    Returns:
        List of values, one per group index in the order specified.

    Raises:
        IndexError: If any group index is out of range.

    Example:
        >>> lrs = get_param_values(optimizer, "lr", [0, 1])
    """
    return [get_param_value(optimizer, param_name, idx) for idx in group_indices]


def get_current_lr(optimizer: Optimizer, group_idx: int = 0) -> float:
    """Get the current learning rate.

    Convenience wrapper around get_param_value for 'lr'.

    Args:
        optimizer: The optimizer to query.
        group_idx: Which param group to query. Default: 0.

    Returns:
        The learning rate value.

    Raises:
        IndexError: If group_idx is out of range.
        KeyError: If 'lr' is not in the param group.

    Example:
        >>> lr = get_current_lr(optimizer)
    """
    lr = get_param_value(optimizer, "lr", group_idx)
    if lr is None:
        raise KeyError(f"'lr' not found in param_group {group_idx}")
    return lr
