"""
Scheduling utilities for optimizer param_group values.

This module provides functions to update optimizer param_groups based on
schedule configurations. It handles both LR scheduling (via PyTorch's
LRScheduler) and custom param_group value scheduling (momentum, betas, etc.).

The main function :func:`update_param_group_schedules` should be called each
step after the LR scheduler step to update any custom scheduled values.

Example:
    In a training loop::

        for step in range(total_steps):
            # ... forward, backward ...
            optimizer.step()
            lr_scheduler.step()

            # Update custom schedules (momentum, etc.)
            update_param_group_schedules(
                optimizer=optimizer,
                scheduler_config=scheduler_config,
                step=step,
                total_steps=total_steps,
            )

See Also:
    - :class:`research_lib.training.configs.SchedulerConfig`
    - :class:`research_lib.training.configs.CustomScheduleConfig`
"""

from typing import List, Tuple, Union

from torch.optim import Optimizer

from .configs import CustomScheduleConfig, SchedulerConfig


def update_single_param_schedule(
    optimizer: Optimizer,
    schedule: CustomScheduleConfig,
    step: int,
    total_steps: int,
) -> None:
    """Update a single param_group value according to its schedule.

    This function modifies the optimizer's param_groups in-place.

    Args:
        optimizer: The optimizer to update.
        schedule: The schedule configuration for this parameter.
        step: Current training step (0-indexed).
        total_steps: Total number of training steps.

    Note:
        If the param_name doesn't exist in param_groups, this function will
        add it (which is harmless but effectless for unknown params).

        If param_index is specified for a tuple/list param, only that index
        is modified. The original tuple is replaced with a new tuple.
    """
    new_value = schedule.get_value(step, total_steps)

    for param_group in optimizer.param_groups:
        if schedule.param_index is not None:
            # Handle tuple/list params (e.g., betas)
            current = param_group.get(schedule.param_name)
            if current is not None and isinstance(current, (tuple, list)):
                new_tuple = list(current)
                new_tuple[schedule.param_index] = new_value
                param_group[schedule.param_name] = tuple(new_tuple)
            # If param doesn't exist or isn't a tuple, skip silently
        else:
            # Scalar param
            param_group[schedule.param_name] = new_value


def update_param_group_schedules(
    optimizer: Optimizer,
    scheduler_config: SchedulerConfig,
    step: int,
    total_steps: int,
) -> None:
    """Update all custom param_group schedules for an optimizer.

    This should be called each step after the LR scheduler step. It handles
    momentum scheduling and any custom schedules defined in the config.

    Args:
        optimizer: The optimizer to update.
        scheduler_config: The scheduler configuration containing custom schedules.
        step: Current training step (0-indexed).
        total_steps: Total number of training steps.

    Example:
        >>> update_param_group_schedules(
        ...     optimizer=muon_optimizer,
        ...     scheduler_config=muon_config.scheduler_config,
        ...     step=current_step,
        ...     total_steps=training_config.total_steps,
        ... )
    """
    for schedule in scheduler_config.get_all_custom_schedules():
        update_single_param_schedule(optimizer, schedule, step, total_steps)


def get_current_lr(optimizer: Optimizer) -> float:
    """Get the current learning rate from an optimizer.

    Args:
        optimizer: The optimizer to query.

    Returns:
        The learning rate from the first param_group.
    """
    return optimizer.param_groups[0]["lr"]


def get_param_group_value(
    optimizer: Optimizer,
    param_name: str,
    param_index: int | None = None,
) -> float | None:
    """Get a value from an optimizer's param_groups.

    Args:
        optimizer: The optimizer to query.
        param_name: The key to look up in param_groups.
        param_index: If the value is a tuple/list, which index to return.

    Returns:
        The value, or None if the param doesn't exist.
    """
    value = optimizer.param_groups[0].get(param_name)
    if value is None:
        return None
    if param_index is not None and isinstance(value, (tuple, list)):
        return value[param_index]
    return value
