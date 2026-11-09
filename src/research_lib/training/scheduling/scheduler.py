"""
Runtime parameter scheduler for applying schedules to optimizers.

This module provides the :class:`ParamScheduler` class, which binds parameter
schedules to an optimizer at runtime. It is analogous to PyTorch's LRScheduler
but supports arbitrary optimizer parameters.

Design Principles:
    - Created AFTER optimizer exists (typically in on_fit_start)
    - STATEFUL: tracks current step count
    - CHECKPOINTABLE: use state_dict/load_state_dict for resuming

Example:
    Basic usage::

        from research_lib.training.scheduling import ParamScheduler, WarmupStableDecaySchedule

        lr_schedule = WarmupStableDecaySchedule(param_name="lr", peak=0.01)
        momentum_schedule = WarmupStableDecaySchedule(param_name="momentum", peak=0.95)

        scheduler = ParamScheduler(
            optimizer=optimizer,
            global_schedules=[lr_schedule, momentum_schedule],
            total_steps=100000,
        )

        for step in range(total_steps):
            train_batch()
            optimizer.step()
            scheduler.step()

    With per-group overrides::

        scheduler = ParamScheduler(
            optimizer=optimizer,
            global_schedules=[lr_schedule],
            group_overrides={
                1: [WarmupStableDecaySchedule(param_name="lr", peak=0.001)],
            },
            total_steps=100000,
        )

    Checkpointing::

        # Save
        checkpoint["scheduler_state"] = scheduler.state_dict()

        # Load
        scheduler.load_state_dict(checkpoint["scheduler_state"])

See Also:
    - :class:`research_lib.training.scheduling.ParamSchedule` for schedule primitives
    - :class:`research_lib.training.scheduling.WarmupStableDecaySchedule` for presets
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from torch.optim import Optimizer

from .schedules import ParamSchedule


class ParamScheduler:
    """Applies parameter schedules to an optimizer at runtime.

    Analogous to torch.optim.lr_scheduler.LRScheduler, but for arbitrary
    optimizer parameters (lr, momentum, betas, etc.).

    This class is STATEFUL—it tracks the current step count. Use state_dict()
    and load_state_dict() for checkpointing.

    Attributes:
        optimizer: The optimizer to schedule.
        global_schedules: Schedules applied to all param groups by default.
        group_overrides: Per-group schedule overrides (replaces global for that group).
        total_steps: Total training steps (for schedule computation).

    Example:
        >>> from research_lib.training.scheduling import ParamScheduler, WarmupStableDecaySchedule
        >>>
        >>> lr_schedule = WarmupStableDecaySchedule(param_name="lr", peak=0.01)
        >>> scheduler = ParamScheduler(
        ...     optimizer=optimizer,
        ...     global_schedules=[lr_schedule],
        ...     total_steps=100000,
        ... )
        >>>
        >>> for step in range(total_steps):
        ...     train_batch()
        ...     optimizer.step()
        ...     scheduler.step()

    Checkpointing:
        >>> # Save
        >>> checkpoint["scheduler_state"] = scheduler.state_dict()
        >>> # Load
        >>> scheduler.load_state_dict(checkpoint["scheduler_state"])

    Note:
        Call scheduler.step() AFTER optimizer.step(). Optimizer step ``s``
        (0-based) runs at ``schedule(s)``: the value for step 0 is applied at
        construction, and each ``step()`` advances the counter and sets the
        value for the new step.

    Note:
        Override semantics: group_overrides[i] completely REPLACES global_schedules
        for group i. They do not merge.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        global_schedules: List[ParamSchedule],
        total_steps: int,
        group_overrides: Optional[Dict[int, List[ParamSchedule]]] = None,
    ) -> None:
        """Initialize the parameter scheduler.

        Args:
            optimizer: The optimizer to schedule.
            global_schedules: Schedules applied to all param groups by default.
            total_steps: Total number of optimizer steps for the training run.
            group_overrides: Per-group schedule overrides. Keys are group indices.
                If a group index is present, those schedules REPLACE (not extend)
                the global_schedules for that group.

        Raises:
            IndexError: If group_overrides contains an index >= num param groups.

        Example:
            Basic usage::

                scheduler = ParamScheduler(
                    optimizer=optimizer,
                    global_schedules=[lr_schedule, momentum_schedule],
                    total_steps=100000,
                )

            With per-group overrides::

                scheduler = ParamScheduler(
                    optimizer=optimizer,
                    global_schedules=[lr_schedule],
                    group_overrides={
                        1: [slower_lr_schedule],  # Different schedule for group 1
                    },
                    total_steps=100000,
                )
        """
        self.optimizer = optimizer
        self.global_schedules = global_schedules
        self.group_overrides = group_overrides or {}
        self.total_steps = total_steps
        self._step_count = 0

        self._validate()
        # Stepping convention: optimizer step ``s`` (0-based) uses
        # ``schedule(s)``.  The value for step 0 is applied here, at
        # construction, so the very first ``optimizer.step()`` already runs
        # at ``schedule(0)`` rather than at whatever LR the optimizer was
        # constructed with.
        self._apply_step(0)

    def _validate(self) -> None:
        """Validate group_overrides against optimizer structure."""
        num_groups = len(self.optimizer.param_groups)
        for idx in self.group_overrides:
            if idx >= num_groups:
                raise IndexError(
                    f"group_overrides contains index {idx}, but optimizer "
                    f"only has {num_groups} param groups (indices 0-{num_groups - 1})"
                )

    def _get_schedules_for_group(self, group_idx: int) -> List[ParamSchedule]:
        """Get applicable schedules for a param group.

        Args:
            group_idx: The param group index.

        Returns:
            List of schedules. Returns group_overrides[group_idx] if present,
            otherwise returns global_schedules.
        """
        if group_idx in self.group_overrides:
            return self.group_overrides[group_idx]
        return self.global_schedules

    def _apply_step(self, count: int) -> None:
        """Set every param group's scheduled value for optimizer step ``count``."""
        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            schedules = self._get_schedules_for_group(group_idx)
            for schedule in schedules:
                param_group[schedule.param_name] = schedule(count, self.total_steps)

    def step(self) -> None:
        """Advance to the next optimizer step and set its scheduled value.

        Call once per optimizer step, **after** ``optimizer.step()``.  The
        counter is always incremented, then the value for the new count is
        applied — but only while ``count < total_steps``.  After the final
        ``optimizer.step()`` (at ``count == total_steps``) the pending value
        would describe a step that never runs, and a user schedule may not
        tolerate being evaluated there (the built-in schedules clamp, so
        they are unaffected), so it is skipped.  Optimizer step ``s``
        (0-based) runs at ``schedule(s)``; the value set after step ``s``
        is what step ``s + 1`` uses.
        """
        self._step_count += 1
        if self._step_count < self.total_steps:
            self._apply_step(self._step_count)

    def get_current_step(self) -> int:
        """Return current step count.

        Returns:
            The number of times step() has been called.
        """
        return self._step_count

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing.

        Returns:
            Dictionary containing step_count for restoration.

        Example:
            >>> state = scheduler.state_dict()
            >>> torch.save({"scheduler": state}, "checkpoint.pt")
        """
        return {"step_count": self._step_count}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint.

        Restores the step count and re-applies the scheduled value for the
        restored step, so a resumed run continues with the exact value its
        next ``optimizer.step()`` should use.  When the saved count is a
        finished schedule (``>= total_steps``), the schedule is applied at
        ``total_steps - 1`` - the last step that actually ran - instead of
        evaluating a step that never runs.

        Args:
            state: Dictionary from state_dict().

        Example:
            >>> checkpoint = torch.load("checkpoint.pt")
            >>> scheduler.load_state_dict(checkpoint["scheduler"])
        """
        self._step_count = int(state["step_count"])
        if self._step_count < self.total_steps:
            self._apply_step(self._step_count)
        elif self.total_steps > 0:
            # A finished schedule: keep the saved count but apply the value
            # of the last step that ran, not the never-to-run step total.
            self._apply_step(self.total_steps - 1)
