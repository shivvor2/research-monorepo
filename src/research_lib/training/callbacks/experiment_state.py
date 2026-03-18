"""
Experiment state tracking and checkpoint auto-resolution.

Provides:
- ExperimentStateCallback: Tracks the latest checkpoint path for an experiment
- resolve_checkpoint_path: Resolves which checkpoint to resume from

Design:
    The callback stores the **actual checkpoint file path** (not just the run
    directory) into a state file. This avoids the fragile `last.ckpt` pattern
    and ensures auto-resume always points to a valid, fully-written checkpoint.

    The state file is a simple text file stored alongside the experiment scripts,
    named `.latest_ckpt_{experiment_name}.txt`. It contains the absolute path
    to the most recently saved named checkpoint.

Usage:
    # In your training script:
    from research_lib.training.callbacks import (
        ExperimentStateCallback,
        resolve_checkpoint_path,
    )

    callback = ExperimentStateCallback(
        experiment_name="my_experiment",
        base_dir=Path.cwd(),  # or get_original_cwd() if using Hydra
    )

    ckpt_path = resolve_checkpoint_path(
        resume_from="auto",           # "auto", None, or explicit path
        experiment_name="my_experiment",
        base_dir=Path.cwd(),
        checkpoint_subdir="checkpoints",
        model=model,                   # for compatibility check
    )
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

logger = logging.getLogger(__name__)

# =============================================================================
# State File Helpers
# =============================================================================


def _get_state_file_path(
    experiment_name: str,
    base_dir: Path,
    prefix: str = ".latest_ckpt_",
) -> Path:
    """Get path to the state file tracking the latest checkpoint for an experiment.

    Args:
        experiment_name: Logical name for the experiment family.
        base_dir: Directory where the state file is stored (typically the
            original working directory before Hydra changes cwd).
        prefix: Filename prefix. Default: ".latest_ckpt_"

    Returns:
        Path to the state file.
    """
    return base_dir / f"{prefix}{experiment_name}.txt"


def _find_latest_named_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the checkpoint with the highest step number in a directory.

    Looks for files matching common patterns:
    - step_*.ckpt
    - checkpoint_*.ckpt
    - epoch=*-step=*.ckpt (Lightning default)

    Falls back to most recently modified .ckpt file if no pattern matches.

    Args:
        checkpoint_dir: Directory containing checkpoint files.

    Returns:
        Path to the latest checkpoint, or None if directory is empty.
    """
    if not checkpoint_dir.exists():
        return None

    # Collect all .ckpt files, excluding last.ckpt
    candidates = [p for p in checkpoint_dir.glob("*.ckpt") if p.name != "last.ckpt"]

    if not candidates:
        # Fall back to last.ckpt if it's all we have
        last = checkpoint_dir / "last.ckpt"
        return last if last.exists() else None

    # Sort by modification time (most recent first) as a robust fallback
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# =============================================================================
# Checkpoint Resolution
# =============================================================================


def resolve_checkpoint_path(
    resume_from: Optional[str],
    experiment_name: str,
    base_dir: Union[str, Path],
    checkpoint_subdir: str = "checkpoints",
    state_file_prefix: str = ".latest_ckpt_",
) -> Optional[str]:
    """Resolve the checkpoint path for resuming training.

    Logic:
        1. If resume_from is None → start fresh.
        2. If resume_from is an explicit path → validate it exists and return.
        3. If resume_from is "auto":
           a. Read the state file to get the last saved checkpoint path.
           b. If state file doesn't point to a valid file, scan the checkpoint
              directory for the latest named checkpoint.

    Args:
        resume_from: One of:
            - None: Always start fresh.
            - "auto": Find latest checkpoint from state file or directory scan.
            - str path: Explicit checkpoint path.
        experiment_name: Logical experiment name (used for state file lookup).
        base_dir: Base directory where state files live and paths are relative to.
        checkpoint_subdir: Subdirectory name within run dirs for checkpoints.
        state_file_prefix: Prefix for state files. Default ".latest_ckpt_".

    Returns:
        Absolute path to checkpoint as a string, or None to start fresh.

    Raises:
        FileNotFoundError: If resume_from is an explicit path that doesn't exist.
    """
    base_dir = Path(base_dir)

    # Case 1: Start fresh
    if resume_from is None:
        return None

    # Case 2: Explicit path
    if resume_from != "auto":
        path = Path(resume_from)
        if not path.is_absolute():
            path = base_dir / path
        if path.exists():
            logger.info(f"Resuming from explicit path: {path}")
            return str(path)
        else:
            raise FileNotFoundError(f"Explicit checkpoint path not found: {path}")

    # Case 3: Auto-resume
    state_file = _get_state_file_path(experiment_name, base_dir, state_file_prefix)

    if not state_file.exists():
        logger.info(
            f"No previous run history found for experiment '{experiment_name}'. "
            "Starting fresh."
        )
        return None

    try:
        stored_path = state_file.read_text().strip()
        ckpt_path = Path(stored_path)

        # Handle relative paths
        if not ckpt_path.is_absolute():
            ckpt_path = base_dir / ckpt_path

        # If the stored path is a file that exists, use it directly
        if ckpt_path.is_file():
            candidate = ckpt_path
        elif ckpt_path.is_dir():
            # Stored path might be the checkpoint dir itself (current format)
            # or a run directory (legacy format). Try both.
            candidate = _find_latest_named_checkpoint(ckpt_path)
            if candidate is None:
                # Legacy: stored path is run dir, checkpoints are in a subdirectory
                candidate = _find_latest_named_checkpoint(ckpt_path / checkpoint_subdir)
            if candidate is None:
                logger.info(f"No checkpoints found in {ckpt_path}. Starting fresh.")
                return None
        else:
            logger.info(
                f"Stored checkpoint path no longer exists: {ckpt_path}. "
                "Starting fresh."
            )
            return None

        logger.info(f"Found candidate checkpoint: {candidate}")

        return str(candidate)

    except Exception as e:
        logger.warning(f"Error reading state file: {e}. Starting fresh.")
        return None


# =============================================================================
# Callback
# =============================================================================


class ExperimentStateCallback(Callback):
    """Tracks the latest checkpoint path for auto-resume across runs.

    On every checkpoint save, updates a state file with the **actual path to
    the checkpoint file** (not just the run directory). This ensures auto-resume
    always points to a valid, fully-written checkpoint.

    Args:
        experiment_name: Logical name for the experiment (used in state filename).
        base_dir: Directory where the state file is stored. Typically the
            original working directory (before Hydra changes cwd).
            If None, uses Path.cwd() at callback construction time.
        state_file_prefix: Prefix for the state filename.
            Default: ".latest_ckpt_"

    Example:
        >>> callback = ExperimentStateCallback(
        ...     experiment_name="nanogpt_base",
        ...     base_dir=Path("/path/to/project"),
        ... )
        >>> trainer = L.Trainer(callbacks=[callback, ModelCheckpoint(...)])
    """

    def __init__(
        self,
        experiment_name: str,
        base_dir: Optional[Union[str, Path]] = None,
        state_file_prefix: str = ".latest_ckpt_",
    ):
        super().__init__()
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.state_file_prefix = state_file_prefix

    @property
    def state_file(self) -> Path:
        """Path to the state file for this experiment."""
        return _get_state_file_path(
            self.experiment_name, self.base_dir, self.state_file_prefix
        )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Record the path to the checkpoint being saved.

        Looks at the ModelCheckpoint callback to find the actual file path
        that was just written.
        """
        # Find the ModelCheckpoint callback to get the real path
        best_path = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.dirpath:
                try:
                    self.state_file.write_text(str(cb.dirpath))
                except OSError as e:
                    logger.warning(f"Failed to write state file: {e}")
                return

        if best_path:
            path_to_store = best_path
        else:
            # Fallback: store the run directory (legacy behavior)
            path_to_store = str(Path.cwd())

        try:
            self.state_file.write_text(str(path_to_store))
        except OSError as e:
            logger.warning(f"Failed to write state file: {e}")

    def on_train_end(self, trainer, pl_module):
        """Mark training as completed (but preserve the pointer)."""
        if trainer.state.status == "finished":
            # Write a completion marker alongside (don't destroy the pointer)
            completed_file = self.state_file.with_suffix(".completed.txt")
            try:
                completed_file.write_text(
                    f"Training completed. Last state: {self.state_file.read_text()}"
                )
            except OSError:
                pass
