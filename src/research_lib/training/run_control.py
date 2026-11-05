"""Run control for one training job: precision, fit guard, shards, skip and
claims.

The pieces here are the **general** layer of run control: nothing about
a project's groups, task names, Hydra config paths, run-ID formats or
folder layouts appears in this module. A different project in this
repository, with a different grouping of runs and a different folder
layout, can use these functions without editing them; the project-specific
bits (which config keys carry the job number and the STOP file, how a run
ID and a group name are derived) are supplied by the caller or by the
project layer.

Three helpers read the record format :class:`RunSummaryCallback` writes
(``status``, ``config_hash``, ``seed``): :func:`should_skip`,
:func:`claim_run` and the shared :data:`RUN_DONE_STATUS` constant.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.trainer.states import TrainerStatus

from research_lib.utils.run_lock import exclusive_lock

logger = logging.getLogger(__name__)

# The lifecycle status at which a run is considered finished: the run
# record's ``status`` field is exactly this when the fit ended normally
# AND the test pass completed on the final weights. One definition,
# shared by the callback that writes the record, the skip check
# (:func:`should_skip`) and the pull that moves records about.
RUN_DONE_STATUS = "test_done"


def disable_tf32() -> dict[str, bool]:
    """fp32 without TF32: both PyTorch TF32 switches off.

    ``torch.set_float32_matmul_precision`` is left untouched: the matmul
    flag covers the reference layers' fp32 matmuls, and the cuDNN flag
    covers BaseConv / the short convs, which run through cuDNN and
    default to TF32 on.

    Returns:
        The values read back, for the run record's ``precision_flags``.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def assert_fit_finished(trainer: L.Trainer) -> None:
    """Raise unless the fit ended normally.

    With Lightning < 2.4, Ctrl-C made ``fit()`` return normally and the
    script would test a half-trained model and record ``test_done``; the
    lock pins >= 2.4, the guard keeps the invariant independent of that.
    ``status == "test_done"`` means fit ended normally (max steps or
    early stop) AND the test pass completed on the final weights.

    Raises:
        RuntimeError: Naming the status, if it is not ``FINISHED``.
    """
    if trainer.state.status != TrainerStatus.FINISHED:
        raise RuntimeError(
            f"fit() ended with trainer.state.status={trainer.state.status!r}, "
            f"not FINISHED; the model is not fully trained, refusing to test"
        )


def in_shard(job_num: int, shard_index: int, num_shards: int) -> bool:
    """Round-robin shard membership: job ``k`` runs only in the process
    with ``shard_index == k % num_shards``.

    All shards must be launched with the identical sweep command; the
    round-robin keeps every shard moving along the sweep's priority order.

    Raises:
        ValueError: If ``num_shards < 1`` or ``shard_index`` is outside
            ``[0, num_shards)``.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index {shard_index} out of range for num_shards {num_shards}"
        )
    return job_num % num_shards == shard_index


def should_skip(record_path: Path, identity: tuple[str, Any]) -> str | None:
    """Skip reason for an existing run record, or ``None`` to run.

    A record is done only when its status is ``test_done`` and its
    ``(config_hash, seed)`` equals this run's identity (the identity
    fields :class:`RunSummaryCallback` writes). Any other record falls
    through: a different identity is refused by ``RunSummaryCallback``;
    a non-done record is overwritten by the re-run; a missing or
    unreadable record runs.

    Args:
        record_path: The candidate run record's JSON path.
        identity: The run's ``(config_hash, seed)``.

    Returns:
        A human-readable skip reason, or ``None`` to run.
    """
    if not record_path.exists():
        return None
    try:
        with open(record_path, encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("status") != RUN_DONE_STATUS:
        return None
    if (record.get("config_hash"), record.get("seed")) != identity:
        return None
    return "already test_done"


@contextmanager
def claim_run(
    lock_path: Path | str,
    record_path: Path | str | None,
    identity: tuple[str, Any],
) -> Generator[str | None, None, None]:
    """Acquire the run lock and decide whether to run.

    Yields a skip reason (or ``None`` to run) while the lock is held:
    the record is read only after this process owns the run, so a run
    that another process just completed is skipped instead of re-run (a
    check before the lock would race with a process that finishes
    between the check and the lock). ``record_path=None`` skips the
    record check (the run always proceeds once the lock is acquired).

    Args:
        lock_path: The per-run lock file.
        record_path: The run's record file, or ``None`` when the skip
            check does not apply (e.g. the smoke and debug configs).
        identity: The run's ``(config_hash, seed)``.

    Yields:
        str | None: The skip reason, or ``None`` to run.
    """
    with exclusive_lock(lock_path) as acquired:
        if not acquired:
            yield "running in another process"
            return
        reason = None
        if record_path is not None:
            reason = should_skip(Path(record_path), identity)
        yield reason