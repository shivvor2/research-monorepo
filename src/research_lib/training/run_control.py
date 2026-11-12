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
import os
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


def read_record(path: Path) -> dict[str, Any] | None:
    """The run record at ``path``, or ``None`` when missing/unparseable or
    not a JSON object.

    Catches ``OSError`` (missing), ``json.JSONDecodeError`` and
    ``UnicodeDecodeError`` (not valid JSON / not UTF-8); a parse result
    that is not a ``dict`` is also unreadable as a record (``None``).

    Args:
        path: The record's JSON path.

    Returns:
        The parsed record dict, or ``None``.
    """
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


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


def _slurm_task_count() -> int | None:
    """The number of tasks a Slurm allocation launched, or ``None`` when
    it cannot be established (no task-count variable, or an unparseable
    one)."""
    for name in ("SLURM_STEP_NUM_TASKS", "SLURM_NTASKS"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return None
    return None


def lightning_launched_rank() -> int:
    """The rank this process holds under Lightning's subprocess launcher.

    Only Lightning's own launcher is supported: the training script is
    started once (rank 0), and at ``Trainer.fit`` Lightning re-executes
    the script for ranks 1..W-1 as child processes.  A child is told
    apart from an ordinary rank-0 start by the environment the launcher
    sets: ``LOCAL_RANK`` plus ``MASTER_PORT``, ``NODE_RANK`` and
    ``WORLD_SIZE`` (lightning 2.6.6, the version pinned in ``uv.lock``;
    it never sets a global ``RANK``).  Lightning also sets ``LOCAL_RANK``
    (0) in the **rank-0** process's environment while ``fit`` runs its
    subprocess launcher; that is harmless for a single run — a child
    only ever exists there, because several devices under a Hydra
    multirun are refused — and it is the reason multiruns with several
    devices are not supported.  Launches that start ranks independently
    are refused here:

    - ``torchrun`` sets ``RANK``, which Lightning never does;
    - a Slurm allocation sets ``SLURM_PROCID`` for every task, including
      a one-task ``sbatch`` script, so it is refused only when several
      tasks were launched (``SLURM_STEP_NUM_TASKS`` / ``SLURM_NTASKS``
      parse to more than one) or the task count cannot be established;
      a one-task allocation returns the ``LOCAL_RANK``-based rank as
      usual (multi-device runs inside Slurm stay unsupported — see
      :func:`refuse_multi_device_under_slurm`);
    - ``LOCAL_RANK > 0`` without the launcher's master variables is not
      a Lightning child and is refused too.  ``MASTER_PORT`` is the
      durable signal: Lightning's cluster environment sets it for its
      children (with ``MASTER_ADDR``, ``NODE_RANK`` and ``WORLD_SIZE``)
      and only removes ``WORLD_SIZE`` in its post-fit teardown, so a
      legitimate child still carries ``MASTER_PORT`` (and the gate that
      calls this after ``fit`` must not require ``WORLD_SIZE``).

    Returns:
        The rank: 0 for the process that runs ``fit``, ``i > 0`` for a
        child rank Lightning started.  At world size 1 this is 0 (no
        rank variables are set).

    Raises:
        RuntimeError: An unsupported launcher's variables are present
            (``RANK``, a multi-task or unparseable Slurm allocation, or
            ``LOCAL_RANK`` without the launcher's ``MASTER_PORT``).
    """
    if "RANK" in os.environ:
        raise RuntimeError(
            "process was started by an unsupported launcher (found RANK); "
            "only Lightning's own subprocess launcher is supported: start "
            "the script once and let Lightning start the other ranks "
            "(torchrun is refused)"
        )
    if "SLURM_PROCID" in os.environ:
        ntasks = _slurm_task_count()
        if ntasks is None or ntasks > 1:
            raise RuntimeError(
                "process was started by an unsupported launcher: a Slurm "
                "allocation with several tasks (SLURM_PROCID is set, and "
                "SLURM_STEP_NUM_TASKS/SLURM_NTASKS are missing or > 1); "
                "only Lightning's own subprocess launcher is supported "
                "for multi-device runs — inside an allocation each job "
                "runs on one device"
            )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank > 0 and "MASTER_PORT" not in os.environ:
        raise RuntimeError(
            f"process was started by an unsupported launcher: LOCAL_RANK="
            f"{local_rank} is set without the launcher's MASTER_PORT; "
            f"only Lightning's own subprocess launcher is supported: "
            f"start the script once and let Lightning start the other "
            f"ranks"
        )
    return local_rank


def refuse_multi_device_under_slurm(devices: Any) -> None:
    """Refuse more than one device inside a Slurm allocation.

    Slurm sets ``SLURM_JOB_ID`` in the environment of every ``sbatch`` /
    ``srun`` job, including a one-task one.  Inside an allocation each
    job runs a single device (the pilots' run machinery is per-machine
    and the run lock is not shared across allocation tasks); multi-device
    runs use Lightning's own launcher outside Slurm.

    Args:
        devices: The configured ``training.devices`` value.

    Raises:
        RuntimeError: When ``SLURM_JOB_ID`` is set and ``devices`` is not
            the single integer 1.
    """
    if "SLURM_JOB_ID" in os.environ and not (isinstance(devices, int) and devices == 1):
        raise RuntimeError(
            f"training.devices={devices} inside a Slurm allocation "
            f"(SLURM_JOB_ID is set) is not supported: each job runs on "
            f"one device inside an allocation; multi-device runs use "
            f"Lightning's own launcher outside Slurm"
        )


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
    record = read_record(record_path)
    if record is None or record.get("status") != RUN_DONE_STATUS:
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
