"""Hydra launch helpers shared by this project's pilots.

The module reads the Hydra job and the project's ``launch:`` block: the
job number (for the shard round-robin), the STOP file in the launch
directory, and the run claim (:func:`claimed_run`). :func:`begin_run`
runs the head of one job in the pilots' shared order — precision flags,
STOP file, shard membership — once, up front, before any run-specific
machinery is built (a rank that Lightning's subprocess launcher started
skips the gate: rank 0 already decided it; children exist only for a
single run, since several devices under a Hydra multirun are refused). A
job outside this process's shard is not started; :func:`claimed_run`
wraps the run lock, the skip check and the re-upload of a skipped
finished run.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from research_lib.training.run_control import (
    claim_run,
    disable_tf32,
    in_shard,
    lightning_launched_rank,
    refuse_multi_device_under_slurm,
)

logger = logging.getLogger(__name__)


def job_num(cfg: DictConfig) -> int:
    """The Hydra job number (0 when not in a multirun).

    ``hydra.job.num`` is set only in a multirun; a single run has no job
    number, and the shard round-robin treats it as job 0.
    """
    job = cast(Any, HydraConfig.get().job)  # a DictConfig at runtime
    return int(OmegaConf.select(job, "num", default=0))


def stop_requested(cfg: DictConfig) -> Path | None:
    """The STOP file path if it exists in the launch directory.

    The launch directory is the original working directory (the
    directory the command was run from; with ``job.chdir`` the job's cwd
    is the job dir, so the STOP file must live next to the launch
    command, not in the job dir).
    """
    path = Path(get_original_cwd()) / str(cfg.launch.stop_file)
    return path if path.exists() else None


def refuse_multi_device_in_multirun(devices: Any) -> None:
    """Refuse more than one device inside a Hydra multirun.

    Lightning's subprocess launcher re-executes its child ranks with the
    current command line, so under a multirun a child re-runs the whole
    sweep itself starting at job 0.  The two processes stay aligned only
    while rank 0 (whose gate can skip jobs — a shard exclusion, a
    completed record, a STOP file) and its child run every job in the
    same order; from the first skipped job on they train **different
    jobs in the same process group**, and for jobs with the same
    parameter shapes DDP averages their gradients silently.  Several
    devices are therefore supported for **single runs only**; a sweep
    runs one device per job (use shards to run several jobs on one
    machine).

    Args:
        devices: The configured ``training.devices`` value.

    Raises:
        RuntimeError: When running under a Hydra multirun and ``devices``
            is not the single integer 1 (``auto`` included).
    """
    if not HydraConfig.initialized() or HydraConfig.get().mode != RunMode.MULTIRUN:
        return
    if not (isinstance(devices, int) and devices == 1):
        raise RuntimeError(
            f"training.devices={devices} under a Hydra multirun is not "
            f"supported: Lightning's subprocess launcher re-runs the whole "
            f"sweep in each child, so a multi-device sweep would train "
            f"different jobs in the same process group; several devices "
            f"work for single runs only, and a sweep runs one device per "
            f"job (use shards to run several jobs on one machine)"
        )


def begin_run(cfg: DictConfig) -> dict[str, bool] | None:
    """The shared head of one job: precision flags, then branch on the
    STOP file and shard membership.

    Precision is set first (fp32 without TF32) in every process, the
    launch-gate ranks included.  Unsupported launchers and multi-device
    multiruns are refused here (see
    :func:`refuse_multi_device_in_multirun`).  A rank that Lightning's
    subprocess launcher started is a child of this job — children exist
    only for a single run, since several devices under a Hydra multirun
    are refused: rank 0 already ran the gate, so the child skips it and
    goes straight to building the run — the STOP file and the shard
    round-robin must not stop a child whose rank 0 decided to run, and
    the child's Hydra job number is not the gate's.  Otherwise: while the
    STOP file exists, no new run starts and the job exits
    (``sys.exit(0)``): in a multirun this ends the whole sweep — Hydra's
    job wrapper catches Exception, not SystemExit — in a single run it
    just exits. A run whose job number is not in this process's shard is
    **not** started; the function returns ``None`` for the caller to
    return normally, so the other shards of the sweep continue.

    Args:
        cfg: The resolved config (``launch.*``).

    Returns:
        ``None`` when this job is not in this process's shard (the caller
        returns without running); otherwise the precision flags, for the
        run record.

    Raises:
        RuntimeError: An unsupported launcher's rank variables are
            present (torchrun / srun), from :func:`lightning_launched_rank`.
    """
    precision_flags = disable_tf32()
    logger.info("Precision flags: %s", precision_flags)

    rank = lightning_launched_rank()
    if rank > 0:
        logger.info(
            "Lightning started this process as child rank %d; the launch "
            "gate (STOP file, shard membership, skip and lock) was decided "
            "by rank 0 and is bypassed here",
            rank,
        )
        return precision_flags

    # A multirun runs one device per job, and inside a Slurm allocation
    # each job runs a single device.
    refuse_multi_device_in_multirun(cfg.training.get("devices"))
    refuse_multi_device_under_slurm(cfg.training.get("devices"))
    stop = stop_requested(cfg)
    if stop is not None:
        logger.info(
            "STOP file found at %s; not starting job %d. Remove it to continue.",
            stop,
            job_num(cfg),
        )
        sys.exit(0)

    if not in_shard(job_num(cfg), cfg.launch.shard_index, cfg.launch.num_shards):
        logger.info(
            "job %d not in shard %d of %d; skipping",
            job_num(cfg),
            cfg.launch.shard_index,
            cfg.launch.num_shards,
        )
        return None
    return precision_flags


@contextmanager
def claimed_run(
    cfg: DictConfig,
    run_id: str,
    identity: tuple[str, Any],
    *,
    protocol_tiers: frozenset[str],
) -> Generator[str | None, None, None]:
    """Acquire the run lock and decide whether to run one job.

    Wraps the general :func:`research_lib.training.run_control.claim_run`:
    the record is checked only when ``launch.skip_completed`` is on **and**
    the run's ``experiment.tier`` is in ``protocol_tiers`` (tooling configs
    always run). A ``"already test_done"`` skip re-checks the W&B artifact
    before yielding (a skipped finished run whose upload failed when it
    finished would otherwise never be (re)uploaded).

    Args:
        cfg: The resolved config (``output.runs_dir``, ``launch.*``,
            ``experiment.tier``).
        run_id: The run ID (names the record and lock).
        identity: The run's ``(config_hash, seed)``.
        protocol_tiers: The pilot's protocol tiers; the skip applies only
            to a run in one of these.

    Yields:
        str | None: The skip reason (``"already test_done"`` or
            ``"running in another process"``), or ``None`` to run.
    """
    lock_path = Path(cfg.output.runs_dir) / f"{run_id}.lock"
    record_path = Path(cfg.output.runs_dir) / f"{run_id}.json"
    tier = str(cfg.experiment.get("tier") or "")
    skip_completed = bool(cfg.launch.skip_completed) and tier in protocol_tiers
    with claim_run(
        lock_path, record_path if skip_completed else None, identity
    ) as reason:
        if reason == "already test_done":
            from .run_lifecycle import ensure_run_record_uploaded

            ensure_run_record_uploaded(cfg, run_id)
        yield reason
