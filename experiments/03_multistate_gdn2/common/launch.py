"""Hydra launch helpers shared by this project's pilots.

The two functions here read the Hydra job and the project's ``launch:``
block: the job number (for the shard round-robin) and the STOP file in
the launch directory. :func:`begin_run` runs the head of one job in the
pilots' shared order — precision flags, STOP file, shard membership —
once, up front, before any run-specific machinery is built.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, cast

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from research_lib.training.run_control import disable_tf32, in_shard

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


def begin_run(cfg: DictConfig) -> dict[str, bool]:
    """The shared head of one job: precision flags, then exit if the STOP
    file exists or this job is not in this process's shard.

    Precision is set first (fp32 without TF32); then, while the STOP file
    exists, no new run starts (in a multirun this ends the whole sweep —
    Hydra's job wrapper catches Exception, not SystemExit — in a single
    run it just exits); then a run whose job number is not in this
    process's shard is skipped. Either exit calls ``sys.exit(0)``.

    Args:
        cfg: The resolved config (``launch.*``).

    Returns:
        The precision flags, for the run record.
    """
    precision_flags = disable_tf32()
    logger.info("Precision flags: %s", precision_flags)

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
        sys.exit(0)
    return precision_flags