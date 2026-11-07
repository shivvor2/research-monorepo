"""General W&B transport: logger keyword arguments and run-record uploads.

A completed run is uploaded as one artifact and pulled back into a local
store; the upload side lives here. This module is the **general** layer:
the logger keyword arguments come in as plain values (a project maps its
own config onto them) and an upload takes explicit local paths and their
artifact-relative names, so no Hydra config path or folder convention
appears in this module.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from research_lib.training.run_control import RUN_DONE_STATUS
from research_lib.training.run_records import ARTIFACT_TYPE

logger = logging.getLogger(__name__)


def wandb_logger_kwargs(
    *,
    project: str,
    name: str,
    group: str,
    job_type: str,
    tags: list[str] | tuple[str, ...],
    config: Any = None,
    entity: str | None = None,
    save_dir: str = "logs/wandb",
    log_model: bool = False,
) -> dict[str, Any]:
    """The ``WandbLogger`` keyword arguments of one run.

    One W&B run per training run, named by the run ID; the caller picks
    the group, job type and tags from its own project conventions and
    passes them as plain values (a project maps its Hydra config onto
    them). The resolved config dict is passed through for browsing.

    Args:
        project: The W&B project name.
        name: The run name.
        group: The run's group.
        job_type: The run's job type.
        tags: The run's tags.
        config: The resolved config dict (or ``None``), for browsing.
        entity: The W&B entity; ``None`` uses the API key's default.
        save_dir: The local log directory.
        log_model: Whether to log the model to the registry.

    Returns:
        The keyword arguments for ``lightning.pytorch.loggers.WandbLogger``.
    """
    return {
        "project": project,
        "entity": entity,
        "name": name,
        "group": group,
        "job_type": job_type,
        "tags": list(tags),
        "config": config,
        "save_dir": save_dir,
        "log_model": log_model,
    }


def upload_init_kwargs(wandb_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The ``wandb.init`` keyword arguments of a short re-upload run.

    The same project / entity / name / group / tags as the training
    logger, but ``job_type="upload"``: a relaunch that skips a finished
    run opens this short run to upload the record that finished run
    missed. The keyword arguments only ``WandbLogger`` understands are
    mapped or dropped for ``wandb.init``: ``save_dir`` becomes ``dir``
    (so the re-upload run writes under the same log directory as
    training), and ``log_model`` is removed.
    """
    kwargs = dict(wandb_kwargs)
    kwargs["dir"] = kwargs.pop("save_dir")
    kwargs.pop("log_model", None)
    kwargs["job_type"] = "upload"
    return kwargs


def upload_run_record_artifact(
    artifact: str,
    metadata: Mapping[str, Any],
    files: Iterable[tuple[Path | str, str]],
    *,
    is_global_zero: bool = True,
) -> bool:
    """Upload a completed run's files as a ``run_record`` artifact.

    Runs only on the global-zero process (a non-zero rank under
    ``devices="auto"`` has no live W&B run and would only log noise).
    The caller passes the artifact metadata (see
    :func:`research_lib.training.run_records.run_record_metadata`) and
    the files as ``(local_path, artifact_relative_name)`` pairs; the
    artifact-relative names are the pull-side contract (a project's
    :class:`~research_lib.training.run_records.ArtifactLayout`).

    An upload failure is logged at ERROR level and does not raise: the
    local record is already done, and a ``--check`` pull reports what
    the canonical store still misses. The job waits for the upload
    (``artifact.wait()``) so it does not end before the artifact is
    complete.

    Args:
        artifact: The artifact collection name (from
            :func:`research_lib.training.run_records.artifact_name`).
        metadata: The artifact metadata (the record's identity and
            provenance).
        files: ``(local_path, artifact_relative_name)`` pairs to add.
        is_global_zero: Whether this process is rank zero (the only one
            with a live W&B run).

    Returns:
        ``True`` when the artifact was uploaded, ``False`` otherwise
        (not rank zero, no live W&B run, or an upload failure).
    """
    if not is_global_zero:
        return False
    try:
        import wandb
        from wandb import Artifact

        if wandb.run is None:
            logger.error("no live W&B run; skipping the artifact upload")
            return False
        artifact_obj = Artifact(artifact, type=ARTIFACT_TYPE, metadata=dict(metadata))
        for local_path, rel_name in files:
            artifact_obj.add_file(str(local_path), name=rel_name)
        wandb.log_artifact(artifact_obj)
        artifact_obj.wait()  # the job must not end before the upload completes
        logger.info("Uploaded run record artifact %s", artifact)
        return True
    except Exception as exc:  # an upload failure never fails a completed run
        logger.error("W&B artifact upload failed for %s: %s", artifact, exc)
        return False


def ensure_remote_record(
    api: Any,
    qualified_name: str,
    record: Mapping[str, Any],
    upload: Any,
) -> bool:
    """Make sure a finished run's record reaches the store as an artifact.

    A relaunched sweep skips a run whose record is already done with the
    same identity, before a fresh job would have uploaded anything, so
    if that run's upload failed when it finished it would stay off the
    store forever. This re-checks the ``run_record`` artifact: when it
    is absent or carries a different ``(config_hash, seed)``, the caller
    supplies an ``upload`` callable (a short W&B run that calls
    :func:`upload_run_record_artifact`). Any problem is logged at ERROR
    level and never raises; a machine without W&B is untouched.

    Args:
        api: A ``wandb.Api``-like object.
        qualified_name: The artifact's entity/project collection path,
            e.g. ``"<entity>/<project>/<name>:latest"``.
        record: The parsed run record.
        upload: A ``() -> None`` callable that performs the upload when
            the record is missing or stale.

    Returns:
        ``True`` when the remote artifact already matches the record,
        ``False`` when a (re)upload was attempted (or skipped).
    """
    try:
        if api.artifact_exists(qualified_name):
            artifact = api.artifact(qualified_name)
            metadata = dict(artifact.metadata or {})
            already = (metadata.get("config_hash"), metadata.get("seed")) == (
                record.get("config_hash"),
                record.get("seed"),
            )
            if already:
                logger.info("run record already on W&B; nothing to (re)upload")
                return True
        upload()
    except Exception as exc:  # a failed check/upload never fails a skipped job
        logger.error("could not ensure run record is on W&B: %s", exc)
    return False


def artifact_is_done(record_path: Path, status: str | None = RUN_DONE_STATUS) -> bool:
    """True when the record at ``record_path`` exists and is done."""
    from research_lib.training.run_records import read_record

    record = read_record(Path(record_path))
    return record is not None and record.get("status") == status
