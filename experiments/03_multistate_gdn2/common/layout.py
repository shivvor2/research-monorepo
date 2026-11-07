"""This project's run layout: where records live, their artifact paths, and
the mapping from our Hydra config to the general run-record and W&B
machinery.

The project's runs live under ``runs/<task>/<run_id>.json`` (plus a
curves directory next to each record and an optional final checkpoint),
with tier as the grouping level. This module supplies the general
:class:`~research_lib.training.run_records.ArtifactLayout` for that
convention, the identity-extraction pieces the general pull needs, the
metadata extras (``task``, ``tier``), and the mapping from our Hydra
config to :func:`research_lib.training.wandb_transport.wandb_logger_kwargs`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from research_lib.training.wandb_transport import wandb_logger_kwargs

from .protocol import TIER_NONE, group_name, tier_of_config


class ProjectLayout:
    """The ``runs/<task>/<run_id>.json`` artifact layout.

    The artifact-relative paths are the upload side of the contract
    (what a completed run's artifact holds); :meth:`local_target` maps an
    artifact-relative path to the local path (relative to the runs
    directory) of the same file. The record sits at
    ``<task>/<run_id>.json``, the curve under ``<task>/curves/<run_id>/``
    and the optional checkpoint at ``<task>/<run_id>.pt`` — locally the
    curve falls under the CSV logger's ``version_0`` directory, matching
    what training writes.
    """

    def record_path(self, metadata: Mapping[str, Any]) -> str:
        return f"{metadata['task']}/{metadata['run_id']}.json"

    def curve_path(self, metadata: Mapping[str, Any]) -> str:
        return f"{metadata['task']}/curves/{metadata['run_id']}/metrics.csv"

    def checkpoint_path(self, metadata: Mapping[str, Any]) -> str | None:
        return f"{metadata['task']}/{metadata['run_id']}.pt"

    def local_target(
        self,
        rel: Path,
        metadata: Mapping[str, Any],
        include_checkpoints: bool,
    ) -> tuple[str, Path] | None:
        task, run_id = str(metadata["task"]), str(metadata["run_id"])
        if rel == Path(task) / f"{run_id}.json":
            return "record", Path(task) / f"{run_id}.json"
        if rel == Path(task) / "curves" / run_id / "metrics.csv":
            return (
                "curves",
                Path(task) / "curves" / run_id / "version_0" / "metrics.csv",
            )
        if include_checkpoints and rel == Path(task) / f"{run_id}.pt":
            return "checkpoint", Path(task) / f"{run_id}.pt"
        return None


LAYOUT = ProjectLayout()
"""The shared instance of :class:`ProjectLayout`."""


def group_of(metadata: Mapping[str, Any]) -> str:
    """The group of an artifact's metadata: its tier, normalised."""
    return group_name(metadata.get("tier"))


def metadata_extras() -> dict[str, str]:
    """The run-record metadata extras: ``{metadata_key: config path}``."""
    return {"task": "task.name", "tier": "experiment.tier"}


def metadata_identity_fields() -> dict[str, Any]:
    """The identity fields :func:`staged_record_problem` checks, each a
    ``{metadata_key: record_extractor}`` entry (the tier is normalised
    through :func:`common.protocol.group_name`, matching the metadata)."""
    return {
        "run_id": lambda record: record.get("run_id"),
        "config_hash": lambda record: record.get("config_hash"),
        "seed": lambda record: record.get("seed"),
        "task": lambda record: ((record.get("config") or {}).get("task") or {}).get(
            "name"
        ),
        "tier": lambda record: tier_of_config(record.get("config") or {}),
    }


def group_of_record(record: Mapping[str, Any]) -> str:
    """The group of a run record: its tier from its config, normalised."""
    config = record.get("config") or {}
    return tier_of_config(config)


def record_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    """The artifact metadata of a run record, with the group fields
    normalised (the tier through :func:`common.protocol.group_name`).

    Args:
        record: The parsed run JSON.

    Returns:
        The metadata dict for a ``run_record`` artifact.
    """
    from research_lib.training.run_records import run_record_metadata

    metadata = run_record_metadata(record, metadata_extras())
    if "tier" in metadata:
        metadata["tier"] = group_name(metadata.get("tier"))
    return metadata


def wandb_kwargs(cfg: DictConfig, run_id: str) -> dict[str, Any]:
    """The ``WandbLogger`` keyword arguments of one run, from our config.

    One W&B run per pilot run, named by the run ID; the tier becomes the
    group, the phase the job type, and the experiment / task / model
    identity the tags, so the runs of the project's later experiments
    stay apart inside the same W&B project (the experiment name is the
    first tag). The resolved config is passed to W&B for browsing.

    Args:
        cfg: The resolved config (``logging.wandb.*``, ``experiment.*``,
            ``task.name``, ``model.*``).
        run_id: The run ID (also the run record's file stem).

    Returns:
        The keyword arguments for ``WandbLogger``.
    """
    tier = cfg.experiment.get("tier", None)
    entity = cfg.logging.wandb.get("entity", None)
    return wandb_logger_kwargs(
        project=str(cfg.logging.wandb.project),
        entity=str(entity) if entity else None,
        name=run_id,
        group=str(tier) if tier else TIER_NONE,
        job_type=f"phase{cfg.experiment.phase}",
        tags=[
            str(cfg.experiment.name),
            str(cfg.task.name),
            str(cfg.model.layout),
            str(cfg.model.gate_mode),
            str(cfg.model.lambda_mode),
            f"d{cfg.model.d_model}",
        ],
        config=OmegaConf.to_container(cfg, resolve=True),
        save_dir="logs/wandb",
        log_model=False,
    )


def record_files(cfg: DictConfig, run_id: str) -> list[tuple[Path, str]]:
    """The ``(local_path, artifact_relative_name)`` files of one run's
    record artifact (the record, its curve and, when saved, its
    checkpoint)."""
    from research_lib.training.run_control import RUN_DONE_STATUS
    from research_lib.training.run_records import read_record

    metadata = {"task": str(cfg.task.name), "run_id": run_id}
    files: list[tuple[Path, str]] = []
    record_path = Path(cfg.output.runs_dir) / f"{run_id}.json"
    if (
        read_record(record_path) is not None
        and (read_record(record_path) or {}).get("status") == RUN_DONE_STATUS
    ):
        files.append((record_path, LAYOUT.record_path(metadata)))
    curves = Path(cfg.logging.csv.save_dir) / run_id / "version_0" / "metrics.csv"
    if curves.exists():
        files.append((curves, LAYOUT.curve_path(metadata)))
    if bool(cfg.logging.wandb.upload_checkpoint):
        checkpoint = Path(cfg.output.runs_dir) / f"{run_id}.pt"
        if checkpoint.exists():
            files.append((checkpoint, LAYOUT.checkpoint_path(metadata)))
    return files
