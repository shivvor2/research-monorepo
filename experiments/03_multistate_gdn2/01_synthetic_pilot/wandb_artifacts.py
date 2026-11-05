"""W&B run-record artifact helpers, shared by ``01_train.py`` (upload)
and ``03_pull.py`` (download).

A completed run is uploaded as one W&B artifact: the run JSON, the
training curve and optionally the final checkpoint, with the identity
and provenance of the record in the artifact metadata. The artifact
paths are the contract between the two scripts:

- ``<task>/<run_id>.json`` — the run record;
- ``<task>/curves/<run_id>/metrics.csv`` — the training curve;
- ``<task>/<run_id>.pt`` — the final checkpoint (optional).
"""

from __future__ import annotations

import re
from typing import Any

ARTIFACT_TYPE = "run_record"
"""The W&B artifact type of a run record."""


def artifact_name(run_id: str) -> str:
    """The artifact collection name of a run record: ``run-<run_id>``.

    W&B artifact names allow only ``[A-Za-z0-9._-]``, exactly the
    characters a generated run ID carries. The collection name is the
    run's, so a re-upload of the same run (a relaunch) creates the next
    version of the same collection.

    Args:
        run_id: The run ID (also the run record's file stem).

    Returns:
        ``run-<run_id>``.

    Raises:
        ValueError: The run ID contains a character outside
            ``[A-Za-z0-9._-]``. Run IDs are generated, so this signals a
            bug; mapping such a character to ``-`` would silently put two
            distinct runs in the same collection (e.g. ``1e+06`` and
            ``1e-06``).
    """
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError(
            f"run ID {run_id!r} contains a character outside [A-Za-z0-9._-]"
        )
    return "run-" + run_id


def run_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """The artifact metadata of a completed run record.

    The identity fields let ``03_pull.py`` decide skip / conflict /
    download from the metadata alone, without fetching the artifact.

    Args:
        record: The parsed run JSON (``config``, ``seed``,
            ``config_hash``, ``git``, ``provenance``, ``status``).

    Returns:
        ``{run_id, tier, task, seed, config_hash, git_commit,
        lock_hash, status}``.
    """
    config = record.get("config", {}) or {}
    return {
        "run_id": record.get("run_id"),
        "tier": (config.get("experiment") or {}).get("tier"),
        "task": (config.get("task") or {}).get("name"),
        "seed": record.get("seed"),
        "config_hash": record.get("config_hash"),
        "git_commit": (record.get("git") or {}).get("commit"),
        "lock_hash": (record.get("provenance") or {}).get("uv.lock"),
        "status": record.get("status"),
    }
