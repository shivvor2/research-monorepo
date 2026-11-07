"""General run-record storage: artifact names, metadata, and a convention-free
pull from a W&B-like artifact store.

A completed run is uploaded as one artifact: the run JSON, a training
curve and optionally a final checkpoint, with the run's identity and
provenance in the artifact metadata. This module is the **general**
layer: it knows the run-record format :class:`RunSummaryCallback`
writes (``status``, ``config_hash``, ``seed``) and nothing about groups,
tasks, Hydra config paths or a particular folder layout. A project
supplies three things:

- an :class:`ArtifactLayout` — how the artifact's files map to local
  paths (the project layer's ``runs/<task>/<run_id>.json`` convention);
- a grouping function ``group_of(metadata) -> str`` and the expected
  counts, for the per-group report;
- the record's identity fields for :func:`staged_record_problem`.

:class:`GroupSummary` is the per-group pull outcome, and :func:`pull`
applies the per-artifact decision to a whole listing. The per-artifact
decision is :func:`pull_decision`; a download is installed atomically
and only after the staged record validates against the artifact's
metadata.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from research_lib.training.run_control import RUN_DONE_STATUS, read_record
from research_lib.utils.io import atomic_write
from research_lib.utils.run_lock import exclusive_lock

logger = logging.getLogger(__name__)

ARTIFACT_TYPE = "run_record"
"""The artifact type of a run record."""


def artifact_name(run_id: str) -> str:
    """The artifact collection name of a run record: ``run-<run_id>``.

    Artifact names allow only ``[A-Za-z0-9._-]``, exactly the characters
    of a well-formed run ID. The collection name is the run's, so a
    re-upload of the same run (a relaunch) creates the next version of
    the same collection.

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


def version_index(version: str | int | None) -> int:
    """The integer N of an artifact version ``"vN"``; -1 when absent or
    malformed.

    Accepts ``int`` too (harmless, keeps the helper robust) and returns
    ``-1`` for ``None`` or anything that does not match ``^v(\\d+)$``, so
    the newest-version pick in :func:`list_run_record_artifacts` never
    compares the raw strings (a lexicographic ``max`` would pick
    ``"v9"`` over ``"v10"``).
    """
    if version is None:
        return -1
    if isinstance(version, int):
        return version
    match = re.fullmatch(r"v(\d+)", str(version))
    return int(match.group(1)) if match else -1


def list_run_record_artifacts(
    api: Any,
    project: str,
    entity: str | None,
    group: str | None = None,
    group_of: Callable[[Mapping[str, Any]], str] | None = None,
) -> list[Any]:
    """The latest version of every ``run_record`` artifact of the project.

    Args:
        api: A ``wandb.Api``-like object (the tests use a fake).
        project: The W&B-style project name.
        entity: The entity; ``None`` uses the API key's default.
        group: Optional filter: keep only artifacts whose group (from
            ``group_of``) equals this.
        group_of: Maps an artifact's metadata to its group name; required
            when ``group`` is set (the general layer has no group
            convention of its own).

    Returns:
        One artifact per collection: the newest version by version
        number (order-independent), found by paging through every
        version of each collection. Collections whose metadata carries no
        run ID are not run records and are skipped.

    Raises:
        ValueError: When ``group`` is set without a ``group_of``.
    """
    if group is not None and group_of is None:
        raise ValueError("group_of is required when filtering by group")
    project_path = f"{entity}/{project}" if entity else project
    artifacts = []
    for collection in api.artifact_collections(project_path, ARTIFACT_TYPE):
        versions = list(collection.artifacts())
        if not versions:
            continue
        artifact = max(versions, key=lambda a: version_index(a.version))
        metadata = dict(artifact.metadata or {})
        if not metadata.get("run_id"):
            continue
        if group is not None and group_of(metadata) != group:
            continue
        artifacts.append(artifact)
    return artifacts


def identity_of(record_or_metadata: Mapping[str, Any]) -> str:
    """The run identity of a record or artifact metadata: ``hash:seed``."""
    return (
        f"{record_or_metadata.get('config_hash')}:" f"{record_or_metadata.get('seed')}"
    )


def run_record_metadata(
    record: Mapping[str, Any], extra_fields: Mapping[str, str] = ()
) -> dict[str, Any]:
    """The artifact metadata of a completed run record.

    The identity fields let the pull decide skip / conflict / download
    from the metadata alone, without fetching the artifact. Fields
    beyond the identity and provenance come from ``extra_fields``, a
    mapping of ``metadata key -> config dotted path``: the project
    layer adds its own identity fields from its config (e.g. the task name).

    Args:
        record: The parsed run JSON (``config``, ``seed``,
            ``config_hash``, ``git``, ``provenance``, ``status``).
        extra_fields: Metadata keys to copy from the record's config, as
            ``{metadata_key: config_dotted_path}``.

    Returns:
        ``{run_id, seed, config_hash, git_commit, lock_hash, status}``
        plus the supplied extra fields.
    """
    config = record.get("config", {}) or {}
    metadata: dict[str, Any] = {
        "run_id": record.get("run_id"),
        "seed": record.get("seed"),
        "config_hash": record.get("config_hash"),
        "git_commit": (record.get("git") or {}).get("commit"),
        "lock_hash": (record.get("provenance") or {}).get("uv.lock"),
        "status": record.get("status"),
    }
    for key, path in dict(extra_fields).items():
        metadata[key] = config_dot(config, path)
    return metadata


def config_dot(config: Mapping[str, Any], dotted: str) -> Any:
    """The value at a dotted config path in an arbitrary mapping, or None."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node


def pull_decision(local: Mapping[str, Any] | None, metadata: Mapping[str, Any]) -> str:
    """What to do with one artifact: ``download``, ``skip`` or ``conflict``.

    - no local record, or a record that is not done -> ``download`` (the
      artifact's record replaces it);
    - a done record with the same ``(config_hash, seed)`` -> ``skip``;
    - a done record with a different identity, or with identity fields
      missing from the artifact metadata -> ``conflict``: reported and
      never overwritten (the safe choice when the artifact cannot prove
      it is the same run).

    Args:
        local: The local record at the artifact's target path, or
            ``None``.
        metadata: The artifact's metadata (``config_hash``, ``seed``).

    Returns:
        One of ``"download"``, ``"skip"``, ``"conflict"``.
    """
    if local is None or local.get("status") != RUN_DONE_STATUS:
        return "download"
    same = (local.get("config_hash"), local.get("seed")) == (
        metadata.get("config_hash"),
        metadata.get("seed"),
    )
    return "skip" if same else "conflict"


def staged_record_problem(
    record: dict | None,
    metadata: Mapping[str, Any],
    identity_fields: Mapping[str, Callable[[Mapping[str, Any]], Any]],
) -> str | None:
    """Why a downloaded record must not be installed, or None if it may.

    A downloaded record may be installed only when it is the completed
    run the artifact's metadata claims the collection to be: it must
    parse, be done (``RUN_DONE_STATUS``), and each identity field the
    caller names must match between the record and the metadata. A
    mismatch writes the wrong record under a run's name, so each is a
    reason not to install.

    Args:
        record: The staged record parsed, or ``None`` when missing or
            unparseable.
        metadata: The artifact's metadata (see
            :func:`run_record_metadata`).
        identity_fields: ``{metadata_key: record_extractor}``; each key
            is compared between the record (via its extractor) and the
            metadata, and a record value of ``None`` (cannot be
            matched) is a reason not to install on its own.

    Returns:
        A human-readable reason, or ``None`` when the record may be
        installed.
    """
    if record is None:
        return "record is missing or unparseable"
    if not isinstance(record, dict):
        return f"record is a {type(record).__name__}, not a JSON object"
    if record.get("status") != RUN_DONE_STATUS:
        return f"status is {record.get('status')!r}, not {RUN_DONE_STATUS}"
    metadata_dict = dict(metadata)
    for key, get_record_value in identity_fields.items():
        record_value = get_record_value(record)
        if record_value is None:
            return f"record has no {key}"
        if str(record_value) != str(metadata_dict.get(key)):
            return (
                f"{key} in the record ({record_value!r}) does not match "
                f"the artifact metadata ({metadata_dict.get(key)!r})"
            )
    return None


@runtime_checkable
class ArtifactLayout(Protocol):
    """A project's run-artifact layout: the paths inside an artifact and
    their local targets, from an artifact's metadata.

    Implementations are small and usually stateless; they are the
    project-layer piece that keeps :func:`pull` free of any one folder
    convention. All returned paths are relative to the runs directory.

    Methods:
        record_path: The artifact-relative path of the run JSON.
        curve_path: The artifact-relative path of the training curve.
        checkpoint_path: The artifact-relative path of the final
            checkpoint, or ``None`` when this layout has none.
        local_target: ``(kind, local-relative-path)`` for an
            artifact-relative path, or ``None`` when that file is not
            part of this layout (or the kind is not wanted).
    """

    def record_path(self, metadata: Mapping[str, Any]) -> str: ...
    def curve_path(self, metadata: Mapping[str, Any]) -> str: ...
    def checkpoint_path(self, metadata: Mapping[str, Any]) -> str | None: ...
    def local_target(
        self,
        rel: Path,
        metadata: Mapping[str, Any],
        include_checkpoints: bool,
    ) -> tuple[str, Path] | None: ...


def atomic_copy(src: Path, target: Path) -> None:
    """Copy one file into place atomically (temp file in the target dir,
    flushed and fsynced, then renamed).

    Reuses :func:`research_lib.utils.io.atomic_write`, so the target never
    holds a partial file (a failed pull leaves no record and the next pull
    retries) and both the file and the rename survive a power loss.
    """

    def _copy_file(f) -> None:
        with open(src, "rb") as s:
            shutil.copyfileobj(s, f)

    atomic_write(target, _copy_file, "wb")


def install_artifact(
    artifact: Any,
    runs_dir: Path,
    layout: ArtifactLayout,
    include_checkpoints: bool,
    identity_fields: Mapping[str, Callable[[Mapping[str, Any]], Any]],
) -> list[Path]:
    """Download one artifact and move its files to their local targets.

    The download goes to a temporary directory first, so an interrupted
    download never leaves partial files in the canonical store. The
    staged record is validated against the artifact's metadata before
    anything touches the runs directory (see :func:`staged_record_problem`):
    a record that is missing, not done, or not the run the metadata
    claims makes the install raise without writing anything, leaving an
    existing local file untouched. The record JSON is moved into place
    last (after the curve and, when requested, the checkpoint), so a
    crash before that final move leaves no new done record and the next
    pull retries the run instead of skipping it.

    Args:
        artifact: A W&B-like public artifact (``metadata``,
            ``download(root=..., path_prefix=...)``).
        runs_dir: The canonical runs directory.
        layout: The project's :class:`ArtifactLayout`.
        include_checkpoints: Install the artifact's checkpoint too.
        identity_fields: ``{metadata_key: record_extractor}`` for
            :func:`staged_record_problem`.

    Returns:
        The local paths written, in the order they were installed
        (companions before the record).
    """
    metadata = dict(artifact.metadata or {})
    prefixes = [
        layout.record_path(metadata),
        layout.curve_path(metadata),
    ]
    if include_checkpoints:
        checkpoint = layout.checkpoint_path(metadata)
        if checkpoint is not None:
            prefixes.append(checkpoint)
    written: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="run-record-") as tmp:
        tmp = Path(tmp)
        for prefix in prefixes:
            artifact.download(root=str(tmp), path_prefix=prefix)
        record_path = Path(layout.record_path(metadata))
        record = read_record(tmp / record_path)
        problem = staged_record_problem(record, metadata, identity_fields)
        if problem is not None:
            raise ValueError(f"cannot install {metadata.get('run_id')}: {problem}")
        # Companions first, the record JSON last.
        order: list[Path] = [
            Path(layout.curve_path(metadata)),
        ]
        if include_checkpoints and layout.checkpoint_path(metadata) is not None:
            order.insert(0, Path(layout.checkpoint_path(metadata)))
        order.append(record_path)
        for rel in order:
            src = tmp / rel
            if not src.exists():
                continue
            hit = layout.local_target(rel, metadata, include_checkpoints)
            if hit is None:
                continue
            _, target = hit
            atomic_copy(src, runs_dir / target)
            written.append(runs_dir / target)
    return written


@dataclass
class GroupSummary:
    """One group's pull outcome: local counts and the artifact decisions."""

    group: str
    local_done: int
    expected: int | None
    pulled: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    conflicts: list[tuple[str, str, str]] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    would_pull: list[str] = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        """Artifact run IDs still not done locally with the artifact's
        identity: the conflicts and the failed downloads."""
        return [run_id for run_id, _, _ in self.conflicts] + [
            run_id for run_id, _ in self.failures
        ]


def local_done_counts(
    runs_dir: Path, group_of_record: Callable[[Mapping[str, Any]], str]
) -> dict[str, int]:
    """Done-record counts per group under ``runs_dir``.

    Counts every ``*.json`` that parses and is ``RUN_DONE_STATUS``, by
    the group its record maps to (``group_of_record(record)``).
    """
    counts: dict[str, int] = {}
    for path in sorted(runs_dir.glob("**/*.json")):
        record = read_record(path)
        if record is None or record.get("status") != RUN_DONE_STATUS:
            continue
        group = group_of_record(record)
        counts[group] = counts.get(group, 0) + 1
    return counts


def pull(
    artifacts: list[Any],
    runs_dir: Path,
    layout: ArtifactLayout,
    identity_fields: Mapping[str, Callable[[Mapping[str, Any]], Any]],
    *,
    group_of: Callable[[Mapping[str, Any]], str],
    group_of_record: Callable[[Mapping[str, Any]], str],
    expected_counts: Mapping[str, int] | None = None,
    include_checkpoints: bool = False,
    dry_run: bool = False,
    group_filter: str | None = None,
) -> dict[str, GroupSummary]:
    """Apply the pull decision to every artifact; one summary per group.

    A real (non-``dry_run``) install takes the run's lock -- the same
    file training holds -- for the whole read-and-install, so a pull on
    a live machine can neither overwrite a running job's record nor race
    the skip / conflict / install decision, which is computed under the
    lock. A lock that is not acquired records a failure and writes
    nothing. ``dry_run`` (report only) takes no lock and writes nothing.

    Args:
        artifacts: The latest version of each ``run_record`` artifact
            (from :func:`list_run_record_artifacts`).
        runs_dir: The canonical runs directory.
        layout: The project's :class:`ArtifactLayout`.
        identity_fields: ``{metadata_key: record_extractor}`` for
            :func:`staged_record_problem`.
        group_of: Maps an artifact's metadata to its group name.
        group_of_record: Maps a run record to its group name. Separate
            from ``group_of`` because an artifact's entire metadata is a
            different shape from a run JSON; local counts must come from
            the records, not the metadata.
        expected_counts: The expected done-count per group, for the
            report.
        include_checkpoints: Also install the artifacts' checkpoints.
        dry_run: Report only: decisions computed and counted, nothing
            downloaded and no lock taken.
        group_filter: Restrict the report to one group (that group is
            still reported when it has no artifacts and no local
            records, so an empty group shows as ``0 / N``).

    Returns:
        One :class:`GroupSummary` per group.
    """
    expected_counts = expected_counts or {}
    local_done = local_done_counts(runs_dir, group_of_record)
    if group_filter is not None:
        group_names = {group_filter}
    else:
        group_names = set(local_done)
        group_names.update(group_of(dict(a.metadata or {})) for a in artifacts)
    summaries = {
        group: GroupSummary(group, local_done.get(group, 0), expected_counts.get(group))
        for group in sorted(group_names)
    }

    for artifact in artifacts:
        metadata = dict(artifact.metadata or {})
        run_id = metadata.get("run_id")
        if not run_id:
            continue  # not one of ours (the listing already skipped these)
        run_id = str(run_id)
        group = group_of(metadata)
        if group_filter is not None and group != group_filter:
            continue
        summary = summaries[group]
        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            summary.failures.append((run_id, "unsafe run_id in artifact metadata"))
            continue
        try:
            hit = layout.local_target(
                Path(layout.record_path(metadata)), metadata, include_checkpoints
            )
            if hit is None:
                summary.failures.append(
                    (run_id, "no local target for the artifact's record")
                )
                continue
            record_rel = hit[1]
            record_path = runs_dir / record_rel
        except Exception as exc:
            summary.failures.append((run_id, str(exc)))
            continue

        # The local target must stay under runs_dir: a task or run_id
        # such as ``../`` in the metadata must not escape the store.
        anchor = runs_dir.resolve()
        try:
            resolved = record_path.resolve()
        except OSError:
            resolved = record_path
        if resolved != anchor and anchor not in resolved.parents:
            summary.failures.append(
                (run_id, "unsafe run_id or task in artifact metadata")
            )
            continue

        if dry_run:
            local = read_record(record_path)
            decision = pull_decision(local, metadata)
            if decision == "skip":
                summary.skipped.append(run_id)
            elif decision == "conflict":
                summary.conflicts.append(
                    (
                        run_id,
                        identity_of(metadata),
                        identity_of(local) if local else "<unreadable>",
                    )
                )
            else:
                summary.would_pull.append(run_id)
            continue

        # Training holds this same file while a run is alive: taking it
        # here keeps a pull on the training machine from overwriting the
        # running job's record. The lock sits next to the run record.
        lock_path = record_path.with_suffix(".lock")
        with exclusive_lock(lock_path) as acquired:
            if not acquired:
                summary.failures.append((run_id, "run lock held by another process"))
                continue
            local = read_record(record_path)
            decision = pull_decision(local, metadata)
            if decision == "skip":
                summary.skipped.append(run_id)
            elif decision == "conflict":
                summary.conflicts.append(
                    (
                        run_id,
                        identity_of(metadata),
                        identity_of(local) if local else "<unreadable>",
                    )
                )
            else:
                try:
                    # install_artifact validates the staged record before
                    # writing and moves the record JSON in last.
                    install_artifact(
                        artifact,
                        runs_dir,
                        layout,
                        include_checkpoints,
                        identity_fields,
                    )
                    summary.pulled.append(run_id)
                except Exception as exc:  # one bad artifact must not stop the pull
                    summary.failures.append((run_id, str(exc)))
    # The report counts the store as it is after the pull: a record
    # installed just now is locally done. A dry run installs nothing,
    # so its local counts are the pre-pull ones.
    after = local_done_counts(runs_dir, group_of_record)
    for summary in summaries.values():
        summary.local_done = after.get(summary.group, 0)
    return summaries


def render_report(summaries: Mapping[str, GroupSummary], artifacts_total: int) -> str:
    """The pull report: one block per group, then the totals."""
    lines = [f"{artifacts_total} run_record artifacts on W&B"]
    for group, s in sorted(summaries.items()):
        lines.append(f"== {group} ==")
        if s.expected is None:
            lines.append(f"  local test_done: {s.local_done} (no expected count)")
        else:
            shortfall = max(0, s.expected - s.local_done)
            lines.append(
                f"  local test_done: {s.local_done} / {s.expected} expected"
                + (f" ({shortfall} short)" if shortfall else "")
            )
        if s.would_pull:
            lines.append(f"  would pull: {len(s.would_pull)}")
        else:
            lines.append(f"  pulled now: {len(s.pulled)}")
        lines.append(f"  skipped: {len(s.skipped)}")
        if s.conflicts:
            lines.append(f"  CONFLICTS ({len(s.conflicts)}, never overwritten):")
            for run_id, artifact_identity, local_identity in s.conflicts:
                lines.append(
                    f"    {run_id}: artifact {artifact_identity} "
                    f"!= local {local_identity}"
                )
        if s.failures:
            lines.append(f"  failures ({len(s.failures)}):")
            for run_id, reason in s.failures:
                lines.append(f"    {run_id}: {reason}")
        missing = s.missing
        lines.append(
            f"  still missing locally: {len(missing)}"
            + (f": {', '.join(missing)}" if missing else "")
        )
    total_conflicts = sum(len(s.conflicts) for s in summaries.values())
    total_failures = sum(len(s.failures) for s in summaries.values())
    if total_conflicts or total_failures:
        lines.append(
            f"pull incomplete: {total_conflicts} conflicts, "
            f"{total_failures} failures"
        )
    else:
        lines.append("pull clean: no conflicts, no failures")
    return "\n".join(lines)
