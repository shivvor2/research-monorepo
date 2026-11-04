"""Copy run records from W&B artifacts into the local runs directory.

Rented machines upload every completed run — its JSON record, its
training curve and optionally its final checkpoint — as a W&B artifact
of type ``run_record`` named ``run-<run_id>``. This script pulls those
artifacts into the canonical local ``runs/`` store that the analysis
reads, without Hydra.

Usage:
    python 03_pull.py                           # pull everything
    python 03_pull.py --tier tier2_gates        # one tier
    python 03_pull.py --check                   # report only, no download
    python 03_pull.py --include-checkpoints     # also the final checkpoints

The decision per artifact, against the local record at
``<runs-dir>/<task>/<run_id>.json``:

- missing -> download it (into a temp dir first, then atomically);
- ``test_done`` with the same ``(config_hash, seed)`` -> skip;
- ``test_done`` with a different identity -> **conflict**: reported,
  never overwritten;
- any other status -> replaced by the artifact's record.

A download is installed only after its staged record is validated
against the artifact's metadata (status ``test_done`` and a matching
identity), and the record JSON is the last file moved into place, so a
crash mid-install leaves no new ``test_done`` record and the next pull
retries. A real install takes the run's lock (``runs/<task>/<run_id>.lock``,
the same file training holds); ``--check`` takes no lock and writes
nothing.

The per-tier report counts the local ``test_done`` records against the
protocol's expected counts and lists the run IDs that are still missing
locally. The exit code is non-zero on any conflict (1) or check
failure (2), so a scripted pull can gate destroying a rented instance.

``--entity`` and ``--project`` default to ``conf/config.yaml``'s
``logging.wandb`` block (the entity, when it is not the personal
default). The listing and downloads use the installed ``wandb``
client's public API: ``Api.artifact_collections(project_name,
type_name)`` enumerates the ``run_record`` collections of the project,
and each collection's versions are reduced to the newest by version
number.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pilot_analysis.protocol import EXPECTED_RUNS
from wandb_artifacts import ARTIFACT_TYPE

from research_lib.utils.run_lock import exclusive_lock

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Defaults (conf/config.yaml) and artifact listing
# --------------------------------------------------------------------------- #


def default_entity_project() -> tuple[str | None, str | None]:
    """The W&B entity and project defaults from ``conf/config.yaml``.

    Returns:
        ``(entity, project)``; the entity is ``None`` when the config
        leaves it empty (the API key's default entity then applies).
    """
    conf_dir = Path(__file__).resolve().parent / "conf"
    wandb_conf = OmegaConf.select(
        OmegaConf.load(conf_dir / "config.yaml"), "logging.wandb", default={}
    )
    entity = OmegaConf.select(wandb_conf, "entity", default=None)
    project = OmegaConf.select(wandb_conf, "project", default=None)
    return (str(entity) if entity else None), (str(project) if project else None)


def artifact_tier(metadata: dict[str, Any]) -> str:
    """The tier of an artifact's metadata (``"none"`` when tier-less)."""
    tier = metadata.get("tier")
    return str(tier) if tier else "none"


def version_index(version: str | int | None) -> int:
    """The integer N of a W&B artifact version ``"vN"``; -1 when absent
    or malformed.

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
    api: Any, project: str, entity: str | None, tier: str | None = None
) -> list[Any]:
    """The latest version of every ``run_record`` artifact of the project.

    Args:
        api: A ``wandb.Api``-like object (the tests use a fake).
        project: The W&B project name.
        entity: The W&B entity; ``None`` uses the API key's default.
        tier: Optional filter: keep only artifacts whose metadata names
            this tier.

    Returns:
        One artifact per collection: the newest version by version
        number (order-independent), found by paging through every
        version of each collection (``collection.artifacts()`` is a
        paginator the public API fully drains in a for/list).
        Collections whose metadata carries no run ID are not run
        records of this pilot and are skipped.
    """
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
        if tier is not None and artifact_tier(metadata) != tier:
            continue
        artifacts.append(artifact)
    return artifacts


# --------------------------------------------------------------------------- #
# Pull decisions
# --------------------------------------------------------------------------- #


def read_record(path: Path) -> dict[str, Any] | None:
    """The run record at ``path``, or ``None`` when missing/unparseable."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def identity_of(record_or_metadata: dict[str, Any]) -> str:
    """The run identity of a record or artifact metadata: ``hash:seed``."""
    return (
        f"{record_or_metadata.get('config_hash')}:" f"{record_or_metadata.get('seed')}"
    )


def pull_decision(local: dict[str, Any] | None, metadata: dict[str, Any]) -> str:
    """What to do with one artifact: ``download``, ``skip`` or ``conflict``.

    - no local record, or a record that is not done -> ``download`` (the
      artifact's record replaces it);
    - ``test_done`` with the same ``(config_hash, seed)`` -> ``skip``;
    - ``test_done`` with a different identity, or with identity fields
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
    if local is None or local.get("status") != "test_done":
        return "download"
    same = (local.get("config_hash"), local.get("seed")) == (
        metadata.get("config_hash"),
        metadata.get("seed"),
    )
    return "skip" if same else "conflict"


def staged_record_problem(record: dict | None, metadata: dict) -> str | None:
    """Why a downloaded record must not be installed, or None if it may.

    A downloaded record may be installed only when it is the completed
    run the artifact's metadata claims the collection to be: it must
    parse, be ``test_done``, and its identity (``run_id``, ``task``,
    tier, ``config_hash``, ``seed``) must match the metadata that names
    where it is installed. A mismatch writes the wrong record under a
    run's name, so it is a reason not to install.

    Args:
        record: The staged ``<task>/<run_id>.json`` parsed, or ``None``
            when missing/unparseable.
        metadata: The artifact's metadata (see
            ``wandb_artifacts.run_record_metadata``).

    Returns:
        A human-readable reason, or ``None`` when the record may be
        installed.
    """
    if record is None:
        return "record is missing or unparseable"
    if not isinstance(record, dict):
        return f"record is a {type(record).__name__}, not a JSON object"
    if record.get("status") != "test_done":
        return f"status is {record.get('status')!r}, not test_done"
    config = record.get("config", {}) or {}
    expected = {
        "run_id": record.get("run_id"),
        "task": (config.get("task") or {}).get("name"),
        "config_hash": record.get("config_hash"),
        "seed": record.get("seed"),
    }
    for field in ("run_id", "task", "config_hash", "seed"):
        if str(expected[field]) != str(metadata.get(field)):
            return (
                f"{field} in the record ({expected[field]!r}) does not match "
                f"the artifact metadata ({metadata.get(field)!r})"
            )
    record_tier = (config.get("experiment") or {}).get("tier")
    if artifact_tier({"tier": record_tier}) != artifact_tier(metadata):
        return "tier in the record does not match the artifact metadata"
    return None


# --------------------------------------------------------------------------- #
# Download and install
# --------------------------------------------------------------------------- #


def classify_local_target(
    rel: Path, task: str, run_id: str, runs_dir: Path, include_checkpoints: bool
) -> tuple[str | None, Path | None]:
    """The local target of one artifact file, by its path in the artifact.

    The artifact's paths are the 01_train.py contract:
    ``<task>/<run_id>.json``, ``<task>/curves/<run_id>/metrics.csv`` and
    ``<task>/<run_id>.pt``. The local targets keep the shapes the
    training script writes: the record and checkpoint next to the run
    JSON, the curve under the CSV logger's ``version_0`` directory.
    Checkpoints are skipped unless ``include_checkpoints`` is set.

    Returns:
        ``(kind, target)`` with kind in ``record`` / ``curves`` /
        ``checkpoint``, or ``(None, None)`` for an unknown file.
    """
    if rel == Path(task) / f"{run_id}.json":
        return "record", runs_dir / task / f"{run_id}.json"
    if rel == Path(task) / "curves" / run_id / "metrics.csv":
        return (
            "curves",
            runs_dir / task / "curves" / run_id / "version_0" / "metrics.csv",
        )
    if include_checkpoints and rel == Path(task) / f"{run_id}.pt":
        return "checkpoint", runs_dir / task / f"{run_id}.pt"
    return None, None


def atomic_copy(src: Path, target: Path) -> None:
    """Copy one file into place atomically.

    A unique temporary file in the target directory, then
    ``os.replace``; the temporary file is removed if the copy raises,
    so a failed pull never leaves a partial file under the record's
    name (the next ``03_pull.py`` run sees no record and retries).
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def install_artifact(
    artifact: Any, runs_dir: Path, include_checkpoints: bool
) -> tuple[list[Path], list[str]]:
    """Download one artifact and move its files to their local targets.

    The download goes to a temporary directory first, so an interrupted
    download never leaves partial files in the canonical store. Each
    kind is fetched under its own ``path_prefix``, so a checkpoint is
    never downloaded when ``include_checkpoints`` is off. The staged
    record is validated against the artifact's metadata before anything
    touches ``runs/`` (see :func:`staged_record_problem`): a record
    that is missing, not ``test_done``, or not the run the metadata
    claims makes the install raise without writing anything, leaving an
    existing local file untouched. The record JSON is moved into place
    last (after the curves and, when requested, the checkpoint), so a
    crash before that final move leaves no new ``test_done`` record and
    the next pull retries the run instead of skipping it.

    Args:
        artifact: A ``wandb`` public artifact (``metadata``,
            ``download(root=...)``) whose files sit at the 01_train.py
            paths.
        runs_dir: The canonical runs directory.
        include_checkpoints: Install the artifact's checkpoint too.

    Returns:
        ``(written, skipped)``: the local paths written, and the
        artifact files that were not installed (unknown paths, and the
        checkpoint when ``include_checkpoints`` is false).
    """
    metadata = dict(artifact.metadata or {})
    run_id = str(metadata["run_id"])
    task = str(metadata["task"])
    written: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="run-record-") as tmp:
        tmp = Path(tmp)
        prefixes = [f"{task}/{run_id}.json", f"{task}/curves/{run_id}/"]
        if include_checkpoints:
            prefixes.append(f"{task}/{run_id}.pt")
        for prefix in prefixes:
            artifact.download(root=str(tmp), path_prefix=prefix)
        record = read_record(tmp / task / f"{run_id}.json")
        problem = staged_record_problem(record, metadata)
        if problem is not None:
            raise ValueError(f"cannot install {run_id}: {problem}")
        # Companions first, the record JSON last.
        order: list[Path] = [Path(task) / "curves" / run_id / "metrics.csv"]
        if include_checkpoints:
            order.insert(0, Path(task) / f"{run_id}.pt")
        order.append(Path(task) / f"{run_id}.json")
        for rel in order:
            src = tmp / rel
            if not src.exists():
                continue
            _, target = classify_local_target(
                rel, task, run_id, runs_dir, include_checkpoints
            )
            if target is None:
                continue
            atomic_copy(src, target)
            written.append(target)
    return written, []


# --------------------------------------------------------------------------- #
# The pull and its per-tier report
# --------------------------------------------------------------------------- #


@dataclass
class TierSummary:
    """One tier's pull outcome: local counts and the artifact decisions."""

    tier: str
    local_done: int
    expected: int | None
    pulled: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    conflicts: list[tuple[str, str, str]] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    would_pull: list[str] = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        """Artifact run IDs still not ``test_done`` locally with the
        artifact's identity: the conflicts and the failed downloads."""
        return [run_id for run_id, _, _ in self.conflicts] + [
            run_id for run_id, _ in self.failures
        ]


def local_test_done_counts(runs_dir: Path) -> dict[str, int]:
    """``test_done`` record counts per tier under ``runs_dir``.

    Counts every ``*.json`` that parses and is ``test_done``, by the
    tier of its config (``"none"`` for the tier-less runs, e.g. the
    smoke config).
    """
    counts: dict[str, int] = {}
    for path in sorted(runs_dir.glob("**/*.json")):
        record = read_record(path)
        if record is None or record.get("status") != "test_done":
            continue
        config = record.get("config", {}) or {}
        tier = (config.get("experiment") or {}).get("tier")
        tier = str(tier) if tier else "none"
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def pull(
    artifacts: list[Any],
    runs_dir: Path,
    include_checkpoints: bool = False,
    dry_run: bool = False,
    tier_filter: str | None = None,
) -> dict[str, TierSummary]:
    """Apply the pull decision to every artifact; one summary per tier.

    A real (non-``--check``) install takes the run's lock -- the same
    ``runs/<task>/<run_id>.lock`` training holds -- for the whole
    read-and-install, so a pull on a live machine can neither overwrite
    a running job's record nor race the skip / conflict / install
    decision, which is computed under the lock. A lock that is not
    acquired records a failure and writes nothing. ``--check``
    (``dry_run``) takes no lock and writes nothing; it is a read-only
    report.

    Args:
        artifacts: The latest version of each ``run_record`` artifact
            (from :func:`list_run_record_artifacts`).
        runs_dir: The canonical runs directory.
        include_checkpoints: Also install the artifacts' checkpoints.
        dry_run: Report only (``--check``): decisions are computed and
            counted, nothing is downloaded and no lock is taken.
        tier_filter: Restrict the report to one tier (that tier is
            still reported when it has no artifacts and no local
            records, so an empty tier shows as ``0 / N``).

    Returns:
        One :class:`TierSummary` per tier.
    """
    local_done = local_test_done_counts(runs_dir)
    if tier_filter is not None:
        tier_names = {tier_filter}
    else:
        tier_names = set(local_done)
        tier_names.update(artifact_tier(dict(a.metadata or {})) for a in artifacts)
    summaries = {
        tier: TierSummary(tier, local_done.get(tier, 0), EXPECTED_RUNS.get(tier))
        for tier in sorted(tier_names)
    }

    for artifact in artifacts:
        metadata = dict(artifact.metadata or {})
        run_id = metadata.get("run_id")
        task = metadata.get("task")
        if not run_id or not task:
            continue  # not one of ours (the listing already skipped these)
        run_id, task = str(run_id), str(task)
        tier = artifact_tier(metadata)
        if tier_filter is not None and tier != tier_filter:
            continue
        summary = summaries[tier]
        record_path = runs_dir / task / f"{run_id}.json"

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

        # Training holds this same file (inside claim_run) while a run
        # is alive: taking it here keeps a pull on the machine that is
        # training from overwriting the running job's record.
        lock_path = runs_dir / task / f"{run_id}.lock"
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
                    install_artifact(artifact, runs_dir, include_checkpoints)
                    summary.pulled.append(run_id)
                except Exception as exc:  # one bad artifact must not stop the pull
                    summary.failures.append((run_id, str(exc)))
    # The report counts the store as it is after the pull: a record
    # installed just now is locally ``test_done``. A dry run installs
    # nothing, so its local counts are the pre-pull ones (the same scan
    # over an untouched store).
    after = local_test_done_counts(runs_dir)
    for summary in summaries.values():
        summary.local_done = after.get(summary.tier, 0)
    return summaries


def render_report(summaries: dict[str, TierSummary], artifacts_total: int) -> str:
    """The pull report: one block per tier, then the totals."""
    lines = [f"{artifacts_total} run_record artifacts on W&B"]
    for tier, s in sorted(summaries.items()):
        lines.append(f"== {tier} ==")
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity (default: conf/config.yaml's logging.wandb.entity)",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="W&B project (default: conf/config.yaml's logging.wandb.project)",
    )
    parser.add_argument("--runs-dir", default="runs", help="run JSON root")
    parser.add_argument(
        "--tier", default=None, help="only this tier (e.g. tier2_gates)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report only: what would be pulled, conflicts, the per-tier counts",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="also download the artifacts' final checkpoints",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    entity, project = args.entity, args.project
    if entity is None or project is None:
        default_entity, default_project = default_entity_project()
        entity = entity or default_entity
        project = project or default_project
    if not project:
        raise SystemExit(
            "no W&B project: pass --project, or set logging.wandb.project in "
            "conf/config.yaml"
        )

    import wandb  # lazy: the pure helpers above must not need it

    api = wandb.Api()
    artifacts = list_run_record_artifacts(api, project, entity, args.tier)
    logger.info(
        "Listing artifacts of %s%s (type %s): %d run records",
        f"{entity}/" if entity else "",
        project,
        ARTIFACT_TYPE,
        len(artifacts),
    )
    summaries = pull(
        artifacts,
        Path(args.runs_dir),
        include_checkpoints=args.include_checkpoints,
        dry_run=args.check,
        tier_filter=args.tier,
    )
    logger.info("%s", render_report(summaries, len(artifacts)))

    if any(s.conflicts for s in summaries.values()):
        return 1
    if any(s.failures for s in summaries.values()):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
