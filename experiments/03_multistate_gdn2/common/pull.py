"""A pilot's thin ``03_pull.py``: the general pull wired to this project's
layout and tier grouping.

:func:`main` parses the same command line as the first pilot's
``03_pull.py``, builds the W&B API client, lists the ``run_record``
artifacts and runs the general :func:`research_lib.training.run_records.pull`
with this project's :class:`~...common.layout.ProjectLayout` and tier
grouping, reporting per tier. A pilot passes its expected counts.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

from omegaconf import OmegaConf

from research_lib.training.run_records import (
    ARTIFACT_TYPE,
    list_run_record_artifacts,
    pull,
    render_report,
)

from .layout import LAYOUT, group_of, metadata_identity_fields

logger = logging.getLogger(__name__)


def default_entity_project(
    conf_config_path: Path | None = None,
) -> tuple[str | None, str | None]:
    """The W&B entity and project defaults from a pilot's ``conf/config.yaml``.

    Args:
        conf_config_path: The pilot's ``conf/config.yaml`` path; defaults
            to the first pilot's (its W&B project is shared).

    Returns:
        ``(entity, project)``; the entity is ``None`` when the config
        leaves it empty (the API key's default entity then applies).
    """
    conf_config_path = conf_config_path or (
        Path(__file__).resolve().parent.parent / "01_synthetic_pilot" / "conf" / "config.yaml"
    )
    wandb_conf = OmegaConf.select(
        OmegaConf.load(conf_config_path), "logging.wandb", default={}
    )
    entity = OmegaConf.select(wandb_conf, "entity", default=None)
    project = OmegaConf.select(wandb_conf, "project", default=None)
    return (str(entity) if entity else None), (str(project) if project else None)


def main(
    expected_counts: Mapping[str, int] | None = None,
    conf_config_path: Path | None = None,
) -> int:
    """The ``03_pull.py`` entry point for one pilot.

    Args:
        expected_counts: The pilot's expected done-count per tier, for
            the report.
        conf_config_path: The pilot's ``conf/config.yaml`` path to read
            the W&B defaults from (defaults to a sibling lookup in the
            first pilot).

    Returns:
        The exit code: 1 on any conflict, 2 on any failure, else 0.
    """
    parser = argparse.ArgumentParser(
        description="Pull run records from W&B into the local runs store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--tier", default=None, help="only this tier (e.g. tier_reserved)"
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
        default_entity, default_project = default_entity_project(conf_config_path)
        entity = entity or default_entity
        project = project or default_project
    if not project:
        raise SystemExit(
            "no W&B project: pass --project, or set logging.wandb.project in "
            "conf/config.yaml"
        )

    # Load the repo-root .env (research_lib.utils.secrets) and fail fast
    # if the key is missing, before the API client is built.
    from research_lib.utils.secrets import check_auth

    check_auth(["WANDB_API_KEY"])

    import wandb  # lazy: the pure helpers above must not need it

    api = wandb.Api()
    artifacts = list_run_record_artifacts(
        api, project, entity, args.tier, group_of=group_of
    )
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
        LAYOUT,
        metadata_identity_fields(),
        group_of=group_of,
        expected_counts=expected_counts,
        include_checkpoints=args.include_checkpoints,
        dry_run=args.check,
        group_filter=args.tier,
    )
    logger.info("%s", render_report(summaries, len(artifacts)))

    if any(s.conflicts for s in summaries.values()):
        return 1
    if any(s.failures for s in summaries.values()):
        return 2
    return 0


def run() -> int:
    """Console wrapper (``sys.exit(main())``)."""
    sys.exit(main())