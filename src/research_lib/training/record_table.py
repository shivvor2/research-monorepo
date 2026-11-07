"""General loading of run records into a flat table.

:func:`load_records` reads every ``*.json`` under a directory, keeps the
records whose ``status`` is the done status, and builds one row per
record from a caller-supplied ``flatten`` function. The column choices
are the caller's; nothing about groups, tasks, layouts or any other
project convention appears here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from research_lib.training.run_control import RUN_DONE_STATUS
from research_lib.training.run_records import read_record

logger = logging.getLogger(__name__)

Flatten = Callable[[Mapping[str, Any]], dict[str, Any]]
"""Build one row's columns from a parsed run record."""


def load_records(
    runs_dir: Path | str,
    flatten: Flatten,
    *,
    status: str = RUN_DONE_STATUS,
) -> pd.DataFrame:
    """Load every done run record under ``runs_dir`` into one row-per-record
    DataFrame.

    Reads every ``*.json`` under the directory; unreadable files are
    skipped with a logged warning, and records whose ``status`` is not
    ``status`` (default the done status) are skipped and reported
    separately at INFO level. Each kept record becomes one row via
    ``flatten(record)``.

    Args:
        runs_dir: The runs directory root.
        flatten: ``record -> dict`` mapping a parsed run record to one
            row's columns.
        status: The lifecycle status to keep (a record has exactly this
            when it is finished).

    Returns:
        A DataFrame with one row per done record, the columns that
        ``flatten`` returns. When no records are kept the DataFrame has
        no columns; the caller's ``flatten`` decides the schema.
    """
    rows: list[dict[str, Any]] = []
    unreadable: list[tuple[Path, str]] = []
    other_status: list[tuple[Path, str]] = []
    for path in sorted(Path(runs_dir).glob("**/*.json")):
        record = read_record(path)
        if record is None:
            unreadable.append((path, "unreadable JSON"))
            continue
        if record.get("status") != status:
            other_status.append((path, str(record.get("status"))))
            continue
        rows.append(dict(flatten(record)))
    for path, reason in unreadable:
        logger.warning("skipping unreadable record %s: %s", path, reason)
    if unreadable:
        logger.warning("skipped %d unreadable record(s)", len(unreadable))
    if other_status:
        logger.info(
            "skipped %d record(s) with status other than %r",
            len(other_status),
            status,
        )
    return pd.DataFrame(rows)
