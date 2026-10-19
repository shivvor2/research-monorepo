"""The report interface: options, the Report container and its type."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class ReportOptions:
    """Bootstrap settings shared by every report.

    Args:
        n_resamples: Number of bootstrap resamples per interval.
        alpha: Two-sided level: the interval covers ``1 - alpha``.
    """

    n_resamples: int = 10_000
    alpha: float = 0.05


@dataclass
class Report:
    """One analysis report: named tables plus its markdown section.

    A report is a pure function of the loaded run table (see
    :func:`pilot_analysis.records.load_runs`); it does no file I/O —
    the command-line entry point writes everything.

    Args:
        name: The subcommand name (and the ``<name>.md`` file stem).
        title: The markdown section title.
        tables: File stem -> table; the CLI writes each as
            ``<stem>.csv``.
        markdown: The report's section: its tables plus its reading,
            ready to drop into ``summary.md``.
    """

    name: str
    title: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    markdown: str = ""


ReportFn = Callable[[pd.DataFrame, ReportOptions], Report]
"""A report: ``(loaded run table, options) -> Report``."""
