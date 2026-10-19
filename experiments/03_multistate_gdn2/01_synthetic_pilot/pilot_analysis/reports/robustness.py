"""R1: Delta_int at the primary endpoint under uniform weight decay."""

from __future__ import annotations

import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import TIER_R1
from pilot_analysis.records import tier_rows
from pilot_analysis.render import markdown_table

from .primary import primary_records

TITLE = "Under uniform weight decay (tier R1)"


def run(table: pd.DataFrame, n_resamples: int, alpha: float) -> pd.DataFrame:
    """Delta_int at the primary endpoint under uniform weight decay (tier R1).

    Reported next to the main result: how much the weight-decay exemption
    of the vector group (lambda, ``A_log``, ``dt_bias``, ...) matters to
    the interaction. R1 rows never enter the pre-registered statistic.
    """
    overwrite = tier_rows(table, TIER_R1)
    overwrite = overwrite[overwrite["task"] == "overwrite"]
    return pd.DataFrame(primary_records(overwrite, n_resamples, alpha))


def render(frame: pd.DataFrame) -> str:
    lines = [f"### {TITLE}", ""]
    if frame.empty:
        lines.append("_No R1 runs present._")
    else:
        lines.append(markdown_table(frame))
    return "\n".join(lines) + "\n\n"


def report(table: pd.DataFrame, options: ReportOptions) -> Report:
    frame = run(table, options.n_resamples, options.alpha)
    return Report(
        name="robustness",
        title=TITLE,
        tables={"r1_primary_endpoint": frame},
        markdown=render(frame),
    )
