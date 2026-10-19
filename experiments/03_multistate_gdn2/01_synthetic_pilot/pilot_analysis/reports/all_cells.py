"""Delta_int for every measured cell, core tier rows only."""

from __future__ import annotations

import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import TIER_CORE
from pilot_analysis.records import tier_rows
from pilot_analysis.render import markdown_section
from pilot_analysis.statistics import did_table_for

TITLE = "Delta_int on every measured cell (core tier)"


def run(
    table: pd.DataFrame, n_resamples: int = 10_000, alpha: float = 0.05
) -> pd.DataFrame:
    """Delta_int for every measured cell, core tier rows only."""
    frames = []
    acc_columns = [c for c in table.columns if c.startswith("test/acc")]
    core = tier_rows(table, TIER_CORE)
    core = core[core["gate_mode"].isin(["full", "kda"])]
    for task in core["task"].dropna().unique():
        sub = core[core["task"] == task]
        for column in acc_columns:
            if sub[column].notna().sum() == 0:
                continue
            result = did_table_for(sub, column, n_resamples=n_resamples, alpha=alpha)
            if not result.empty:
                result.insert(0, "task", task)
                frames.append(result)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def report(table: pd.DataFrame, options: ReportOptions) -> Report:
    frame = run(table, options.n_resamples, options.alpha)
    return Report(
        name="all_cells",
        title=TITLE,
        tables={"did_all_cells": frame},
        markdown=markdown_section(TITLE, frame),
    )
