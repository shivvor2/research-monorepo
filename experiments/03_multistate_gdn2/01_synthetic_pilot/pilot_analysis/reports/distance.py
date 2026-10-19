"""Accuracy and leakage by write-distance bin, per layout and gate."""

from __future__ import annotations

import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import TIER_CORE
from pilot_analysis.records import tier_rows
from pilot_analysis.render import markdown_section

TITLE = "Write distance (core tier)"


def run(table: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and leakage by write-distance bin, per layout and gate.

    Core tier rows only. The run JSONs carry the per-bin metrics from
    ``RecallMetrics`` (``test/acc/dist={lo}-{hi}`` and
    ``test/leak/dist={lo}-{hi}``); this table averages them over seeds per
    (layout, gate mode), so the distance response of each row of the grid
    is readable next to the class-binned numbers.
    """
    core = tier_rows(table, TIER_CORE)
    core = core[core["task"] == "overwrite"]
    if core.empty:
        return pd.DataFrame()
    dist_columns = sorted(
        c for c in core.columns if c.startswith(("test/acc/dist=", "test/leak/dist="))
    )
    records = []
    for (layout, gate_mode), group in core.groupby(["layout", "gate_mode"]):
        for column in dist_columns:
            values = group[column].dropna()
            if values.empty:
                continue
            records.append(
                {
                    "layout": layout,
                    "gate_mode": gate_mode,
                    "metric": column,
                    "n_seeds": int(values.size),
                    "mean": float(values.mean()),
                }
            )
    return pd.DataFrame(records)


def report(table: pd.DataFrame, options: ReportOptions) -> Report:
    frame = run(table)
    return Report(
        name="distance",
        title=TITLE,
        tables={"write_distance": frame},
        markdown=markdown_section(TITLE, frame),
    )
