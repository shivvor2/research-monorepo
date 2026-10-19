"""Lambda / erase statistics per slot, averaged per (layout, gate_mode)."""

from __future__ import annotations

import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import TIER_CORE
from pilot_analysis.records import tier_rows
from pilot_analysis.render import markdown_section

TITLE = "Lambda / erase statistics per slot (core tier)"


def run(table: pd.DataFrame) -> pd.DataFrame:
    """Lambda / erase statistics per slot, averaged per (layout, gate_mode).

    Core tier rows only, like every other statistic: calibration, smoke,
    R1, long-T and lambda-control runs repeat core configurations and
    must not mix in.
    """
    mixer_cols = [c for c in table.columns if c.startswith("mixer/")]
    if not mixer_cols:
        return pd.DataFrame()
    rows = tier_rows(table, TIER_CORE)
    rows = rows[rows["layout"] != "single"]
    records = []
    for (layout, gate_mode), group in rows.groupby(["layout", "gate_mode"]):
        record = {"layout": layout, "gate_mode": gate_mode, "n_runs": len(group)}
        for column in sorted(mixer_cols):
            values = group[column].dropna()
            if values.empty:
                continue
            record[column] = float(values.mean())
        records.append(record)
    return pd.DataFrame(records)


def report(table: pd.DataFrame, options: ReportOptions) -> Report:
    frame = run(table)
    return Report(
        name="mixer",
        title=TITLE,
        tables={"mixer_stats": frame},
        markdown=markdown_section(TITLE, frame),
    )
