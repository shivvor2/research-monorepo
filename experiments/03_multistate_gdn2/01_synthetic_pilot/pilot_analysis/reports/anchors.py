"""Tier 6 anchors: our Gated DeltaNet numbers next to the published ones.

The published values are from the Log-Linear paper's Table 2 (Gated
DeltaNet / with Log-Linear on MQAR), the sanity anchor for the pilot's
own gate implementation.
"""

from __future__ import annotations

import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import LOG_LINEAR_STD_2, LOG_LINEAR_TABLE_2, TIER_ANCHORS
from pilot_analysis.records import tier_rows
from pilot_analysis.render import markdown_section

TITLE = "Anchors next to Log-Linear Table 2"


def run(table: pd.DataFrame) -> pd.DataFrame:
    """Tier 6 anchor table next to Log-Linear Table 2."""
    anchors = tier_rows(table, TIER_ANCHORS)
    if anchors.empty:
        return pd.DataFrame()
    records = []
    for (d_model, layout), group in anchors.groupby(["d_model", "layout"]):
        acc = group["test/acc"].dropna()
        record = {
            "d_model": d_model,
            "layout": layout,
            "n_seeds": int(acc.size),
            "acc_mean": float(acc.mean()) if acc.size else None,
            "acc_std": float(acc.std()) if acc.size > 1 else None,
            "log_linear_acc": LOG_LINEAR_TABLE_2.get(int(d_model), {}).get(layout),
            "log_linear_std": LOG_LINEAR_STD_2.get(int(d_model), {}).get(layout),
        }
        records.append(record)
    return pd.DataFrame(records).sort_values(["d_model", "layout"])


def report(table: pd.DataFrame, options: ReportOptions) -> Report:
    frame = run(table)
    return Report(
        name="anchors",
        title=TITLE,
        tables={"anchors": frame},
        markdown=markdown_section(TITLE, frame),
    )
