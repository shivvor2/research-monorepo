"""Long-sequence tier: Delta_int on the long, sparse MQAR variant."""

from __future__ import annotations

import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import TIER_LONG_T
from pilot_analysis.records import tier_rows
from pilot_analysis.render import markdown_section
from pilot_analysis.statistics import bootstrap_ci, decision_rule, did

TITLE = "Long-sequence tier (Delta_int)"


def run(table: pd.DataFrame, n_resamples: int, alpha: float) -> pd.DataFrame:
    """Delta_int on the long-sequence tier, per accuracy cell.

    The long-sequence tier repeats the core grid on the long, sparse MQAR
    variant; its own table, never mixed into the T=256 statistic.
    """
    records = []
    long_t = tier_rows(table, TIER_LONG_T)
    if long_t.empty:
        return pd.DataFrame()
    acc_columns = [
        c
        for c in long_t.columns
        if c == "test/acc" or (c.startswith("test/acc/slice=") and "/class=" not in c)
    ]
    for column in acc_columns:
        sub = long_t[long_t[column].notna()]
        if sub.empty:
            continue
        for layout in ("ws", "ll"):
            deltas = did(sub, column, layout)
            if deltas.empty:
                continue
            mean, low, high = bootstrap_ci(
                deltas.values, n_resamples=n_resamples, alpha=alpha
            )
            records.append(
                {
                    "layout": layout,
                    "metric": column,
                    "n_seeds": int(deltas.size),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "reading": decision_rule(low, high, primary=False),
                }
            )
    return pd.DataFrame(records)


def report(table: pd.DataFrame, options: ReportOptions) -> Report:
    frame = run(table, options.n_resamples, options.alpha)
    return Report(
        name="long_t",
        title=TITLE,
        tables={"long_t_did": frame},
        markdown=markdown_section(TITLE, frame),
    )
