"""Matched bytes: the main effect against the value-dim-expanded single
state.

Main-effect claims (multi-state vs single-state) must control for
recurrent-state bytes: the multi-state layouts carry 10 slots' worth of
state (window 8 + 2 ws slots; the ll slot count at T=256 is also 10),
so the matched control is single-state ``full`` with the value dimension
expanded to the same slot count (``expand_v = 10``). The table reports
the per-seed paired difference (multi-state full) - (matched single
full) on overall accuracy, per task and layout.
"""

from __future__ import annotations

import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import TIER_CORE, TIER_MATCHED_BYTES
from pilot_analysis.records import tier_rows
from pilot_analysis.render import markdown_section
from pilot_analysis.statistics import bootstrap_ci

TITLE = "Matched bytes (main effect vs the expanded single state)"

METRIC = "test/acc"


def run(table: pd.DataFrame, n_resamples: int, alpha: float) -> pd.DataFrame:
    """(multi-state full) - (matched single full) per task and layout."""
    core = tier_rows(table, TIER_CORE)
    t5 = tier_rows(table, TIER_MATCHED_BYTES)
    if core.empty or t5.empty or METRIC not in table.columns:
        return pd.DataFrame()
    records = []
    for task in sorted(set(core["task"].dropna()) & set(t5["task"].dropna())):
        base = t5[
            (t5["task"] == task)
            & (t5["layout"] == "single")
            & (t5["gate_mode"] == "full")
        ]
        for layout in ("ws", "ll"):
            rows = core[
                (core["task"] == task)
                & (core["layout"] == layout)
                & (core["gate_mode"] == "full")
            ]
            values = []
            for seed in sorted(set(rows["seed"].dropna()) & set(base["seed"].dropna())):
                cell_x = rows[(rows["seed"] == seed)][METRIC]
                cell_b = base[(base["seed"] == seed)][METRIC]
                if len(cell_x) == 1 and len(cell_b) == 1:
                    values.append(float(cell_x.iloc[0]) - float(cell_b.iloc[0]))
            if not values:
                continue
            mean, low, high = bootstrap_ci(
                pd.Series(values, dtype=float).values,
                n_resamples=n_resamples,
                alpha=alpha,
            )
            records.append(
                {
                    "task": task,
                    "layout": layout,
                    "metric": METRIC,
                    "n_seeds": len(values),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "mean_matched_single": float(base[METRIC].mean()),
                    "mean_multistate": float(rows[METRIC].mean()),
                }
            )
    return pd.DataFrame(records)


def report(table: pd.DataFrame, options: ReportOptions) -> Report:
    frame = run(table, options.n_resamples, options.alpha)
    return Report(
        name="matched_bytes",
        title=TITLE,
        tables={"matched_bytes": frame},
        markdown=markdown_section(TITLE, frame),
    )
