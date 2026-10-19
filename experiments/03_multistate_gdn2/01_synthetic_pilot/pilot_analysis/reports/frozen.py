"""Predictions 2 and 3: transported vs the frozen control.

Prediction 2: transported (ws) beats frozen (fs) on cross-block
overwrite, with FS's errors concentrated on the stale value (leakage).
Prediction 3: frozen beats transported on never-overwritten keys
(``single`` class), with a smaller gap under GDN-2 gates than under KDA
gates. Both pair the core tier's ws rows against the frozen-control
tier's fs rows, joined by seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import TIER_CORE, TIER_FROZEN
from pilot_analysis.records import tier_rows
from pilot_analysis.render import markdown_section
from pilot_analysis.statistics import bootstrap_ci

TITLE = "Predictions 2 and 3: transported vs frozen control"


def _ws_fs_diff(ws_rows: pd.DataFrame, fs_rows: pd.DataFrame, column: str) -> list:
    values = []
    seeds = sorted(set(ws_rows["seed"].dropna()) & set(fs_rows["seed"].dropna()))
    for seed in seeds:
        cell_ws = ws_rows[(ws_rows["seed"] == seed)][column]
        cell_fs = fs_rows[(fs_rows["seed"] == seed)][column]
        if len(cell_ws) == 1 and len(cell_fs) == 1:
            values.append(float(cell_ws.iloc[0]) - float(cell_fs.iloc[0]))
    return values


def run(table: pd.DataFrame, n_resamples: int, alpha: float) -> pd.DataFrame:
    """Per-seed ws - fs paired differences (predictions 2 and 3)."""
    records = []

    def _add(name, layout, gate, values):
        if len(values) == 0:
            return
        mean, low, high = bootstrap_ci(
            np.asarray(values, dtype=float), n_resamples=n_resamples, alpha=alpha
        )
        records.append(
            {
                "prediction": name,
                "layout": layout,
                "gate_mode": gate,
                "n_seeds": len(values),
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )

    distant = [
        c
        for c in table.columns
        if c.startswith("test/acc/slice=") and c.endswith("/class=distant")
    ]
    leak = [
        c
        for c in table.columns
        if c.startswith("test/leak/slice=") and c.endswith("/class=distant")
    ]
    single_cols = [
        c
        for c in table.columns
        if c.startswith("test/acc/slice=") and c.endswith("/class=single")
    ]

    core = tier_rows(table, TIER_CORE)
    core_ow = core[core["task"] == "overwrite"]
    t3 = tier_rows(table, TIER_FROZEN)
    t3_ow = t3[t3["task"] == "overwrite"]

    # P2: transported >> frozen on cross-block overwrite (accuracy on the
    # distant class, and old-value leakage).
    for gate in ("full", "kda"):
        ws_rows = core_ow[(core_ow["gate_mode"] == gate) & (core_ow["layout"] == "ws")]
        fs_rows = t3_ow[(t3_ow["gate_mode"] == gate) & (t3_ow["layout"] == "fs")]
        for column in distant + leak:
            _add(
                f"P2 ws - fs ({column})",
                "ws-fs",
                gate,
                _ws_fs_diff(ws_rows, fs_rows, column),
            )

    # P3: frozen >= transported on never-overwritten keys.
    for gate in ("full", "kda"):
        ws_rows = core_ow[(core_ow["gate_mode"] == gate) & (core_ow["layout"] == "ws")]
        fs_rows = t3_ow[(t3_ow["gate_mode"] == gate) & (t3_ow["layout"] == "fs")]
        for column in single_cols:
            _add(
                f"P3 ws - fs ({column})",
                "ws-fs",
                gate,
                _ws_fs_diff(ws_rows, fs_rows, column),
            )
    return pd.DataFrame(records)


def report(table: pd.DataFrame, options: ReportOptions) -> Report:
    frame = run(table, options.n_resamples, options.alpha)
    return Report(
        name="frozen",
        title=TITLE,
        tables={"predictions_frozen": frame},
        markdown=markdown_section(TITLE, frame),
    )
