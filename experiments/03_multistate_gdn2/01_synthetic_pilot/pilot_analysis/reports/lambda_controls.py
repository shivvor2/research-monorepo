"""Prediction 4: lambda-static gives no gain over single-state GDN-2
once the number of keys exceeds d_k.

The lambda-control tier's ``static`` rows against the core tier's
single-state ``full`` rows, joined by seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import TIER_CORE, TIER_LAMBDA
from pilot_analysis.records import tier_rows
from pilot_analysis.render import markdown_section
from pilot_analysis.statistics import bootstrap_ci

TITLE = "Prediction 4: lambda static vs single-state full"


def run(table: pd.DataFrame, n_resamples: int, alpha: float) -> pd.DataFrame:
    """Per-seed (lambda-static multi-state) - (single-state full) differences."""
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

    core = tier_rows(table, TIER_CORE)
    core_ow = core[core["task"] == "overwrite"]
    t4 = tier_rows(table, TIER_LAMBDA)
    t4_ow = t4[t4["task"] == "overwrite"]
    base = core_ow[(core_ow["layout"] == "single") & (core_ow["gate_mode"] == "full")]
    distant = [
        c
        for c in table.columns
        if c.startswith("test/acc/slice=") and c.endswith("/class=distant")
    ]
    for layout in ("ws", "ll"):
        static = t4_ow[(t4_ow["lambda_mode"] == "static") & (t4_ow["layout"] == layout)]
        for column in distant:
            values = []
            for seed in sorted(
                set(static["seed"].dropna()) & set(base["seed"].dropna())
            ):
                cell_x = static[
                    (static["seed"] == seed) & (static["gate_mode"] == "full")
                ][column]
                cell_s = base[(base["seed"] == seed)][column]
                if len(cell_x) == 1 and len(cell_s) == 1:
                    values.append(float(cell_x.iloc[0]) - float(cell_s.iloc[0]))
            _add(f"P4 static - single-full ({column})", layout, "full", values)
    return pd.DataFrame(records)


def report(table: pd.DataFrame, options: ReportOptions) -> Report:
    frame = run(table, options.n_resamples, options.alpha)
    return Report(
        name="lambda_controls",
        title=TITLE,
        tables={"predictions_lambda": frame},
        markdown=markdown_section(TITLE, frame),
    )
