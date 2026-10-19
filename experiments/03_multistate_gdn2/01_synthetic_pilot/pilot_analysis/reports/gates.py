"""Prediction 1: gate decomposition (b_only / w_only recovered fractions).

The prediction: ``b_only`` (the decoupled channel-wise erase gate alone)
recovers most of Delta_int, ``w_only`` little. The effect is the
per-seed paired difference (layout, b/w_only) - (single, b/w_only) on
the gate-decomposition tier; the reference is the same difference at
``full`` on the core tier; the reported number is the recovered
fraction, joined by seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import TIER_CORE, TIER_GATES
from pilot_analysis.records import tier_rows
from pilot_analysis.render import markdown_section
from pilot_analysis.statistics import bootstrap_ci, paired_diff

TITLE = "Prediction 1: b_only recovers most of Delta_int, w_only little"


def run(table: pd.DataFrame, n_resamples: int, alpha: float) -> pd.DataFrame:
    """Recovered fractions of Delta_int per (layout, b/w_only gate)."""
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
    t2 = tier_rows(table, TIER_GATES)
    t2_ow = t2[t2["task"] == "overwrite"]
    distant = [
        c
        for c in table.columns
        if c.startswith("test/acc/slice=") and c.endswith("/class=distant")
    ]
    for column in distant:
        for layout in ("ws", "ll"):
            reference = paired_diff(core_ow, column, layout, "full")
            for gate in ("b_only", "w_only"):
                effect = paired_diff(t2_ow, column, layout, gate)
                if effect.empty or reference.empty:
                    continue
                common = effect.index.intersection(reference.index)
                if len(common) == 0:
                    continue
                recovered = (
                    (effect[common] / reference[common])
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                _add(
                    f"P1 recovered fraction ({column})",
                    layout,
                    gate,
                    recovered.values,
                )
    return pd.DataFrame(records)


def report(table: pd.DataFrame, options: ReportOptions) -> Report:
    frame = run(table, options.n_resamples, options.alpha)
    return Report(
        name="gates",
        title=TITLE,
        tables={"predictions_gates": frame},
        markdown=markdown_section(TITLE, frame),
    )
