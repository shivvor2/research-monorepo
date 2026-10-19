"""Matched bytes: the main effect against the value-dim-expanded single
state.

Main-effect claims (multi-state vs single-state) must control for
recurrent-state bytes: the multi-state layouts carry 10 slots' worth of
state (window 8 + 2 ws slots; the ll slot count at T=256 is also 10),
so the matched control is single-state ``full`` with the value dimension
expanded to the same slot count (``expand_v = 10``). The table reports
the per-seed paired difference (multi-state full) - (matched single
full) on overall accuracy, per task and layout.

The pairing goes through the shared per-seed contrast helper on the
union of the core and matched-bytes rows: a seed enters only when both
of its cells select exactly one row, and a missing or duplicated cell
drops the seed with a warning instead of passing silently. Both means of
a row describe the paired seeds only (the core tier carries ten seeds,
the matched tier five), so the columns of one row never mix seed sets.
"""

from __future__ import annotations

import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import TIER_CORE, TIER_MATCHED_BYTES
from pilot_analysis.records import tier_rows
from pilot_analysis.render import markdown_section
from pilot_analysis.statistics import bootstrap_ci, paired_contrast

TITLE = "Matched bytes (main effect vs the expanded single state)"

METRIC = "test/acc"


def run(table: pd.DataFrame, n_resamples: int, alpha: float) -> pd.DataFrame:
    """(multi-state full) - (matched single full) per task and layout.

    Within each task, the per-seed contrast pairs the core tier's
    ``(layout, full)`` cell against the matched tier's
    ``(single, full)`` cell; the means describe the paired seeds only.
    """
    core = tier_rows(table, TIER_CORE)
    t5 = tier_rows(table, TIER_MATCHED_BYTES)
    if core.empty or t5.empty or METRIC not in table.columns:
        return pd.DataFrame()
    records = []
    for task in sorted(set(core["task"].dropna()) & set(t5["task"].dropna())):
        paired = pd.concat(
            [core[core["task"] == task], t5[t5["task"] == task]], ignore_index=True
        )
        for layout in ("ws", "ll"):
            contrast = paired_contrast(
                paired,
                METRIC,
                [
                    ({"tier": TIER_CORE, "layout": layout, "gate_mode": "full"}, 1.0),
                    (
                        {
                            "tier": TIER_MATCHED_BYTES,
                            "layout": "single",
                            "gate_mode": "full",
                        },
                        -1.0,
                    ),
                ],
                name=f"matched_bytes_{layout}",
            )
            if contrast.empty:
                continue
            seeds = contrast.index
            multistate = paired[
                (paired["tier"] == TIER_CORE)
                & (paired["layout"] == layout)
                & (paired["gate_mode"] == "full")
                & (paired["seed"].isin(seeds))
            ]
            matched_single = paired[
                (paired["tier"] == TIER_MATCHED_BYTES)
                & (paired["layout"] == "single")
                & (paired["gate_mode"] == "full")
                & (paired["seed"].isin(seeds))
            ]
            mean, low, high = bootstrap_ci(
                contrast.to_numpy(dtype=float),
                n_resamples=n_resamples,
                alpha=alpha,
            )
            records.append(
                {
                    "task": task,
                    "layout": layout,
                    "metric": METRIC,
                    "n_seeds": int(contrast.size),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "mean_matched_single": float(
                        matched_single[METRIC].astype(float).mean()
                    ),
                    "mean_multistate": float(multistate[METRIC].astype(float).mean()),
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
