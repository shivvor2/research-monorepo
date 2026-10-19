"""Paired contrasts, bootstrap CIs and pre-registered decision rules."""

from __future__ import annotations

import pandas as pd

from research_lib.utils.stats import (
    Factor,
    bootstrap_ci,
    interaction_contrast,
    paired_contrast,
)


def decision_rule(low: float, high: float, primary: bool) -> str:
    """The pre-registered reading of a Delta_int CI."""
    if low > 0:
        return "CI entirely above 0: supports the hypothesis at pilot scale"
    if high < 0:
        return (
            "CI entirely below 0: transport makes selective erase redundant; "
            "informative, report it"
        )
    if primary:
        return (
            "CI includes 0 on the distant, K=2d_k cell: evidence AGAINST the "
            "mechanism (this cell stresses (a)+(b)+(c))"
        )
    return "CI includes 0: uninformative about the hypothesis on this cell"


def did(table: pd.DataFrame, metric: str, layout: str) -> pd.Series:
    """Delta_int per seed for one multi-state layout.

    The experiment's statistic: the 2x2 interaction of the layout factor
    (``layout`` vs ``single``) with the gate factor (``full`` vs ``kda``).
    ``layout="single"`` raises (the factor's levels coincide), so the
    all-cells loop skips it naturally.
    """
    return interaction_contrast(
        table,
        metric,
        Factor("layout", layout, "single"),
        Factor("gate_mode", "full", "kda"),
        name=f"did_{layout}",
    )


def paired_diff(
    table: pd.DataFrame, metric: str, layout: str, gate_mode: str
) -> pd.Series:
    """Per-seed paired difference ``(layout-gate) - (single-gate)``."""
    return paired_contrast(
        table,
        metric,
        [
            ({"layout": layout, "gate_mode": gate_mode}, 1.0),
            ({"layout": "single", "gate_mode": gate_mode}, -1.0),
        ],
        name=f"diff_{layout}_{gate_mode}",
    )


def did_table_for(
    table: pd.DataFrame,
    metric: str,
    layouts: tuple[str, ...] = ("ws", "ll", "fs"),
    n_resamples: int = 10_000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Delta_int per layout for one metric, with bootstrap CIs."""
    records = []
    for layout in layouts:
        sub = table[table[metric].notna()]
        try:
            deltas = did(sub, metric, layout)
        except ValueError:
            continue
        if deltas.empty:
            continue
        mean, low, high = bootstrap_ci(
            deltas.values, n_resamples=n_resamples, alpha=alpha
        )
        records.append(
            {
                "layout": layout,
                "metric": metric,
                "n_seeds": int(deltas.size),
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "reading": decision_rule(low, high, primary=False),
            }
        )
    return pd.DataFrame(records)
