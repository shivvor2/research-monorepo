"""Shared analysis pieces for this project's reports.

Everything here is free of a specific tier's constants: it works on a
loaded run table (rows from
:func:`research_lib.training.records.load_records` / ``common.records.flatten``)
and takes the column names and the gate / layout levels it needs as
arguments. The per-pilot protocol (screening cells, decision thresholds,
tier filtering) lives in each pilot's own ``protocol.py``.

The primary statistic of the project stems from
:class:`research_lib.utils.stats.paired_contrast` /
:func:`interaction_contrast` and :func:`bootstrap_ci`; this module wires
them into the report-level readings: a gap column at a gate pair, a
paired gap over seeds with a bootstrap CI, and Δ_int tables over an
arbitrary layout list.
"""

from __future__ import annotations

import pandas as pd

from research_lib.utils.stats import (
    Factor,
    bootstrap_ci,
    interaction_contrast,
    paired_contrast,
)


def gap_pairs(
    table: pd.DataFrame,
    metric: str,
    high_gate: str,
    low_gate: str,
) -> pd.Series:
    """Per-seed paired difference ``high_gate - low_gate`` on ``metric``.

    Pairs rows by their other identity columns (what remains once the
    gate factor is stripped, i.e. seed and layout). Uses
    :func:`research_lib.utils.stats.paired_contrast`.

    Args:
        table: The loaded run table.
        metric: The metric column.
        high_gate: The gate level subtracted from (e.g. ``"full"``).
        low_gate: The gate level subtracted (e.g. ``"kda"``).

    Returns:
        A ``Series`` of per-unit (per-seed) paired differences.
    """
    return paired_contrast(
        table,
        metric,
        [
            ({"gate_mode": high_gate}, 1.0),
            ({"gate_mode": low_gate}, -1.0),
        ],
        name=f"gap_{high_gate}_{low_gate}",
    )


def delta_int(
    table: pd.DataFrame,
    metric: str,
    layout: str,
    high_gate: str = "full",
    low_gate: str = "kda",
) -> pd.Series:
    """Δ_int per seed: the 2×2 interaction of ``layout`` vs single-state
    with the gate factor (``high_gate`` vs ``low_gate``).

    ``layout="single"`` raises (the factor's levels coincide), so a
    reporting loop skips the reference row naturally.

    Args:
        table: The loaded run table.
        metric: The metric column.
        layout: The multi-state layout level.
        high_gate: The leading gate level.
        low_gate: The reference gate level.

    Returns:
        A per-seed ``Series`` of the interaction contrasts.
    """
    return interaction_contrast(
        table,
        metric,
        Factor("layout", layout, "single"),
        Factor("gate_mode", high_gate, low_gate),
        name=f"did_{layout}",
    )


def delta_int_table(
    table: pd.DataFrame,
    metric: str,
    layouts: tuple[str, ...] = ("ws", "ll"),
    n_resamples: int = 10_000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Δ_int per layout for one metric, with percentile bootstrap CIs.

    Args:
        table: The loaded run table.
        metric: The metric column.
        layouts: The multi-state layouts to report (a pilot may extend).
        n_resamples: Bootstrap resamples per interval.
        alpha: Two-sided level: the interval covers ``1 - alpha``.

    Returns:
        A DataFrame with one row per layout: ``layout``, ``metric``,
        ``n_seeds``, ``mean``, ``ci_low``, ``ci_high``.
    """
    records = []
    for layout in layouts:
        sub = table[table[metric].notna()]
        deltas = delta_int(sub, metric, layout, "full", "kda")
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
            }
        )
    return pd.DataFrame(records)


def paired_gap_report(
    table: pd.DataFrame,
    metric: str,
    high_gate: str = "full",
    low_gate: str = "kda",
    n_resamples: int = 10_000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """The paired ``high_gate - low_gate`` gap over seeds, with a bootstrap
    CI and both gates' means.

    Args:
        table: The loaded run table (a single-state screening cell; the
            caller filters).
        metric: The metric column.
        high_gate: The leading gate level.
        low_gate: The reference gate level.
        n_resamples: Bootstrap resamples per interval.
        alpha: Two-sided level.

    Returns:
        One row: ``n_seeds``, ``mean_gap``, ``ci_low``, ``ci_high``,
        ``mean_high``, ``mean_low``.
    """
    gaps = gap_pairs(table, metric, high_gate, low_gate)
    if gaps.empty:
        return pd.DataFrame(
            columns=[
                "n_seeds",
                "mean_gap",
                "ci_low",
                "ci_high",
                "mean_high",
                "mean_low",
            ]
        )
    mean, low, high = bootstrap_ci(gaps.values, n_resamples=n_resamples, alpha=alpha)
    best_high = _mean_over(table, metric, high_gate)
    best_low = _mean_over(table, metric, low_gate)
    return pd.DataFrame(
        [
            {
                "n_seeds": int(gaps.size),
                "mean_gap": mean,
                "ci_low": low,
                "ci_high": high,
                "mean_high": best_high,
                "mean_low": best_low,
            }
        ]
    )


def _mean_over(table: pd.DataFrame, metric: str, gate_mode: str) -> float | None:
    sub = table[table["gate_mode"] == gate_mode][metric].dropna()
    return float(sub.mean()) if not sub.empty else None


def gap_column(
    table: pd.DataFrame, metric: str, high_gate: str = "full", low_gate: str = "kda"
) -> pd.Series:
    """A ``full - kda`` gap column at each row's point (by its identity).

    Args:
        table: The loaded run table (already sliced to one selected point
            per layout/gate).
        metric: The metric column.
        high_gate: The leading gate level.
        low_gate: The reference gate level.

    Returns:
        A ``Series`` of ``high_gate`` value minus ``low_gate`` value per
        intersecting layout/seed, NaN where either side is absent.
    """
    pivot = table.pivot_table(
        index="layout",
        columns="gate_mode",
        values=metric,
        aggfunc="first",
    )
    if high_gate not in pivot.columns or low_gate not in pivot.columns:
        return pd.Series(index=table.index, dtype=float)
    return table["layout"].map(pivot[high_gate] - pivot[low_gate])
