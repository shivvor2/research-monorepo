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

import numpy as np
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


def missing_runs(
    table: pd.DataFrame,
    expected_counts: dict[str, int],
    done_status: str = "test_done",
) -> pd.DataFrame:
    """Per-tier expected vs observed done runs, plus unexpected tiers.

    Every tier in ``expected_counts`` gets a row (observed ``0`` when no
    done run is present); every tier that has runs in ``table`` but no
    expected count is appended as ``expected=0`` (a tier with runs but
    no declaration, flagged by ``known=False``).

    Args:
        table: The loaded run table (a ``tier`` column).
        expected_counts: Expected done-count per tier.
        done_status: The run status that counts as done.

    Returns:
        ``DataFrame`` with ``tier``, ``expected``, ``observed``,
        ``missing``, ``known``.
    """
    tiers = list(expected_counts)
    present = (
        set(table["tier"].dropna().astype(str)) if "tier" in table.columns else set()
    )
    if "tier" in table.columns and "status" in table.columns:
        observed = (
            table[table["status"].fillna("").astype(str) == done_status]
            .groupby("tier")["run_id"]
            .nunique()
        )
    else:
        observed = pd.Series(dtype=int)
    # Tiers present in the runs but not declared expected: report them.
    for tier in sorted(present - set(tiers)):
        tiers.append(tier)
    rows = []
    for tier in tiers:
        obs = int(observed.get(tier, 0) or 0)
        exp = int(expected_counts.get(tier, 0))
        rows.append(
            {
                "tier": tier,
                "expected": exp,
                "observed": obs,
                "missing": max(0, exp - obs),
                "known": tier in expected_counts,
            }
        )
    return pd.DataFrame(rows)


_TIE_ATOL = 1e-12
"""Epsilon for "equal means": values come from JSON, so equal means are
symmetric; the epsilon only guards floating arrival order."""


def per_seed_mean(
    rows: pd.DataFrame,
    by: str | list[str],
    selection_seeds: tuple[int, ...],
    best: str = "best_value",
    step: str = "best_step",
) -> pd.DataFrame:
    """One row per ``by`` value: per-seed best values and steps, then means.

    Each ``by`` group is summarised across the selection seeds: the
    per-seed ``best``/``step`` values, the number of measured seeds, and
    the mean ``best`` (and mean ``step`` over the seeds that measured
    one). The base columns are whatever the caller's table carries
    (e.g. the learning rate), then the seed columns.

    Args:
        rows: The completed rows to summarise.
        by: One column or a list to group by.
        selection_seeds: The seeds averaged over.
        best: The best-value column.
        step: The best-step column.

    Returns:
        One row per ``by`` value, with ``n_seeds``, the per-seed
        columns and the means.
    """
    by_list = [by] if isinstance(by, str) else by
    if best == step:
        raise ValueError("best and step must name different columns")
    columns = [
        *by_list,
        "n_seeds",
        *(f"seed={seed}" for seed in selection_seeds),
        f"mean_{best}",
        f"mean_{step}",
    ]
    if rows.empty or any(b not in rows.columns for b in by_list):
        return pd.DataFrame(columns=columns)
    if best not in rows.columns or step not in rows.columns:
        return pd.DataFrame(columns=columns)
    records = []
    for value, group in rows.groupby(by_list, dropna=False):
        if len(by_list) == 1:
            record = {by_list[0]: value[0]}
        else:
            record = dict(zip(by_list, value))
        record["n_seeds"] = int(group[best].notna().sum())
        for seed in selection_seeds:
            cell = group[group["seed"] == seed][best]
            record[f"seed={seed}"] = float(cell.iloc[0]) if len(cell) == 1 else None
        values = group[best].dropna()
        record[f"mean_{best}"] = float(values.mean()) if len(values) else None
        steps = group[step].dropna()
        record[f"mean_{step}"] = float(steps.mean()) if len(steps) else None
        records.append(record)
    return pd.DataFrame(records)


def select_lr(per_lr: pd.DataFrame) -> float | None:
    """The selected learning rate for one configuration.

    Best mean accuracy first; ties go to the fewer mean steps to the
    best value, then to the smaller LR.

    Args:
        per_lr: One row per learning rate (indexed by ``lr``) with the
            columns ``mean_acc`` and ``mean_steps``.

    Returns:
        The selected ``lr``, or ``None`` when nothing was measured.
    """
    per_lr = per_lr[per_lr["mean_acc"].notna()]
    if per_lr.empty:
        return None
    best_acc = per_lr["mean_acc"].max()
    tied = per_lr[np.isclose(per_lr["mean_acc"], best_acc, rtol=0.0, atol=_TIE_ATOL)]
    best_steps = tied["mean_steps"].min()  # skips NaN
    if not np.isnan(best_steps):
        tied = tied[
            np.isclose(tied["mean_steps"], best_steps, rtol=0.0, atol=_TIE_ATOL)
        ]
    return float(tied.index.min())


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
