"""Paired contrasts and bootstrap intervals over tabular data.

The general setting: a table with one row per experimental unit's
condition, a column identifying the unit (``unit``, e.g. a seed: the
same unit is observed under several conditions), and a numeric outcome
column. Pairing by unit cancels unit-level noise.

- :func:`paired_contrast` computes, per unit, the weighted sum
  ``Σ coefficient · value(cell)`` over a set of cells; each cell is a
  ``(column == value filters, coefficient)`` pair selecting one row of
  the unit. A contrast over two cells is a difference; a one-cell
  contrast with coefficient 1 is the value itself.
- :func:`interaction_contrast` is the common 2×2 case: a
  difference-in-differences over two binary factors,
  ``(a_t,b_t − a_c,b_t) − (a_t,b_c − a_c,b_c)``, per unit.
- :func:`bootstrap_ci` gives the percentile bootstrap interval of any
  statistic over the per-unit values.

Run tests with::

    pytest src/research_lib/utils/tests/test_stats.py -v
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Factor(NamedTuple):
    """A binary factor of a contrast.

    Args:
        column: The table column the factor varies over.
        treated: The treated level.
        control: The control level (distinct from ``treated``).
    """

    column: str
    treated: Any
    control: Any


def paired_contrast(
    table: pd.DataFrame,
    metric: str,
    cells: Sequence[tuple[Mapping[str, Any], float]],
    *,
    unit: str = "seed",
    name: str | None = None,
) -> pd.Series:
    """Per unit: ``Σ coefficient · value(cell)``, indexed by unit, sorted.

    A unit contributes its contrast only when every cell selects exactly
    one row for it; a unit missing any cell, or with a duplicated cell,
    is dropped and reported with a warning naming this function, the
    metric and the dropped units.

    Args:
        table: One row per unit and condition.
        metric: Name of the numeric outcome column.
        cells: ``(filters, coefficient)`` pairs. ``filters`` is a mapping
            of ``column == value`` conditions selecting the cell's rows
            (they may be empty, selecting every row); the cell's
            contribution is ``coefficient *`` its metric value.
        unit: The column identifying the paired units.
        name: A label for the contrast (the returned Series' name and
            the log messages); defaults to ``"contrast"``.

    Returns:
        pd.Series: One contrast value per kept unit, sorted by unit.

    Raises:
        ValueError: ``metric``, ``unit`` or a filter column missing from
            ``table``, or ``cells`` empty.
    """
    label = name if name is not None else "paired_contrast"
    if not cells:
        raise ValueError("cells must be non-empty")
    filter_columns = sorted({column for filters, _ in cells for column in filters})
    required = [unit, metric, *filter_columns]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"table is missing the required columns {missing}")
    contrasts: dict[Any, float] = {}
    dropped: list[Any] = []
    for u in sorted(table[unit].dropna().unique()):
        rows = table[table[unit] == u]
        values: list[float] = []
        ok = True
        for filters, coefficient in cells:
            selected = rows
            for column, value in filters.items():
                selected = selected[selected[column] == value]
            if len(selected) != 1:
                ok = False
                break
            values.append(coefficient * float(selected[metric].iloc[0]))
        if ok:
            contrasts[u] = sum(values)
        else:
            dropped.append(u)
    if dropped:
        logger.warning(
            "%s(metric=%r): dropped units missing or duplicating a cell: %s",
            label,
            metric,
            dropped,
        )
    return pd.Series(
        contrasts, name=name if name is not None else "contrast"
    ).sort_index()


def interaction_contrast(
    table: pd.DataFrame,
    metric: str,
    factor_a: Factor,
    factor_b: Factor,
    *,
    unit: str = "seed",
    name: str | None = None,
) -> pd.Series:
    """2×2 difference-in-differences, per unit.

    ``(a_t, b_t − a_c, b_t) − (a_t, b_c − a_c, b_c)`` for the treated /
    control levels of the two factors: whether the effect of factor A
    differs between the levels of factor B. A thin wrapper over
    :func:`paired_contrast` with the four cells and coefficients
    ``+1, −1, −1, +1``.

    Args:
        table: One row per unit and condition.
        metric: Name of the numeric outcome column.
        factor_a: The first binary factor (its difference is taken).
        factor_b: The second binary factor (its levels split the two
            differences).
        unit: The column identifying the paired units.
        name: A label for the contrast; defaults to
            ``"interaction_contrast"``.

    Returns:
        pd.Series: One difference-in-differences per kept unit, sorted
        by unit.

    Raises:
        ValueError: A factor's treated and control levels are equal
            (the contrast would vanish), or any validation of
            :func:`paired_contrast`.
    """
    if factor_a.treated == factor_a.control:
        raise ValueError(f"factor_a's treated and control levels are equal: {factor_a}")
    if factor_b.treated == factor_b.control:
        raise ValueError(f"factor_b's treated and control levels are equal: {factor_b}")
    cells = [
        (
            {factor_a.column: factor_a.treated, factor_b.column: factor_b.treated},
            1.0,
        ),
        (
            {factor_a.column: factor_a.control, factor_b.column: factor_b.treated},
            -1.0,
        ),
        (
            {factor_a.column: factor_a.treated, factor_b.column: factor_b.control},
            -1.0,
        ),
        (
            {factor_a.column: factor_a.control, factor_b.column: factor_b.control},
            1.0,
        ),
    ]
    return paired_contrast(
        table,
        metric,
        cells,
        unit=unit,
        name=name if name is not None else "interaction_contrast",
    )


def bootstrap_ci(
    values: np.ndarray | pd.Series,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: Callable[..., Any] = np.mean,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI over per-unit values.

    Args:
        values: 1-D array-like of per-unit values.
        n_resamples: Number of bootstrap resamples. Defaults to 10,000.
        alpha: Two-sided level: the CI covers ``1 - alpha``. Defaults to
            0.05.
        seed: Seed of the resampling RNG (fixed, so the reported CI is
            reproducible). Defaults to 0.
        statistic: The statistic to interval-estimate. It must accept an
            ``axis`` keyword, as numpy reductions do: it is applied as
            ``statistic(values)`` for the point estimate and
            ``statistic(resamples, axis=1)`` for the bootstrap
            distribution.

    Returns:
        tuple: ``(estimate, low, high)`` — the statistic's value on
        ``values`` and the ``alpha/2`` / ``1 - alpha/2`` percentiles of
        its bootstrap distribution. The CI of a constant series is
        degenerate.

    Raises:
        ValueError: Empty ``values``, non-positive ``n_resamples`` or
            ``alpha`` outside ``(0, 1)``.
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("values must be non-empty")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    estimate = float(statistic(values))
    rng = np.random.default_rng(seed)
    index = rng.integers(0, values.size, size=(n_resamples, values.size))
    distribution = np.asarray(statistic(values[index], axis=1))
    low, high = np.quantile(distribution, [alpha / 2.0, 1.0 - alpha / 2.0])
    return estimate, float(low), float(high)
