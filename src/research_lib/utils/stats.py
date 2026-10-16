"""Statistics helpers for the multi-state GDN-2 pilot analysis.

Per-seed difference-in-differences
:func:`did_per_seed` computes the protocol's primary statistic
(``06_experiment_protocol.md`` §6):

.. math::

    \\Delta_i = (a_i^{X\\text{-full}} - a_i^{X\\text{-kda}})
              - (a_i^{\\text{single-full}} - a_i^{\\text{single-kda}})

for a multi-state layout ``X``, pairing runs by seed (seed ``i`` fixes
the parameter init and the data stream, so cell values with the same
``i`` share data).  :func:`bootstrap_ci` gives the percentile bootstrap
CI over the per-seed values.

Run tests with::

    pytest src/research_lib/utils/tests/test_stats.py -v
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ("layout", "gate_mode", "seed")


def did_per_seed(table: pd.DataFrame, metric: str, layout: str) -> pd.Series:
    """Per-seed Δ_int for a multi-state layout.

    For each seed ``i`` present in ``table``:

    .. math::

        \\Delta_i = (a_i^{X\\text{-full}} - a_i^{X\\text{-kda}})
                  - (a_i^{\\text{single-full}} - a_i^{\\text{single-kda}})

    with ``X = layout``.  Seeds missing any of the four cells (or with a
    duplicated cell) are dropped and reported in a log message.

    Args:
        table: One row per run with columns ``layout``, ``gate_mode``,
            ``seed`` and the metric column.
        metric: Name of the metric column (e.g. a ``RecallMetrics`` key
            such as ``"acc"``).
        layout: The multi-state layout ``X`` (``"ws"`` / ``"fs"`` /
            ``"ll"``).

    Returns:
        pd.Series: Δ per seed, indexed by seed, named
        ``"did_{layout}"``.

    Raises:
        ValueError: Missing required columns, unknown metric column, or
            ``layout == "single"`` (the single-state rows are the shared
            baseline of the statistic).
    """
    layout = str(layout)
    if layout == "single":
        raise ValueError(
            "layout must be a multi-state layout (ws/fs/ll); the single-state "
            "rows are the shared baseline of the statistic"
        )
    for column in (*_REQUIRED_COLUMNS, metric):
        if column not in table.columns:
            raise ValueError(f"table is missing the required column {column!r}")
    deltas: dict[int, float] = {}
    dropped: list[int] = []
    for seed in sorted(table["seed"].unique()):
        cells = {}
        for lay, gate in (
            (layout, "full"),
            (layout, "kda"),
            ("single", "full"),
            ("single", "kda"),
        ):
            rows = table[
                (table["layout"] == lay)
                & (table["gate_mode"] == gate)
                & (table["seed"] == seed)
            ]
            cells[(lay, gate)] = rows[metric]
        if any(len(values) != 1 for values in cells.values()):
            dropped.append(seed)
            continue
        x_full = float(cells[(layout, "full")].iloc[0])
        x_kda = float(cells[(layout, "kda")].iloc[0])
        s_full = float(cells[("single", "full")].iloc[0])
        s_kda = float(cells[("single", "kda")].iloc[0])
        deltas[int(seed)] = (x_full - x_kda) - (s_full - s_kda)
    if dropped:
        logger.warning(
            "did_per_seed(layout=%r, metric=%r): dropped seeds missing or "
            "duplicating one of the four cells: %s",
            layout,
            metric,
            dropped,
        )
    return pd.Series(deltas, dtype=float, name=f"did_{layout}").sort_index()


def bootstrap_ci(
    values: np.ndarray | pd.Series,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over the given per-seed values.

    Args:
        values: 1-D array-like of per-seed values.
        n_resamples: Number of bootstrap resamples. Defaults to 10,000
            (the protocol's choice).
        alpha: Two-sided level: the CI covers ``1 - alpha``. Defaults to
            0.05.
        seed: Seed of the resampling RNG (fixed, so the reported CI is
            reproducible). Defaults to 0.

    Returns:
        tuple: ``(mean, low, high)`` — the sample mean and the
        ``alpha/2`` / ``1 - alpha/2`` percentiles of the bootstrap mean
        distribution.  The CI of a constant series is degenerate.

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
    mean = float(values.mean())
    rng = np.random.default_rng(seed)
    index = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[index].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return mean, float(low), float(high)
