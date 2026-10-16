"""
Tests for the pilot statistics helpers.

Target: src/research_lib/utils/stats.py
Run with: pytest src/research_lib/utils/tests/test_stats.py -v
"""

import numpy as np
import pandas as pd
import pytest

from research_lib.utils.stats import bootstrap_ci, did_per_seed


def make_table(seed_values=(0, 1, 2), acc_fn=None) -> pd.DataFrame:
    """A hand-built run table: layout {single, ws, ll} x gate {full, kda} x seeds."""
    rows = []
    for seed in seed_values:
        for layout in ("single", "ws", "ll"):
            for gate_mode in ("full", "kda"):
                acc = (
                    acc_fn(layout, gate_mode, seed)
                    if acc_fn is not None
                    else 0.5
                    + 0.01 * seed
                    + 0.1 * (layout != "single")
                    + 0.05 * (gate_mode == "full")
                )
                rows.append(
                    {
                        "layout": layout,
                        "gate_mode": gate_mode,
                        "seed": seed,
                        "acc": acc,
                    }
                )
    return pd.DataFrame(rows)


class TestDidPerSeed:
    """Tests for did_per_seed."""

    def test_exact_per_seed_differences(self):
        """Δ_i matches the hand-computed difference-in-differences."""
        table = make_table()
        for layout in ("ws", "ll"):
            deltas = did_per_seed(table, "acc", layout)
            for seed in (0, 1, 2):
                x_full = 0.5 + 0.01 * seed + 0.1 + 0.05
                x_kda = 0.5 + 0.01 * seed + 0.1
                s_full = 0.5 + 0.01 * seed + 0.05
                s_kda = 0.5 + 0.01 * seed
                expected = (x_full - x_kda) - (s_full - s_kda)
                assert deltas.loc[seed] == pytest.approx(expected)

    def test_returns_series_indexed_by_seed(self):
        deltas = did_per_seed(make_table(), "acc", "ws")
        assert isinstance(deltas, pd.Series)
        assert list(deltas.index) == [0, 1, 2]
        assert deltas.name == "did_ws"

    def test_missing_cells_are_dropped(self, caplog):
        """A seed missing any of the four cells is dropped and reported."""
        table = make_table()
        table = table[~((table["seed"] == 1) & (table["layout"] == "ws"))]
        deltas = did_per_seed(table, "acc", "ws")
        assert list(deltas.index) == [0, 2]
        assert any("dropped" in r.message for r in caplog.records)

    def test_duplicated_cells_are_dropped(self, caplog):
        """A seed with a duplicated cell is dropped (ambiguous), not averaged."""
        table = pd.concat(
            [make_table(), make_table(seed_values=(1,))], ignore_index=True
        )
        deltas = did_per_seed(table, "acc", "ws")
        assert list(deltas.index) == [0, 2]

    def test_all_missing_yields_empty_series(self):
        table = make_table()
        table = table[table["layout"] != "single"]
        deltas = did_per_seed(table, "acc", "ws")
        assert deltas.empty

    def test_rejects_single_layout(self):
        with pytest.raises(ValueError):
            did_per_seed(make_table(), "acc", "single")

    def test_rejects_missing_columns(self):
        table = make_table().drop(columns=["gate_mode"])
        with pytest.raises(ValueError):
            did_per_seed(table, "acc", "ws")
        table = make_table().drop(columns=["acc"])
        with pytest.raises(ValueError):
            did_per_seed(table, "acc", "ws")


class TestBootstrapCI:
    """Tests for bootstrap_ci."""

    def test_constant_series_has_degenerate_ci(self):
        mean, low, high = bootstrap_ci(np.full(10, 0.7))
        assert mean == pytest.approx(0.7)
        assert low == pytest.approx(0.7)
        assert high == pytest.approx(0.7)

    def test_ci_brackets_the_mean_and_is_deterministic(self):
        values = np.array([0.1, 0.2, 0.3, 0.6])
        first = bootstrap_ci(values, seed=3)
        second = bootstrap_ci(values, seed=3)
        mean, low, high = first
        assert first == second
        # The percentile CI brackets the sample mean and lies within the
        # data range (it need NOT reach the sample extremes: resample means
        # concentrate around the mean).
        assert low <= mean <= high
        assert values.min() <= low and high <= values.max()

    def test_respects_alpha(self):
        values = np.arange(20) / 20.0
        _, low95, high95 = bootstrap_ci(values, alpha=0.05, seed=0)
        _, low90, high90 = bootstrap_ci(values, alpha=0.10, seed=0)
        assert low90 >= low95 - 1e-12
        assert high90 <= high95 + 1e-12

    def test_empty_values_raise(self):
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([]))

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            bootstrap_ci(np.ones(3), n_resamples=0)
        with pytest.raises(ValueError):
            bootstrap_ci(np.ones(3), alpha=0.0)
