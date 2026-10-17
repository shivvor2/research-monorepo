"""
Tests for the paired-contrast and bootstrap helpers.

Target: src/research_lib/utils/stats.py
Run with: pytest src/research_lib/utils/tests/test_stats.py -v
"""

import logging

import numpy as np
import pandas as pd
import pytest

from research_lib.utils.stats import (
    Factor,
    bootstrap_ci,
    interaction_contrast,
    paired_contrast,
)


def make_table(seed_values=(0, 1, 2), acc_fn=None) -> pd.DataFrame:
    """A hand-built table: layout {single, ws, ll} x gate {full, kda} x seeds."""
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


# The bootstrap_ci outputs of the implementation before the statistic
# parameter was added (captured on the same inputs and seeds; the default
# must stay bit-identical, RNG calls included).
_BOOTSTRAP_EXPECTED = {
    "a": [0.30000000000000004, 0.15, 0.5],
    "b": [0.4749999999999999, 0.34750000000000003, 0.5975],
    "c": [0.7, 0.7, 0.7],
    "d": [0.45, 0.16666666666666666, 0.7333333333333334],
}


class TestPairedContrast:
    """paired_contrast: weighted sums of cells, paired by unit."""

    def test_two_cells_is_a_difference(self):
        # (ws, full) - (single, full), per seed.
        table = make_table()
        cells = [
            ({"layout": "ws", "gate_mode": "full"}, 1.0),
            ({"layout": "single", "gate_mode": "full"}, -1.0),
        ]
        result = paired_contrast(table, "acc", cells, name="ws_minus_single")
        assert list(result.index) == [0, 1, 2]
        for seed in (0, 1, 2):
            assert result.loc[seed] == pytest.approx(0.1)
        assert result.name == "ws_minus_single"

    def test_four_cells_weighted_sum(self):
        # A weighted sum over four cells: 2*ws-full - ll-kda + single-kda - 1/2*single-full.
        table = make_table()
        cells = [
            ({"layout": "ws", "gate_mode": "full"}, 2.0),
            ({"layout": "ll", "gate_mode": "kda"}, -1.0),
            ({"layout": "single", "gate_mode": "kda"}, 1.0),
            ({"layout": "single", "gate_mode": "full"}, -0.5),
        ]
        result = paired_contrast(table, "acc", cells)
        for seed in (0, 1, 2):
            s = 0.5 + 0.01 * seed
            expected = 2.0 * (s + 0.1 + 0.05) - (s + 0.1) + s - 0.5 * (s + 0.05)
            assert result.loc[seed] == pytest.approx(expected)

    def test_empty_filters_select_every_row_of_the_unit(self):
        # In a table with one row per unit, the empty filter is that row.
        table = make_table()
        table = table[(table["layout"] == "ll") & (table["gate_mode"] == "kda")]
        result = paired_contrast(table, "acc", [({}, 1.0)])
        for seed in (0, 1, 2):
            assert result.loc[seed] == pytest.approx(0.5 + 0.01 * seed + 0.1)

    def test_missing_and_duplicated_cells_are_dropped(self, caplog):
        table = make_table()
        # Seed 1 misses its ws row; seed 2 duplicates its ll-kda row.
        table = table[~((table["seed"] == 1) & (table["layout"] == "ws"))]
        table = pd.concat([table, make_table(seed_values=(2,))], ignore_index=True)
        cells = [
            ({"layout": "ws", "gate_mode": "full"}, 1.0),
            ({"layout": "single", "gate_mode": "full"}, -1.0),
        ]
        with caplog.at_level(logging.WARNING):
            result = paired_contrast(table, "acc", cells, name="my_contrast")
        assert list(result.index) == [0]
        assert any(
            "my_contrast" in r.message and "dropped" in r.message
            for r in caplog.records
        )
        assert any("acc" in r.message for r in caplog.records)

    def test_all_missing_yields_empty_series(self, caplog):
        table = make_table()
        table = table[table["layout"] != "single"]
        with caplog.at_level(logging.WARNING):
            result = paired_contrast(
                table, "acc", [({"layout": "single"}, 1.0)], name="c"
            )
        assert result.empty

    def test_validation(self):
        table = make_table()
        with pytest.raises(ValueError, match="non-empty"):
            paired_contrast(table, "acc", [])
        with pytest.raises(ValueError, match="required columns"):
            paired_contrast(table, "missing_metric", [({"layout": "ws"}, 1.0)])
        with pytest.raises(ValueError, match="required columns"):
            paired_contrast(table, "acc", [({"missing_col": "ws"}, 1.0)])
        with pytest.raises(ValueError, match="required columns"):
            paired_contrast(table, "acc", [({"layout": "ws"}, 1.0)], unit="run")


class TestInteractionContrast:
    """interaction_contrast: the 2x2 difference-in-differences."""

    def test_hand_computed_did(self):
        """The per-seed DiD matches the hand-computed differences.

        With the default acc_fn the full-gate offset is the same for the
        multi-state and the single layout, so the 2x2 DiD cancels to
        exactly 0.0 for ws and ll - the paired-contrast arithmetic is
        checked against the by-hand value.
        """
        table = make_table()
        for layout in ("ws", "ll"):
            result = interaction_contrast(
                table,
                "acc",
                Factor("layout", layout, "single"),
                Factor("gate_mode", "full", "kda"),
                name=f"did_{layout}",
            )
            assert list(result.index) == [0, 1, 2]
            assert result.name == f"did_{layout}"
            for seed in (0, 1, 2):
                assert result.loc[seed] == pytest.approx(0.0)

    def test_did_with_nonzero_interaction(self):
        # Gate full helps ws more than single: DiD = 0.03 per hand:
        # (0.13 - 0.02) - (0.10 - 0.02) = 0.11 - 0.08.
        def acc(layout, gate, seed):
            base = {"single": 0.5, "ws": 0.55, "ll": 0.60}[layout]
            off = {"full": 0.10, "kda": 0.02}[gate]
            if layout == "ws":
                off = {"full": 0.13, "kda": 0.02}[gate]
            return base + off + 0.001 * seed

        table = make_table(acc_fn=acc)
        result = interaction_contrast(
            table,
            "acc",
            Factor("layout", "ws", "single"),
            Factor("gate_mode", "full", "kda"),
        )
        for seed in (0, 1, 2):
            assert result.loc[seed] == pytest.approx(0.03)

    def test_unit_column_is_configurable(self):
        table = make_table(seed_values=(7,))
        table = table.rename(columns={"seed": "rep"})
        result = interaction_contrast(
            table,
            "acc",
            Factor("layout", "ws", "single"),
            Factor("gate_mode", "full", "kda"),
            unit="rep",
        )
        assert list(result.index) == [7]

    def test_equal_factor_levels_rejected(self):
        table = make_table()
        with pytest.raises(ValueError, match="factor_a"):
            interaction_contrast(
                table,
                "acc",
                Factor("layout", "ws", "ws"),
                Factor("gate_mode", "full", "kda"),
            )
        with pytest.raises(ValueError, match="factor_b"):
            interaction_contrast(
                table,
                "acc",
                Factor("layout", "ws", "single"),
                Factor("gate_mode", "full", "full"),
            )

    def test_missing_columns_raise(self):
        table = make_table()
        with pytest.raises(ValueError, match="required columns"):
            interaction_contrast(
                table,
                "missing",
                Factor("layout", "ws", "single"),
                Factor("gate_mode", "full", "kda"),
            )


class TestBootstrapCI:
    """Bootstrap intervals, including a custom statistic."""

    def test_default_is_bit_identical_to_the_previous_implementation(self):
        # Captured on the same inputs before the statistic parameter was
        # added; same RNG calls, same order.
        values = np.array([0.1, 0.2, 0.3, 0.6])
        assert bootstrap_ci(values, seed=3) == tuple(_BOOTSTRAP_EXPECTED["a"])
        values = np.arange(20) / 20.0
        assert bootstrap_ci(values, seed=0) == tuple(_BOOTSTRAP_EXPECTED["b"])
        values = np.full(10, 0.7)
        assert bootstrap_ci(values, seed=5, n_resamples=50) == tuple(
            _BOOTSTRAP_EXPECTED["c"]
        )
        values = np.array([0.05, 0.4, 0.9])
        assert bootstrap_ci(values, alpha=0.10, seed=11) == tuple(
            _BOOTSTRAP_EXPECTED["d"]
        )

    def test_constant_series_has_degenerate_ci(self):
        mean, low, high = bootstrap_ci(np.full(10, 0.7))
        assert mean == pytest.approx(0.7)
        assert low == pytest.approx(0.7)
        assert high == pytest.approx(0.7)

    def test_custom_statistic_median(self):
        values = np.array([0.1, 0.2, 0.6, 9.0])
        estimate, low, high = bootstrap_ci(values, seed=0, statistic=np.median)
        # The point estimate is the median (0.4), not the mean (2.475).
        assert estimate == pytest.approx(0.4)
        assert low <= estimate <= high
        mean_estimate, _, _ = bootstrap_ci(values, seed=0)
        assert mean_estimate == pytest.approx(2.475)

    def test_respects_alpha(self):
        values = np.arange(20) / 20.0
        _, low95, high95 = bootstrap_ci(values, alpha=0.05, seed=0)
        _, low90, high90 = bootstrap_ci(values, alpha=0.10, seed=0)
        assert low90 >= low95 - 1e-12
        assert high90 <= high95 + 1e-12

    def test_deterministic_given_the_seed(self):
        values = np.array([0.1, 0.2, 0.3, 0.6])
        assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)

    def test_input_validation(self):
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([]))
        with pytest.raises(ValueError):
            bootstrap_ci(np.ones(3), n_resamples=0)
        with pytest.raises(ValueError):
            bootstrap_ci(np.ones(3), alpha=0.0)
