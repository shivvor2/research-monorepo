"""
Tests for the pilot analysis: tier-based row selection.

Target: experiments/03_multistate_gdn2/01_synthetic_pilot/02_analyze.py
Run with: pytest experiments/03_multistate_gdn2/01_synthetic_pilot/test_02_analyze.py -v

Every statistic must select its rows by tier: several tiers repeat core
configurations (the matched-state tier, the lambda controls, R1 and the
long-sequence tier), so to the per-seed DiD pairing those rows read as
duplicated cells and would drop seeds; calibration tiers and smoke runs
must never enter any statistic. The synthetic table below has rows from
EVERY tier.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SPEC = importlib.util.spec_from_file_location(
    "pilot_analyze", Path(__file__).resolve().parent / "02_analyze.py"
)
analyze = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze)

N_RESAMPLES = 200
ALPHA = 0.05

# Core-tier accuracy constants (no seed dependence, so the per-seed DiD is
# hand-computable): Delta_int(ll) = (0.70 - 0.50) - (0.50 - 0.40) = 0.10,
# Delta_int(ws) = (0.60 - 0.45) - 0.10 = 0.05.
CORE_ACC = {
    "single": {"full": 0.50, "kda": 0.40},
    "ws": {"full": 0.60, "kda": 0.45},
    "ll": {"full": 0.70, "kda": 0.50},
}
# R1 runs the core grid under uniform decay with its own constants.
R1_ACC = {
    "single": {"full": 0.80, "kda": 0.70},
    "ws": {"full": 0.85, "kda": 0.75},
    "ll": {"full": 0.90, "kda": 0.80},
}
# Long-T runs the core grid on the long MQAR variant.
LONG_ACC = {
    "single": {"full": 0.40, "kda": 0.30},
    "ws": {"full": 0.50, "kda": 0.35},
    "ll": {"full": 0.60, "kda": 0.40},
}
# Every other tier gets sentinel values: if any of these rows leaked into a
# statistic, the seed count or the mean would change.
SENTINEL = 0.99

PRIMARY = "test/acc/slice=32/class=distant"
SINGLE_CLASS = "test/acc/slice=32/class=single"
DISTANT_LEAK = "test/leak/slice=32/class=distant"


def _row(tier, task, layout, gate, seed, acc, **extra):
    row = {
        "run_id": f"{tier}-{task}-{layout}-{gate}-s{seed}",
        "json_path": f"runs/{task}/{tier}-{task}-{layout}-{gate}-s{seed}.json",
        "task": task,
        "layout": layout,
        "gate_mode": gate,
        "lambda_mode": extra.get("lambda_mode", "mlp"),
        "d_model": extra.get("d_model", 32),
        "num_heads": extra.get("num_heads", 2),
        "head_dim": extra.get("head_dim"),
        "seed": seed,
        "tier": tier,
        "decay_scheme": extra.get("decay_scheme", "grouped"),
        "steps_completed": 100,
        "best_value": acc,
        "best_step": 100,
        "status": "test_done",
        "test/acc": acc,
        "test/acc/slice=32": acc,
        PRIMARY: acc,
        SINGLE_CLASS: acc,
        DISTANT_LEAK: 0.0,
        "test/acc/dist=32-64": acc,
        "test/leak/dist=32-64": 0.0,
    }
    return row


@pytest.fixture
def table() -> pd.DataFrame:
    """A synthetic run table containing rows from EVERY tier."""
    rows = []

    def add(tier, task, layouts, gates, seeds, accs, **extra):
        for seed in seeds:
            for layout in layouts:
                for gate in gates:
                    rows.append(
                        _row(
                            tier, task, layout, gate, seed, accs[layout][gate], **extra
                        )
                    )

    # Core tier: seeds 0-9 (the primary statistic's ten seeds).
    add(
        "tier1_core",
        "overwrite",
        ("single", "ws", "ll"),
        ("full", "kda"),
        range(10),
        CORE_ACC,
    )
    # Gate decomposition, frozen control, lambda controls, matched bytes:
    # seeds 0-4, sentinel values, same configurations as the core tier.
    add(
        "tier2_gates",
        "overwrite",
        ("single", "ws", "ll"),
        ("b_only", "w_only"),
        range(5),
        dict.fromkeys(CORE_ACC, {"b_only": SENTINEL, "w_only": SENTINEL}),
    )
    add(
        "tier3_frozen",
        "overwrite",
        ("fs",),
        ("full", "kda"),
        range(5),
        {"fs": {"full": SENTINEL, "kda": SENTINEL}},
    )
    add(
        "tier4_lambda",
        "overwrite",
        ("ws", "ll"),
        ("full",),
        range(5),
        dict.fromkeys(("ws", "ll"), {"full": SENTINEL}),
        lambda_mode="static",
    )
    add(
        "tier4_lambda",
        "overwrite",
        ("ws", "ll"),
        ("full",),
        range(5),
        dict.fromkeys(("ws", "ll"), {"full": SENTINEL}),
        lambda_mode="lin",
    )
    add(
        "tier5_matched_bytes",
        "overwrite",
        ("single",),
        ("full",),
        range(5),
        {"single": {"full": SENTINEL}},
    )
    # R1: the core grid under uniform decay, seeds 0-4.
    add(
        "R1",
        "overwrite",
        ("single", "ws", "ll"),
        ("full", "kda"),
        range(5),
        R1_ACC,
        decay_scheme="uniform",
    )
    # Long-sequence tier: the core grid on the long MQAR variant, seeds 0-9.
    add(
        "tier_longT",
        "mqar",
        ("single", "ws", "ll"),
        ("full", "kda"),
        range(10),
        LONG_ACC,
    )
    # Anchors: gdn x {single, ll} x d in {16, 32, 64}, seeds 0-4.
    for d_model, num_heads in ((16, 1), (32, 2), (64, 2)):
        add(
            "tier6_anchors",
            "mqar",
            ("single", "ll"),
            ("gdn",),
            range(5),
            dict.fromkeys(("single", "ll"), {"gdn": SENTINEL}),
            d_model=d_model,
            num_heads=num_heads,
        )
    # Calibration tiers: seeds 100-101, core configurations with sentinel
    # values (a leak adds seeds 100/101 to any statistic).
    for tier in ("calib_dmodel", "calib_harness", "calib_lr", "calib_lr_anchors"):
        add(
            tier,
            "overwrite",
            ("single", "ws", "ll"),
            ("full", "kda"),
            (100, 101),
            dict.fromkeys(CORE_ACC, {"full": SENTINEL, "kda": SENTINEL}),
        )
        add(
            tier,
            "mqar",
            ("single", "ws", "ll"),
            ("full", "kda"),
            (100, 101),
            dict.fromkeys(CORE_ACC, {"full": SENTINEL, "kda": SENTINEL}),
        )
    # A smoke run (tier None) must never enter a statistic either.
    rows.append(_row(None, "overwrite", "ws", "full", 0, SENTINEL))
    return pd.DataFrame(rows)


class TestTierSelection:
    """Rows are selected by tier, never by excluding tiers."""

    def test_primary_uses_exactly_the_ten_core_seeds(self, table):
        primary = analyze.analyze_primary(table, N_RESAMPLES, ALPHA)
        assert set(primary["layout"]) == {"ll", "ws"}
        for _, record in primary.iterrows():
            assert record["n_seeds"] == 10, record.to_dict()
        # Hand-computed means: Delta_int(ll) = 0.10, Delta_int(ws) = 0.05.
        ll = primary[primary["layout"] == "ll"].iloc[0]
        ws = primary[primary["layout"] == "ws"].iloc[0]
        assert ll["mean"] == pytest.approx(0.10, abs=1e-12)
        assert ws["mean"] == pytest.approx(0.05, abs=1e-12)

    def test_r1_reports_five_seeds_next_to_the_main_result(self, table):
        r1 = analyze.analyze_r1(table, N_RESAMPLES, ALPHA)
        assert set(r1["layout"]) == {"ll", "ws"}
        for _, record in r1.iterrows():
            assert record["n_seeds"] == 5, record.to_dict()
        # Hand-computed: Delta_int(ll) = (0.90 - 0.80) - 0.10 = 0.00,
        # Delta_int(ws) = (0.85 - 0.75) - 0.10 = 0.00.
        for _, record in r1.iterrows():
            assert record["mean"] == pytest.approx(0.0, abs=1e-12)

    def test_long_t_gets_its_own_table(self, table):
        long_t = analyze.analyze_long_t(table, N_RESAMPLES, ALPHA)
        assert not long_t.empty
        acc_rows = long_t[long_t["metric"] == "test/acc"]
        assert set(acc_rows["layout"]) == {"ll", "ws"}
        for _, record in acc_rows.iterrows():
            assert record["n_seeds"] == 10, record.to_dict()
        # Hand-computed: Delta_int(ll) = (0.60 - 0.40) - (0.40 - 0.30) = 0.10.
        ll = acc_rows[acc_rows["layout"] == "ll"].iloc[0]
        assert ll["mean"] == pytest.approx(0.10, abs=1e-12)

    def test_all_cells_uses_the_core_tier_only(self, table):
        cells = analyze.analyze_all_cells(table)
        # tier4 / tier5 / R1 repeat core configurations with sentinel values;
        # if they leaked in, their seeds (0-4) would drop as duplicates and
        # the seed count would fall below ten.
        distant = cells[cells["metric"] == PRIMARY]
        assert set(distant["layout"]) == {"ws", "ll"}
        for _, record in distant.iterrows():
            assert record["n_seeds"] == 10, record.to_dict()

    def test_predictions_pair_their_own_tiers_with_the_core(self, table):
        predictions = analyze.analyze_predictions(table, N_RESAMPLES, ALPHA)
        names = set(predictions["prediction"])
        gates = set(predictions["gate_mode"])
        # P1 exists (tier 2 joined to core): b_only/w_only recovered
        # fractions, laid out per layout.
        p1 = predictions[predictions["prediction"].str.startswith("P1")]
        assert not p1.empty
        assert set(p1["gate_mode"]) == {"b_only", "w_only"}
        assert set(p1["layout"]) == {"ws", "ll"}
        # P2/P3 exist (tier 3 fs joined to core ws), P4 exists (tier 4
        # static joined to core single-full).
        assert any(n.startswith("P2") for n in names)
        assert any(n.startswith("P3") for n in names)
        assert any(n.startswith("P4") for n in names)
        # Gates never mix: the predictions carry their own tiers' gates.
        assert gates <= {"b_only", "w_only", "full", "kda"}
        # Every prediction pairs by seed: tier 2-4 have seeds 0-4 only.
        for _, record in predictions.iterrows():
            assert record["n_seeds"] <= 5, record.to_dict()

    def test_calibration_tiers_never_enter_any_statistic(self, table):
        # Sentinel-valued calib rows at seeds 100/101 share the core
        # configurations; any leak would add seeds to the primary table.
        primary = analyze.analyze_primary(table, N_RESAMPLES, ALPHA)
        for _, record in primary.iterrows():
            assert record["n_seeds"] == 10
            assert record["ci_low"] <= record["mean"] <= record["ci_high"]
        r1 = analyze.analyze_r1(table, N_RESAMPLES, ALPHA)
        for _, record in r1.iterrows():
            assert record["n_seeds"] == 5
        long_t = analyze.analyze_long_t(table, N_RESAMPLES, ALPHA)
        for _, record in long_t.iterrows():
            assert record["n_seeds"] == 10


class TestRenderSummary:
    """summary.md renders the R1 table next to the main result."""

    def test_summary_contains_r1_and_long_t_sections(self, table):
        primary = analyze.analyze_primary(table, N_RESAMPLES, ALPHA)
        r1 = analyze.analyze_r1(table, N_RESAMPLES, ALPHA)
        long_t = analyze.analyze_long_t(table, N_RESAMPLES, ALPHA)
        cells = analyze.analyze_all_cells(table)
        predictions = analyze.analyze_predictions(table, N_RESAMPLES, ALPHA)
        distance = analyze.analyze_distance(table)
        anchors = analyze.analyze_anchors(table)
        missing = analyze.analyze_missing_runs(table)
        incomplete = analyze.analyze_incomplete(table)
        summary = analyze.render_summary(
            primary,
            r1,
            long_t,
            cells,
            predictions,
            distance,
            anchors,
            missing,
            incomplete,
        )
        assert "## Primary endpoint" in summary
        assert "### Under uniform weight decay (tier R1)" in summary
        assert "## Long-sequence tier" in summary
        assert "## Write distance" in summary
        assert "## Incomplete runs" in summary
        # The R1 table carries n_seeds, mean and CI columns.
        assert "n_seeds" in summary
        assert "ci_low" in summary

    def test_distance_table_uses_core_tier_rows_only(self, table):
        distance = analyze.analyze_distance(table)
        # Per (layout, gate) x distance-metric column, averaged over the
        # core tier's ten seeds. Tiers 4/5 and R1 repeat the core
        # configurations with sentinel values; a leak would change the
        # means or the seed counts.
        assert not distance.empty
        acc_rows = distance[distance["metric"] == "test/acc/dist=32-64"]
        assert set(acc_rows["layout"]) == {"single", "ws", "ll"}
        assert set(acc_rows["gate_mode"]) == {"full", "kda"}
        for _, record in acc_rows.iterrows():
            assert record["n_seeds"] == 10, record.to_dict()
            assert record["mean"] == pytest.approx(
                CORE_ACC[record["layout"]][record["gate_mode"]]
            )


class TestIncompleteRunsNeverEnterTheStatistics:
    """Only complete runs (status == 'test_done') enter the statistics,
    like the missing-run report: a failed record that still carries test
    metrics (e.g. a test-only relaunch that then failed) must not."""

    def test_failed_core_row_excluded_from_primary(self, table):
        table = table.copy()
        mask = (
            (table["tier"] == "tier1_core")
            & (table["seed"] == 0)
            & (table["layout"] == "ll")
            & (table["gate_mode"] == "full")
        )
        assert mask.sum() == 1  # one run to corrupt
        table.loc[mask, "status"] = "failed"
        table.loc[mask, PRIMARY] = 9.99  # would poison the ll/full cell

        primary = analyze.analyze_primary(table, N_RESAMPLES, ALPHA)
        ll = primary[primary["layout"] == "ll"].iloc[0]
        # Seed 0's DiD is uncomputable (its ll/full cell is incomplete):
        # the seed is dropped, and the sentinel never enters the mean.
        assert ll["n_seeds"] == 9
        ws = primary[primary["layout"] == "ws"].iloc[0]
        assert ws["n_seeds"] == 10  # untouched cells keep every seed

        # The same row is listed under incomplete runs.
        incomplete = analyze.analyze_incomplete(table)
        assert len(incomplete) == 1
        assert incomplete.iloc[0]["status"] == "failed"


class TestLoadRuns:
    """The loader flattens the run JSON (one row per run file)."""

    def test_early_stopped_loaded_from_stopping(self, tmp_path):
        run = {
            "run_id": "r0",
            "config": {
                "model": {
                    "layout": "ws",
                    "gate_mode": "full",
                    "lambda_mode": "mlp",
                    "d_model": 32,
                    "num_heads": 2,
                    "head_dim": None,
                },
                "task": {"name": "mqar"},
                "experiment": {"tier": "tier1_core", "phase": 1},
                "optimizer": {"decay_scheme": "grouped"},
            },
            "seed": 0,
            "timing": {"steps_completed": 10},
            "stopping": {
                "early_stopped": True,
                "best_value": 0.9,
                "best_step": 10,
            },
            "status": "test_done",
            "test": {"acc": 0.5},
        }
        (tmp_path / "r0.json").write_text(json.dumps(run))
        table = analyze.load_runs(tmp_path)
        assert len(table) == 1
        assert bool(table.iloc[0]["early_stopped"]) is True
        assert table.iloc[0]["best_value"] == 0.9


class TestMixerStatsCoreTier:
    """analyze_mixer_stats selects the core tier's rows like every other
    statistic: calibration, smoke, R1, long-T and lambda-control runs
    must never mix in."""

    def test_non_core_rows_never_mix_in(self):
        rows = []
        for layout in ("ws", "ll"):
            for seed in range(2):
                rows.append(
                    {
                        "tier": "tier1_core",
                        "layout": layout,
                        "gate_mode": "full",
                        "run_id": f"c-{layout}-{seed}",
                        "mixer/lambda_mean": 1.0,
                    }
                )
        # A non-core row with sentinel mixer values: never mixed in.
        rows.append(
            {
                "tier": "R1",
                "layout": "ws",
                "gate_mode": "full",
                "run_id": "r1-sentinel",
                "mixer/lambda_mean": 9.0,
            }
        )
        out = analyze.analyze_mixer_stats(pd.DataFrame(rows))
        ws = out[(out["layout"] == "ws") & (out["gate_mode"] == "full")].iloc[0]
        assert ws["mixer/lambda_mean"] == 1.0  # not (1 + 1 + 9) / 3
        assert ws["n_runs"] == 2


class TestAllCellsSettings:
    """--alpha / --n-resamples must reach did_all_cells.csv too."""

    def test_alpha_narrows_the_all_cells_intervals(self):
        # 4 seeds x (ws | single) x (full | kda): the ws DiD has 4 deltas.
        acc = {"ws": {"full": 0.60, "kda": 0.45}, "single": {"full": 0.50, "kda": 0.40}}
        rows = []
        for seed in range(4):
            for layout in ("ws", "single"):
                for gate in ("full", "kda"):
                    rows.append(
                        {
                            "tier": "tier1_core",
                            "task": "mqar",
                            "layout": layout,
                            "gate_mode": gate,
                            "seed": seed,
                            "run_id": f"r-{layout}-{gate}-{seed}",
                            "test/acc": acc[layout][gate] + 0.01 * seed,
                        }
                    )
        table = pd.DataFrame(rows)
        # bootstrap_ci seeds its RNG, so the two calls are deterministic.
        wide = analyze.analyze_all_cells(table, 500, 0.5)
        narrow = analyze.analyze_all_cells(table, 500, 0.05)
        w = wide[wide["layout"] == "ws"].iloc[0]
        n = narrow[narrow["layout"] == "ws"].iloc[0]
        assert (w["ci_high"] - w["ci_low"]) < (n["ci_high"] - n["ci_low"])


class TestIncompleteRuns:
    """The missing-run report separates incomplete runs by status."""

    def test_incomplete_runs_listed_with_status(self, table):
        # One core-tier run is "interrupted": it is listed with its
        # status, and the tier's complete count drops by one (59 of 60).
        table = table.copy()
        table.loc[table.index[0], "status"] = "interrupted"
        incomplete = analyze.analyze_incomplete(table)
        assert len(incomplete) == 1
        assert set(incomplete["status"]) == {"interrupted"}
        assert incomplete.iloc[0]["run_id"] == table.iloc[0]["run_id"]
        missing = analyze.analyze_missing_runs(table)
        tier = table.iloc[0]["tier"]
        row = missing[missing["tier"] == tier].iloc[0]
        assert row["observed"] == 59
        assert row["missing"] == row["expected"] - 59

    def test_complete_fixture_has_no_incomplete_runs(self, table):
        assert analyze.analyze_incomplete(table).empty
