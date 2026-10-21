"""
Tests for the pilot analysis package: report registry, tier-based row
selection, and byte-identity with the pre-split single-script outputs.

Target: experiments/03_multistate_gdn2/01_synthetic_pilot/pilot_analysis/
Run with: pytest experiments/03_multistate_gdn2/01_synthetic_pilot/test_02_analyze.py -v

Every statistic must select its rows by tier: several tiers repeat core
configurations (the matched-state tier, the lambda controls, R1 and the
long-sequence tier), so to the per-seed DiD pairing those rows read as
duplicated cells and would drop seeds; calibration tiers and smoke runs
must never enter any statistic. The synthetic table below has rows from
EVERY tier.

The fixtures under ``fixtures/pre_split/`` hold the CSVs and summary the
single-script analysis produced on this same synthetic table before the
package split. The split reports must reproduce them byte for byte,
with one exception by construction: the single combined predictions
table is now one table per prediction report, and their concatenation
(in registry order) must reproduce the old file; the summary replaces
the single predictions section with the per-prediction sections and
adds the matched-bytes and mixer sections.
"""

import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from pilot_analysis import (
    REPORT_TITLES,
    REPORTS,
    SUMMARY_EXCLUDES,
    ReportOptions,
    render,
)
from pilot_analysis.protocol import PROTOCOL_TIERS
from pilot_analysis.records import load_runs
from pilot_analysis.reports import (
    all_cells,
    anchors,
    completeness,
    distance,
    frozen,
    gates,
    lambda_controls,
    long_t,
    matched_bytes,
    mixer,
    primary,
    robustness,
)

_SRC = Path(__file__).resolve().parents[3] / "src"

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

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pre_split"


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
        "layer_pattern": extra.get("layer_pattern", "zoology"),
        "state_mixer": extra.get("state_mixer", "none"),
        "lr": extra.get("lr"),
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


def synthetic_table() -> pd.DataFrame:
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


@pytest.fixture
def table() -> pd.DataFrame:
    return synthetic_table()


@pytest.fixture
def options() -> ReportOptions:
    return ReportOptions(n_resamples=N_RESAMPLES, alpha=ALPHA)


# The command-line entry point (loaded the same way test_01_train.py loads
# the training script; the analysis itself is imported from the package).
_CLI_SPEC = importlib.util.spec_from_file_location(
    "pilot_cli", Path(__file__).resolve().parent / "02_analyze.py"
)
cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(cli)


class TestTierSelection:
    """Rows are selected by tier, never by excluding tiers."""

    def test_primary_uses_exactly_the_ten_core_seeds(self, table, options):
        frame = primary.run(table, options.n_resamples, options.alpha)
        assert set(frame["layout"]) == {"ll", "ws"}
        for _, record in frame.iterrows():
            assert record["n_seeds"] == 10, record.to_dict()
        # Hand-computed means: Delta_int(ll) = 0.10, Delta_int(ws) = 0.05.
        ll = frame[frame["layout"] == "ll"].iloc[0]
        ws = frame[frame["layout"] == "ws"].iloc[0]
        assert ll["mean"] == pytest.approx(0.10, abs=1e-12)
        assert ws["mean"] == pytest.approx(0.05, abs=1e-12)

    def test_r1_reports_five_seeds_next_to_the_main_result(self, table, options):
        frame = robustness.run(table, options.n_resamples, options.alpha)
        assert set(frame["layout"]) == {"ll", "ws"}
        for _, record in frame.iterrows():
            assert record["n_seeds"] == 5, record.to_dict()
        # Hand-computed: Delta_int(ll) = (0.90 - 0.80) - 0.10 = 0.00,
        # Delta_int(ws) = (0.85 - 0.75) - 0.10 = 0.00.
        for _, record in frame.iterrows():
            assert record["mean"] == pytest.approx(0.0, abs=1e-12)

    def test_long_t_gets_its_own_table(self, table, options):
        frame = long_t.run(table, options.n_resamples, options.alpha)
        assert not frame.empty
        acc_rows = frame[frame["metric"] == "test/acc"]
        assert set(acc_rows["layout"]) == {"ll", "ws"}
        for _, record in acc_rows.iterrows():
            assert record["n_seeds"] == 10, record.to_dict()
        # Hand-computed: Delta_int(ll) = (0.60 - 0.40) - (0.40 - 0.30) = 0.10.
        ll = acc_rows[acc_rows["layout"] == "ll"].iloc[0]
        assert ll["mean"] == pytest.approx(0.10, abs=1e-12)

    def test_all_cells_uses_the_core_tier_only(self, table):
        frame = all_cells.run(table)
        # tier4 / tier5 / R1 repeat core configurations with sentinel values;
        # if they leaked in, their seeds (0-4) would drop as duplicates and
        # the seed count would fall below ten.
        distant = frame[frame["metric"] == PRIMARY]
        assert set(distant["layout"]) == {"ws", "ll"}
        for _, record in distant.iterrows():
            assert record["n_seeds"] == 10, record.to_dict()

    def test_predictions_pair_their_own_tiers_with_the_core(self, table, options):
        p1 = gates.run(table, options.n_resamples, options.alpha)
        p23 = frozen.run(table, options.n_resamples, options.alpha)
        p4 = lambda_controls.run(table, options.n_resamples, options.alpha)
        # P1 exists (the gate-decomposition tier joined to the core):
        # b_only/w_only recovered fractions, laid out per layout.
        assert not p1.empty
        assert set(p1["gate_mode"]) == {"b_only", "w_only"}
        assert set(p1["layout"]) == {"ws", "ll"}
        assert all(n.startswith("P1") for n in p1["prediction"])
        # P2/P3 exist (the frozen-control tier's fs rows joined to the
        # core tier's ws rows) with their own gates.
        assert not p23.empty
        assert all(n.startswith(("P2", "P3")) for n in p23["prediction"])
        assert set(p23["gate_mode"]) == {"full", "kda"}
        assert set(p23["layout"]) == {"ws-fs"}
        # P4 exists (the lambda-static rows joined to core single-full).
        assert not p4.empty
        assert all(n.startswith("P4") for n in p4["prediction"])
        assert set(p4["layout"]) == {"ws", "ll"}
        # Every prediction pairs by seed: tiers 2-4 have seeds 0-4 only.
        for frame in (p1, p23, p4):
            for _, record in frame.iterrows():
                assert record["n_seeds"] <= 5, record.to_dict()

    def test_calibration_tiers_never_enter_any_statistic(self, table, options):
        # Sentinel-valued calib rows at seeds 100/101 share the core
        # configurations; any leak would add seeds to the primary table.
        frame = primary.run(table, options.n_resamples, options.alpha)
        for _, record in frame.iterrows():
            assert record["n_seeds"] == 10
            assert record["ci_low"] <= record["mean"] <= record["ci_high"]
        r1 = robustness.run(table, options.n_resamples, options.alpha)
        for _, record in r1.iterrows():
            assert record["n_seeds"] == 5
        long_frame = long_t.run(table, options.n_resamples, options.alpha)
        for _, record in long_frame.iterrows():
            assert record["n_seeds"] == 10

    def test_distance_table_uses_core_tier_rows_only(self, table):
        frame = distance.run(table)
        # Per (layout, gate) x distance-metric column, averaged over the
        # core tier's ten seeds. Tiers 4/5 and R1 repeat the core
        # configurations with sentinel values; a leak would change the
        # means or the seed counts.
        assert not frame.empty
        acc_rows = frame[frame["metric"] == "test/acc/dist=32-64"]
        assert set(acc_rows["layout"]) == {"single", "ws", "ll"}
        assert set(acc_rows["gate_mode"]) == {"full", "kda"}
        for _, record in acc_rows.iterrows():
            assert record["n_seeds"] == 10, record.to_dict()
            assert record["mean"] == pytest.approx(
                CORE_ACC[record["layout"]][record["gate_mode"]]
            )

    def test_matched_bytes_pairs_tier5_with_the_core(self, table, options):
        frame = matched_bytes.run(table, options.n_resamples, options.alpha)
        # The matched row is single-state full with the expanded value
        # dim; the effect is (core layout full) - (matched single full),
        # paired on the five shared seeds.
        assert set(frame["layout"]) == {"ws", "ll"}
        for _, record in frame.iterrows():
            assert record["n_seeds"] == 5, record.to_dict()
        ws = frame[frame["layout"] == "ws"].iloc[0]
        # (0.60 - 0.99) hand-computed from the sentinels.
        assert ws["mean"] == pytest.approx(0.60 - SENTINEL, abs=1e-12)

    def test_matched_bytes_means_use_the_paired_seeds_only(self, options):
        # Core seeds 0-9 with seed-dependent values, matched seeds 0-4
        # only: both means of one row must describe the same seed set -
        # the paired seeds (0-4), never the whole core column.
        rows = [
            _row("tier1_core", "overwrite", "ws", "full", seed, 0.5 + 0.01 * seed)
            for seed in range(10)
        ]
        rows += [
            _row(
                "tier5_matched_bytes",
                "overwrite",
                "single",
                "full",
                seed,
                0.3 + 0.01 * seed,
            )
            for seed in range(5)
        ]
        frame = matched_bytes.run(
            pd.DataFrame(rows), options.n_resamples, options.alpha
        )
        assert len(frame) == 1
        record = frame.iloc[0]
        assert record["n_seeds"] == 5
        # Hand-computed over seeds 0-4: 0.52, 0.32, and their difference.
        assert record["mean_multistate"] == pytest.approx(0.52)
        assert record["mean_matched_single"] == pytest.approx(0.32)
        assert record["mean"] == pytest.approx(0.20, abs=1e-12)

    def test_matched_bytes_duplicated_cell_drops_that_seed(self, options, caplog):
        # A duplicated matched-bytes record for one seed: that seed
        # cannot pair, drops from the table, and the shared pairing
        # helper logs it (a silent drop would hide the record).
        rows = []
        for seed in (0, 1):
            rows.append(_row("tier1_core", "overwrite", "ws", "full", seed, 0.6))
            rows.append(
                _row("tier5_matched_bytes", "overwrite", "single", "full", seed, 0.3)
            )
        duplicate = dict(rows[-1])
        duplicate["run_id"] = "tier5-duplicate"
        rows.append(duplicate)
        with caplog.at_level(logging.WARNING, logger="research_lib.utils.stats"):
            frame = matched_bytes.run(
                pd.DataFrame(rows), options.n_resamples, options.alpha
            )
        assert len(frame) == 1
        record = frame.iloc[0]
        # Seed 0 dropped (its matched cell is duplicated); seed 1 pairs.
        assert record["n_seeds"] == 1
        assert record["mean"] == pytest.approx(0.3, abs=1e-12)
        assert any(
            "dropped units" in record_.getMessage() and "0" in record_.getMessage()
            for record_ in caplog.records
        )

    def test_anchors_table_lists_published_numbers(self, table, options):
        frame = anchors.run(table)
        assert set(frame["layout"]) == {"single", "ll"}
        assert set(frame["d_model"]) == {16, 32, 64}
        d32_single = frame[
            (frame["d_model"] == 32) & (frame["layout"] == "single")
        ].iloc[0]
        assert d32_single["log_linear_acc"] == pytest.approx(79.0)
        assert d32_single["log_linear_std"] == pytest.approx(2.1)
        d32_ll = frame[(frame["d_model"] == 32) & (frame["layout"] == "ll")].iloc[0]
        assert d32_ll["log_linear_acc"] == pytest.approx(84.4)


class TestSummary:
    """The summary concatenates the registry's sections in order."""

    def test_summary_contains_every_report_section(self, table, options):
        sections = [
            REPORTS[name](table, options).markdown
            for name in REPORTS
            if name not in SUMMARY_EXCLUDES
        ]
        summary = (
            render.SUMMARY_TITLE + "\n\n" + "".join(sections) + render.SUMMARY_NOTES
        )
        assert summary.startswith("# Synthetic pilot analysis\n\n")
        assert "## Primary endpoint" in summary
        assert "### Under uniform weight decay (tier R1)" in summary
        assert "## Long-sequence tier" in summary
        assert "## Write distance" in summary
        assert "## Incomplete runs" in summary
        assert "## Notes and limitations" in summary
        assert summary.endswith("never enter any statistic.\n")
        # The calibration report is never part of the summary.
        assert "## Calibration readings" not in summary

    def test_primary_empty_note_and_r1_section(self):
        empty = primary.render(pd.DataFrame())
        assert empty == (
            "## Primary endpoint (core tier, overwrite, distant, K = 2 d_k)\n\n"
            "_No core-tier runs with the primary endpoint present._\n"
        )
        r1_empty = robustness.render(pd.DataFrame())
        assert r1_empty.startswith("### Under uniform weight decay (tier R1)\n\n")
        assert "_No R1 runs present._" in r1_empty


class TestPreSplitOutputs:
    """The split reports reproduce the pre-split outputs byte for byte.

    The fixtures hold what the single-script analysis produced on the
    same synthetic table (bootstrap settings included) before the
    package split.
    """

    # report name -> CSV stems (the pre-split file names stay as they are).
    UNCHANGED = {
        "primary": ("primary_endpoint",),
        "robustness": ("r1_primary_endpoint",),
        "all_cells": ("did_all_cells",),
        "distance": ("write_distance",),
        "long_t": ("long_t_did",),
        "anchors": ("anchors",),
        "mixer": ("mixer_stats",),
        "completeness": ("missing_runs", "incomplete_runs"),
    }

    def test_unchanged_tables_match_the_pre_split_csvs(self, table, options):
        for name, stems in self.UNCHANGED.items():
            report = REPORTS[name](table, options)
            assert tuple(report.tables.keys()) == stems
            for table_stem, frame in report.tables.items():
                expected = (FIXTURES / f"{table_stem}.csv").read_text()
                assert (
                    frame.to_csv(index=False) == expected
                ), f"{name}/{table_stem} differs from the pre-split CSV"

    def test_split_predictions_concatenate_to_the_pre_split_file(self, table, options):
        frames = [
            gates.report(table, options).tables["predictions_gates"],
            frozen.report(table, options).tables["predictions_frozen"],
            lambda_controls.report(table, options).tables["predictions_lambda"],
        ]
        combined = pd.concat(frames, ignore_index=True)
        expected = (FIXTURES / "predictions.csv").read_text()
        assert combined.to_csv(index=False) == expected

    def test_summary_matches_the_pre_split_rendering(self, table, options):
        reports = {name: REPORTS[name](table, options) for name in REPORTS}
        new_summary = (
            render.SUMMARY_TITLE
            + "\n\n"
            + "".join(
                reports[name].markdown
                for name in REPORTS
                if name not in SUMMARY_EXCLUDES
            )
            + render.SUMMARY_NOTES
        )
        expected = (FIXTURES / "summary.md").read_text()
        assert new_summary == expected


class TestReportRegistry:
    """Every registered report runs on the synthetic table and returns
    a Report with unique table stems."""

    def test_every_report_runs_on_the_table(self, table, options):
        from pilot_analysis import Report

        stems = []
        for name, fn in REPORTS.items():
            report = fn(table, options)
            assert isinstance(report, Report)
            assert report.name == name
            assert report.markdown, name
            for stem in report.tables:
                assert "/" not in stem and ".." not in stem
                stems.append((name, stem))
        all_stems = [stem for _, stem in stems]
        assert len(all_stems) == len(set(all_stems)), stems

    def test_all_excludes_calibration(self):
        names, is_all = cli.expand_names(["all"])
        assert is_all
        assert names == [n for n in REPORTS if n not in SUMMARY_EXCLUDES]
        assert "calibration" not in names
        assert "calibration" in REPORTS

    def test_explicit_names_keep_order_and_deduplicate(self):
        names, is_all = cli.expand_names(["primary", "gates", "primary"])
        assert names == ["primary", "gates"]
        assert not is_all

    def test_unknown_name_is_an_error(self):
        with pytest.raises(SystemExit, match="unknown report"):
            cli.expand_names(["no_such_report"])

    def test_unknown_name_is_an_error_even_alongside_all(self):
        # `all` on the line must not swallow a mistyped name: every
        # requested name other than `all` is validated.
        with pytest.raises(SystemExit, match="unknown report"):
            cli.expand_names(["all", "no_such_report"])

    def test_all_plus_an_excluded_report_appends_it(self):
        # `all calibration` runs every report plus calibration, the
        # excluded report appended after the expansion, once.
        names, is_all = cli.expand_names(["all", "calibration"])
        assert is_all
        assert names == [n for n in REPORTS if n not in SUMMARY_EXCLUDES] + [
            "calibration"
        ]

    def test_all_plus_an_included_report_does_not_duplicate(self):
        names, is_all = cli.expand_names(["all", "primary"])
        assert is_all
        assert names == [n for n in REPORTS if n not in SUMMARY_EXCLUDES]

    def test_titles_cover_the_registry(self):
        assert set(REPORT_TITLES) == set(REPORTS)


class TestSweepTierConfigs:
    """Every sweep file's tier is a protocol tier, so a new tier can
    never silently skip the frozen-choice gate or the expected counts."""

    def test_every_sweep_tier_is_a_protocol_tier(self):
        from omegaconf import OmegaConf

        sweep_dir = Path(__file__).resolve().parent / "conf" / "sweep"
        sweep_files = sorted(sweep_dir.glob("*.yaml"))
        assert sweep_files  # sanity: we found the sweep files
        for path in sweep_files:
            cfg = OmegaConf.load(path)
            tier = OmegaConf.select(cfg, "experiment.tier")
            assert tier is not None, f"{path.name} has no experiment.tier"
            assert (
                str(tier) in PROTOCOL_TIERS
            ), f"{path.name} sets tier={tier!r}, which is not a protocol tier"


class TestIncompleteRunsNeverEnterTheStatistics:
    """Only complete runs (status == 'test_done') enter the statistics,
    like the missing-run report: a failed record that still carries test
    metrics (e.g. a test-only relaunch that then failed) must not."""

    def test_failed_core_row_excluded_from_primary(self, table, options):
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

        frame = primary.run(table, options.n_resamples, options.alpha)
        ll = frame[frame["layout"] == "ll"].iloc[0]
        # Seed 0's DiD is uncomputable (its ll/full cell is incomplete):
        # the seed is dropped, and the sentinel never enters the mean.
        assert ll["n_seeds"] == 9
        ws = frame[frame["layout"] == "ws"].iloc[0]
        assert ws["n_seeds"] == 10  # untouched cells keep every seed

        # The same row is listed under incomplete runs.
        incomplete = completeness.incomplete_runs(table)
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
                    "layer_pattern": "zoology",
                    "state_mixer": "none",
                },
                "task": {"name": "mqar"},
                "experiment": {"tier": "tier1_core", "phase": 1},
                "optimizer": {"lr": 0.003, "decay_scheme": "grouped"},
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
        table = load_runs(tmp_path)
        assert len(table) == 1
        assert bool(table.iloc[0]["early_stopped"]) is True
        assert table.iloc[0]["best_value"] == 0.9
        assert table.iloc[0]["lr"] == 0.003
        assert table.iloc[0]["layer_pattern"] == "zoology"

    def test_unreadable_json_skipped_with_warning(self, tmp_path, caplog):
        (tmp_path / "bad.json").write_text("{not json")
        with pytest.raises(SystemExit):
            load_runs(tmp_path)
        assert any("bad.json" in str(r.message) for r in caplog.records)


class TestMixerStatsCoreTier:
    """The mixer-statistics report selects the core tier's rows like
    every other statistic: calibration, smoke, R1, long-T and
    lambda-control runs must never mix in."""

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
        out = mixer.run(pd.DataFrame(rows))
        ws = out[(out["layout"] == "ws") & (out["gate_mode"] == "full")].iloc[0]
        assert ws["mixer/lambda_mean"] == 1.0  # not (1 + 1 + 9) / 3
        assert ws["n_runs"] == 2


class TestAllCellsSettings:
    """--alpha / --n-resamples must reach the all-cells table too."""

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
        wide = all_cells.run(table, 500, 0.5)
        narrow = all_cells.run(table, 500, 0.05)
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
        incomplete = completeness.incomplete_runs(table)
        assert len(incomplete) == 1
        assert set(incomplete["status"]) == {"interrupted"}
        assert incomplete.iloc[0]["run_id"] == table.iloc[0]["run_id"]
        missing = completeness.missing_runs(table)
        tier = table.iloc[0]["tier"]
        row = missing[missing["tier"] == tier].iloc[0]
        assert row["observed"] == 59
        assert row["missing"] == row["expected"] - 59

    def test_complete_fixture_has_no_incomplete_runs(self, table):
        assert completeness.incomplete_runs(table).empty


def _run_json(run_id, layout, gate, acc, seed=0, tier="tier1_core"):
    return {
        "run_id": run_id,
        "config": {
            "model": {
                "layout": layout,
                "gate_mode": gate,
                "lambda_mode": "mlp",
                "d_model": 32,
                "num_heads": 2,
                "head_dim": None,
                "layer_pattern": "zoology",
                "state_mixer": "none",
            },
            "task": {"name": "overwrite"},
            "experiment": {"tier": tier, "phase": 1},
            "optimizer": {"lr": 0.003, "decay_scheme": "grouped"},
        },
        "seed": seed,
        "status": "test_done",
        "stopping": {"best_value": acc, "best_step": 10},
        "test": {"acc": acc, "acc/slice=32/class=distant": acc},
    }


class TestCommandLine:
    """End-to-end: the CLI loads run JSONs, writes every report's
    outputs, and `all` never writes the calibration report."""

    @pytest.fixture
    def runs_dir(self, tmp_path) -> Path:
        runs = tmp_path / "runs"
        runs.mkdir()
        for layout in ("single", "ws"):
            for gate in ("full", "kda"):
                run_id = f"r-{layout}-{gate}"
                (runs / f"{run_id}.json").write_text(
                    json.dumps(_run_json(run_id, layout, gate, 0.5))
                )
        return runs

    @staticmethod
    def _run_cli(*args, cwd: Path) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "02_analyze.py", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def test_all_writes_summary_and_every_csv(self, runs_dir, tmp_path):
        out_dir = tmp_path / "analysis"
        result = self._run_cli(
            "all",
            "--runs-dir",
            str(runs_dir),
            "--out-dir",
            str(out_dir),
            cwd=Path(__file__).resolve().parent,
        )
        assert result.returncode == 0, result.stderr
        assert (out_dir / "summary.md").exists()
        assert (out_dir / "primary.md").exists()
        assert (out_dir / "primary_endpoint.csv").exists()
        # `all` never writes the calibration report.
        assert not (out_dir / "calibration.md").exists()
        # Every registered report's markdown exists.
        for name in REPORTS:
            if name in SUMMARY_EXCLUDES:
                continue
            assert (out_dir / f"{name}.md").exists(), name
        summary = (out_dir / "summary.md").read_text()
        assert "## Notes and limitations" in summary

    def test_calibration_report_runs_on_its_own(self, runs_dir, tmp_path):
        out_dir = tmp_path / "analysis"
        result = self._run_cli(
            "calibration",
            "--runs-dir",
            str(runs_dir),
            "--out-dir",
            str(out_dir),
            cwd=Path(__file__).resolve().parent,
        )
        assert result.returncode == 0, result.stderr
        assert (out_dir / "calibration.md").exists()
        assert (out_dir / "calibration_lr.csv").exists()
        assert not (out_dir / "summary.md").exists()

    def test_list_prints_every_report_name(self):
        result = self._run_cli("--list", cwd=Path(__file__).resolve().parent)
        assert result.returncode == 0, result.stderr
        for name in REPORTS:
            assert f"{name}: " in result.stdout

    def test_unknown_report_fails_loudly(self, runs_dir, tmp_path):
        result = self._run_cli(
            "no_such_report",
            "--runs-dir",
            str(runs_dir),
            cwd=Path(__file__).resolve().parent,
        )
        assert result.returncode != 0
        assert "unknown report" in result.stderr

    def test_unknown_report_fails_loudly_even_with_all(self, runs_dir, tmp_path):
        # `all no_such_report` must fail the same way: `all` never
        # hides a mistyped name.
        result = self._run_cli(
            "all",
            "no_such_report",
            "--runs-dir",
            str(runs_dir),
            cwd=Path(__file__).resolve().parent,
        )
        assert result.returncode != 0
        assert "unknown report" in result.stderr

    def test_all_plus_calibration_runs_calibration_outside_the_summary(
        self, runs_dir, tmp_path
    ):
        out_dir = tmp_path / "analysis"
        result = self._run_cli(
            "all",
            "calibration",
            "--runs-dir",
            str(runs_dir),
            "--out-dir",
            str(out_dir),
            cwd=Path(__file__).resolve().parent,
        )
        assert result.returncode == 0, result.stderr
        # Every report ran, calibration included: it wrote its own
        # markdown and tables.
        for name in REPORTS:
            assert (out_dir / f"{name}.md").exists(), name
        assert (out_dir / "calibration_lr.csv").exists()
        # The summary covers every report except calibration: its
        # section never enters it.
        summary = (out_dir / "summary.md").read_text()
        assert "## Calibration readings" not in summary
        for name in REPORTS:
            if name in SUMMARY_EXCLUDES:
                continue
            assert REPORT_TITLES[name] in summary, name
