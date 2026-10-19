"""
Tests for the calibration report: the pre-registered Phase-1 readings.

Target: experiments/03_multistate_gdn2/01_synthetic_pilot/pilot_analysis/
Run with: pytest experiments/03_multistate_gdn2/01_synthetic_pilot/test_calibration_report.py -v

Every reading uses the best validation accuracy during the run, averaged
over the selection seeds (100, 101); only completed runs count. The
width verdict picks the smallest d whose mean lands in the pre-registered
window (fallback: 32). The LR verdict picks the best mean, then the
fewer mean steps to the best value, then the smaller LR. The report
computes the readings; the frozen-protocol snippet mirrors them.
"""

import pandas as pd
import pytest
from omegaconf import OmegaConf
from pilot_analysis.protocol import CALIBRATION_TIERS, DMODEL_FALLBACK, SELECTION_SEEDS
from pilot_analysis.reports import calibration


def _calib_row(
    tier,
    seed,
    best_value,
    *,
    best_step=100,
    d_model=32,
    layout="single",
    gate_mode="full",
    task="mqar",
    layer_pattern="zoology",
    lr=0.003,
    status="test_done",
):
    return {
        "run_id": f"{tier}-{task}-{layout}-{gate_mode}-d{d_model}-lr{lr}-s{seed}",
        "task": task,
        "layout": layout,
        "gate_mode": gate_mode,
        "lambda_mode": "mlp",
        "d_model": d_model,
        "num_heads": 2,
        "head_dim": None,
        "layer_pattern": layer_pattern,
        "state_mixer": "none",
        "lr": lr,
        "seed": seed,
        "tier": tier,
        "status": status,
        "best_value": best_value,
        "best_step": best_step,
    }


def _dmodel_table(means, *, best_step=100):
    """A calib_dmodel run table: one mean per d_model over both seeds."""
    rows = []
    for d_model, values in means.items():
        for seed, value in zip(SELECTION_SEEDS, values, strict=True):
            rows.append(
                _calib_row(
                    "calib_dmodel", seed, value, d_model=d_model, best_step=best_step
                )
            )
    return pd.DataFrame(rows)


class TestDModelVerdict:
    """The width reading: smallest d in the window, else the fallback."""

    def test_smallest_in_range_wins(self):
        table = _dmodel_table({32: (0.6, 0.8), 64: (0.7, 0.9)})
        readings = calibration.run(table)
        assert readings["dmodel_choice"] == (
            32,
            "d_model=32: the smallest d whose mean accuracy lies in [0.5, 0.95]",
        )

    def test_only_the_larger_is_in_range(self):
        table = _dmodel_table({32: (0.99, 0.99), 64: (0.7, 0.9)})
        readings = calibration.run(table)
        assert readings["dmodel_choice"][0] == 64

    def test_none_in_range_falls_back(self):
        table = _dmodel_table({32: (0.99, 0.99), 64: (0.99, 0.99)})
        readings = calibration.run(table)
        d, reading = readings["dmodel_choice"]
        assert d == DMODEL_FALLBACK
        assert "fallback" in reading

    def test_no_records_leaves_the_choice_unset(self):
        readings = calibration.run(
            pd.DataFrame(_calib_row("tier1_core", 0, 0.5), index=[0])
        )
        assert readings["dmodel_choice"] == (None, None)
        snippet = OmegaConf.create(readings["snippet"])
        assert snippet.d_model is None

    def test_bounds_are_inclusive(self):
        table = _dmodel_table({32: (0.5, 0.5), 64: (0.95, 0.95)})
        readings = calibration.run(table)
        assert readings["dmodel_choice"][0] == 32


class TestReadingRowSelection:
    """Readings use completed selection-seed rows only."""

    def test_incomplete_and_nonselection_rows_never_enter(self):
        rows = _dmodel_table({32: (0.7, 0.8)}).to_dict("records")
        # A failed run with a poison value, and a seed-0 run: neither
        # may move the mean (0.7 + 0.8) / 2 = 0.75.
        rows.append(_calib_row("calib_dmodel", 100, 0.1, d_model=32, status="failed"))
        rows.append(_calib_row("calib_dmodel", 0, 0.1, d_model=32))
        table = pd.DataFrame(rows)
        readings = calibration.run(table)
        d32 = readings["dmodel"].set_index("d_model").loc[32]
        assert d32["mean"] == pytest.approx(0.75)
        assert d32["n_seeds"] == 2
        # The tier's observed count covers every completed record (the
        # seed-0 run is a completed record of the tier, so it counts
        # there), while the failed one does not; the reading above used
        # only the selection seeds.
        missing = readings["missing"].set_index("tier").loc["calib_dmodel"]
        assert missing["observed"] == 3
        assert missing["missing"] == missing["expected"] - 3

    def test_missing_table_lists_only_calibration_tiers(self):
        table = _dmodel_table({32: (0.7, 0.8)})
        readings = calibration.run(table)
        assert set(readings["missing"]["tier"]) == set(CALIBRATION_TIERS)


class TestHarnessReading:
    """The harness table lists the numbers, next to the published ones."""

    def test_harness_table_marks_no_verdict(self):
        rows = _dmodel_table({32: (0.7, 0.8)}).to_dict("records")
        for seed in SELECTION_SEEDS:
            for pattern, value in (("zoology", 0.85), ("mixer_x2", 0.55)):
                rows.append(
                    _calib_row(
                        "calib_harness",
                        seed,
                        value,
                        layer_pattern=pattern,
                        gate_mode="gdn",
                    )
                )
        table = pd.DataFrame(rows)
        readings = calibration.run(table)
        harness = readings["harness"].set_index("layer_pattern")
        # Per-seed values and the mean, next to the published number at
        # the width chosen by the width step (32 here).
        assert harness.loc["zoology", "mean"] == pytest.approx(0.85)
        assert harness.loc["zoology", "seed=100"] == pytest.approx(0.85)
        assert harness.loc["mixer_x2", "mean"] == pytest.approx(0.55)
        assert harness.loc["zoology", "log_linear_acc"] == pytest.approx(79.0)
        assert harness.loc["zoology", "log_linear_std"] == pytest.approx(2.1)
        # The markdown records that the choice is a judgment, not a
        # verdict.
        markdown = calibration.render(readings)
        assert "judgment" in markdown

    def test_harness_without_a_width_uses_its_own_d_model(self):
        rows = [
            _calib_row(
                "calib_harness",
                seed,
                0.85,
                layer_pattern="zoology",
                gate_mode="gdn",
                d_model=64,
            )
            for seed in SELECTION_SEEDS
        ]
        readings = calibration.run(pd.DataFrame(rows))
        harness = readings["harness"].set_index("layer_pattern")
        assert harness.loc["zoology", "log_linear_acc"] == pytest.approx(99.0)


class TestLRSelection:
    """The LR reading: best mean, then fewer steps, then smaller LR."""

    @staticmethod
    def _per_lr(mean_acc, mean_steps):
        return pd.DataFrame(
            {"mean_acc": mean_acc, "mean_steps": mean_steps},
            index=pd.Index([0.001, 0.003, 0.01], name="lr"),
        )

    def test_highest_mean_wins(self):
        per_lr = self._per_lr([0.5, 0.8, 0.7], [10, 20, 30])
        assert calibration.select_lr(per_lr) == 0.003

    def test_tie_goes_to_fewer_steps(self):
        per_lr = self._per_lr([0.8, 0.8, 0.7], [30, 20, 10])
        assert calibration.select_lr(per_lr) == 0.003

    def test_full_tie_goes_to_the_smaller_lr(self):
        per_lr = self._per_lr([0.8, 0.8, 0.8], [20, 20, 20])
        assert calibration.select_lr(per_lr) == 0.001

    def test_unmeasured_steps_do_not_block_selection(self):
        # A run that never recorded its best step (NaN): the LR that did
        # record one wins the step tie-break over it.
        per_lr = self._per_lr([0.8, 0.8, 0.7], [float("nan"), 40, 30])
        assert calibration.select_lr(per_lr) == 0.003

    def test_nothing_measured_gives_no_selection(self):
        per_lr = self._per_lr([None, None, None], [None, None, None])
        assert calibration.select_lr(per_lr) is None

    def test_selection_is_marked_in_the_table(self):
        rows = []
        for lr, accs, steps in (
            (0.001, (0.5, 0.52), (50, 54)),
            (0.003, (0.8, 0.82), (20, 24)),
            (0.01, (0.7, 0.72), (30, 34)),
        ):
            for seed, value, step in zip(SELECTION_SEEDS, accs, steps, strict=True):
                rows.append(
                    _calib_row(
                        "calib_lr",
                        seed,
                        value,
                        lr=lr,
                        best_step=step,
                        task="overwrite",
                        layout="ws",
                        gate_mode="full",
                    )
                )
        table = pd.DataFrame(rows)
        readings = calibration.run(table)
        lr_table = readings["lr"]
        assert list(lr_table["lr"]) == [0.001, 0.003, 0.01]
        selected = lr_table[lr_table["selected"]]
        assert len(selected) == 1
        assert selected.iloc[0]["lr"] == 0.003
        assert selected.iloc[0]["n_seeds"] == 2
        assert selected.iloc[0]["mean_best_acc"] == pytest.approx(0.81)
        assert selected.iloc[0]["mean_best_step"] == pytest.approx(22)

    def test_rows_without_lr_never_enter_a_selection(self):
        rows = [
            _calib_row(
                "calib_lr",
                seed,
                0.8,
                lr=0.003,
                best_step=10,
                task="overwrite",
                layout="ws",
                gate_mode="full",
            )
            for seed in SELECTION_SEEDS
        ]
        # A run with no LR recorded (inherited): no grid point.
        rows.append(
            _calib_row(
                "calib_lr",
                100,
                0.99,
                lr=None,
                best_step=5,
                task="overwrite",
                layout="ws",
                gate_mode="full",
            )
        )
        readings = calibration.run(pd.DataFrame(rows))
        lr_table = readings["lr"]
        assert len(lr_table) == 1
        assert lr_table.iloc[0]["lr"] == 0.003

    def test_anchor_lr_keyed_by_d_model_and_layout(self):
        rows = []
        for d_model, layout, lr, accs in (
            (16, "single", 0.001, (0.4, 0.42)),
            (16, "single", 0.003, (0.6, 0.62)),
            (16, "ll", 0.001, (0.5, 0.52)),
            (16, "ll", 0.003, (0.4, 0.44)),
        ):
            for seed, value in zip(SELECTION_SEEDS, accs, strict=True):
                rows.append(
                    _calib_row(
                        "calib_lr_anchors",
                        seed,
                        value,
                        lr=lr,
                        d_model=d_model,
                        layout=layout,
                        gate_mode="gdn",
                    )
                )
        readings = calibration.run(pd.DataFrame(rows))
        anchors = readings["lr_anchors"]
        selected = anchors[anchors["selected"]]
        assert {
            (int(r["d_model"]), r["layout"], r["lr"]) for _, r in selected.iterrows()
        } == {
            (16, "single", 0.003),
            (16, "ll", 0.001),
        }


class TestSnippet:
    """The snippet parses as YAML and mirrors the readings."""

    def _full_table(self):
        rows = _dmodel_table({32: (0.7, 0.8), 64: (0.99, 0.99)}).to_dict("records")
        for seed, value in zip(SELECTION_SEEDS, (0.85, 0.9), strict=True):
            rows.append(_calib_row("calib_harness", seed, value, gate_mode="gdn"))
        for layout in ("single", "ws", "ll"):
            for gate_mode in ("full", "kda"):
                for lr, accs in ((0.001, (0.4, 0.5)), (0.003, (0.8, 0.9))):
                    for seed, value in zip(SELECTION_SEEDS, accs, strict=True):
                        rows.append(
                            _calib_row(
                                "calib_lr",
                                seed,
                                value,
                                lr=lr,
                                task="overwrite",
                                layout=layout,
                                gate_mode=gate_mode,
                            )
                        )
        for d_model, layout, lr, accs in (
            (16, "single", 0.001, (0.5, 0.6)),
            (16, "single", 0.003, (0.7, 0.8)),
            (32, "ll", 0.001, (0.9, 0.95)),
        ):
            for seed, value in zip(SELECTION_SEEDS, accs, strict=True):
                rows.append(
                    _calib_row(
                        "calib_lr_anchors",
                        seed,
                        value,
                        lr=lr,
                        d_model=d_model,
                        layout=layout,
                        gate_mode="gdn",
                    )
                )
        return pd.DataFrame(rows)

    def test_snippet_parses_and_matches_the_readings(self):
        readings = calibration.run(self._full_table())
        snippet = OmegaConf.create(readings["snippet"])
        # The width reading (the smaller d is in range).
        assert snippet.d_model == 32
        # The harness block stays null: its reading is a judgment.
        assert snippet.harness.layer_pattern is None
        assert snippet.harness.state_mixer is None
        # The selected LRs land in the table, keyed by the frozen file's
        # structure; uncalibrated entries stay null.
        assert snippet.lr_table.overwrite.single.full == 0.003
        assert snippet.lr_table.overwrite.single.kda == 0.003
        assert snippet.lr_table.overwrite.ws.full == 0.003
        assert snippet.lr_table.mqar.ll.kda is None  # no mqar rows in the table
        assert snippet.lr_table.overwrite.single.b_only is None
        assert snippet.lr_table.overwrite.fs.full is None
        assert snippet.lr_table.overwrite.single.gdn is None
        # Anchor LRs, keyed by d_model and layout; unmeasured stay null.
        # (YAML integer keys parse as ints under OmegaConf.)
        assert snippet.anchor_lr[16].single == 0.003
        assert snippet.anchor_lr[16].ll is None
        assert snippet.anchor_lr[32].ll == 0.001
        assert snippet.anchor_lr[64].single is None

    def test_snippet_nulls_when_nothing_has_run(self):
        readings = calibration.run(
            pd.DataFrame(_calib_row("tier1_core", 0, 0.5), index=[0])
        )
        snippet = OmegaConf.create(readings["snippet"])
        assert snippet.d_model is None
        assert snippet.lr_table.mqar.single.full is None
        assert snippet.anchor_lr[64].ll is None

    def test_snippet_floats_are_stable(self):
        readings = calibration.run(self._full_table())
        assert "0.003" in readings["snippet"]
        assert "0.001" in readings["snippet"]
        assert "0.0010000" not in readings["snippet"]

    def test_report_markdown_embeds_the_snippet(self):
        from pilot_analysis import ReportOptions
        from pilot_analysis.reports import calibration as calib

        report = calib.report(self._full_table(), ReportOptions())
        assert report.markdown.startswith("## Calibration readings")
        assert "```yaml" in report.markdown
        assert set(report.tables) == {
            "calibration_dmodel",
            "calibration_harness",
            "calibration_lr",
            "calibration_lr_anchors",
        }
