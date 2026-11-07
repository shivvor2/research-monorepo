"""
Tests for the structured-recall pilot's reports on hand-made run tables:
the calibration LR selection and the single-state gap, the positive
control's known-gap numbers, and the completeness counts.

Target: experiments/03_multistate_gdn2/02_structured_recall_pilot/02_analyze.py
Run with: pytest experiments/03_multistate_gdn2/02_structured_recall_pilot/test_02_analyze.py -v
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

_PILOT = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "pilot_analyze", _PILOT / "02_analyze.py"
)
_analyze = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_analyze)


METRIC = "test/acc/slice=64"


def _row(
    run_id,
    task="mqar",
    layout="single",
    gate="full",
    lam="mlp",
    mixer="gated_delta",
    lr=0.003,
    seed=0,
    tier="calib_lr",
    best=None,
    step=None,
    status="test_done",
    **metric,
):
    row = {
        "run_id": run_id,
        "task": task,
        "layout": layout,
        "gate_mode": gate,
        "lambda_mode": lam,
        "d_model": 64,
        "mixer": mixer,
        "lr": lr,
        "seed": seed,
        "tier": tier,
        "status": status,
        "best_value": best,
        "best_step": step,
        "decay_scheme": "grouped",
    }
    for k, v in metric.items():
        row[k] = v
    return row


def _table(rows):
    return pd.DataFrame(rows)


class TestCalibration:
    def test_lr_selection_and_gap_at_the_selected_point(self, tmp_path):
        rows = []
        # gate full: lr 0.01 beats 0.003; screening cell 0.95 at 0.01.
        for seed in range(3):
            rows.append(
                _row(
                    f"f{seed}",
                    gate="full",
                    lr=0.003,
                    seed=seed,
                    best=0.8,
                    step=10,
                    **{METRIC: 0.9},
                )
            )
            rows.append(
                _row(
                    f"f{seed}b",
                    gate="full",
                    lr=0.01,
                    seed=seed,
                    best=0.9,
                    step=30,
                    **{METRIC: 0.95},
                )
            )
        # gate kda: only 0.01; cell 0.5.
        for seed in range(3):
            rows.append(
                _row(
                    f"k{seed}",
                    gate="kda",
                    lr=0.01,
                    seed=seed,
                    best=0.7,
                    step=25,
                    **{METRIC: 0.5},
                )
            )
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        single = df[(df["kind"] == "single") & (df["lambda_mode"] == "mlp")]
        full = single[single["gate_mode"] == "full"].iloc[0]
        kda = single[single["gate_mode"] == "kda"].iloc[0]
        assert full["selected_lr"] == pytest.approx(0.01)
        assert kda["selected_lr"] == pytest.approx(0.01)
        assert full["cell"] == pytest.approx(0.95)
        assert kda["cell"] == pytest.approx(0.5)
        assert full["gap_full_kda"] == pytest.approx(0.45)
        assert kda["gap_full_kda"] == pytest.approx(0.45)

    def test_lr_tie_is_broken_by_fewer_steps(self, tmp_path):
        # Both LRs average the same best value; 0.003 reaches it in fewer
        # steps, so it is selected.
        rows = [
            _row(
                "a", gate="full", lr=0.003, seed=0, best=0.8, step=10, **{METRIC: 0.9}
            ),
            _row(
                "b", gate="full", lr=0.003, seed=1, best=0.8, step=20, **{METRIC: 0.9}
            ),
            _row("c", gate="full", lr=0.01, seed=0, best=0.8, step=30, **{METRIC: 0.9}),
            _row("d", gate="full", lr=0.01, seed=1, best=0.8, step=40, **{METRIC: 0.9}),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")].iloc[0]
        assert full["selected_lr"] == pytest.approx(0.003)

    def test_attention_gets_its_own_row(self, tmp_path):
        rows = [
            _row(
                "att0",
                task="mqar",
                layout="ws",
                mixer="attention",
                lr=0.003,
                seed=0,
                best=0.99,
                step=5,
                tier="calib_ws",
                **{METRIC: 0.99},
            ),
            _row(
                "att1",
                task="mqar",
                layout="ws",
                mixer="attention",
                lr=0.003,
                seed=1,
                best=0.98,
                step=6,
                tier="calib_ws",
                **{METRIC: 0.98},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        att = df[df["kind"] == "attention"].iloc[0]
        assert att["layout"] == "ws"
        assert att["mixer"] == "attention"
        assert att["selected_lr"] == pytest.approx(0.003)
        assert att["mean_best_value"] == pytest.approx(0.985)


class TestPositiveControl:
    def test_known_gap_over_seeds(self, tmp_path):
        rows = [
            _row(
                "p0",
                gate="full",
                seed=0,
                tier="tier1_core",
                best=0.6,
                step=5,
                **{METRIC: 0.6},
            ),
            _row(
                "p1",
                gate="full",
                seed=1,
                tier="tier1_core",
                best=0.7,
                step=5,
                **{METRIC: 0.7},
            ),
            _row(
                "p2",
                gate="kda",
                seed=0,
                tier="tier1_core",
                best=0.4,
                step=5,
                **{METRIC: 0.4},
            ),
            _row(
                "p3",
                gate="kda",
                seed=1,
                tier="tier1_core",
                best=0.5,
                step=5,
                **{METRIC: 0.5},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.positive_control(_table(rows), out)
        df = pd.read_csv(out / "positive_control.csv")
        assert df["n_seeds"].iloc[0] == 2
        assert df["mean_gap"].iloc[0] == pytest.approx(0.2)
        assert df["mean_high"].iloc[0] == pytest.approx(0.65)
        assert df["mean_low"].iloc[0] == pytest.approx(0.45)

    def test_duplicate_seed_rows_raise_naming_runs(self, tmp_path):
        rows = [
            _row(
                "p0",
                gate="full",
                seed=0,
                tier="tier1_core",
                best=0.6,
                step=5,
                **{METRIC: 0.6},
            ),
            _row(
                "p0dup",
                gate="full",
                seed=0,
                tier="tier1_core",
                best=0.66,
                step=5,
                **{METRIC: 0.66},
            ),
            _row(
                "p1",
                gate="kda",
                seed=0,
                tier="tier1_core",
                best=0.4,
                step=5,
                **{METRIC: 0.4},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="p0"):
            _analyze.positive_control(_table(rows), out)


class TestCompleteness:
    def test_expected_and_unexpected_tiers(self, tmp_path, monkeypatch):
        # expected tier1_core -> 5 done runs; only 3 present -> missing 2.
        monkeypatch.setattr(_analyze._protocol, "EXPECTED_RUNS", {"tier1_core": 5})
        rows = [
            _row("a0", tier="tier1_core", best=0.6),
            _row("a1", tier="tier1_core", best=0.6),
            _row("a2", tier="tier1_core", best=0.6),
            _row("a3", tier="tier1_core", status="interrupted", best=None),
            _row("c0", tier="calib_lr", best=0.6),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.completeness(_table(rows), out)
        df = pd.read_csv(out / "completeness.csv")
        core = df[df["tier"] == "tier1_core"].iloc[0]
        assert core["expected"] == 5
        assert core["observed"] == 3  # the interrupted one does not count
        assert core["missing"] == 2
        assert bool(core["known"]) is True
        calib = df[df["tier"] == "calib_lr"].iloc[0]
        assert calib["expected"] == 0  # present with runs, no expected count
        assert bool(calib["known"]) is False


class TestCalibrationDecay:
    def test_two_decay_schemes_give_two_rows_and_one_snippet_entry(self, tmp_path):
        # One cell run under two decay schemes: two rows, each with its
        # own selected LR; the frozen snippet holds a single entry (the
        # harness's grouped decay only).
        rows = [
            _row(
                "g0", gate="full", lr=0.003, seed=0, best=0.8, step=10, **{METRIC: 0.9}
            ),
            _row(
                "g1", gate="full", lr=0.003, seed=1, best=0.8, step=10, **{METRIC: 0.9}
            ),
            _row(
                "u0", gate="full", lr=0.01, seed=0, best=0.9, step=10, **{METRIC: 0.95}
            ),
            _row(
                "u1", gate="full", lr=0.01, seed=1, best=0.9, step=10, **{METRIC: 0.95}
            ),
        ]
        rows[2]["decay_scheme"] = "uniform"
        rows[3]["decay_scheme"] = "uniform"
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        gd = df[(df["kind"] == "single") & (df["decay_scheme"] == "grouped")].iloc[0]
        un = df[(df["kind"] == "single") & (df["decay_scheme"] == "uniform")].iloc[0]
        assert gd["selected_lr"] == pytest.approx(0.003)
        assert un["selected_lr"] == pytest.approx(0.01)
        # The snippet holds one LR for the cell.
        snippet = (out / "frozen_protocol_snippet.yaml").read_text()
        assert snippet.count("full: 0.003") == 1
        assert "full: 0.01" not in snippet

    def test_n_seeds_is_the_selected_lrs_count(self, tmp_path):
        # The best LR (0.01) is measured on one fewer seed than 0.003.
        rows = [
            _row(
                "g0", gate="full", lr=0.003, seed=0, best=0.8, step=10, **{METRIC: 0.9}
            ),
            _row(
                "g1", gate="full", lr=0.003, seed=1, best=0.8, step=10, **{METRIC: 0.9}
            ),
            _row(
                "u0", gate="full", lr=0.01, seed=0, best=0.9, step=10, **{METRIC: 0.95}
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")].iloc[0]
        assert full["selected_lr"] == pytest.approx(0.01)
        assert full["n_seeds"] == 1
