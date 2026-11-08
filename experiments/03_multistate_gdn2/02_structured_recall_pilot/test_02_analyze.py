"""
Tests for the structured-recall pilot's reports on hand-made run tables:
the calibration LR selection and the single-state gap, the positive
control's known-gap numbers, the completeness counts, the frozen snippet
round-trip and the empty-folder behaviour.

The tests register a toy task (``protocol`` fixture) so they don't depend
on real tier names: MQAR is smoke and debug only and registers no tiers.
Calibration rows use the shared selection seeds (100, 101); the positive
control reads tier-1 seeds with no seed filter.

Target: experiments/03_multistate_gdn2/02_structured_recall_pilot/02_analyze.py
Run with: pytest experiments/03_multistate_gdn2/02_structured_recall_pilot/test_02_analyze.py -v
"""

import importlib.util
import sys
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

_spec1 = importlib.util.spec_from_file_location("pilot_01", _PILOT / "01_train.py")
_01 = importlib.util.module_from_spec(_spec1)
assert _spec1 is not None and _spec1.loader is not None
_spec1.loader.exec_module(_01)
_frozen_lr = _01._frozen_lr

METRIC = "test/acc/slice=64"
PRIMARY = "test/acc/slice=64/class=sibling"
FALLBACK = "test/acc/slice=32/class=sibling"

TOY_CALIBRATION = ("calib_lr", "calib_ws")
TOY_CONTROL = "tier1_core"


def _row(
    run_id,
    task="toy",
    layout="single",
    gate="full",
    lam="mlp",
    mixer="gated_delta",
    lr=0.003,
    seed=100,
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


def _lr_cfg(layout="ws", gate="full", mixer="gated_delta"):
    return OmegaConf.create(
        {
            "task": {"name": "toy"},
            "model": {"mixer": mixer, "layout": layout, "gate_mode": gate},
        }
    )


@pytest.fixture
def protocol(monkeypatch):
    """Register a toy task so calibration / positive-control tests don't
    depend on real tier names (MQAR is smoke/debug only). The selection
    seeds stay the shared (100, 101) re-export from ``common.protocol``.
    """
    proto = _analyze._protocol
    monkeypatch.setattr(
        proto,
        "TASK_TIERS",
        {
            "toy": proto.TaskTiers(
                calibration=TOY_CALIBRATION, positive_control=TOY_CONTROL
            )
        },
    )
    monkeypatch.setattr(proto, "CALIBRATION_TIERS", frozenset(TOY_CALIBRATION))
    monkeypatch.setattr(
        proto, "PROTOCOL_TIERS", frozenset((*TOY_CALIBRATION, TOY_CONTROL))
    )
    monkeypatch.setattr(proto, "SCREENING_CELLS", {"toy": METRIC})
    return proto


class TestCalibration:
    def test_lr_selection_and_gap_at_the_selected_point(self, tmp_path, protocol):
        rows = []
        # gate full: lr 0.01 beats 0.003; screening cell 0.95 at 0.01.
        for seed in (100, 101):
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
        for seed in (100, 101):
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

    def test_lr_tie_is_broken_by_fewer_steps(self, tmp_path, protocol):
        # Both LRs average the same best value; 0.003 reaches it in fewer
        # steps, so it is selected.
        rows = [
            _row(
                "a",
                gate="full",
                lr=0.003,
                seed=100,
                best=0.8,
                step=10,
                **{METRIC: 0.9},
            ),
            _row(
                "b",
                gate="full",
                lr=0.003,
                seed=101,
                best=0.8,
                step=20,
                **{METRIC: 0.9},
            ),
            _row(
                "c",
                gate="full",
                lr=0.01,
                seed=100,
                best=0.8,
                step=30,
                **{METRIC: 0.9},
            ),
            _row(
                "d",
                gate="full",
                lr=0.01,
                seed=101,
                best=0.8,
                step=40,
                **{METRIC: 0.9},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")].iloc[0]
        assert full["selected_lr"] == pytest.approx(0.003)

    def test_attention_gets_its_own_row(self, tmp_path, protocol):
        rows = [
            _row(
                "att0",
                task="toy",
                layout="ws",
                mixer="attention",
                lr=0.003,
                seed=100,
                best=0.99,
                step=5,
                tier="calib_ws",
                **{METRIC: 0.99},
            ),
            _row(
                "att1",
                task="toy",
                layout="ws",
                mixer="attention",
                lr=0.003,
                seed=101,
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


class TestCalibrationDamage:
    HIGH = "untouched"
    LOW = "sibling"

    @pytest.fixture
    def damage_protocol(self, protocol, monkeypatch):
        monkeypatch.setattr(
            protocol,
            "TASK_DAMAGE",
            {"toy": protocol.DamageReading(self.HIGH, self.LOW, slices=(16, 32))},
        )
        return protocol

    def _rows(self, seed, **extra):
        return _row(
            f"f{seed}",
            gate="full",
            lr=0.01,
            seed=seed,
            best=0.9,
            step=10,
            **{METRIC: 0.95, **extra},
        )

    def test_damage_columns_per_slice_at_the_selected_point(
        self, tmp_path, damage_protocol
    ):
        rows = [
            self._rows(
                seed,
                **{
                    "test/acc/slice=16/class=untouched": 0.9,
                    "test/acc/slice=16/class=sibling": 0.7,
                    "test/acc/slice=32/class=untouched": 0.8,
                    "test/acc/slice=32/class=sibling": 0.5,
                },
            )
            for seed in (100, 101)
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")].iloc[0]
        assert full["selected_lr"] == pytest.approx(0.01)
        assert full["damage/slice=16"] == pytest.approx(0.2)
        assert full["damage/slice=32"] == pytest.approx(0.3)
        assert "damage" not in df.columns  # no pooled column

    def test_damage_uses_only_the_selected_lr(self, tmp_path, damage_protocol):
        # The selected LR is 0.01; an unselected LR carries a wildly
        # different damage reading that must not leak into the value.
        rows = []
        for seed in (100, 101):
            rows.append(
                self._rows(
                    seed,
                    **{
                        "test/acc/slice=16/class=untouched": 0.9,
                        "test/acc/slice=16/class=sibling": 0.7,
                    },
                )
            )
            rows.append(
                _row(
                    f"old{seed}",
                    gate="full",
                    lr=0.003,
                    seed=seed,
                    best=0.2,
                    step=10,
                    **{
                        METRIC: 0.1,
                        "test/acc/slice=16/class=untouched": 1.0,
                        "test/acc/slice=16/class=sibling": 0.0,
                    },
                )
            )
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")].iloc[0]
        assert full["selected_lr"] == pytest.approx(0.01)
        assert full["damage/slice=16"] == pytest.approx(0.2)
        assert pd.isna(full["damage/slice=32"])

    def test_multi_state_rows_get_damage_at_their_selected_lr(
        self, tmp_path, damage_protocol
    ):
        rows = []
        for seed in (100, 101):
            rows.append(
                _row(
                    f"m{seed}",
                    layout="ws",
                    gate="full",
                    lr=0.01,
                    seed=seed,
                    best=0.8,
                    step=10,
                    **{
                        METRIC: 0.9,
                        "test/acc/slice=16/class=untouched": 0.9,
                        "test/acc/slice=16/class=sibling": 0.7,
                        "test/acc/slice=32/class=untouched": 0.8,
                        "test/acc/slice=32/class=sibling": 0.5,
                    },
                )
            )
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        multi = df[df["kind"] == "multi"].iloc[0]
        assert multi["selected_lr"] == pytest.approx(0.01)
        assert multi["damage/slice=16"] == pytest.approx(0.2)
        assert multi["damage/slice=32"] == pytest.approx(0.3)

    def test_missing_slice_columns_give_none(self, tmp_path, damage_protocol):
        rows = [
            self._rows(
                seed,
                **{
                    "test/acc/slice=16/class=untouched": 0.9,
                    "test/acc/slice=16/class=sibling": 0.7,
                },
            )
            for seed in (100, 101)
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")].iloc[0]
        assert full["damage/slice=16"] == pytest.approx(0.2)
        assert pd.isna(full["damage/slice=32"])

    def test_no_damage_reading_means_no_damage_columns(self, tmp_path, protocol):
        rows = [self._rows(100)]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        assert not any(c.startswith("damage") for c in df.columns)


class TestPositiveControl:
    def test_known_gap_over_seeds(self, tmp_path, protocol):
        rows = [
            _row(
                "p0",
                gate="full",
                seed=0,
                tier=TOY_CONTROL,
                best=0.6,
                step=5,
                **{METRIC: 0.6},
            ),
            _row(
                "p1",
                gate="full",
                seed=1,
                tier=TOY_CONTROL,
                best=0.7,
                step=5,
                **{METRIC: 0.7},
            ),
            _row(
                "p2",
                gate="kda",
                seed=0,
                tier=TOY_CONTROL,
                best=0.4,
                step=5,
                **{METRIC: 0.4},
            ),
            _row(
                "p3",
                gate="kda",
                seed=1,
                tier=TOY_CONTROL,
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

    def test_damage_columns_per_gate_and_slice(self, tmp_path, protocol, monkeypatch):
        monkeypatch.setattr(
            protocol,
            "TASK_DAMAGE",
            {"toy": protocol.DamageReading("untouched", "sibling", slices=(16, 32))},
        )
        rows = [
            _row(
                "p0",
                gate="full",
                seed=0,
                tier=TOY_CONTROL,
                best=0.6,
                step=5,
                **{
                    METRIC: 0.6,
                    "test/acc/slice=16/class=untouched": 0.9,
                    "test/acc/slice=16/class=sibling": 0.7,
                    "test/acc/slice=32/class=untouched": 0.8,
                    "test/acc/slice=32/class=sibling": 0.5,
                },
            ),
            _row(
                "p1",
                gate="full",
                seed=1,
                tier=TOY_CONTROL,
                best=0.7,
                step=5,
                **{
                    METRIC: 0.7,
                    "test/acc/slice=16/class=untouched": 0.95,
                    "test/acc/slice=16/class=sibling": 0.65,
                    "test/acc/slice=32/class=untouched": 0.85,
                    "test/acc/slice=32/class=sibling": 0.55,
                },
            ),
            _row(
                "p2",
                gate="kda",
                seed=0,
                tier=TOY_CONTROL,
                best=0.4,
                step=5,
                **{
                    METRIC: 0.4,
                    "test/acc/slice=16/class=untouched": 0.8,
                    "test/acc/slice=16/class=sibling": 0.7,
                    "test/acc/slice=32/class=untouched": 0.7,
                    "test/acc/slice=32/class=sibling": 0.6,
                },
            ),
            _row(
                "p3",
                gate="kda",
                seed=1,
                tier=TOY_CONTROL,
                best=0.5,
                step=5,
                **{
                    METRIC: 0.5,
                    "test/acc/slice=16/class=untouched": 0.82,
                    "test/acc/slice=16/class=sibling": 0.7,
                    "test/acc/slice=32/class=untouched": 0.72,
                    "test/acc/slice=32/class=sibling": 0.6,
                },
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.positive_control(_table(rows), out)
        df = pd.read_csv(out / "positive_control.csv")
        assert df["damage_full/slice=16"].iloc[0] == pytest.approx(0.25)
        assert df["damage_full/slice=32"].iloc[0] == pytest.approx(0.3)
        assert df["damage_kda/slice=16"].iloc[0] == pytest.approx(0.11)
        assert df["damage_kda/slice=32"].iloc[0] == pytest.approx(0.11)
        assert "damage_full" not in df.columns

    def test_duplicate_seed_rows_raise_naming_runs(self, tmp_path, protocol):
        rows = [
            _row(
                "p0",
                gate="full",
                seed=0,
                tier=TOY_CONTROL,
                best=0.6,
                step=5,
                **{METRIC: 0.6},
            ),
            _row(
                "p0dup",
                gate="full",
                seed=0,
                tier=TOY_CONTROL,
                best=0.66,
                step=5,
                **{METRIC: 0.66},
            ),
            _row(
                "p1",
                gate="kda",
                seed=0,
                tier=TOY_CONTROL,
                best=0.4,
                step=5,
                **{METRIC: 0.4},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="p0"):
            _analyze.positive_control(_table(rows), out)

    def test_screening_cell_without_tiers_is_skipped(self, tmp_path, monkeypatch):
        # A task in SCREENING_CELLS but not in TASK_TIERS is skipped with
        # a log line, not an error.
        proto = _analyze._protocol
        monkeypatch.setattr(proto, "SCREENING_CELLS", {"mqar": METRIC})
        monkeypatch.setattr(proto, "TASK_TIERS", {})
        out = tmp_path / "out"
        out.mkdir()
        # A zero-row table that still carries the schema (what ``_load``
        # returns on an empty folder).
        empty = _table([_row("x")]).iloc[0:0]
        _analyze.positive_control(empty, out)
        assert (out / "positive_control.csv").read_text().strip() == ""


class TestScreeningCell:
    """Direct unit tests of the pure fallback rule."""

    @pytest.fixture
    def fallback_protocol(self, protocol, monkeypatch):
        monkeypatch.setattr(protocol, "SCREENING_CELLS", {"toy": PRIMARY})
        monkeypatch.setattr(protocol, "SCREENING_FALLBACKS", {"toy": (FALLBACK, 0.15)})
        return protocol

    def test_both_below_floor_returns_fallback(self, fallback_protocol):
        assert (
            fallback_protocol.screening_cell("toy", {"full": 0.1, "kda": 0.05})
            == FALLBACK
        )

    def test_one_above_floor_keeps_primary(self, fallback_protocol):
        assert (
            fallback_protocol.screening_cell("toy", {"full": 0.2, "kda": 0.05})
            == PRIMARY
        )

    def test_missing_gate_keeps_primary(self, fallback_protocol):
        assert fallback_protocol.screening_cell("toy", {"full": 0.1}) == PRIMARY
        assert (
            fallback_protocol.screening_cell("toy", {"full": 0.1, "kda": None})
            == PRIMARY
        )

    def test_task_without_fallback_keeps_primary(self, protocol):
        # The protocol fixture registers a screening cell but no fallback.
        assert protocol.screening_cell("toy", {"full": 0.0, "kda": 0.0}) == METRIC


class TestScreeningFallback:
    @pytest.fixture
    def fallback_protocol(self, protocol, monkeypatch):
        monkeypatch.setattr(protocol, "SCREENING_CELLS", {"toy": PRIMARY})
        monkeypatch.setattr(protocol, "SCREENING_FALLBACKS", {"toy": (FALLBACK, 0.15)})
        return protocol

    def _calib_rows(self, full_primary, kda_primary):
        rows = []
        for seed in (100, 101):
            rows.append(
                _row(
                    f"f{seed}",
                    gate="full",
                    lr=0.01,
                    seed=seed,
                    best=0.9,
                    step=10,
                    **{PRIMARY: full_primary, FALLBACK: 0.6},
                )
            )
            rows.append(
                _row(
                    f"k{seed}",
                    gate="kda",
                    lr=0.01,
                    seed=seed,
                    best=0.7,
                    step=10,
                    **{PRIMARY: kda_primary, FALLBACK: 0.4},
                )
            )
        return rows

    def test_calibration_fallback_triggers_when_both_below(
        self, tmp_path, fallback_protocol
    ):
        rows = self._calib_rows(full_primary=0.1, kda_primary=0.05)
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")].iloc[0]
        kda = df[(df["kind"] == "single") & (df["gate_mode"] == "kda")].iloc[0]
        assert full["cell_metric"] == FALLBACK
        assert full["cell"] == pytest.approx(0.6)
        assert kda["cell"] == pytest.approx(0.4)
        assert full["gap_full_kda"] == pytest.approx(0.2)

    def test_calibration_no_fallback_when_one_gate_above(
        self, tmp_path, fallback_protocol
    ):
        rows = self._calib_rows(full_primary=0.2, kda_primary=0.05)
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")].iloc[0]
        assert full["cell_metric"] == PRIMARY
        assert full["cell"] == pytest.approx(0.2)

    def test_calibration_no_fallback_when_one_gate_missing(
        self, tmp_path, fallback_protocol
    ):
        rows = [
            _row(
                f"f{seed}",
                gate="full",
                lr=0.01,
                seed=seed,
                best=0.9,
                step=10,
                **{PRIMARY: 0.1, FALLBACK: 0.6},
            )
            for seed in (100, 101)
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")].iloc[0]
        assert full["cell_metric"] == PRIMARY
        assert full["cell"] == pytest.approx(0.1)

    def test_calibration_fallback_missing_column_keeps_primary(
        self, tmp_path, fallback_protocol
    ):
        # Both gates are below the floor, but the fallback column is absent:
        # the report keeps the primary cells instead of raising KeyError.
        rows = self._calib_rows(full_primary=0.1, kda_primary=0.05)
        for row in rows:
            row.pop(FALLBACK)
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")].iloc[0]
        assert full["cell_metric"] == PRIMARY
        assert full["cell"] == pytest.approx(0.1)

    def _control_rows(self, full_primary, kda_primary):
        rows = []
        for seed in (0, 1):
            rows.append(
                _row(
                    f"f{seed}",
                    gate="full",
                    seed=seed,
                    tier=TOY_CONTROL,
                    best=0.6,
                    step=5,
                    **{PRIMARY: full_primary, FALLBACK: 0.6},
                )
            )
            rows.append(
                _row(
                    f"k{seed}",
                    gate="kda",
                    seed=seed,
                    tier=TOY_CONTROL,
                    best=0.4,
                    step=5,
                    **{PRIMARY: kda_primary, FALLBACK: 0.4},
                )
            )
        return rows

    def test_positive_control_fallback_triggers_when_both_below(
        self, tmp_path, fallback_protocol
    ):
        rows = self._control_rows(full_primary=0.1, kda_primary=0.05)
        out = tmp_path / "out"
        out.mkdir()
        _analyze.positive_control(_table(rows), out)
        df = pd.read_csv(out / "positive_control.csv")
        assert df["metric"].iloc[0] == FALLBACK
        assert df["mean_high"].iloc[0] == pytest.approx(0.6)
        assert df["mean_low"].iloc[0] == pytest.approx(0.4)
        assert df["mean_gap"].iloc[0] == pytest.approx(0.2)

    def test_positive_control_no_fallback_when_one_gate_above(
        self, tmp_path, fallback_protocol
    ):
        rows = self._control_rows(full_primary=0.2, kda_primary=0.05)
        out = tmp_path / "out"
        out.mkdir()
        _analyze.positive_control(_table(rows), out)
        df = pd.read_csv(out / "positive_control.csv")
        assert df["metric"].iloc[0] == PRIMARY
        assert df["mean_high"].iloc[0] == pytest.approx(0.2)

    def test_positive_control_no_fallback_when_one_gate_missing(
        self, tmp_path, fallback_protocol
    ):
        # The full gate has only the primary cell; the kda gate has only the
        # fallback cell (its primary value is missing). The rule needs both
        # gates' primary readings, so it does not trigger and there is no
        # paired gap: the report is empty. A wrongly triggered fallback would
        # see both gates' fallback cells and produce a row.
        rows = []
        for seed in (0, 1):
            rows.append(
                _row(
                    f"f{seed}",
                    gate="full",
                    seed=seed,
                    tier=TOY_CONTROL,
                    best=0.6,
                    step=5,
                    **{PRIMARY: 0.1, FALLBACK: 0.6},
                )
            )
            rows.append(
                _row(
                    f"k{seed}",
                    gate="kda",
                    seed=seed,
                    tier=TOY_CONTROL,
                    best=0.4,
                    step=5,
                    **{PRIMARY: float("nan"), FALLBACK: 0.4},
                )
            )
        out = tmp_path / "out"
        out.mkdir()
        _analyze.positive_control(_table(rows), out)
        df = pd.read_csv(out / "positive_control.csv")
        assert df.empty


class TestCompleteness:
    def test_expected_and_unexpected_tiers(self, tmp_path, monkeypatch):
        # expected tier1_core -> 5 done runs; only 3 present -> missing 2.
        monkeypatch.setattr(_analyze._protocol, "EXPECTED_RUNS", {TOY_CONTROL: 5})
        rows = [
            _row("a0", tier=TOY_CONTROL, best=0.6),
            _row("a1", tier=TOY_CONTROL, best=0.6),
            _row("a2", tier=TOY_CONTROL, best=0.6),
            _row("a3", tier=TOY_CONTROL, status="interrupted", best=None),
            _row("c0", tier="calib_lr", best=0.6),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.completeness(_table(rows), out)
        df = pd.read_csv(out / "completeness.csv")
        core = df[df["tier"] == TOY_CONTROL].iloc[0]
        assert core["expected"] == 5
        assert core["observed"] == 3  # the interrupted one does not count
        assert core["missing"] == 2
        assert bool(core["known"]) is True
        calib = df[df["tier"] == "calib_lr"].iloc[0]
        assert calib["expected"] == 0  # present with runs, no expected count
        assert bool(calib["known"]) is False


class TestCalibrationDecay:
    def test_two_decay_schemes_give_two_rows_and_one_snippet_entry(
        self, tmp_path, protocol
    ):
        # One cell run under two decay schemes: two rows, each with its
        # own selected LR; the frozen snippet holds a single entry (the
        # harness's grouped decay only).
        rows = [
            _row(
                "g0",
                gate="full",
                lr=0.003,
                seed=100,
                best=0.8,
                step=10,
                **{METRIC: 0.9},
            ),
            _row(
                "g1",
                gate="full",
                lr=0.003,
                seed=101,
                best=0.8,
                step=10,
                **{METRIC: 0.9},
            ),
            _row(
                "u0",
                gate="full",
                lr=0.01,
                seed=100,
                best=0.9,
                step=10,
                **{METRIC: 0.95},
            ),
            _row(
                "u1",
                gate="full",
                lr=0.01,
                seed=101,
                best=0.9,
                step=10,
                **{METRIC: 0.95},
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

    def test_n_seeds_is_the_selected_lrs_count(self, tmp_path, protocol):
        # The best LR (0.01) is measured on one fewer seed than 0.003.
        rows = [
            _row(
                "g0",
                gate="full",
                lr=0.003,
                seed=100,
                best=0.8,
                step=10,
                **{METRIC: 0.9},
            ),
            _row(
                "g1",
                gate="full",
                lr=0.003,
                seed=101,
                best=0.8,
                step=10,
                **{METRIC: 0.9},
            ),
            _row(
                "u0",
                gate="full",
                lr=0.01,
                seed=100,
                best=0.9,
                step=10,
                **{METRIC: 0.95},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")].iloc[0]
        assert full["selected_lr"] == pytest.approx(0.01)
        assert full["n_seeds"] == 1


class TestFrozenSnippet:
    def test_single_and_multi_gated_and_attention_read_back(self, tmp_path, protocol):
        # Every grouped-decay selection (single + multi gated-delta, and
        # the attention rate) lands in the snippet in the shape
        # ``_frozen_lr`` reads, and reads back verbatim.
        rows = [
            _row(
                "s0",
                layout="single",
                gate="full",
                lr=0.003,
                seed=100,
                best=0.8,
                **{METRIC: 0.9},
            ),
            _row(
                "s1",
                layout="single",
                gate="kda",
                lr=0.001,
                seed=100,
                best=0.8,
                **{METRIC: 0.9},
            ),
            _row("m0", layout="ws", gate="full", lr=0.005, seed=100, best=0.8),
            _row(
                "a0",
                layout="ws",
                mixer="attention",
                lr=0.006,
                seed=100,
                best=0.8,
                tier="calib_ws",
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        snippet = out / "frozen_protocol_snippet.yaml"
        frozen = OmegaConf.load(snippet)  # parses as YAML
        assert _frozen_lr(_lr_cfg(layout="single", gate="full"), frozen) == 0.003
        assert _frozen_lr(_lr_cfg(layout="single", gate="kda"), frozen) == 0.001
        assert _frozen_lr(_lr_cfg(layout="ws", gate="full"), frozen) == 0.005
        assert _frozen_lr(_lr_cfg(mixer="attention"), frozen) == 0.006

    def test_multiple_attention_rows_raise_naming_layouts(self, tmp_path, protocol):
        rows = [
            _row(
                "a0",
                layout="ws",
                mixer="attention",
                lr=0.006,
                seed=100,
                best=0.8,
                tier="calib_ws",
            ),
            _row(
                "a1",
                layout="ll",
                mixer="attention",
                lr=0.007,
                seed=101,
                best=0.8,
                tier="calib_ws",
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="ws"):
            _analyze.calibration(_table(rows), out)


class TestEmptyRunsDir:
    def test_main_writes_all_three_without_error(self, tmp_path, monkeypatch):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        out_dir = tmp_path / "analysis"
        out_dir.mkdir()
        monkeypatch.setattr(_analyze._protocol, "EXPECTED_RUNS", {"tier0": 4})
        monkeypatch.setattr(
            sys,
            "argv",
            ["02_analyze.py", "--runs-dir", str(runs_dir), "--out-dir", str(out_dir)],
        )
        _analyze.main()
        calib = out_dir / "calibration.csv"
        pos = out_dir / "positive_control.csv"
        comp = out_dir / "completeness.csv"
        assert calib.exists() and pos.exists() and comp.exists()
        assert calib.read_text().strip() == ""
        assert pos.read_text().strip() == ""
        df = pd.read_csv(comp)
        assert len(df) == 1
        assert df["tier"].iloc[0] == "tier0"
        assert df["missing"].iloc[0] == 4


class TestSweepTiersRegistered:
    def test_every_sweep_tier_is_a_protocol_tier(self):
        # conf/sweep has no configs now (MQAR is smoke/debug only), so
        # the loop is empty; this catches a sweep file whose tier isn't
        # registered in PROTOCOL_TIERS.
        sweep_dir = _PILOT / "conf" / "sweep"
        registered = _analyze._protocol.PROTOCOL_TIERS
        for path in sorted(sweep_dir.glob("*.yaml")):
            cfg = OmegaConf.load(path)
            tier = str(OmegaConf.select(cfg, "experiment.tier"))
            assert (
                tier in registered
            ), f"{path.name}: tier {tier!r} not in PROTOCOL_TIERS"
