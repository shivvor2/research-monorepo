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
    weight_decay=0.1,
    seq_len=0,
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
        "weight_decay": weight_decay,
        "seq_len": seq_len,
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
            # A competing LR with lower accuracy and a different damage
            # reading: it must not be selected, so its values must not leak
            # into the reported damage.
            rows.append(
                _row(
                    f"m{seed}old",
                    layout="ws",
                    gate="full",
                    lr=0.003,
                    seed=seed,
                    best=0.2,
                    step=10,
                    **{
                        METRIC: 0.1,
                        "test/acc/slice=16/class=untouched": 1.0,
                        "test/acc/slice=16/class=sibling": 0.0,
                        "test/acc/slice=32/class=untouched": 1.0,
                        "test/acc/slice=32/class=sibling": 0.0,
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

    def test_exactly_at_floor_keeps_primary(self, fallback_protocol):
        # The rule is strictly below the floor: a gate reading exactly 0.15
        # (one gate or both) keeps the primary cell.
        assert (
            fallback_protocol.screening_cell("toy", {"full": 0.15, "kda": 0.05})
            == PRIMARY
        )
        assert (
            fallback_protocol.screening_cell("toy", {"full": 0.1, "kda": 0.15})
            == PRIMARY
        )
        assert (
            fallback_protocol.screening_cell("toy", {"full": 0.15, "kda": 0.15})
            == PRIMARY
        )


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

    def _gap_cell_rows(self):
        """An ordinary cell (both gates above the floor, so no fallback
        triggers) that does produce a gap: a table with this cell has a
        ``gap_full_kda`` column even when the fallback cell's gap is
        unreported."""
        rows = []
        for seed in (100, 101):
            rows.append(
                _row(
                    f"g{seed}",
                    gate="full",
                    lam="gated",
                    lr=0.01,
                    seed=seed,
                    best=0.9,
                    step=10,
                    **{PRIMARY: 0.4, FALLBACK: 0.6},
                )
            )
            rows.append(
                _row(
                    f"kg{seed}",
                    gate="kda",
                    lam="gated",
                    lr=0.01,
                    seed=seed,
                    best=0.7,
                    step=10,
                    **{PRIMARY: 0.3, FALLBACK: 0.4},
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

    def _no_gap_rows(self, *, drop_fallback_column):
        """The unreadable-fallback cell (both gates below the floor, mlp)
        plus an ordinary gap-producing cell (gated), for the no-gap tests.
        With ``drop_fallback_column`` the fallback metric is removed from
        every row, so the column is absent from the table."""
        rows = self._calib_rows(full_primary=0.1, kda_primary=0.05)
        rows += self._gap_cell_rows()
        if drop_fallback_column:
            for row in rows:
                row.pop(FALLBACK)
        else:
            for row in rows:
                if row["lambda_mode"] == "mlp" and row["gate_mode"] == "kda":
                    row[FALLBACK] = float("nan")
        return rows

    def _assert_unreadable_fallback_cell_no_gap(self, df):
        single = df[df["kind"] == "single"]
        mlp = single[single["lambda_mode"] == "mlp"]
        full = mlp[mlp["gate_mode"] == "full"].iloc[0]
        kda = mlp[mlp["gate_mode"] == "kda"].iloc[0]
        assert full["cell_metric"] == PRIMARY
        assert kda["cell_metric"] == PRIMARY
        assert full["cell"] == pytest.approx(0.1)
        assert kda["cell"] == pytest.approx(0.05)
        # The column exists (the ordinary cell below set it) and these rows
        # hold NaN: per-row, this cell has NO gap, not a whole-CSV absence.
        assert pd.isna(full.get("gap_full_kda", float("nan")))
        assert pd.isna(kda.get("gap_full_kda", float("nan")))
        gated = single[single["lambda_mode"] == "gated"]
        gd_full = gated[gated["gate_mode"] == "full"].iloc[0]
        gd_kda = gated[gated["gate_mode"] == "kda"].iloc[0]
        assert gd_full["gap_full_kda"] == pytest.approx(0.1)
        assert gd_kda["gap_full_kda"] == pytest.approx(0.1)
        assert gd_full["cell_metric"] == PRIMARY

    def test_calibration_fallback_missing_column_reports_no_gap(
        self, tmp_path, fallback_protocol
    ):
        # Both gates are below the floor, but the fallback column is absent:
        # the report keeps the primary cells for inspection and this cell
        # reports NO gap (the primary gap would be noise the protocol flagged
        # as uninformative). The ordinary cell in the same table has its gap.
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(self._no_gap_rows(drop_fallback_column=True)), out)
        df = pd.read_csv(out / "calibration.csv")
        assert "gap_full_kda" in df.columns
        self._assert_unreadable_fallback_cell_no_gap(df)

    def test_calibration_fallback_missing_reading_reports_no_gap(
        self, tmp_path, fallback_protocol
    ):
        # Both gates are below the floor and the fallback column exists, but
        # the kda gate has no fallback value at its selected LR: both cells
        # stay on the primary metric (never a primary/fallback mix) and this
        # cell reports no gap, while the ordinary cell keeps its own.
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(self._no_gap_rows(drop_fallback_column=False)), out)
        df = pd.read_csv(out / "calibration.csv")
        assert "gap_full_kda" in df.columns
        self._assert_unreadable_fallback_cell_no_gap(df)

    def test_positive_control_fallback_missing_column_skips_task(
        self, tmp_path, fallback_protocol
    ):
        # Both gates are below the floor, but the fallback column is absent:
        # there is no valid screening reading, so the task gets no row at
        # all instead of a primary-gap row.
        rows = self._control_rows(full_primary=0.1, kda_primary=0.05)
        for row in rows:
            row.pop(FALLBACK)
        out = tmp_path / "out"
        out.mkdir()
        _analyze.positive_control(_table(rows), out)
        assert (out / "positive_control.csv").read_text().strip() == ""

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
        # The no-row case still carries the report schema, so read_csv works
        # and yields an empty frame.
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

    def test_snippet_writes_an_attention_entry_per_task(
        self, tmp_path, protocol, monkeypatch
    ):
        # The attention calibration tier feeds lr_table.<task>.attention for
        # every task that runs it, each under its own task heading.
        monkeypatch.setattr(
            protocol,
            "TASK_TIERS",
            {
                "toy": protocol.TaskTiers(
                    calibration=TOY_CALIBRATION, positive_control=TOY_CONTROL
                ),
                "entity_field": protocol.TaskTiers(
                    calibration=("calib_ef_attention",),
                    positive_control="tier1_ef_single",
                ),
            },
        )
        monkeypatch.setattr(
            protocol,
            "CALIBRATION_TIERS",
            frozenset((*TOY_CALIBRATION, "calib_ef_attention")),
        )
        rows = [
            _row(
                "ea0",
                task="entity_field",
                layout="single",
                mixer="attention",
                lr=0.003,
                seed=100,
                best=0.8,
                tier="calib_ef_attention",
            ),
            _row(
                "a0",
                task="toy",
                layout="single",
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
        snippet = (out / "frozen_protocol_snippet.yaml").read_text()
        assert "  entity_field:\n    attention: 0.003" in snippet
        assert "  toy:\n    attention: 0.006" in snippet

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
        # The message names the task and the two layouts, not a %-tuple.
        with pytest.raises(RuntimeError, match="task toy.*\\['ll', 'ws'\\]"):
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
        sweep_dir = _PILOT / "conf" / "sweep"
        registered = _analyze._protocol.PROTOCOL_TIERS
        for path in sorted(sweep_dir.glob("*.yaml")):
            cfg = OmegaConf.load(path)
            tier = str(OmegaConf.select(cfg, "experiment.tier"))
            assert (
                tier in registered
            ), f"{path.name}: tier {tier!r} not in PROTOCOL_TIERS"

    def test_every_sweep_tier_pins_its_task(self):
        # Every sweep owns its task (a sweep must not run the pilot's
        # default-less task under another tier's name by accident).
        sweep_dir = _PILOT / "conf" / "sweep"
        for path in sorted(sweep_dir.glob("*.yaml")):
            cfg = OmegaConf.load(path)
            # Assert on the raw value: ``str(None) == "None"` is truthy, so
            # converting before asserting would let an unpinned sweep pass.
            task = cfg.get("hydra", {}).get("sweeper", {}).get("params", {}).get("task")
            assert task, f"{path.name}: no task pinned in the sweeper params"


class TestWeightDecaySelection:
    def test_selects_pair_not_lr_pool(self, tmp_path, protocol):
        # Two weight decays at one LR with different best values: selecting
        # the (lr, wd) pair picks the better decay; pooling by LR alone
        # would average them and reach a different reading.
        rows = [
            _row("r1", lr=1e-3, weight_decay=0.0, seed=100, best=0.8, **{METRIC: 0.79}),
            _row("r2", lr=1e-3, weight_decay=0.1, seed=100, best=0.6, **{METRIC: 0.55}),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        row = df.iloc[0]
        assert row["selected_lr"] == 1e-3
        assert row["selected_weight_decay"] == 0.0
        assert row["mean_best_value"] == 0.8  # not the pooled (0.8+0.6)/2
        assert row["cell"] == 0.79

    def test_single_decay_unchanged_apart_from_column(self, tmp_path, protocol):
        rows = [
            _row("a", lr=1e-3, weight_decay=0.1, seed=100, best=0.8, **{METRIC: 0.79}),
            _row("b", lr=5e-4, weight_decay=0.1, seed=100, best=0.7, **{METRIC: 0.68}),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        assert list(df["selected_weight_decay"]) == [0.1]
        assert list(df["selected_lr"]) == [1e-3]

    def test_double_decay_selection_reads_screening_fallback_damage(
        self, tmp_path, protocol, monkeypatch
    ):
        # One gate at one LR run under two weight decays; selection picks
        # (0.01, 0.0). Screening (both gates below the floor at the primary
        # cell, so the fallback cell is read), and damage are all read at
        # that exact pair: a filter on lr alone (which mixes the decays)
        # must fail every one of these assertions.
        monkeypatch.setattr(protocol, "SCREENING_CELLS", {"toy": PRIMARY})
        monkeypatch.setattr(protocol, "SCREENING_FALLBACKS", {"toy": (FALLBACK, 0.15)})
        monkeypatch.setattr(
            protocol,
            "TASK_DAMAGE",
            {"toy": protocol.DamageReading("untouched", "sibling", slices=(16,))},
        )
        rows = []
        for seed in (100, 101):
            rows.append(
                _row(
                    f"wf{seed}",
                    gate="full",
                    lr=0.01,
                    weight_decay=0.0,
                    seed=seed,
                    best=0.9,
                    step=10,
                    **{
                        PRIMARY: 0.1,
                        FALLBACK: 0.8,
                        "test/acc/slice=16/class=untouched": 0.9,
                        "test/acc/slice=16/class=sibling": 0.4,
                    },
                )
            )
            rows.append(
                _row(
                    f"wx{seed}",
                    gate="full",
                    lr=0.01,
                    weight_decay=0.1,
                    seed=seed,
                    best=0.4,
                    step=10,
                    **{
                        PRIMARY: 0.1,
                        FALLBACK: 0.2,
                        "test/acc/slice=16/class=untouched": 0.2,
                        "test/acc/slice=16/class=sibling": 0.8,
                    },
                )
            )
        for seed in (100, 101):
            rows.append(
                _row(
                    f"wk{seed}",
                    gate="kda",
                    lr=0.01,
                    weight_decay=0.0,
                    seed=seed,
                    best=0.3,
                    step=10,
                    **{PRIMARY: 0.1, FALLBACK: 0.4},
                )
            )
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")].iloc[0]
        assert full["selected_lr"] == 0.01
        assert full["selected_weight_decay"] == 0.0
        # Fallback cell at the selected pair (0.8), not the lr-pooled
        # (0.8 + 0.2)/2 = 0.5.
        assert full["cell"] == pytest.approx(0.8)
        assert full["cell_metric"] == FALLBACK
        # Damage at the selected pair (0.5), not the lr-pooled -0.05.
        assert full["damage/slice=16"] == pytest.approx(0.5)


class TestMadBandAndSafeguard:
    def test_band_near_edge_and_stuck(self, tmp_path):
        # Run a sits in the band (0.18 both) with a monitored history that
        # never doubles its first value -> look-stuck on the same row. Run b
        # sits above the ceiling and peaks above twice its first value, so
        # it is not flagged on the first clause and (its tail is not flat)
        # not on the second either.
        rows = [
            _row(
                "a",
                task="mad_fuzzy",
                tier="calib_mad_single",
                lr=1e-3,
                weight_decay=0.1,
                seed=100,
                best=0.18,
                **{"test/acc_mad": 0.18},
                monitored_first=0.1,
                monitored_max=0.15,
                monitored_last_half_min=0.12,
                monitored_last_half_max=0.15,
            ),
            _row(
                "b",
                task="mad_fuzzy",
                gate="kda",
                tier="calib_mad_single",
                lr=1e-3,
                weight_decay=0.1,
                seed=100,
                best=0.95,
                **{"test/acc_mad": 0.9},
                monitored_first=0.8,
                monitored_max=1.9,
                monitored_last_half_min=0.9,
                monitored_last_half_max=0.95,
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        full = df[df["gate_mode"] == "full"].iloc[0]
        kda = df[df["gate_mode"] == "kda"].iloc[0]
        assert full["band"] == "in_band"
        assert full["near_edge"] is True or str(full["near_edge"]) == "True"
        assert kda["band"] == "above_ceiling"
        # The same safeguard row that lists run ``a`` must flag it stuck.
        stuck = pd.read_csv(out / "safeguard.csv")
        a = stuck[stuck["run_id"] == "a"]
        assert len(a) == 1 and bool(a.iloc[0]["looks_stuck"]) is True
        b = stuck[stuck["run_id"] == "b"]
        assert len(b) == 1 and bool(b.iloc[0]["looks_stuck"]) is False

    def test_peak_above_twice_early_then_fall_is_not_stuck(self):
        row = {
            "monitored_first": 0.1,
            "monitored_max": 0.9,
            "monitored_last_half_min": 0.08,
            "monitored_last_half_max": 0.13,  # 0.13 - 0.1 = 0.03 > 0.02
        }
        assert _analyze._looks_stuck(pd.Series(row)) is False

    def test_climbs_then_flat_tail_is_stuck_only_when_flat_both_ways(self):
        stuck = {
            "monitored_first": 0.1,
            "monitored_max": 0.9,
            "monitored_last_half_min": 0.09,
            "monitored_last_half_max": 0.11,
        }
        assert _analyze._looks_stuck(pd.Series(stuck)) is True
        not_stuck = {
            "monitored_first": 0.1,
            "monitored_max": 0.9,
            "monitored_last_half_min": 0.06,
            "monitored_last_half_max": 0.11,
        }
        assert _analyze._looks_stuck(pd.Series(not_stuck)) is False


class TestSeqLenCells:
    @staticmethod
    def _grid_rows(seq_len, winner):
        """The full MAD grid (lr x weight decay) at ``seq_len`` with
        ``winner`` the unique best point."""
        rows = []
        for seed in (100, 101):
            for lr in (1e-4, 5e-4, 1e-3):
                for wd in (0.0, 0.1):
                    best = 0.9 if (lr, wd) == winner else 0.7
                    rows.append(
                        _row(
                            f"r{seq_len}-{lr:g}-{wd:g}-{seed}",
                            gate="full",
                            lr=lr,
                            weight_decay=wd,
                            seq_len=seq_len,
                            seed=seed,
                            best=best,
                            step=10,
                            **{METRIC: best + 0.05},
                        )
                    )
        return rows

    def test_per_length_selection_picks_each_lengths_own_winner(
        self, tmp_path, protocol
    ):
        # Both lengths run the full grid with a different (lr, wd) winning at
        # each. Selection inside each length picks that length's winner; a
        # selection pooled across lengths would fail these assertions.
        rows = self._grid_rows(512, (5e-4, 0.0)) + self._grid_rows(256, (1e-3, 0.1))
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out, {"toy": 512})
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")]
        by_len = {row["seq_len"]: row for _, row in full.iterrows()}
        assert by_len[512]["selected_lr"] == pytest.approx(5e-4)
        assert by_len[512]["selected_weight_decay"] == pytest.approx(0.0)
        assert by_len[256]["selected_lr"] == pytest.approx(1e-3)
        assert by_len[256]["selected_weight_decay"] == pytest.approx(0.1)

    def test_rows_split_into_one_set_per_length(self, tmp_path, protocol):
        rows = [
            _row(
                "x0",
                gate="full",
                lr=1e-3,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                **{METRIC: 0.82},
                **{"seq_len": 512},
            ),
            _row(
                "x1",
                gate="full",
                lr=1e-3,
                weight_decay=0.0,
                seed=101,
                best=0.8,
                **{METRIC: 0.82},
                **{"seq_len": 512},
            ),
            _row(
                "y0",
                gate="full",
                lr=5e-4,
                weight_decay=0.0,
                seed=100,
                best=0.7,
                **{METRIC: 0.72},
                **{"seq_len": 256},
            ),
            _row(
                "y1",
                gate="full",
                lr=5e-4,
                weight_decay=0.0,
                seed=101,
                best=0.7,
                **{METRIC: 0.72},
                **{"seq_len": 256},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out, {"toy": 512})
        df = pd.read_csv(out / "calibration.csv")
        full = df[(df["kind"] == "single") & (df["gate_mode"] == "full")]
        assert set(full["seq_len"]) == {512, 256}
        by_len = {row["seq_len"]: row for _, row in full.iterrows()}
        assert by_len[512]["selected_lr"] == pytest.approx(1e-3)
        assert by_len[256]["selected_lr"] == pytest.approx(5e-4)

    def test_mixed_lengths_refuse_a_snippet_naming_them(self, tmp_path, protocol):
        rows = [
            _row(
                "x0",
                gate="full",
                lr=1e-3,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                **{METRIC: 0.82},
                **{"seq_len": 512},
            ),
            _row(
                "y0",
                gate="full",
                lr=5e-4,
                weight_decay=0.0,
                seed=100,
                best=0.7,
                **{METRIC: 0.72},
                **{"seq_len": 256},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="256.*512|512.*256"):
            _analyze.calibration(_table(rows), out)

    def test_seq_len_override_picks_one_length_for_the_snippet(
        self, tmp_path, protocol
    ):
        rows = [
            _row(
                "x0",
                gate="full",
                lr=1e-3,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                **{METRIC: 0.82},
                **{"seq_len": 512},
            ),
            _row(
                "y0",
                gate="full",
                lr=5e-4,
                weight_decay=0.0,
                seed=100,
                best=0.7,
                **{METRIC: 0.72},
                **{"seq_len": 256},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out, {"toy": 512})
        snippet = OmegaConf.load(out / "frozen_protocol_snippet.yaml")
        assert _frozen_lr(_lr_cfg(layout="single", gate="full"), snippet) == 1e-3

    def test_override_with_no_matching_rows_is_an_error(self, tmp_path, protocol):
        rows = [
            _row(
                "x0",
                gate="full",
                lr=1e-3,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                **{METRIC: 0.82},
                **{"seq_len": 512},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="1024.*512|512.*1024"):
            _analyze.calibration(_table(rows), out, {"toy": 1024})

    def test_one_length_across_gated_and_attention_rows(self, tmp_path, protocol):
        # Gated single-state rows at 512 and an attention row at 256: one
        # length per task over the combined rows, so freeze at 512 with an
        # override. The gated and attention entries both come from 512.
        rows = [
            _row(
                "g0",
                gate="full",
                lr=1e-3,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                **{METRIC: 0.82},
                **{"seq_len": 512},
            ),
            _row(
                "a0",
                layout="ws",
                mixer="attention",
                lr=6e-3,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                tier="calib_ws",
                **{"seq_len": 256},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out, {"toy": 512})
        snippet = OmegaConf.load(out / "frozen_protocol_snippet.yaml")
        assert _frozen_lr(_lr_cfg(layout="single", gate="full"), snippet) == 1e-3
        # The attention row was at 256, so it is not frozen at 512.
        assert "attention" not in str(snippet.lr_table.toy)


class TestMissingWeightDecay:
    def test_no_weight_decay_raises_naming_runs(self, tmp_path, protocol):
        rows = [
            _row("w0", gate="full", lr=1e-3, seed=100, best=0.8, weight_decay=None),
            _row("w1", gate="kda", lr=1e-3, seed=100, best=0.8, weight_decay=None),
        ]
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="w0"):
            _analyze.calibration(_table(rows), out)


class TestNoBandWithoutRegistration:
    def test_unregistered_task_gets_no_band_columns(self, tmp_path, protocol):
        rows = [
            _row(
                "a0",
                gate="full",
                lr=1e-3,
                weight_decay=0.1,
                seed=100,
                best=0.5,
                **{METRIC: 0.5},
            )
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        assert "band" not in df.columns
        assert "near_edge" not in df.columns


class TestSweepComposeTaskLen:
    def test_sweep_composes_with_the_length_override(self, monkeypatch):
        # The MAD sweeps pin task=mad_fuzzy; the length is an ordinary task
        # key, so task.seq_len=256 composes and max_seq_len (and t_max)
        # follow via the config resolver.
        from common import conf_resolvers  # noqa: F401  (registers the resolver)
        from hydra import compose, initialize

        monkeypatch.chdir(_PILOT)
        with initialize(version_base=None, config_path="conf"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "+sweep=calib_mad_single",
                    "task=mad_fuzzy",
                    "task.seq_len=256",
                ],
            )
            assert cfg.task.name == "mad_fuzzy"
            assert cfg.task.seq_len == 256
            assert cfg.task.max_seq_len == 272
            assert OmegaConf.select(cfg, "model.t_max") == 272


class TestPositiveControlSeqLen:
    def test_mixed_length_tier_refuses_without_override(self, tmp_path, protocol):
        rows = [
            _row(
                "p0",
                gate="full",
                seed=0,
                tier=TOY_CONTROL,
                best=0.6,
                step=5,
                **{METRIC: 0.6},
                seq_len=512,
            ),
            _row(
                "p1",
                gate="kda",
                seed=0,
                tier=TOY_CONTROL,
                best=0.4,
                step=5,
                **{METRIC: 0.4},
                seq_len=256,
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="256.*512|512.*256"):
            _analyze.positive_control(_table(rows), out)

    def test_override_picks_one_length(self, tmp_path, protocol):
        # Both seeds' paired gates at 512, plus a stray 256 row that the
        # override drops before the gap is read.
        rows = [
            _row(
                "p0",
                gate="full",
                seed=0,
                tier=TOY_CONTROL,
                best=0.6,
                step=5,
                **{METRIC: 0.6},
                seq_len=512,
            ),
            _row(
                "p1",
                gate="kda",
                seed=0,
                tier=TOY_CONTROL,
                best=0.4,
                step=5,
                **{METRIC: 0.4},
                seq_len=256,
            ),
            _row(
                "p2",
                gate="full",
                seed=1,
                tier=TOY_CONTROL,
                best=0.7,
                step=5,
                **{METRIC: 0.7},
                seq_len=512,
            ),
            _row(
                "p3",
                gate="kda",
                seed=1,
                tier=TOY_CONTROL,
                best=0.5,
                step=5,
                **{METRIC: 0.5},
                seq_len=512,
            ),
            _row(
                "p4",
                gate="kda",
                seed=0,
                tier=TOY_CONTROL,
                best=0.4,
                step=5,
                **{METRIC: 0.4},
                seq_len=512,
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.positive_control(_table(rows), out, {"toy": 512})
        df = pd.read_csv(out / "positive_control.csv")
        assert df["n_seeds"].iloc[0] == 2
        assert df["mean_gap"].iloc[0] == pytest.approx(0.2)


class TestSeqLenOverrideValidation:
    def test_unknown_task_key_refused(self, tmp_path, protocol):
        rows = [
            _row(
                "a0",
                gate="full",
                lr=1e-3,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                **{METRIC: 0.8},
            )
        ]
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="no rows"):
            _analyze.calibration(_table(rows), out, {"toy2": 512})

    def test_single_length_nonmatching_override_refused(self, tmp_path, protocol):
        rows = [
            _row(
                "p0",
                gate="full",
                seed=0,
                tier=TOY_CONTROL,
                best=0.6,
                step=5,
                **{METRIC: 0.6},
                seq_len=512,
            )
        ]
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="matches no rows"):
            _analyze.positive_control(_table(rows), out, {"toy": 256})

    def test_multi_length_nonmatching_override_refused(self, tmp_path, protocol):
        rows = [
            _row(
                "p0",
                gate="full",
                seed=0,
                tier=TOY_CONTROL,
                best=0.6,
                step=5,
                **{METRIC: 0.6},
                seq_len=512,
            ),
            _row(
                "p1",
                gate="full",
                seed=0,
                tier=TOY_CONTROL,
                best=0.6,
                step=5,
                **{METRIC: 0.6},
                seq_len=256,
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="matches no rows"):
            _analyze.positive_control(_table(rows), out, {"toy": 1024})


class TestCompletenessPerLength:
    def test_two_lengths_read_as_complete_groups(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_analyze._protocol, "EXPECTED_RUNS", {TOY_CONTROL: 2})
        rows = [
            _row("a0", seed=0, tier=TOY_CONTROL, best=0.6, step=5, seq_len=512),
            _row("a1", seed=1, tier=TOY_CONTROL, best=0.6, step=5, seq_len=512),
            _row("b0", seed=0, tier=TOY_CONTROL, best=0.6, step=5, seq_len=256),
            _row("b1", seed=1, tier=TOY_CONTROL, best=0.6, step=5, seq_len=256),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.completeness(_table(rows), out)
        df = pd.read_csv(out / "completeness.csv")
        assert set(df["seq_len"]) == {512, 256}
        assert (df["observed"] == 2).all()
        assert (df["missing"] == 0).all()
        assert (df["expected"] == 2).all()


class TestCompletenessMissingLength:
    def test_missing_length_task_counts_beside_another_tasks_override(
        self, tmp_path, monkeypatch
    ):
        # Task ``a`` has no seq_len at all (the column is float / NA); the
        # override is for task ``b``. Both tasks' done runs are counted and
        # the override filters only ``b``.
        monkeypatch.setattr(
            _analyze._protocol, "EXPECTED_RUNS", {"a_tier": 1, "b_tier": 1}
        )
        rows = [
            _row("a0", task="a", tier="a_tier", best=0.6, step=5, seq_len=None),
            _row("b0", task="b", tier="b_tier", best=0.6, step=5, seq_len=512),
            _row("b1", task="b", tier="b_tier", best=0.6, step=5, seq_len=512),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.completeness(_table(rows), out, {"b": 512})
        # keep_default_na=False: the empty seq_len in the CSV must stay empty
        # instead of being read back as NaN in a mixed column.
        df = pd.read_csv(out / "completeness.csv", keep_default_na=False)
        a_row = df[df["tier"] == "a_tier"].iloc[0]
        assert a_row["observed"] == 1
        assert a_row["seq_len"] == ""  # the missing length writes empty
        b_row = df[df["tier"] == "b_tier"].iloc[0]
        assert b_row["observed"] == 2
        assert b_row["seq_len"] == "512"

    def test_float_seq_len_column_override_keeps_the_length_runs(
        self, tmp_path, monkeypatch
    ):
        # One row without seq_len makes the column float ("512.0"); the
        # override must still match the 512 rows numerically and drop only
        # the not-at-512 ones.
        monkeypatch.setattr(
            _analyze._protocol, "EXPECTED_RUNS", {"calib_mad_single": 12}
        )
        rows = [
            _row(
                "x0",
                task="mad_fuzzy",
                tier="calib_mad_single",
                best=0.6,
                step=5,
                seq_len=512,
            ),
            _row(
                "x1",
                task="mad_fuzzy",
                tier="calib_mad_single",
                best=0.6,
                step=5,
                seq_len=512,
            ),
            _row(
                "y0",
                task="mad_fuzzy",
                tier="calib_mad_single",
                best=0.6,
                step=5,
                seq_len=None,
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.completeness(_table(rows), out, {"mad_fuzzy": 512})
        df = pd.read_csv(out / "completeness.csv")
        assert len(df) == 1
        cell = df.iloc[0]
        assert cell["seq_len"] == 512
        assert cell["observed"] == 2


class TestOwnTierCalibration:
    def test_mismatched_tier_row_excluded_not_silent(
        self, tmp_path, protocol, monkeypatch, caplog
    ):
        monkeypatch.setattr(
            protocol,
            "TASK_TIERS",
            {
                "toy": protocol.TaskTiers(
                    calibration=("calib_lr", "calib_ws"), positive_control=TOY_CONTROL
                ),
                "ef": protocol.TaskTiers(
                    calibration=("calib_ef_single",), positive_control="tier1_ef_single"
                ),
            },
        )
        monkeypatch.setattr(
            protocol,
            "CALIBRATION_TIERS",
            frozenset(("calib_lr", "calib_ws", "calib_ef_single")),
        )
        rows = [
            _row(
                "good",
                task="toy",
                gate="full",
                lr=1e-3,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                **{METRIC: 0.8},
                tier="calib_lr",
            ),
            _row(
                "stray",
                task="toy",
                gate="kda",
                lr=1e-3,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                **{METRIC: 0.8},
                tier="calib_ef_single",
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        with caplog.at_level("WARNING"):
            _analyze.calibration(_table(rows), out)
        df = pd.read_csv(out / "calibration.csv")
        assert len(df) == 1 and df["gate_mode"].iloc[0] == "full"
        assert "stray" in caplog.text


class TestBandMetric:
    def test_band_rows_on_another_monitor_refused(
        self, tmp_path, protocol, monkeypatch
    ):
        monkeypatch.setattr(
            protocol,
            "TASK_SETTING_BANDS",
            {"toy": protocol.SettingBand(0.15, 0.75, 0.05, "val/acc_toy")},
        )
        rows = [
            _row(
                "a",
                gate="full",
                lr=1e-3,
                weight_decay=0.0,
                seed=100,
                best=0.18,
                **{METRIC: 0.18},
                monitor_key="val/loss",
            )
        ]
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="band"):
            _analyze.calibration(_table(rows), out)

    def test_band_row_on_the_band_metric_passes(self, tmp_path, protocol, monkeypatch):
        monkeypatch.setattr(
            protocol,
            "TASK_SETTING_BANDS",
            {"toy": protocol.SettingBand(0.15, 0.75, 0.05, "val/acc_toy")},
        )
        rows = [
            _row(
                "a",
                gate="full",
                lr=1e-3,
                weight_decay=0.0,
                seed=100,
                best=0.18,
                **{METRIC: 0.18},
                monitor_key="val/acc_toy",
            )
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)  # no raise


class TestMissingLearningRate:
    def test_no_lr_raises_naming_runs(self, tmp_path, protocol):
        rows = [
            _row(
                "w0",
                gate="full",
                lr=None,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                **{METRIC: 0.8},
            )
        ]
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="learning rate"):
            _analyze.calibration(_table(rows), out)


class TestFrozenConflict:
    def test_conflicting_values_at_a_cell_refused(self, tmp_path, protocol):
        rows = [
            _row(
                "x",
                gate="full",
                lr=0.003,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                **{METRIC: 0.8},
            ),
            _row(
                "y",
                gate="full",
                lr=0.004,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                lam="gated",
                **{METRIC: 0.8},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(RuntimeError, match="conflicting"):
            _analyze.calibration(_table(rows), out)

    def test_identical_values_at_a_cell_coexist(self, tmp_path, protocol):
        rows = [
            _row(
                "x",
                gate="full",
                lr=0.003,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                **{METRIC: 0.8},
            ),
            _row(
                "y",
                gate="full",
                lr=0.003,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                lam="gated",
                **{METRIC: 0.8},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out)  # identical values coexist


class TestFrozenLengthWritten:
    def test_snippet_writes_the_frozen_length(self, tmp_path, protocol, monkeypatch):
        monkeypatch.setattr(protocol, "FROZEN_TASK_KEYS", {"toy": ("seq_len",)})
        rows = [
            _row(
                "x0",
                gate="full",
                lr=1e-3,
                weight_decay=0.0,
                seed=100,
                best=0.8,
                **{METRIC: 0.82},
                **{"seq_len": 512},
            ),
            _row(
                "y0",
                gate="full",
                lr=5e-4,
                weight_decay=0.0,
                seed=100,
                best=0.7,
                **{METRIC: 0.72},
                **{"seq_len": 256},
            ),
        ]
        out = tmp_path / "out"
        out.mkdir()
        _analyze.calibration(_table(rows), out, {"toy": 512})
        snippet = (out / "frozen_protocol_snippet.yaml").read_text()
        assert "task_settings:\n  toy:\n    seq_len: 512" in snippet
        assert (
            "lr_table:\n  toy:\n    gated_delta:\n      single:\n        full: 0.001"
            in snippet
        )
