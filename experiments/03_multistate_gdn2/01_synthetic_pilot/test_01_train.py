"""
Tests for the pilot training script: LR resolution and inheritance.

Target: experiments/03_multistate_gdn2/01_synthetic_pilot/01_train.py
Run with: pytest experiments/03_multistate_gdn2/01_synthetic_pilot/test_01_train.py -v

The core configurations look up their own frozen-table entries; the other
tiers inherit (no other LR is tuned): b_only/w_only -> (layout, full);
fs -> (ws, same gate); lambdas -> the mlp entry of (layout, gate); matched
bytes -> (single, full); R1 and the long-sequence tier -> the core entries;
the anchors -> their own entries keyed by d_model.
"""

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.trainer.states import TrainerFn, TrainerStatus
from omegaconf import OmegaConf

from research_lib.utils.run_lock import exclusive_lock

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SPEC = importlib.util.spec_from_file_location(
    "pilot_train", Path(__file__).resolve().parent / "01_train.py"
)
train = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train)


def make_cfg(
    *,
    task="overwrite",
    layout="single",
    gate_mode="full",
    lr=None,
    tier=None,
    d_model=32,
    num_heads=2,
    head_dim=None,
    num_keys=(8, 16, 32),
    layer_pattern="zoology",
    state_mixer="none",
):
    return OmegaConf.create(
        {
            "optimizer": {"lr": lr},
            "task": {"name": task, "num_keys": list(num_keys)},
            "model": {
                "layout": layout,
                "gate_mode": gate_mode,
                "d_model": d_model,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "layer_pattern": layer_pattern,
                "state_mixer": state_mixer,
            },
            "experiment": {"tier": tier},
        }
    )


@pytest.fixture
def frozen():
    """A filled-in example frozen protocol.

    lr_table: 0.001 / 0.003 / 0.01 patterns per (task, layout, gate);
    anchor_lr: per (d_model, layout).
    """
    lr_table = {}
    for task in ("mqar", "overwrite"):
        lr_table[task] = {}
        for i, layout in enumerate(("single", "ws", "ll")):
            lr_table[task][layout] = {
                gate: 0.001 + 0.001 * i + (0.0005 if gate == "full" else 0)
                for gate in ("kda", "full")
            }
    anchor_lr = {
        d: {
            layout: 0.01 + 0.001 * d + (0.0005 if layout == "ll" else 0)
            for layout in ("single", "ll")
        }
        for d in (16, 32, 64)
    }
    return OmegaConf.create({"lr_table": lr_table, "anchor_lr": anchor_lr})


class TestLrSource:
    """The lookup key after LR inheritance, per tier."""

    def test_core_configurations_lookup_their_own_entries(self, frozen):
        for task in ("mqar", "overwrite"):
            for layout in ("single", "ws", "ll"):
                for gate in ("full", "kda"):
                    cfg = make_cfg(
                        task=task, layout=layout, gate_mode=gate, tier="tier1_core"
                    )
                    assert train.resolve_lr(cfg, frozen) == pytest.approx(
                        frozen.lr_table[task][layout][gate]
                    )

    def test_gate_decomposition_inherits_full(self, frozen):
        # tier 2: b_only / w_only use the LR of (layout, full).
        for gate in ("b_only", "w_only"):
            cfg = make_cfg(layout="ll", gate_mode=gate, tier="tier2_gates")
            key = train.lr_source(cfg)
            assert key[:3] == ("overwrite", "ll", "full")
            assert key[3] is None
            assert train.resolve_lr(cfg, frozen) == pytest.approx(
                frozen.lr_table["overwrite"]["ll"]["full"]
            )

    def test_frozen_control_inherits_ws(self, frozen):
        # tier 3: fs uses (ws, same gate).
        for gate in ("full", "kda"):
            cfg = make_cfg(layout="fs", gate_mode=gate, tier="tier3_frozen")
            assert train.lr_source(cfg)[:3] == ("overwrite", "ws", gate)
            assert train.resolve_lr(cfg, frozen) == pytest.approx(
                frozen.lr_table["overwrite"]["ws"][gate]
            )

    def test_lambda_controls_use_the_mlp_entry(self, frozen):
        # tier 4: lambda static / lin use the (layout, gate) entry, which is
        # the lambda-mlp entry (the table is keyed by gate mode only).
        cfg = OmegaConf.create(
            {
                "optimizer": {"lr": None},
                "task": {"name": "overwrite"},
                "model": {
                    "layout": "ws",
                    "gate_mode": "full",
                    "d_model": 32,
                    "lambda_mode": "static",
                },
                "experiment": {"tier": "tier4_lambda"},
            }
        )
        assert train.lr_source(cfg)[:3] == ("overwrite", "ws", "full")
        assert train.resolve_lr(cfg, frozen) == pytest.approx(
            frozen.lr_table["overwrite"]["ws"]["full"]
        )

    def test_matched_bytes_uses_single_full(self, frozen):
        # tier 5: the matched-bytes configuration IS (single, full).
        cfg = make_cfg(layout="single", gate_mode="full", tier="tier5_matched_bytes")
        assert train.lr_source(cfg)[:3] == ("overwrite", "single", "full")
        assert train.resolve_lr(cfg, frozen) == pytest.approx(
            frozen.lr_table["overwrite"]["single"]["full"]
        )

    def test_r1_and_long_t_use_the_core_entries(self, frozen):
        # R1 runs the core overwrite grid; long-T the core mqar grid.
        cfg = make_cfg(layout="ws", gate_mode="kda", tier="R1")
        assert train.resolve_lr(cfg, frozen) == pytest.approx(
            frozen.lr_table["overwrite"]["ws"]["kda"]
        )
        cfg = make_cfg(task="mqar", layout="ll", gate_mode="full", tier="tier_longT")
        assert train.resolve_lr(cfg, frozen) == pytest.approx(
            frozen.lr_table["mqar"]["ll"]["full"]
        )

    def test_anchors_use_their_own_entries_by_d_model(self, frozen):
        for d in (16, 32, 64):
            for layout in ("single", "ll"):
                cfg = make_cfg(
                    task="mqar",
                    layout=layout,
                    gate_mode="gdn",
                    tier="tier6_anchors",
                    d_model=d,
                )
                key = train.lr_source(cfg)
                assert key[3] == d
                assert train.resolve_lr(cfg, frozen) == pytest.approx(
                    frozen.anchor_lr[d][layout]
                )


class TestResolveLr:
    """Resolution, explicit overrides, logging and loud failures."""

    def test_explicit_optimizer_lr_wins(self, frozen):
        cfg = make_cfg(lr=0.02, layout="fs", gate_mode="gdn", tier="R1")
        assert train.resolve_lr(cfg, frozen) == pytest.approx(0.02)

    def test_inheritance_logs_the_entry_used(self, frozen, caplog):
        cfg = make_cfg(layout="fs", gate_mode="full", tier="tier3_frozen")
        with caplog.at_level("INFO", logger=train.logger.name):
            lr = train.resolve_lr(cfg, frozen)
        assert lr == pytest.approx(frozen.lr_table["overwrite"]["ws"]["full"])
        assert any("inherited from entry" in r.message for r in caplog.records)
        # ... naming the inherited entry (ws, full) and the requested one (fs).
        assert any(
            "layout=ws, gate=full) for requested (layout=fs, gate=full)" in r.message
            for r in caplog.records
        )

    def test_direct_entry_logs_the_entry_used(self, frozen, caplog):
        cfg = make_cfg(layout="ws", gate_mode="full", tier="tier1_core")
        with caplog.at_level("INFO", logger=train.logger.name):
            train.resolve_lr(cfg, frozen)
        assert any("frozen LR table" in r.message for r in caplog.records)

    def test_unknown_configuration_fails_loudly(self, frozen):
        # gdn outside the anchor tier has no entry anywhere.
        cfg = make_cfg(gate_mode="gdn", tier="tier1_core")
        with pytest.raises(RuntimeError, match="gate_mode='gdn'"):
            train.resolve_lr(cfg, frozen)
        # A missing table cell fails loudly.
        broken = OmegaConf.create(
            {
                "lr_table": {"overwrite": {"ws": {"full": None}}},
                "anchor_lr": {},
            }
        )
        cfg = make_cfg(layout="ws", gate_mode="full", tier="tier1_core")
        with pytest.raises(RuntimeError, match="empty"):
            train.resolve_lr(cfg, broken)
        cfg = make_cfg(layout="ll", gate_mode="full", tier="tier1_core")
        with pytest.raises(RuntimeError, match="no frozen-protocol LR entry"):
            train.resolve_lr(cfg, broken)
        # A missing anchor entry fails loudly.
        cfg = make_cfg(
            task="mqar",
            layout="single",
            gate_mode="gdn",
            tier="tier6_anchors",
            d_model=32,
        )
        with pytest.raises(RuntimeError, match="anchor"):
            train.resolve_lr(cfg, broken)

    def test_every_tier_sweep_resolves_from_the_example_table(self, frozen):
        """Each sweep file's parameter grid resolves without error."""
        tasks = ("mqar", "overwrite")
        layouts = ("single", "ws", "ll")
        gates = ("kda", "full")
        grids = {
            "tier1_core": [
                {"task": task, "layout": layout, "gate_mode": gate}
                for task in tasks
                for layout in layouts
                for gate in gates
            ],
            "tier2_gates": [
                {"task": task, "layout": layout, "gate_mode": gate}
                for task in tasks
                for layout in layouts
                for gate in ("b_only", "w_only")
            ],
            "tier3_frozen": [
                {"task": task, "layout": "fs", "gate_mode": gate}
                for task in tasks
                for gate in gates
            ],
            "tier4_lambda": [
                {
                    "task": task,
                    "layout": layout,
                    "gate_mode": "full",
                    "lambda_mode": mode,
                }
                for task in tasks
                for layout in ("ws", "ll")
                for mode in ("static", "lin")
            ],
            "tier5_matched_bytes": [
                {"task": task, "layout": "single", "gate_mode": "full"}
                for task in tasks
            ],
            "R1": [
                {"task": "overwrite", "layout": layout, "gate_mode": gate}
                for layout in layouts
                for gate in gates
            ],
            "tier_longT": [
                {"task": "mqar", "layout": layout, "gate_mode": gate}
                for layout in layouts
                for gate in gates
            ],
            "tier6_anchors": [
                {
                    "task": "mqar",
                    "layout": layout,
                    "gate_mode": "gdn",
                    "d_model": d_model,
                }
                for layout in ("single", "ll")
                for d_model in (16, 32, 64)
            ],
        }
        for tier, variants in grids.items():
            for variant in variants:
                cfg = OmegaConf.create(
                    {
                        "optimizer": {"lr": None},
                        "task": {"name": variant["task"]},
                        "model": {
                            "layout": variant["layout"],
                            "gate_mode": variant["gate_mode"],
                            "d_model": variant.get("d_model", 32),
                            "lambda_mode": variant.get("lambda_mode", "mlp"),
                        },
                        "experiment": {"tier": tier},
                    }
                )
                lr = train.resolve_lr(cfg, frozen)
                assert 0 < lr < 1, (tier, variant, lr)


class TestOverwriteTaskFollowsDModel:
    """The overwrite task's largest K must equal 2 * d_k."""

    @pytest.fixture
    def frozen_with_tasks(self, frozen):
        return OmegaConf.merge(
            frozen,
            {
                "overwrite_task_by_d_model": {
                    32: "overwrite",
                    64: "overwrite_dk32",
                }
            },
        )

    def test_matching_config_passes(self, frozen_with_tasks):
        # d_model=32, num_heads=2 -> d_k=16 -> largest K = 32.
        cfg = make_cfg(d_model=32, num_heads=2, num_keys=(8, 16, 32))
        train.check_overwrite_k(cfg, frozen_with_tasks)
        # d_model=64, num_heads=2 -> d_k=32 -> largest K = 64.
        cfg = make_cfg(d_model=64, num_heads=2, num_keys=(16, 32, 64))
        train.check_overwrite_k(cfg, frozen_with_tasks)
        # head_dim overrides d_model / num_heads.
        cfg = make_cfg(d_model=64, num_heads=2, head_dim=16, num_keys=(8, 16, 32))
        train.check_overwrite_k(cfg, frozen_with_tasks)

    def test_mismatch_names_the_right_task_file(self, frozen_with_tasks):
        # d_model=64 (d_k=32, largest K must be 64) but the d_k=16 task file.
        cfg = make_cfg(d_model=64, num_heads=2, num_keys=(8, 16, 32))
        with pytest.raises(RuntimeError) as exc:
            train.check_overwrite_k(cfg, frozen_with_tasks)
        assert "K = 2 * d_k = 64" in str(exc.value)
        assert "use task=overwrite_dk32 (conf/task/overwrite_dk32.yaml)" in str(
            exc.value
        )
        # The reverse: d_model=32 with the d_k=32 task file.
        cfg = make_cfg(d_model=32, num_heads=2, num_keys=(16, 32, 64))
        with pytest.raises(RuntimeError) as exc:
            train.check_overwrite_k(cfg, frozen_with_tasks)
        assert "use task=overwrite (conf/task/overwrite.yaml)" in str(exc.value)

    def test_mismatch_without_the_record_still_fails_loudly(self, frozen):
        # No overwrite_task_by_d_model record: generic hint, still loud.
        cfg = make_cfg(d_model=64, num_heads=2, num_keys=(8, 16, 32))
        with pytest.raises(RuntimeError) as exc:
            train.check_overwrite_k(cfg, frozen)
        assert "largest K is 32" in str(exc.value)
        assert "conf/task/overwrite_dk32.yaml" in str(exc.value)

    def test_non_overwrite_tasks_are_untouched(self, frozen_with_tasks):
        cfg = make_cfg(task="mqar", d_model=64, num_keys=(8, 16, 32))
        train.check_overwrite_k(cfg, frozen_with_tasks)  # no raise


GATE_MODES = ("full", "b_only", "w_only", "kda", "gdn")


def model_cfg(**overrides):
    """A model section mirroring conf/config.yaml's (Hydra passes this)."""
    model: dict[str, Any] = {
        "vocab_size": 64,
        "d_model": 16,
        "n_layer": 2,
        "layer_pattern": "zoology",
        "state_mixer": "none",
        "ff_mult": 4,
        "num_heads": 1,
        "head_dim": None,
        "expand_v": 1.0,
        "gate_mode": "full",
        "layout": "single",
        "lambda_mode": "mlp",
        "window": 4,
        "chunk_size": 16,
        "t_max": 32,
        "output_gate_activation": "silu",
        "mixer_mode": "chunk",
        "base_conv_kernel": 3,
    }
    model.update(overrides)
    return OmegaConf.create({"model": model})


def sweeper_values(value) -> list[str]:
    """Every value of a --multirun sweeper parameter, in order.

    ``"a,b,c"`` -> ``["a", "b", "c"]``; ``"range(0,5)"`` ->
    ``["0", ..., "4"]``; a single value passes through.
    """
    value = str(value)
    if value.startswith("range("):
        inner = value[value.index("(") + 1 : value.rindex(")")]
        parts = [p.strip() for p in inner.split(",")]
        start, stop = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) > 2 else 1
        return [str(v) for v in range(start, stop, step)]
    return [v.strip() for v in value.split(",")]


def first_sweeper_value(value) -> str:
    """The first value of a --multirun sweeper parameter."""
    return sweeper_values(value)[0]


class TestFrozenChoices:
    """The frozen calibration choices (width, harness) are checked against
    the run's config, in every phase, once the frozen field is filled."""

    def frozen(self, d_model=None, layer_pattern=None, state_mixer=None):
        return OmegaConf.create(
            {
                "d_model": d_model,
                "harness": {
                    "layer_pattern": layer_pattern,
                    "state_mixer": state_mixer,
                },
            }
        )

    def test_d_model_match_passes(self):
        train.check_frozen_choices(
            make_cfg(d_model=64, tier="tier1_core"), self.frozen(d_model=64)
        )

    def test_d_model_mismatch_names_the_fix(self):
        with pytest.raises(RuntimeError) as exc:
            train.check_frozen_choices(
                make_cfg(d_model=32, tier="tier1_core"), self.frozen(d_model=64)
            )
        message = str(exc.value)
        assert "d_model" in message
        assert "32" in message and "64" in message
        assert (
            "set `model.d_model: 64` in conf/config.yaml, "
            "or pass `model.d_model=64`" in message
        )

    @pytest.mark.parametrize(
        "tier", ["calib_dmodel", "calib_lr_anchors", "tier6_anchors"]
    )
    def test_d_model_exemptions(self, tier):
        # These tiers choose or override the width themselves.
        train.check_frozen_choices(
            make_cfg(d_model=32, tier=tier), self.frozen(d_model=64)
        )

    def test_harness_match_passes(self):
        train.check_frozen_choices(
            make_cfg(tier="tier1_core", layer_pattern="mixer_x2", state_mixer="mlp"),
            self.frozen(layer_pattern="mixer_x2", state_mixer="mlp"),
        )

    def test_harness_mismatch_names_the_fix(self):
        with pytest.raises(RuntimeError) as exc:
            train.check_frozen_choices(
                make_cfg(tier="tier1_core", layer_pattern="zoology"),
                self.frozen(layer_pattern="mixer_x2"),
            )
        message = str(exc.value)
        assert "layer_pattern" in message
        assert "zoology" in message and "mixer_x2" in message
        assert "set `model.layer_pattern: mixer_x2` in conf/config.yaml" in message

    def test_anchors_are_not_exempt_from_the_harness(self):
        # The harness is shared by every tier, so the anchor numbers are
        # read in the same harness as the main grid.
        with pytest.raises(RuntimeError, match="layer_pattern"):
            train.check_frozen_choices(
                make_cfg(tier="tier6_anchors", layer_pattern="zoology"),
                self.frozen(layer_pattern="mixer_x2"),
            )

    def test_calib_harness_is_exempt(self):
        train.check_frozen_choices(
            make_cfg(tier="calib_harness", layer_pattern="zoology"),
            self.frozen(layer_pattern="mixer_x2", state_mixer="mlp"),
        )

    def test_tooling_configs_skip_the_check(self):
        """The smoke and debug configs are tooling, not protocol tiers:
        the frozen choices must not apply to them, or filling in the
        frozen file breaks the end-to-end smoke test and
        --config-name=debug (both run at d_model 16)."""
        from hydra import compose, initialize_config_dir

        conf_dir = Path(__file__).resolve().parent / "conf"
        frozen = self.frozen(d_model=64, layer_pattern="mixer_x2", state_mixer="mlp")
        with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
            for name in ("test", "debug"):
                cfg = compose(config_name=name)
                assert str(cfg.model.d_model) == "16"  # contradicts frozen
                train.check_frozen_choices(cfg, frozen)  # no raise

    def test_protocol_tiers_are_still_checked(self):
        # The allow-list must not disable the check for protocol tiers.
        with pytest.raises(RuntimeError, match="d_model"):
            train.check_frozen_choices(
                make_cfg(d_model=32, tier="tier1_core"), self.frozen(d_model=64)
            )

    def test_null_frozen_fields_skip_every_check(self):
        train.check_frozen_choices(
            make_cfg(
                tier="tier1_core",
                d_model=32,
                layer_pattern="zoology",
                state_mixer="none",
            ),
            self.frozen(),
        )


class TestSmokePathIndependentOfFrozenProtocol:
    """The smoke path (config-name=test) must not depend on the frozen
    file's choices, so filling conf/frozen_protocol.yaml in can never
    break the suite."""

    def test_smoke_path_passes_with_a_fully_filled_frozen_protocol(self):
        from hydra import compose, initialize_config_dir

        conf_dir = Path(__file__).resolve().parent / "conf"
        # Worst case: every frozen field filled, contradicting the smoke
        # config's own choices everywhere.
        lr_table = {
            task: {
                layout: {gate: 0.01 for gate in ("kda", "full")}
                for layout in ("single", "ws", "fs", "ll")
            }
            for task in ("mqar", "overwrite")
        }
        frozen = OmegaConf.create(
            {
                "frozen": True,
                "d_model": 64,
                "harness": {
                    "layer_pattern": "mixer_x2",
                    "state_mixer": "mlp",
                },
                "lr_table": lr_table,
                "anchor_lr": {d: {"single": 0.01, "ll": 0.01} for d in (16, 32, 64)},
                "overwrite_task_by_d_model": {
                    32: "overwrite",
                    64: "overwrite_dk32",
                },
            }
        )
        with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
            cfg = compose(config_name="test")
        # The frozen-file-dependent steps of main(), in order.
        # 1. The phase gate: the smoke config is phase 1 regardless.
        assert int(cfg.experiment.phase) < 2
        # 2. The frozen-choice gate: smoke is not a protocol tier.
        train.check_frozen_choices(cfg, frozen)
        # 3. The LR: the smoke config sets its own.
        assert train.resolve_lr(cfg, frozen) == pytest.approx(0.003)
        # 4. The overwrite-K check: mqar is untouched.
        train.check_overwrite_k(cfg, frozen)


class TestSweepConfigsCompose:
    """Every sweep file must compose through Hydra, exactly as
    ``--multirun`` would.

    Hydra struct mode rejects overrides of keys missing from
    ``conf/config.yaml`` ("Key ... is not in struct"), so a default that
    lives only in Python makes a tier unlaunchable. The other tests build
    configs with ``make_cfg``; only a real composition catches this.
    """

    def test_every_sweep_file_composes(self):
        from hydra import compose, initialize_config_dir

        conf_dir = Path(__file__).resolve().parent / "conf"
        sweeps = sorted((conf_dir / "sweep").glob("*.yaml"))
        assert len(sweeps) >= 10  # sanity: we found the sweep group
        for path in sweeps:
            stem = path.stem
            # The sweeper parameters come from the sweep file (compose()
            # returns the config without the hydra section). DictConfig
            # .get does not traverse dotted paths; use OmegaConf.select.
            params = OmegaConf.select(
                OmegaConf.load(path), "hydra.sweeper.params", default={}
            )
            overrides = [f"+sweep={stem}"] + [
                f"{key}={first_sweeper_value(value)}" for key, value in params.items()
            ]
            # The first value of every sweeper parameter must be a valid
            # override (this is what --multirun submits first).
            with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
                compose(config_name="config", overrides=overrides)


class TestRunIdUniqueness:
    """No two runs of one sweep tier may share a run ID.

    ``make_run_id`` must carry every identity dimension the sweep varies:
    two runs that differ only in a key it omits would overwrite each
    other's run JSON and checkpoint (``calib_harness`` sweeps
    ``layer_pattern``, which the ID did not carry).
    """

    @pytest.mark.slow
    def test_every_sweep_combination_has_a_unique_run_id(self):
        # ~75 s: every combination of every sweep file composes.
        import itertools

        from hydra import compose, initialize_config_dir

        conf_dir = Path(__file__).resolve().parent / "conf"
        placeholder_lr = 0.003
        for path in sorted((conf_dir / "sweep").glob("*.yaml")):
            stem = path.stem
            params = OmegaConf.select(
                OmegaConf.load(path), "hydra.sweeper.params", default={}
            )
            names = list(params.keys())
            grids = [sweeper_values(params[name]) for name in names]
            seen: set[str] = set()
            with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
                for combo in itertools.product(*grids):
                    overrides = [f"+sweep={stem}"] + [
                        f"{name}={value}" for name, value in zip(names, combo)
                    ]
                    cfg = compose(config_name="config", overrides=overrides)
                    lr = cfg.optimizer.lr
                    run_id = train.make_run_id(
                        cfg, placeholder_lr if lr is None else float(lr)
                    )
                    assert run_id not in seen, (stem, combo, run_id)
                    seen.add(run_id)


class TestBuildModel:
    """The experiment's builder: make_mixer + zoology_mixers + ZoologyLM.

    The gated-delta-specific model tests moved here from the harness
    tests when ZoologyLM became mixer-agnostic.
    """

    @pytest.mark.parametrize("gate_mode", GATE_MODES)
    def test_param_counts_identical_across_gate_modes(self, gate_mode):
        model = train.build_model(model_cfg(layout="ws", gate_mode=gate_mode))
        counts = {name: tuple(p.shape) for name, p in model.named_parameters()}
        reference = train.build_model(model_cfg(layout="ws", gate_mode="full"))
        ref_counts = {name: tuple(p.shape) for name, p in reference.named_parameters()}
        assert counts.keys() == ref_counts.keys()
        for key in ref_counts:
            assert counts[key] == ref_counts[key], key

    @pytest.mark.parametrize("layout", ["ws", "fs", "ll"])
    def test_layouts_differ_only_by_lambda_parameters(self, layout):
        single = train.build_model(model_cfg(layout="single"))
        multi = train.build_model(model_cfg(layout=layout))
        single_names = {name for name, _ in single.named_parameters()}
        multi_names = {name for name, _ in multi.named_parameters()}
        lambda_names = {n for n in multi_names if "lambda_mixer" in n}
        assert multi_names - single_names == lambda_names
        assert single_names - multi_names == set()
        for name, p in single.named_parameters():
            assert multi.state_dict()[name].shape == p.shape, name

    def test_state_bytes_accounting(self):
        from research_lib.layers.attention import MultistateGatedDelta

        for layout in ("single", "ws", "fs", "ll"):
            model = train.build_model(model_cfg(layout=layout))
            layers = [
                module
                for module in model.modules()
                if isinstance(module, MultistateGatedDelta)
            ]
            assert len(layers) == 1  # one mixer block in the zoology pattern
            layer = layers[0]
            expected = (
                layer.projections.num_heads
                * layer.num_slots
                * layer.projections.head_k_dim
                * layer.projections.head_v_dim
                * 4
            )
            assert layer.state_bytes() == expected

    def test_builder_forward_shapes_both_patterns(self):
        for pattern in ("zoology", "mixer_x2"):
            model = train.build_model(model_cfg(layer_pattern=pattern))
            logits = model(torch.zeros(2, 12, dtype=torch.long))
            assert logits.shape == (2, 12, 64)


class TestPrecisionFlags:
    """B05: fp32 without TF32 - both PyTorch TF32 switches off."""

    def test_both_tf32_flags_off(self):
        flags = train.set_precision_flags()
        assert flags == {
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False


class TestAssertFitFinished:
    """B08: the post-fit guard passes only for a normally finished fit."""

    def test_finished_passes(self):
        trainer = SimpleNamespace(state=SimpleNamespace(status=TrainerStatus.FINISHED))
        train.assert_fit_finished(trainer)  # no raise

    @pytest.mark.parametrize(
        "status", [TrainerStatus.INTERRUPTED, TrainerStatus.RUNNING]
    )
    def test_not_finished_raises(self, status):
        trainer = SimpleNamespace(state=SimpleNamespace(status=status))
        with pytest.raises(RuntimeError, match="not FINISHED"):
            train.assert_fit_finished(trainer)


class TestShouldSkip:
    """B10: the skip logic - done + same identity skips; everything else
    falls through to the run."""

    def _record(self, tmp_path, status="test_done", config_hash="h", seed=0):
        path = tmp_path / "run.json"
        path.write_text(
            json.dumps({"status": status, "config_hash": config_hash, "seed": seed})
        )
        return path

    def test_missing_record_runs(self, tmp_path):
        assert train.should_skip(tmp_path / "nope.json", ("h", 0)) is None

    def test_done_same_identity_skips(self, tmp_path):
        path = self._record(tmp_path)
        assert train.should_skip(path, ("h", 0)) == "already test_done"

    def test_done_other_identity_runs(self, tmp_path):
        path = self._record(tmp_path)
        assert train.should_skip(path, ("h", 1)) is None  # other seed
        assert train.should_skip(path, ("g", 0)) is None  # other config

    @pytest.mark.parametrize("status", ["running", "interrupted", "failed", "fit_done"])
    def test_non_done_statuses_run(self, tmp_path, status):
        path = self._record(tmp_path, status=status)
        assert train.should_skip(path, ("h", 0)) is None

    def test_unreadable_record_runs(self, tmp_path):
        path = tmp_path / "run.json"
        path.write_text("{not json")
        assert train.should_skip(path, ("h", 0)) is None


class TestClaimRun:
    """The skip decision is made while holding the run lock: a run that
    another process completed between its pre-lock check and the lock
    acquisition is skipped, not re-run."""

    def test_skips_a_run_completed_while_waiting(self, tmp_path):
        lock_path = tmp_path / "run.lock"
        record_path = tmp_path / "run.json"
        identity = ("h", 0)
        # Another process is running the run: the lock is held and the
        # record says `running`.
        with exclusive_lock(lock_path):
            record_path.write_text(
                json.dumps({"status": "running", "config_hash": "h", "seed": 0})
            )
            # The other process finishes while this process is waiting:
            # test_done is written before the lock is released.
            record_path.write_text(
                json.dumps({"status": "test_done", "config_hash": "h", "seed": 0})
            )
        # The lock is free now. Claiming the run must skip it: the
        # record is read only after the lock is acquired, so the
        # completed run is not re-run.
        with train.claim_run(lock_path, record_path, identity) as reason:
            assert reason == "already test_done"

    def test_held_lock_skips_without_reading_the_record(self, tmp_path):
        lock_path = tmp_path / "run.lock"
        record_path = tmp_path / "run.json"
        with exclusive_lock(lock_path):
            with train.claim_run(lock_path, record_path, ("h", 0)) as reason:
                assert reason == "running in another process"

    def test_no_record_check_proceeds(self, tmp_path):
        # record_path=None: the skip check does not apply (e.g. the smoke
        # and debug configs); the run proceeds once the lock is acquired.
        lock_path = tmp_path / "run.lock"
        with train.claim_run(lock_path, None, ("h", 0)) as reason:
            assert reason is None


class TestStopRequested:
    """B11: the STOP file lives in the launch directory (the original
    working directory, not the job dir)."""

    def _cfg(self):
        return OmegaConf.create({"launch": {"stop_file": "STOP"}})

    def test_no_stop_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(train, "get_original_cwd", lambda: str(tmp_path))
        assert train.stop_requested(self._cfg()) is None

    def test_stop_file_found(self, tmp_path, monkeypatch):
        stop = tmp_path / "STOP"
        stop.touch()
        monkeypatch.setattr(train, "get_original_cwd", lambda: str(tmp_path))
        assert train.stop_requested(self._cfg()) == stop

    def test_stop_file_elsewhere_is_not_seen(self, tmp_path, monkeypatch):
        other = tmp_path / "other"
        other.mkdir()
        (other / "STOP").touch()
        monkeypatch.setattr(train, "get_original_cwd", lambda: str(tmp_path))
        assert train.stop_requested(self._cfg()) is None


class TestInShard:
    """B12: round-robin shard membership, with validation."""

    def test_round_robin(self):
        assert train.in_shard(0, 0, 2) is True
        assert train.in_shard(1, 0, 2) is False
        assert train.in_shard(1, 1, 2) is True
        assert train.in_shard(2, 0, 2) is True
        assert train.in_shard(2, 1, 2) is False
        assert train.in_shard(0, 0, 1) is True
        assert train.in_shard(7, 0, 1) is True

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="num_shards"):
            train.in_shard(0, 0, 0)
        with pytest.raises(ValueError, match="num_shards"):
            train.in_shard(0, 0, -1)
        with pytest.raises(ValueError, match="shard_index"):
            train.in_shard(0, 2, 2)
        with pytest.raises(ValueError, match="shard_index"):
            train.in_shard(0, -1, 2)


class TestSkipOnlyProtocolTiers:
    """B13: the smoke and debug configs are tooling, not protocol tiers:
    they always run, whatever launch.skip_completed says."""

    def test_smoke_and_debug_tiers_are_not_protocol_tiers(self):
        from hydra import compose, initialize_config_dir

        conf_dir = Path(__file__).resolve().parent / "conf"
        with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
            for name in ("test", "debug"):
                cfg = compose(config_name=name)
                assert str(cfg.experiment.tier) not in train.PROTOCOL_TIERS

    def test_protocol_tiers_are_all_present(self):
        for tier in (
            "calib_dmodel",
            "calib_harness",
            "calib_lr",
            "calib_lr_anchors",
            "tier1_core",
            "tier2_gates",
            "tier3_frozen",
            "tier4_lambda",
            "tier5_matched_bytes",
            "tier6_anchors",
            "tier_longT",
            "R1",
        ):
            assert tier in train.PROTOCOL_TIERS


class TestSeedMajorSweepOrder:
    """B14: tier sweeps run overwrite before mqar, seed-major: a partial
    tier still yields complete paired Delta_i at the primary endpoint
    (overwrite, class distant) before any MQAR run starts."""

    def test_tier_sweeps_are_seed_major(self):
        conf_dir = Path(__file__).resolve().parent / "conf"
        tier_files = sorted((conf_dir / "sweep").glob("tier*.yaml"))
        assert tier_files  # sanity: we found the tier files
        for path in tier_files:
            params = OmegaConf.select(
                OmegaConf.load(path), "hydra.sweeper.params", default={}
            )
            names = list(params.keys())
            grids = [sweeper_values(params[name]) for name in names]
            jobs = [dict(zip(names, combo)) for combo in itertools.product(*grids)]
            # Overwrite jobs come before MQAR jobs (Hydra's BasicSweeper
            # iterates the Cartesian product in key order).
            seen_mqar = False
            for job in jobs:
                task = str(job.get("task", ""))
                if task.startswith("mqar"):
                    seen_mqar = True
                else:
                    assert not seen_mqar, (path.name, job)
            # Within one task, the seed is non-decreasing along the job
            # order (seed-major: a partial tier covers every seed of the
            # first task before moving on).
            for task in {str(job.get("task", "")) for job in jobs}:
                seeds = [
                    int(job["seed"]) for job in jobs if str(job.get("task")) == task
                ]
                assert seeds == sorted(seeds), (path.name, task)

    def test_calibration_sweeps_are_untouched(self):
        """The calib_* files keep their own order (they are not tiers)."""
        conf_dir = Path(__file__).resolve().parent / "conf"
        calib_files = sorted((conf_dir / "sweep").glob("calib*.yaml"))
        assert calib_files
        for path in calib_files:
            params = OmegaConf.select(
                OmegaConf.load(path), "hydra.sweeper.params", default={}
            )
            names = list(params.keys())
            # The seed is NOT the first or second key in the calibration
            # sweeps (they vary the choice dimension fastest).
            assert "seed" not in names[:2], path.name


class TestCreateLoggers:
    """B15: the CSV logger writes next to the run records:
    runs/<task>/curves/<run_id>/version_0."""

    def test_csv_logger_under_runs_curves(self, tmp_path):
        runs_dir = tmp_path / "runs" / "mqar"
        cfg = OmegaConf.create(
            {
                "logging": {
                    "csv": {"enabled": True, "save_dir": str(runs_dir / "curves")},
                    "wandb": {"enabled": False},
                }
            }
        )
        loggers = train.create_loggers(cfg, "run-1")
        csv_logger = next(logger for logger in loggers if isinstance(logger, CSVLogger))
        log_dir = Path(csv_logger.log_dir)
        assert log_dir == runs_dir / "curves" / "run-1" / "version_0"

    def test_csv_disabled_returns_only_wandb(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "logging": {
                    "csv": {"enabled": False, "save_dir": "logs/csv"},
                    "wandb": {"enabled": False},
                }
            }
        )
        assert train.create_loggers(cfg, "run-1") is None

    def test_wandb_logger_keeps_the_project(self, tmp_path):
        from lightning.pytorch.loggers import WandbLogger

        cfg = OmegaConf.create(
            {
                "logging": {
                    "csv": {"enabled": False, "save_dir": "logs/csv"},
                    "wandb": {
                        "enabled": True,
                        "project": "MultistateGDN2",
                        "entity": None,
                    },
                },
                "experiment": {"name": "multistate_gdn2_synthetic_pilot"},
            }
        )
        loggers = train.create_loggers(cfg, "run-1")
        wandb_logger = next(
            logger for logger in loggers if isinstance(logger, WandbLogger)
        )
        # WandbLogger.name is the project name (the run name lives on
        # wandb's Run, which needs a live session).
        assert wandb_logger.name == "MultistateGDN2"
