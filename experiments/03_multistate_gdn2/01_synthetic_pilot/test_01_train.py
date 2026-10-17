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
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

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
