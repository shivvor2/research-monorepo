"""
Tests for the shared project package: the run-ID task-key option, the task
and harness registries, and the shared analysis readings.

Target: experiments/03_multistate_gdn2/common/
Run with: pytest experiments/03_multistate_gdn2/common/tests/test_common.py -v
"""

import pandas as pd
import pytest
from common.analysis import delta_int_table, gap_column, paired_gap_report
from common.harnesses import build_zoology
from common.registries import config_type_of_task, require_harness
from common.run_id import make_run_id
from omegaconf import OmegaConf

from research_lib.architectures.zoology_lm import BaseConv
from research_lib.data.synthetic import MQARConfig, OverwriteConfig, require_task
from research_lib.layers.attention import CausalSelfAttention, MultistateGatedDelta


def _cfg(*, reserved_filler=0, tier=None):
    return OmegaConf.create(
        {
            "task": {
                "name": "overwrite",
                "reserved_filler": reserved_filler,
            },
            "model": {
                "layout": "single",
                "gate_mode": "full",
                "lambda_mode": "mlp",
                "d_model": 32,
                "layer_pattern": "zoology",
                "state_mixer": "none",
                "expand_v": 1.0,
            },
            "optimizer": {"decay_scheme": "grouped"},
            "experiment": {"tier": tier},
            "seed": 0,
        }
    )


def _ef_cfg(**task_overrides):
    """A default entity-field config (the real task config's values)."""
    task = {
        "name": "entity_field",
        "vocab_size": 8192,
        "seq_len": 256,
        "num_fields": 4,
        "num_entities": [2, 4, 8],
        "update_fraction": 0.75,
        "placement_probs": [0.3333333333, 0.3333333333, 0.3333333334],
        "block_size": 16,
        "distant_min_blocks": 4,
        "reserved_filler": 512,
        "max_placement_attempts": 100,
    }
    task.update(task_overrides)
    return OmegaConf.create(
        {
            "task": task,
            "model": {
                "layout": "single",
                "gate_mode": "full",
                "lambda_mode": "mlp",
                "d_model": 32,
                "layer_pattern": "zoology",
                "state_mixer": "none",
                "expand_v": 1.0,
            },
            "optimizer": {"decay_scheme": "grouped"},
            "experiment": {"tier": None},
            "seed": 0,
        }
    )


class TestRunIdTaskKeys:
    def test_off_reproduce_same_id_for_task_key_difference(self):
        a = make_run_id(_cfg(reserved_filler=0), 0.003, include_task_keys=False)
        b = make_run_id(_cfg(reserved_filler=4), 0.003, include_task_keys=False)
        assert a == b

    def test_on_differs_by_task_key(self):
        a = make_run_id(_cfg(reserved_filler=0), 0.003, include_task_keys=True)
        b = make_run_id(_cfg(reserved_filler=4), 0.003, include_task_keys=True)
        assert a != b
        assert "-reserved_filler-4" in b

    def test_tier_leading_prefix(self):
        run_id = make_run_id(_cfg(tier="tier_a"), 0.003, include_task_keys=True)
        assert run_id.startswith("tier_a_")

    def test_default_mixer_and_harness_are_omitted(self):
        run_id = make_run_id(_cfg(), 0.003, include_task_keys=True)
        assert "-mix-" not in run_id
        assert "-h-" not in run_id
        assert "-wd-" not in run_id

    def test_non_default_mixer_and_harness_are_appended(self):
        cfg = _cfg()
        cfg.model.mixer = "attention"
        cfg.harness = {"name": "mad"}
        run_id = make_run_id(cfg, 0.003, include_task_keys=True)
        assert "-mix-attention" in run_id
        assert "-h-mad" in run_id

    def test_default_entity_field_config_has_no_task_suffix(self):
        # A YAML decimal that rounds the default (1/3) and a ListConfig
        # that equals the dataclass tuple are not changes.
        run_id = make_run_id(_ef_cfg(), 0.003, include_task_keys=True)
        assert "placement_probs" not in run_id
        assert "num_entities" not in run_id
        assert run_id == "entity_field-single-full-mlp-d32-lr0.003-s0"

    def test_changed_update_fraction_adds_a_suffix(self):
        run_id = make_run_id(
            _ef_cfg(update_fraction=0.5), 0.003, include_task_keys=True
        )
        assert "-update_fraction-0.5" in run_id


class TestRegistries:
    def test_config_types_come_from_the_general_registry(self):
        for name in ("mqar", "overwrite"):
            assert config_type_of_task(name) is require_task(name).config_type
        assert config_type_of_task("overwrite") is OverwriteConfig
        assert config_type_of_task("mqar") is MQARConfig

    def test_task_spec_carries_class_names(self):
        assert require_task("overwrite").class_names == (
            "same",
            "adjacent",
            "distant",
            "single",
        )
        assert require_task("mqar").class_names is None

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            config_type_of_task("nope")

    def test_unknown_harness_raises(self):
        with pytest.raises(ValueError):
            require_harness("nope")


def _harness_cfg(*, mixer=None):
    cfg = OmegaConf.create(
        {
            "model": {
                "vocab_size": 64,
                "d_model": 32,
                "num_heads": 2,
                "head_dim": 16,
                "expand_v": 1.0,
                "gate_mode": "full",
                "layout": "single",
                "lambda_mode": "mlp",
                "window": 4,
                "chunk_size": 16,
                "t_max": 32,
                "mixer_mode": "chunk",
                "output_gate_activation": "sigmoid",
                "n_layer": 2,
                "layer_pattern": "zoology",
                "base_conv_kernel": 3,
                "state_mixer": "none",
                "ff_mult": 2.0,
            }
        }
    )
    if mixer is not None:
        cfg.model.mixer = mixer
    return cfg


def _mixer_kinds(model) -> list:
    return [block.mixer for block in model.blocks]


def _managed_mixers(model) -> list:
    # The harness's ``zoology`` pattern alternates BaseConv conv blocks
    # with the managed mixer; the managed ones are what the config chose.
    return [m for m in _mixer_kinds(model) if not isinstance(m, BaseConv)]


class TestHarnessBuildsTheRunsMixer:
    def test_default_builds_gated_delta_only(self):
        model = build_zoology(_harness_cfg())
        managed = _managed_mixers(model)
        assert managed
        assert all(isinstance(m, MultistateGatedDelta) for m in managed)
        assert not any(isinstance(m, CausalSelfAttention) for m in managed)

    def test_attention_config_builds_attention_only(self):
        model = build_zoology(_harness_cfg(mixer="attention"))
        managed = _managed_mixers(model)
        assert managed
        assert all(isinstance(m, CausalSelfAttention) for m in managed)
        assert not any(isinstance(m, MultistateGatedDelta) for m in managed)

    def test_attention_ignores_the_gated_delta_head_dim(self):
        # head_dim is the gated-delta key width and must not set the
        # attention control's rotary dimension (which stays the head width).
        cfg = _harness_cfg(mixer="attention")
        cfg.model.head_dim = 10  # not a divisor of 32, irrelevant to attention
        model = build_zoology(cfg)
        for m in _managed_mixers(model):
            assert m.rotary_dim in (None, 16)

    def test_unknown_mixer_raises(self):
        with pytest.raises(ValueError):
            build_zoology(_harness_cfg(mixer="nope"))


class TestAnalysis:
    def test_gap_column(self):
        df = pd.DataFrame(
            {
                "layout": ["single", "single", "ws", "ws", "ll", "ll"],
                "gate_mode": ["full", "kda", "full", "kda", "full", "kda"],
                "metric": [0.9, 0.8, 0.7, 0.6, 0.55, 0.6],
            }
        )
        gap = gap_column(df, "metric")
        # single: 0.1; ws: 0.1; ll: -0.05, in the original row order.
        assert list(gap.round(4)) == [0.1, 0.1, 0.1, 0.1, -0.05, -0.05]

    def test_paired_gap_report_over_seeds(self):
        df = pd.DataFrame(
            {
                "layouts": ["single"] * 4,
                "gate_mode": ["full", "kda"] * 2,
                "seed": [0, 0, 1, 1],
                "metric": [0.6, 0.4, 0.7, 0.5],
            }
        ).rename(columns={"layouts": "layout"})
        report = paired_gap_report(df, "metric")
        assert report["n_seeds"].iloc[0] == 2
        assert report["mean_gap"].iloc[0] == pytest.approx(0.2)
        assert report["mean_high"].iloc[0] == pytest.approx(0.65)
        assert report["mean_low"].iloc[0] == pytest.approx(0.45)

    def test_delta_int_table_over_an_arbitrary_layout_list(self):
        # A full factorial of gate x layout over two seeds: the interaction
        # is 0 for both ws and ll under this symmetric data.
        rows = []
        for layout in ("single", "ws", "ll"):
            for gate in ("full", "kda"):
                for seed in (0, 1):
                    rows.append((layout, gate, seed, 0.5 + 0.1 * (seed + 1)))
        df = pd.DataFrame(rows, columns=["layout", "gate_mode", "seed", "metric"])
        out = delta_int_table(df, "metric", layouts=("ws", "ll"))
        assert set(out["layout"]) == {"ws", "ll"}
        for _, row in out.iterrows():
            assert row["mean"] == pytest.approx(0.0, abs=1e-9)

    def test_delta_int_catches_a_ws_only_gate_effect(self):
        # Raise only ws/full by 0.4 across both seeds: the interaction is
        # +0.4 for ws and 0 for ll (a sign/pairing slip would show here).
        rows = []
        for layout in ("single", "ws", "ll"):
            for gate in ("full", "kda"):
                for seed in (0, 1):
                    value = 0.5
                    if layout == "ws" and gate == "full":
                        value += 0.4
                    rows.append((layout, gate, seed, value))
        df = pd.DataFrame(rows, columns=["layout", "gate_mode", "seed", "metric"])
        out = delta_int_table(df, "metric", layouts=("ws", "ll"))
        ws_row = out[out["layout"] == "ws"].iloc[0]
        ll_row = out[out["layout"] == "ll"].iloc[0]
        assert ws_row["mean"] == pytest.approx(0.4, abs=1e-9)
        assert ll_row["mean"] == pytest.approx(0.0, abs=1e-9)
