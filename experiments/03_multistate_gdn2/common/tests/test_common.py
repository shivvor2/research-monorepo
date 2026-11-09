"""
Tests for the shared project package: the run-ID task-key option, the task
and harness registries, and the shared analysis readings.

Target: experiments/03_multistate_gdn2/common/
Run with: pytest experiments/03_multistate_gdn2/common/tests/test_common.py -v
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from common.analysis import delta_int_table, gap_column, paired_gap_report
from common.harnesses import build_mad, build_zoology
from common.records import flatten as records_flatten
from common.registries import config_type_of_task, require_harness
from common.run_id import HASH_EXCLUDE, make_run_id
from omegaconf import OmegaConf

from research_lib.architectures.zoology_lm import BaseConv
from research_lib.data.synthetic import MQARConfig, OverwriteConfig, require_task
from research_lib.layers.attention import CausalSelfAttention, MultistateGatedDelta
from research_lib.training.callbacks import config_hash


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

    def test_attention_config_drops_the_gated_delta_segment(self):
        cfg = _ef_cfg()
        cfg.model.mixer = "attention"
        run_id = make_run_id(cfg, 0.003, include_task_keys=True)
        assert run_id == "entity_field-d32-lr0.003-mix-attention-s0"

    def test_attention_ignores_layout_gate_and_lambda_in_the_id(self):
        base = _ef_cfg()
        base.model.mixer = "attention"
        base_id = make_run_id(base, 0.003, include_task_keys=True)
        for key, value in (
            ("layout", "ws"),
            ("gate_mode", "kda"),
            ("lambda_mode", "gated"),
        ):
            cfg = _ef_cfg()
            cfg.model.mixer = "attention"
            cfg.model[key] = value
            assert make_run_id(cfg, 0.003, include_task_keys=True) == base_id

    def test_attention_ignores_expand_v_in_the_id(self):
        cfg = _ef_cfg()
        cfg.model.mixer = "attention"
        cfg.model.expand_v = 2.0
        assert make_run_id(cfg, 0.003, include_task_keys=True) == (
            "entity_field-d32-lr0.003-mix-attention-s0"
        )

    def test_attention_same_id_different_identity_is_refused(self, tmp_path):
        # Two attention runs that differ only in a gated-delta setting share
        # a run ID; their config hashes still differ, so the identity check
        # refuses the second loudly instead of overwriting the record.
        from common.run_lifecycle import create_callbacks_reset_curves

        a = _ef_cfg()
        a.model.mixer = "attention"
        b = _ef_cfg()
        b.model.mixer = "attention"
        b.model.gate_mode = "kda"
        run_id = make_run_id(a, 0.003, include_task_keys=True)
        assert run_id == make_run_id(b, 0.003, include_task_keys=True)

        runs = tmp_path / "runs"
        curves = tmp_path / "curves"
        runs.mkdir(parents=True)
        curves_dir = curves / run_id
        curves_dir.mkdir(parents=True)
        (curves_dir / "metrics.csv").write_text("step,val/acc\n0,0.5\n")
        identity = (
            config_hash(OmegaConf.to_container(a, resolve=True), HASH_EXCLUDE),
            a.seed,
        )
        (runs / f"{run_id}.json").write_text(
            json.dumps({"config_hash": identity[0], "seed": identity[1]})
        )
        c = OmegaConf.create(
            {
                "seed": 0,
                "output": {"runs_dir": str(runs)},
                "training": {
                    "early_stop_threshold": 0.99,
                    "collect_mixer_stats": False,
                },
                "logging": {"csv": {"enabled": True, "save_dir": str(curves)}},
                "optimizer": {"decay_scheme": "grouped"},
                "experiment": {"name": "t", "tier": "tier_x"},
            }
        )
        with pytest.raises(FileExistsError):
            create_callbacks_reset_curves(c, run_id, {}, lr=0.003)


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


def _mad_model_cfg(**overrides):
    base = {
        "d_model": 128,
        "num_mixers": 2,
        "mixer": "gated_delta",
        "attention_num_heads": 16,
        "num_heads": 8,
        "expand_v": 1.0,
        "chunk_size": 16,
        "window": 8,
        "t_max": 528,
        "gate_mode": "full",
        "layout": "single",
        "lambda_mode": "mlp",
        "mixer_mode": "chunk",
        "output_gate_activation": "silu",
        "vocab_size": 16,
    }
    base.update(overrides)
    return OmegaConf.create({"model": base})


class TestBuildMadAttention:
    def test_attention_uses_its_own_head_layout(self):
        cfg = _mad_model_cfg(mixer="attention")
        model = build_mad(cfg)
        mixer = model.blocks[0].mixer
        assert isinstance(mixer, CausalSelfAttention)
        # MAD's attention layout: 16 heads of width 8, rotary on the full
        # head width — distinct from the gated-delta 8 heads of width 16.
        assert mixer.num_heads == 16
        assert mixer.rotary_dim == 8
        assert 128 // mixer.num_heads == 8

    def test_gated_delta_uses_the_gated_layout(self):
        cfg = _mad_model_cfg(mixer="gated_delta")
        model = build_mad(cfg)
        mixer = model.blocks[0].mixer
        assert isinstance(mixer, MultistateGatedDelta)
        # The gated-delta layer keeps its own 8 heads of width 16.
        assert mixer.projections.num_heads == 8
        assert mixer.projections.head_k_dim == 16

    def test_attention_heads_must_divide_width(self):
        from common.harnesses import build_mad

        cfg = _mad_model_cfg(mixer="attention", attention_num_heads=12)
        with pytest.raises(ValueError, match="attention_num_heads"):
            build_mad(cfg)

    def test_gated_delta_ignores_undivisible_attention_heads(self):
        # d=104 is even in gated heads of width 8 but not divisible by the
        # attention count of 16; only attention validates/derives that
        # layout, so the gated-delta run builds.
        from common.harnesses import build_mad

        cfg = _mad_model_cfg(d_model=104, mixer="gated_delta", attention_num_heads=16)
        model = build_mad(cfg)
        assert isinstance(model.blocks[0].mixer, MultistateGatedDelta)


class TestRunLength:
    def test_fixed_set_by_epochs(self):
        from common.run_lifecycle import RunLength, run_length

        cfg = OmegaConf.create(
            {
                "training": {"epochs": 200, "batch_size": 128, "val_every_epochs": 1},
                "task": {"num_train": 12800},
            }
        )
        length = run_length(cfg, require_task("mad_fuzzy").regime)
        assert length == RunLength(
            steps_per_epoch=100, max_steps=20000, val_check_interval=100
        )

    def test_streaming_by_steps(self):
        from common.run_lifecycle import run_length

        from research_lib.data.synthetic import TaskRegime

        cfg = OmegaConf.create(
            {"training": {"max_steps": 500, "val_check_interval": 50}}
        )
        length = run_length(cfg, TaskRegime.STREAMING)
        assert length.max_steps == 500 and length.val_check_interval == 50
        assert length.steps_per_epoch is None

    def test_fixed_set_requires_epochs(self):
        from common.run_lifecycle import run_length

        from research_lib.data.synthetic import TaskRegime

        cfg = OmegaConf.create(
            {"training": {"batch_size": 2}, "task": {"num_train": 10}}
        )
        with pytest.raises(ValueError, match="epochs"):
            run_length(cfg, TaskRegime.FIXED_SET)

    def test_fixed_set_on_multiple_devices_is_an_error(self):
        from common.run_lifecycle import RunLength, run_length

        from research_lib.data.synthetic import TaskRegime

        cfg = OmegaConf.create(
            {
                "training": {
                    "epochs": 2,
                    "batch_size": 2,
                    "val_every_epochs": 1,
                    "devices": 2,
                },
                "task": {"num_train": 10},
            }
        )
        with pytest.raises(RuntimeError, match="single-device"):
            run_length(cfg, TaskRegime.FIXED_SET)
        # ``auto`` is not resolved (a multi-GPU machine can mean several
        # devices); it still raises the same clear error, not a confusing
        # ``int('auto')`` ValueError.
        cfg.training.devices = "auto"
        with pytest.raises(RuntimeError, match="auto"):
            run_length(cfg, TaskRegime.FIXED_SET)
        # Single device is allowed.
        cfg.training.devices = 1
        length = run_length(cfg, TaskRegime.FIXED_SET)
        assert length == RunLength(
            steps_per_epoch=5, max_steps=10, val_check_interval=5
        )

    def test_epoch_cosine_on_streaming_is_an_error(self):
        import torch as _torch
        from common.run_lifecycle import build_module

        from research_lib.data.synthetic import TaskRegime

        cfg = OmegaConf.create(
            {
                "optimizer": {
                    "schedule": "epoch_cosine",
                    "min_lr": 1e-6,
                    "decay_scheme": "grouped",
                    "betas": (0.9, 0.999),
                },
                "training": {"max_steps": 10, "epochs": 5, "batch_size": 2},
                "task": {"num_train": 20},
            }
        )
        model = _torch.nn.Linear(4, 4)
        with pytest.raises(ValueError, match="fixed-set"):
            build_module(
                cfg, model, 1e-3, (2,), weight_decay=0.0, regime=TaskRegime.STREAMING
            )


class TestExtraMetrics:
    def test_resolves_named_metrics(self):
        from common.registries import build_extra_metrics

        metrics = build_extra_metrics(
            ["macro_token_accuracy", "token_accuracy"], vocab_size=16
        )
        assert len(metrics) == 2
        assert type(metrics[0]).__name__ == "MacroTokenAccuracy"

    def test_empty_is_empty(self):
        from common.registries import build_extra_metrics

        assert build_extra_metrics((), 16) == ()

    def test_unknown_raises(self):
        from common.registries import build_extra_metrics

        with pytest.raises(ValueError, match="unknown extra metric"):
            build_extra_metrics(["no_such_metric"], 16)


class TestFlattenWeightDecayFallback:
    def test_top_level_field_wins(self):
        row = records_flatten(
            {
                "run_id": "r1",
                "weight_decay": 0.1,
                "config": {"optimizer": {"weight_decay": 0.0}},
                "lr_resolved": 1e-3,
            }
        )
        assert row["weight_decay"] == 0.1

    def test_missing_top_level_falls_back_to_config(self):
        # A record written before the resolved weight decay became a
        # top-level field still lands in its calibration cell.
        row = records_flatten(
            {
                "run_id": "r2",
                "config": {"optimizer": {"weight_decay": 0.0}},
                "lr_resolved": 1e-3,
            }
        )
        assert row["weight_decay"] == 0.0

    def test_neither_keeps_none(self):
        # ``0.0`` must survive a truthiness check, so a missing value stays
        # ``None`` (which the calibration reading then refuses).
        row = records_flatten({"run_id": "r3", "config": {}, "lr_resolved": 1e-3})
        assert row["weight_decay"] is None

    def test_seq_len_read_from_the_task_config(self):
        row = records_flatten(
            {"config": {"task": {"seq_len": 512}}, "lr_resolved": 1e-3}
        )
        assert row["seq_len"] == 512


class TestMonitoredValueHistory:
    def _trainer(self, *, sanity=False, fn=None, metrics=None):
        from lightning.pytorch.trainer.states import TrainerFn

        return SimpleNamespace(
            sanity_checking=sanity,
            state=SimpleNamespace(fn=fn if fn is not None else TrainerFn.FITTING),
            callback_metrics=metrics or {},
        )

    def test_sanity_and_out_of_fit_validations_are_skipped(self):
        from common.run_lifecycle import MonitoredValueHistory

        cb = MonitoredValueHistory("val/acc")
        # The coach's default sanity check (and any validation run outside a
        # fit, e.g. in test/validate) must not be recorded.
        cb.on_validation_end(
            self._trainer(sanity=True, metrics={"val/acc": torch.tensor(0.1)}), None
        )
        from lightning.pytorch.trainer.states import TrainerFn

        cb.on_validation_end(
            self._trainer(fn=TrainerFn.TESTING, metrics={"val/acc": torch.tensor(0.2)}),
            None,
        )
        assert cb.history == []
        cb.on_validation_end(
            self._trainer(metrics={"val/acc": torch.tensor(0.3)}), None
        )
        assert cb.history[0] == pytest.approx(0.3)

    def test_default_sanity_steps_record_one_value_per_epoch(self):
        from common.run_lifecycle import MonitoredValueHistory

        cb = MonitoredValueHistory("val/acc")
        trainer = self._trainer(metrics={"val/acc": torch.tensor(0.5)})
        # Two sanity-check validations (the default num_sanity_val_steps=2)
        # are skipped, then one fit validation per epoch is recorded.
        for _ in range(2):
            cb.on_validation_end(
                self._trainer(sanity=True, metrics={"val/acc": torch.tensor(0.0)}),
                None,
            )
        cb.on_validation_end(trainer, None)  # epoch 1
        cb.on_validation_end(trainer, None)  # epoch 2
        assert cb.history == [0.5, 0.5]

    def test_state_dict_load_state_dict_keeps_history(self):
        from common.run_lifecycle import MonitoredValueHistory

        cb = MonitoredValueHistory("val/acc")
        cb.history = [0.1, 0.2, 0.3]
        resumed = MonitoredValueHistory("val/acc")  # history empty until load
        resumed.load_state_dict(cb.state_dict())
        assert resumed.history == [0.1, 0.2, 0.3]

    def test_summary_reads_the_callback_history(self):
        from common.run_lifecycle import (
            MonitoredValueHistory,
            monitored_history_summary,
        )

        cb = MonitoredValueHistory("val/acc")
        cb.history = [0.4, 0.5, 0.6, 0.45, 0.5, 0.55]
        trainer = SimpleNamespace(callbacks=[cb])
        summary = monitored_history_summary(trainer, None)
        assert summary["monitored_first"] == 0.4
        assert summary["monitored_max"] == 0.6
        assert summary["monitored_last_half_min"] == 0.45
        assert summary["monitored_last_half_max"] == 0.55


class TestMadSeqLenRunId:
    def test_length_change_alters_the_id(self):
        # The MAD default seq_len is 512 (no suffix); 256 is a non-default
        # task field, so its run ID differs from the 512 one.
        def cfg_for(seq_len):
            return OmegaConf.create(
                {
                    "task": {"name": "mad_fuzzy", "seq_len": seq_len},
                    "model": {
                        "layout": "single",
                        "gate_mode": "full",
                        "lambda_mode": "mlp",
                        "d_model": 128,
                    },
                    "optimizer": {"decay_scheme": "grouped"},
                    "experiment": {"tier": "calib_mad_single"},
                    "seed": 100,
                }
            )

        a = make_run_id(cfg_for(512), 0.003, include_task_keys=True)
        b = make_run_id(cfg_for(256), 0.003, include_task_keys=True)
        assert a != b
        assert "-seq_len-256" in b
        assert "-seq_len-512" not in a
