"""
Tests for the frozen-protocol LR lookup: a null-safe walk treats a
missing or empty table node as absent, and attention has its own row.

Target: experiments/03_multistate_gdn2/02_structured_recall_pilot/01_train.py
Run with: pytest experiments/03_multistate_gdn2/02_structured_recall_pilot/test_frozen_lr.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_PILOT = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("pilot_01_train", _PILOT / "01_train.py")
_01 = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_01)
_frozen_lr = _01._frozen_lr


def _cfg(task="mqar", layout="ws", gate="full", mixer=None):
    cfg = OmegaConf.create(
        {
            "task": {"name": task},
            "model": {"layout": layout, "gate_mode": gate},
        }
    )
    if mixer is not None:
        cfg.model.mixer = mixer
    return cfg


def _frozen(entries):
    return OmegaConf.create({"lr_table": entries or {}})


class TestFrozenLR:
    def test_empty_table_raises(self):
        with pytest.raises(RuntimeError, match="fill conf/frozen_protocol"):
            _frozen_lr(_cfg(), _frozen({}))

    def test_null_node_raises_instead_of_attributexrr(self):
        # A hand-edited node present but null: the walk treats it as absent.
        frozen = OmegaConf.create(
            {"lr_table": {"mqar": {"gated_delta": {"ws": {"full": None}}}}}
        )
        with pytest.raises(RuntimeError, match="fill conf/frozen_protocol"):
            _frozen_lr(_cfg(), frozen)

    def test_missing_gate_raises(self):
        # The frozen table holds only ``kda``; a ``full`` run has no entry.
        frozen = _frozen({"mqar": {"gated_delta": {"ws": {"kda": 1e-3}}}})
        with pytest.raises(RuntimeError, match="fill conf/frozen_protocol"):
            _frozen_lr(_cfg(gate="full"), frozen)
        # The present gate reads fine.
        assert _frozen_lr(_cfg(gate="kda"), frozen) == 1e-3

    def test_gated_delta_lookup(self):
        frozen = _frozen({"mqar": {"gated_delta": {"ws": {"full": 3e-3, "kda": 1e-3}}}})
        assert _frozen_lr(_cfg(mixer="gated_delta"), frozen) == 3e-3

    def test_attention_uses_its_own_row(self):
        frozen = _frozen({"mqar": {"attention": 6e-3}})
        # The attention run has no layout/gate row.
        assert _frozen_lr(_cfg(mixer="attention"), frozen) == 6e-3
        with pytest.raises(RuntimeError):
            _frozen_lr(_cfg(mixer="attention"), _frozen({"mqar": {}}))


def _frozen_weight_decay(cfg, frozen):
    return _01._frozen_weight_decay(cfg, frozen)


def _frozen_table(entries, table="lr_table"):
    return OmegaConf.create({table: entries or {}})


def _harness_cfg(
    harness: str,
    d_model: int,
    tier: str = "tier1_mad_single",
    num_mixers: int = 2,
):
    # Each harness's model matches its real config: the MAD harness has no
    # layer pattern / state mixer (and does have num_mixers), so a check
    # that wrongly reads those keys for MAD can fail.
    model = {
        "d_model": d_model,
        "mixer": "gated_delta",
        "layout": "single",
        "gate_mode": "full",
    }
    if harness == "zoology":
        model.update({"layer_pattern": "zoology", "state_mixer": "none"})
    else:
        model["num_mixers"] = num_mixers
    return OmegaConf.create(
        {
            "experiment": {"tier": tier},
            "harness": {"name": harness},
            "model": model,
        }
    )


class TestFrozenWeightDecay:
    def test_mirrors_the_lr_lookup(self):
        frozen = _frozen_table(
            {"mqar": {"gated_delta": {"ws": {"full": 0.1, "kda": 0.0}}}},
            table="weight_decay_table",
        )
        assert _frozen_weight_decay(_cfg(gate="full"), frozen) == 0.1
        assert _frozen_weight_decay(_cfg(gate="kda"), frozen) == 0.0

    def test_attention_uses_its_own_row(self):
        frozen = _frozen_table({"mqar": {"attention": 0.0}}, table="weight_decay_table")
        assert _frozen_weight_decay(_cfg(mixer="attention"), frozen) == 0.0

    def test_missing_raises(self):
        with pytest.raises(RuntimeError, match="weight_decay_table"):
            _frozen_weight_decay(_cfg(), _frozen_table({}, table="weight_decay_table"))


class TestFrozenChoicesPerHarness:
    def test_mad_passes_when_only_zoology_frozen(self):
        frozen = OmegaConf.create(
            {
                "harness_choices": {
                    "zoology": {"d_model": 16, "layer_pattern": "zoology"},
                    "mad": {"d_model": None},
                }
            }
        )
        _01.check_frozen_choices(_harness_cfg("mad", d_model=128), frozen)

    def test_mad_wrong_d_model_refused(self):
        frozen = OmegaConf.create({"harness_choices": {"mad": {"d_model": 128}}})
        with pytest.raises(RuntimeError, match="d_model"):
            _01.check_frozen_choices(_harness_cfg("mad", d_model=64), frozen)

    def test_zoology_is_checked_as_before(self):
        frozen = OmegaConf.create(
            {
                "harness_choices": {
                    "zoology": {
                        "d_model": 16,
                        "layer_pattern": "zoology",
                        "state_mixer": "none",
                    }
                }
            }
        )
        _01.check_frozen_choices(_harness_cfg("zoology", d_model=16), frozen)
        with pytest.raises(RuntimeError, match="d_model"):
            _01.check_frozen_choices(_harness_cfg("zoology", d_model=32), frozen)

    def test_smoke_tiers_are_not_checked(self):
        frozen = OmegaConf.create({"harness_choices": {"mad": {"d_model": 128}}})
        _01.check_frozen_choices(_harness_cfg("mad", d_model=64, tier="smoke"), frozen)


def _task_cfg(task="mad_fuzzy", seq_len=512, tier="tier1_mad_single"):
    """A protocol-tier cfg with a task block (for the length freeze)."""
    cfg = _harness_cfg("mad", d_model=128, tier=tier)
    cfg.task = OmegaConf.create({"name": task, "seq_len": seq_len})
    return cfg


def _complete_frozen_mad(length=512):
    """A fully-frozen protocol the MAD run side accepts."""
    return OmegaConf.create(
        {
            "frozen": True,
            "harness_choices": {"mad": {"d_model": 128, "num_mixers": 2}},
            "task_settings": {"mad_fuzzy": {"seq_len": length}},
        }
    )


class TestFrozenCompleteness:
    def test_frozen_true_missing_harness_entry_raises(self):
        frozen = OmegaConf.create(
            {
                "frozen": True,
                "harness_choices": {},
                "task_settings": {"mad_fuzzy": {"seq_len": 512}},
            }
        )
        with pytest.raises(RuntimeError, match="harness_choices"):
            _01.check_frozen_choices(_task_cfg(), frozen)

    def test_frozen_true_null_key_raises(self):
        frozen = OmegaConf.create(
            {
                "frozen": True,
                "harness_choices": {"mad": {"d_model": None, "num_mixers": 2}},
                "task_settings": {"mad_fuzzy": {"seq_len": 512}},
            }
        )
        with pytest.raises(RuntimeError, match="d_model"):
            _01.check_frozen_choices(_task_cfg(), frozen)

    def test_frozen_empty_entry_is_refused(self):
        # An entry edited down to {} must not pass the completeness check.
        frozen = OmegaConf.create(
            {
                "frozen": True,
                "harness_choices": {"mad": {}},
                "task_settings": {"mad_fuzzy": {"seq_len": 512}},
            }
        )
        with pytest.raises(RuntimeError, match="still null"):
            _01.check_frozen_choices(_task_cfg(), frozen)

    def test_frozen_true_missing_task_settings_raises(self):
        frozen = OmegaConf.create(
            {"frozen": True, "harness_choices": {"mad": {"d_model": 128, "num_mixers": 2}}}
        )
        with pytest.raises(RuntimeError, match="task_settings"):
            _01.check_frozen_choices(_task_cfg(), frozen)

    def test_unfrozen_null_passes(self):
        frozen = OmegaConf.create(
            {
                "frozen": False,
                "harness_choices": {"mad": {"d_model": None, "num_mixers": 2}},
                "task_settings": {"mad_fuzzy": {"seq_len": None}},
            }
        )
        _01.check_frozen_choices(_task_cfg(), frozen)

    def test_frozen_length_mismatch_refused(self):
        frozen = _complete_frozen_mad(length=512)
        # A protocol-tier run at a length other than the frozen one.
        with pytest.raises(RuntimeError, match="seq_len"):
            _01.check_frozen_choices(_task_cfg(seq_len=256), frozen)
        # The frozen length passes.
        _01.check_frozen_choices(_task_cfg(seq_len=512), frozen)

    def test_num_mixers_is_compared(self):
        # A MAD run with the wrong frozen num_mixers is refused even when
        # unfrozen (the key is set, so it is compared).
        frozen = _complete_frozen_mad()
        with pytest.raises(RuntimeError, match="num_mixers"):
            _01.check_frozen_choices(_harness_cfg("mad", d_model=128, num_mixers=1), frozen)
        _01.check_frozen_choices(_harness_cfg("mad", d_model=128, num_mixers=2), frozen)

    def test_frozen_entity_field_with_no_entry_passes(self):
        # entity_field has no FROZEN_TASK_KEYS, so its phase-2 run needs no
        # task_settings entry once frozen; a MAD entry is present for the
        # other task but not required here.
        frozen = OmegaConf.create(
            {
                "frozen": True,
                "harness_choices": {
                    "zoology": {
                        "d_model": 16,
                        "layer_pattern": "zoology",
                        "state_mixer": "none",
                    }
                },
                "task_settings": {"mad_fuzzy": {"seq_len": 512}},
            }
        )
        cfg = _harness_cfg("zoology", d_model=16, tier="tier1_mad_single")
        cfg.task = OmegaConf.create({"name": "entity_field", "seq_len": 256})
        _01.check_frozen_choices(cfg, frozen)  # no raise

    def test_frozen_mad_missing_task_entry_refused(self):
        # MAD has FROZEN_TASK_KEYS, so a frozen phase-2 run with no entry is
        # refused (an unfilled entry would otherwise never be checked).
        frozen = OmegaConf.create(
            {"frozen": True, "harness_choices": {"mad": {"d_model": 128, "num_mixers": 2}}}
        )
        with pytest.raises(RuntimeError, match="task_settings"):
            _01.check_frozen_choices(_task_cfg(), frozen)


def _frozen_yaml():
    return OmegaConf.load(
        Path(__file__).resolve().parent / "conf" / "frozen_protocol.yaml"
    )


class TestFixedSetPadding:
    def test_derived_padding_passes(self):
        cfg = OmegaConf.create(
            {"task": {"seq_len": 512, "max_seq_len": 528}, "model": {"chunk_size": 16}}
        )
        assert _01.assert_fixed_set_padding(cfg) == 528

    def test_mismatched_padding_refused(self):
        # 1024 is a valid chunk multiple but not the derived 528: a hand-set
        # value must be refused, not silently change the padding / t_max.
        cfg = OmegaConf.create(
            {"task": {"seq_len": 512, "max_seq_len": 1024}, "model": {"chunk_size": 16}}
        )
        with pytest.raises(ValueError, match="528"):
            _01.assert_fixed_set_padding(cfg)


class TestFrozenChoicesAndYaml:
    def test_yaml_lists_exactly_the_protocol_frozen_keys(self):
        # The YAML entry keys must match the protocol's required keys exactly,
        # so a required key cannot be added to one and not the other.
        yaml = _frozen_yaml()
        from structured_recall_analysis import protocol as _proto

        assert dict(yaml.harness_choices.get("zoology", {})) == dict(
            (k, None) for k in _proto.FROZEN_HARNESS_KEYS["zoology"]
        )
        assert dict(yaml.harness_choices.get("mad", {})) == dict(
            (k, None) for k in _proto.FROZEN_HARNESS_KEYS["mad"]
        )
        assert dict(yaml.task_settings.get("mad_fuzzy", {})) == dict(
            (k, None) for k in _proto.FROZEN_TASK_KEYS["mad_fuzzy"]
        )

    def test_mad_model_has_no_layer_pattern_keys(self):
        # The MAD fixture matches the real MAD model config (D2): it must not
        # carry layer_pattern / state_mixer, so a check that wrongly reads
        # those keys for MAD cannot be masked.
        mad = _harness_cfg("mad", d_model=128)
        assert "layer_pattern" not in mad.model
        assert "state_mixer" not in mad.model
        assert mad.model.get("num_mixers") == 2
