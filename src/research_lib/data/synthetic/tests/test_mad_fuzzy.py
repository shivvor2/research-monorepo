"""
Tests for the MAD fuzzy in-context recall generator and batch mapping.

Covers: bit-for-bit reproduction of MAD's generated instances (training and
test style, across the stored settings including noise and single-query),
the single-generator train/test split, config validation, and the mapping
of a MAD row to the repository's unshifted-label batch format (training /
evaluation labels, right-padding, slice identity).

The fixtures were generated with MAD's own
:func:`generate_fuzzy_in_context_recall_instance` at ``athms/mad-lab``,
commit ``0f49a45``: ``np.random.default_rng(seed)`` for seeds 0 and 1, one
fresh generator per split for the per-setting fixtures and one shared
generator for the reference-split fixtures, at the settings and counts
listed below (the noise / single-query pair uses fewer test instances).

Run with: pytest src/research_lib/data/synthetic/tests/test_mad_fuzzy.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from ..datamodule import SyntheticRecallDataModule
from ..mad_fuzzy import (
    MadFuzzyConfig,
    fuzzy_recall_dataset,
    fuzzy_recall_instance,
    mad_batch_row,
    mad_reference_split,
)
from ..registry import TaskRegime, require_task

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The stored settings (fixture-generator side mirrors these; see the
# reference directory's README): name -> MadFuzzyConfig kwargs.
SETTINGS = {
    "v16_l128": {
        "vocab_size": 16,
        "seq_len": 128,
        "k_motif_size": 3,
        "v_motif_size": 3,
        "multi_query": True,
        "noise_vocab_size": 0,
        "frac_noise": 0.0,
    },
    "v16_l512": {
        "vocab_size": 16,
        "seq_len": 512,
        "k_motif_size": 3,
        "v_motif_size": 3,
        "multi_query": True,
        "noise_vocab_size": 0,
        "frac_noise": 0.0,
    },
    "v32_l128": {
        "vocab_size": 32,
        "seq_len": 128,
        "k_motif_size": 3,
        "v_motif_size": 3,
        "multi_query": True,
        "noise_vocab_size": 0,
        "frac_noise": 0.0,
    },
    "noise_single_v48_l128": {
        "vocab_size": 48,
        "seq_len": 128,
        "k_motif_size": 3,
        "v_motif_size": 3,
        "multi_query": False,
        "noise_vocab_size": 16,
        "frac_noise": 0.2,
    },
}
# Instances per split per setting (the noise/single-query pair has fewer test).
N_TRAIN = {"v16_l128": 20, "v16_l512": 20, "v32_l128": 20, "noise_single_v48_l128": 20}
N_TEST = {"v16_l128": 20, "v16_l512": 20, "v32_l128": 20, "noise_single_v48_l128": 10}


def _shift(config, inputs, targets, is_training):
    """MAD's own shift of our pre-shift arrays (per token, the last axis):
    ``inputs[..., :-1]`` paired with ``inputs[..., 1:]`` (training) or
    ``targets[..., 1:]`` (test)."""
    return inputs[..., :-1], (inputs if is_training else targets)[..., 1:]


class TestMatchesMAD:
    def test_instances_reproduce_mad_bit_for_bit(self):
        # For every seed and setting, regeneration from a fresh
        # ``default_rng(seed)`` reproduces the stored MAD outputs exactly.
        for seed in (0, 1):
            for name, kwargs in SETTINGS.items():
                fixture = np.load(FIXTURES / f"mad_fuzzy_s{seed}_{name}.npz")
                for split in ("train", "test"):
                    is_training = split == "train"
                    num = N_TRAIN[name] if is_training else N_TEST[name]
                    cfg = MadFuzzyConfig(**kwargs)
                    inputs, targets = fuzzy_recall_dataset(
                        cfg,
                        num,
                        is_training,
                        np.random.default_rng(seed),
                    )
                    x, y = _shift(cfg, inputs, targets, is_training)
                    fx = fixture[f"{split}_x"]
                    fy = fixture[f"{split}_y"]
                    assert x.shape == fx.shape and y.shape == fy.shape
                    assert np.array_equal(x, fx), f"seed {seed} {name} {split} inputs"
                    assert np.array_equal(y, fy), f"seed {seed} {name} {split} targets"

    def test_padding_produces_mad_lengths(self):
        # Left padding yields length seq_len + 1, and (multi-query) the
        # pad token is vocab_size - 1.
        cfg = MadFuzzyConfig(vocab_size=16, seq_len=128)
        for training in (True, False):
            inputs, _ = fuzzy_recall_instance(cfg, training, np.random.default_rng(0))
            assert len(inputs) == 129
            assert inputs[0] in (cfg.vocab_size - 1,)


class TestReferenceSplit:
    def test_single_generator_training_then_test(self):
        # ``mad_reference_split`` uses ONE generator (the training set is
        # drawn first, then the test set continues the same stream); its
        # shifted outputs reproduce the reference split fixture.
        for seed in (0, 1):
            cfg = MadFuzzyConfig(seq_len=128)
            split = mad_reference_split(cfg, 20, 10, seed=seed)
            fixture = np.load(FIXTURES / f"mad_fuzzy_reference_split_s{seed}.npz")
            for name in ("train", "test"):
                is_training = name == "train"
                x, y = _shift(cfg, split[name][0], split[name][1], is_training)
                assert np.array_equal(x, fixture[f"{name}_x"]), name
                assert np.array_equal(y, fixture[f"{name}_y"]), name


class TestBatchMapping:
    def test_training_labels_equal_inputs(self):
        cfg = MadFuzzyConfig(seq_len=32, vocab_size=16)
        inputs, _ = fuzzy_recall_instance(cfg, True, np.random.default_rng(0))
        row = mad_batch_row(
            cfg, inputs, np.zeros_like(inputs), pad_to=48, is_training=True
        )
        assert row["input_ids"].shape == (48,)
        assert row["labels"].shape == (48,)
        # Every training position is a target (labels == inputs), the
        # first position never being scored; the right pad is the pad
        # token / -100.
        assert torch.equal(
            row["labels"][: len(inputs)], row["input_ids"][: len(inputs)]
        )
        assert int(row["input_ids"][-1]) == cfg.vocab_size - 1
        assert int(row["labels"][-1]) == -100
        assert len(inputs) == 33 and int(row["slice_id"]) == 32

    def test_evaluation_labels_are_the_scored_targets(self):
        cfg = MadFuzzyConfig(seq_len=32, vocab_size=16)
        inputs, targets = fuzzy_recall_instance(cfg, False, np.random.default_rng(0))
        row = mad_batch_row(cfg, inputs, targets, pad_to=48, is_training=False)
        # Scored positions carry the value token; others -100.
        mask = targets != -100
        assert torch.equal(
            row["labels"][: len(targets)][mask], torch.from_numpy(targets[mask])
        )
        assert int((row["labels"][: len(targets)][~mask] == -100).sum()) == int(
            (~mask).sum()
        )
        # The right-padded region is -100.
        assert int(row["labels"][-1]) == -100

    def test_training_labelled_batch_matches_mad_return(self):
        # After MAD's own shift, a training batch's labels equal MAD's
        # ``inputs[1:]`` (its training targets).
        cfg = MadFuzzyConfig(seq_len=32, vocab_size=16)
        inputs, _ = fuzzy_recall_instance(cfg, True, np.random.default_rng(0))
        row = mad_batch_row(
            cfg, inputs, np.zeros_like(inputs), pad_to=48, is_training=True
        )
        assert torch.equal(row["labels"][1 : len(inputs)], torch.from_numpy(inputs[1:]))


class TestConfigValidation:
    def test_too_short_sequence_rejected(self):
        with pytest.raises(ValueError, match="seq_len"):
            MadFuzzyConfig(vocab_size=16, seq_len=12)  # 2*(3+3) = 12, must be > 12

    def test_bad_motif_sizes_rejected(self):
        with pytest.raises(ValueError, match="k_motif_size"):
            MadFuzzyConfig(vocab_size=16, seq_len=64, k_motif_size=0)
        with pytest.raises(ValueError, match="v_motif_size"):
            MadFuzzyConfig(vocab_size=16, seq_len=64, v_motif_size=-1)

    def test_value_vocab_too_small_for_motif(self):
        # Isolate the value-vocabulary check: v_motif_size 9 exceeds the
        # value vocabulary of 8 at vocab 16 (multi-query pads V-1 and the
        # non-special range halves into 7 keys / 8 values), while seq_len
        # stays well above the placement floor so only that check fires.
        with pytest.raises(ValueError, match="value vocabulary"):
            MadFuzzyConfig(vocab_size=16, seq_len=64, v_motif_size=9)

    def test_noise_requires_noise_vocab(self):
        with pytest.raises(ValueError, match="frac_noise"):
            MadFuzzyConfig(vocab_size=48, seq_len=64, frac_noise=0.2)
        # frac_noise out of [0, 1)
        with pytest.raises(ValueError, match="frac_noise"):
            MadFuzzyConfig(vocab_size=48, seq_len=64, frac_noise=1.0)

    def test_vocab_too_small_for_motifs(self):
        with pytest.raises(ValueError, match="key vocabulary"):
            # vocab 4 -> key vocab (4-1)//2 = 1 < k_motif_size 3
            MadFuzzyConfig(vocab_size=4, seq_len=64, k_motif_size=3)

    def test_odd_length_is_allowed(self):
        # MAD's generator does not require an even length; only the
        # placement-range floor applies.
        cfg = MadFuzzyConfig(vocab_size=16, seq_len=33)
        out = fuzzy_recall_instance(cfg, True, np.random.default_rng(0))
        assert out[0].shape == (34,)


class TestRegime:
    def test_mad_fuzzy_resolves_as_fixed_set(self):
        spec = require_task("mad_fuzzy")
        assert spec.regime is TaskRegime.FIXED_SET
        assert spec.dataset_fn is not None and spec.row_fn is not None
        assert spec.generator is None

    def test_streaming_datamodule_refuses_fixed_set(self):
        with pytest.raises(ValueError, match="fixed set"):
            SyntheticRecallDataModule(
                task="mad_fuzzy",
                task_config=MadFuzzyConfig(seq_len=64),
            )

    def test_generators_include_streaming_only(self):
        from ..registry import generators

        names = set(generators())
        assert "mqar" in names and "overwrite" in names
        assert "mad_fuzzy" not in names

    def test_task_spec_validation(self):
        from ..registry import TaskSpec

        with pytest.raises(ValueError, match="dataset_fn"):
            TaskSpec(
                "bad",
                MadFuzzyConfig,
                regime=TaskRegime.FIXED_SET,
                dataset_fn=fuzzy_recall_dataset,  # no row_fn
            )
        with pytest.raises(ValueError, match="declares a streaming generator"):
            TaskSpec(
                "bad",
                MadFuzzyConfig,
                generator=lambda *a, **k: {},
                regime=TaskRegime.FIXED_SET,
                dataset_fn=fuzzy_recall_dataset,
                row_fn=lambda c, p: mad_batch_row,
            )
        # A fixed-set task always takes its slices from ``config.slices``;
        # it cannot take a ``slice_arg_of`` (that is a streaming-task mapper).
        with pytest.raises(ValueError, match="slice_arg_of"):
            TaskSpec(
                "bad",
                MadFuzzyConfig,
                regime=TaskRegime.FIXED_SET,
                dataset_fn=fuzzy_recall_dataset,
                row_fn=lambda c, p: mad_batch_row,
                slice_arg_of=lambda slice_value, config: slice_value,
            )

    def test_entity_field_resolves_streaming_with_slice_arg_of(self):
        from ..registry import require_task

        spec = require_task("entity_field")
        assert spec.regime is TaskRegime.STREAMING
        assert spec.slice_arg_of is not None
        assert spec.slice_kwarg == "num_entities"

    def test_task_spec_field_order(self):
        # The pre-regime fields keep the streaming order (slice_arg_of right
        # after class_names); the regime fields come after them.
        from ..registry import TaskSpec

        names = [f.name for f in TaskSpec.__dataclass_fields__.values()]
        at = {name: i for i, name in enumerate(names)}
        assert at["slice_kwarg"] < at["class_names"] < at["slice_arg_of"]
        assert at["slice_arg_of"] < at["regime"] < at["dataset_fn"] < at["row_fn"]


class TestSingleQueryPadToken:
    def test_pad_token_is_vocab_minus_two_in_single_query(self):
        cfg = MadFuzzyConfig(
            vocab_size=24,
            seq_len=64,
            multi_query=False,
            noise_vocab_size=0,
        )
        assert cfg.pad_token == 22
        assert cfg.copy_prefix == 23
        inputs, _ = fuzzy_recall_instance(cfg, False, np.random.default_rng(0))
        row = mad_batch_row(
            cfg, inputs, np.zeros_like(inputs), pad_to=80, is_training=True
        )
        # The right-pad and (where left-padding occurs) the left-pad both
        # use the single-query pad token, and the copy prefix appears.
        assert int(row["input_ids"][-1]) == 22
        assert int(row["input_ids"][0]) == 22
        assert 23 in (row["input_ids"][: len(inputs)].tolist())
