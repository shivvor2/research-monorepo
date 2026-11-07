"""
Tests for the MAD fuzzy in-context recall generator and batch mapping.

Covers: bit-for-bit reproduction of MAD's generated instances (training and
test style, across the stored settings including noise and single-query),
the single-generator train/test split, and the mapping of a MAD row to the
repository's unshifted-label batch format (training/evaluation labels,
right-padding, slice identity).

Run with: pytest src/research_lib/data/synthetic/tests/test_mad_fuzzy.py -v
"""

from pathlib import Path

import numpy as np
import torch

from ..mad_fuzzy import (
    MadFuzzyConfig,
    fuzzy_recall_dataset,
    fuzzy_recall_instance,
    mad_batch_row,
    mad_reference_split,
)

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
