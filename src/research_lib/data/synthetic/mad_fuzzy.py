"""Fuzzy in-context recall: the generator behind the MAD-style harness.

The task ("fuzzy" in-context recall from Poli et al., 2024, "Mechanistic
Design and Scaling of Hybrid Architectures", arXiv:2403.17844): a
sequence of key/value motifs in which a key can be a short adjacent run
of 1..k tokens and a value a short run of 1..v tokens.  A value is
scored when the same key has been presented earlier in the sequence
(keys keep one value within a sequence), and the final probe's value is
always scored.  A model has to recall an old association from a fuzzy
key, in memory, rather than copy an adjacent write.

This module re-implements MAD's generator
(https://github.com/athms/mad-lab, commit ``0f49a45``,
``mad/data/instances.py``,
:func:`generate_fuzzy_in_context_recall_instance`) call for call on the
same ``numpy`` RNG, so the reference reproduction and our runs see
bit-identical data.  MAD's code is not imported; only the algorithm is
transcribed.  The quirks are kept: the value map is per key length, the
probe may appear more than once, and the noise and single-query
branches exist even though our setting uses ``multi_query=True`` with no
noise.

Token layout (vocab 16, multi-query): the pad token is 15; keys are
tokens 0-6 and values 7-14 (``non_special_vocab_size = vocab_size - 1``,
halved).  Every sequence is returned padded on the left to ``seq_len +
1`` tokens.

The return values are the **un-shifted** arrays: ``inputs`` and
``targets`` each of length ``seq_len + 1``, aligned position by position
(``targets[i]`` is the token scored at position ``i``, or ``-100``).
MAD's own return values are the shifted ``inputs[:-1]`` and ``inputs[1:]``
/training ``targets[1:]``/test; applying the same shift to these arrays
reproduces them exactly.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
"""Label reserved for non-score positions (repo convention)."""


@dataclass(frozen=True)
class MadFuzzyConfig:
    """Configuration for the fuzzy in-context recall generator.

    Args:
        vocab_size: Vocabulary size ``V``.  With ``multi_query`` the pad
            token is ``V - 1``; keys are ``[0, (V-1)//2)`` and values
            ``[(V-1)//2, V-1)``.  MAD's setting uses 16.
        seq_len: Sequence length ``T`` (must be longer than the longest
            probe placement range, i.e. ``T > 2 * (k_motif_size +
            v_motif_size)``); the generator pads every instance to ``T +
            1`` tokens and the datamodule right-pads to the first multiple
            of the chunk size above ``T + 1``.
        k_motif_size: Maximum key motif length (1..k adjacent key tokens).
        v_motif_size: Maximum value motif length (1..v adjacent value
            tokens).
        multi_query: Whether every presented key is probed again (the
            default; single-query keeps only the final probe).
        noise_vocab_size: Draw a noise vocabulary of this size out of the
            non-special range (0 disables noise).
        frac_noise: Fraction of positions drawn as noise (0 disables noise;
            requires ``noise_vocab_size > 0`` to be nonzero).

    Raises:
        ValueError: A motif or noise option cannot be satisfied: a
            non-positive vocab/motif size, a sequence too short for the
            probe placement range, an out-of-range ``frac_noise``, noise
            without a noise vocabulary, or a key/value vocabulary too small
            to hold the motif sizes.  All checks run before any random
            draw, so a valid config's stream is unchanged.
    """

    vocab_size: int = 16
    seq_len: int = 512
    k_motif_size: int = 3
    v_motif_size: int = 3
    multi_query: bool = True
    noise_vocab_size: int = 0
    frac_noise: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if self.k_motif_size < 1:
            raise ValueError(f"k_motif_size must be >= 1, got {self.k_motif_size}")
        if self.v_motif_size < 1:
            raise ValueError(f"v_motif_size must be >= 1, got {self.v_motif_size}")
        if self.noise_vocab_size < 0:
            raise ValueError(
                f"noise_vocab_size must be >= 0, got {self.noise_vocab_size}"
            )
        if self.seq_len <= 2 * (self.k_motif_size + self.v_motif_size):
            raise ValueError(
                f"seq_len must be > 2*(k_motif_size + v_motif_size) = "
                f"{2 * (self.k_motif_size + self.v_motif_size)} so the probe "
                f"placement range is non-empty, got {self.seq_len}"
            )
        if not 0.0 <= self.frac_noise < 1.0:
            raise ValueError(f"frac_noise must be in [0, 1), got {self.frac_noise}")
        if self.frac_noise > 0 and self.noise_vocab_size == 0:
            raise ValueError(
                f"frac_noise > 0 ({self.frac_noise}) requires a nonzero "
                "noise_vocab_size"
            )
        non_special = self.non_special_vocab_size
        key_size = non_special // 2
        value_size = non_special - key_size
        if key_size < self.k_motif_size:
            raise ValueError(
                f"the key vocabulary has {key_size} tokens, fewer than "
                f"k_motif_size={self.k_motif_size}"
            )
        if value_size < self.v_motif_size:
            raise ValueError(
                f"the value vocabulary has {value_size} tokens, fewer than "
                f"v_motif_size={self.v_motif_size}"
            )

    @property
    def non_special_vocab_size(self) -> int:
        """The non-special/non-pad/non-noise token count (the keys and
        values share it, halved)."""
        base = self.vocab_size - 1 if self.multi_query else self.vocab_size - 2
        return base - self.noise_vocab_size

    @property
    def pad_token(self) -> int:
        """The left-padding token: ``V - 1`` (multi-query) or ``V - 2``
        (single-query, where ``V - 1`` is the copy prefix)."""
        return self.vocab_size - 1 if self.multi_query else self.vocab_size - 2

    @property
    def copy_prefix(self) -> int:
        """The copy-prefix token, ``V - 1``, used only in single-query mode."""
        return self.vocab_size - 1

    @property
    def slices(self) -> tuple[int, ...]:
        """The task's slice values: the sequence length (one slice)."""
        return (self.seq_len,)


def _exists(obj) -> bool:
    """MAD's ``exists``: present and not the empty string."""
    return obj is not None and obj != ""  # type: ignore[comparison-overlap]


def _vocab_permutations(
    vocab: Iterable[int], motif_size: int, rng: np.random.Generator
) -> list[tuple[int, ...]]:
    """All ``motif_size``-permutations of ``vocab``, shuffled in place.

    Reproduces MAD's :func:`generate_vocab_permutations`: the
    permutations are generated in ``itertools.permutations`` order and
    then ``rng.shuffle`` is applied to the resulting list.
    """
    values = list(itertools.permutations(sorted(vocab), motif_size))
    rng.shuffle(values)
    return values


def _as_int_array(items: list[int]) -> np.ndarray:
    return np.array(items, dtype=np.int64)


def mad_batch_row(
    config: MadFuzzyConfig,
    inputs: np.ndarray,
    targets: np.ndarray,
    pad_to: int,
    is_training: bool = False,
) -> dict[str, torch.Tensor]:
    """Map one MAD instance (pre-shift) row to the repo's batch keys.

    The repo convention: ``input_ids`` and unshifted ``labels`` of equal
    length, ``labels[p]`` predicted from position ``p - 1``, ``-100`` at
    non-scored positions.  MAD's pre-shift ``(inputs, targets)`` arrays
    are exactly that; the batch right-pads both to ``pad_to`` (inputs
    with the pad token, labels with ``-100``).  Training labels equal the
    inputs (every next token is a target, as MAD trains); evaluation
    labels are the scored ``targets``.  After the same shift MAD applies,
    the result equals MAD's returned arrays.

    Args:
        config: The generator config (for the pad token and slice value).
        inputs: Pre-shift inputs of length ``seq_len + 1``.
        targets: Pre-shift targets of length ``seq_len + 1``.
        pad_to: Right-pad both arrays to this length (a multiple of the
            chunk size).
        is_training: Whether this row is for training (labels = inputs).

    Returns:
        Dict with ``input_ids`` (int64, ``(pad_to,)``), ``labels`` (int64,
        ``(pad_to,)``) and ``slice_id`` (int, the sequence length).
    """
    seq_len = len(inputs)
    if len(targets) != seq_len:
        raise ValueError(
            f"inputs ({seq_len}) and targets ({len(targets)}) lengths differ"
        )
    if pad_to < seq_len:
        raise ValueError(f"pad_to ({pad_to}) must be >= the row length ({seq_len})")
    pad_token = config.pad_token
    n_pad = pad_to - seq_len
    input_ids = np.concatenate([inputs, np.full(n_pad, pad_token, dtype=inputs.dtype)])
    base = inputs if is_training else targets
    labels = np.concatenate([base, np.full(n_pad, IGNORE_INDEX, dtype=base.dtype)])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "slice_id": torch.tensor(int(config.seq_len), dtype=torch.long),
    }


def fuzzy_recall_instance(
    config: MadFuzzyConfig,
    is_training: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """One instance as MAD generates it.

    Args:
        config: The generator config.
        is_training: Training instances draw variable-length keys and
            values; test instances always prompt with the full key size.
        rng: The numpy generator to advance (MAD's caller owns it; the
            same stream produces the training set then the test set).

    Returns:
        Tuple ``(inputs, targets)``, each an int64 array of length
        ``config.seq_len + 1``, aligned position by position
        (``targets[i]`` is ``-100`` or the token scored at position
        ``i``).  These are MAD's pre-shift arrays; MAD's return values
        are ``inputs[:-1]`` paired with ``inputs[1:]`` (training) or
        ``targets[1:]`` (test).
    """
    seq_len = int(config.seq_len)
    k_motif_size = int(config.k_motif_size)
    v_motif_size = int(config.v_motif_size)
    noise_vocab_size = int(config.noise_vocab_size)
    copy_prefix = config.copy_prefix
    pad_token = config.pad_token
    non_special_vocab_size = config.non_special_vocab_size
    key_vocab = np.arange(non_special_vocab_size // 2)
    value_vocab = np.arange(non_special_vocab_size // 2, non_special_vocab_size)

    # Keys (and per-size values): the shuffled permutation lists.  MAD
    # draws values per motif size; training draws keys at every motif
    # size, test only at the maximum.
    if is_training:
        keys = {
            s: _vocab_permutations(key_vocab, s, rng)
            for s in range(1, k_motif_size + 1)
        }
    else:
        keys = {k_motif_size: _vocab_permutations(key_vocab, k_motif_size, rng)}
    values = {
        s: _vocab_permutations(value_vocab, s, rng) for s in range(1, v_motif_size + 1)
    }

    if config.frac_noise > 0:
        assert noise_vocab_size > 0, "noise_vocab_size must be >0 if frac_noise > 0"
        noise_vocab = np.arange(
            non_special_vocab_size, non_special_vocab_size + noise_vocab_size
        )

    # The probe (key/value whose value is always scored).
    k_probe_size = rng.choice(list(keys.keys())) if is_training else k_motif_size
    v_probe_size = rng.choice(list(values.keys()))
    k_probe = tuple(rng.choice(keys[int(k_probe_size)]))
    v_probe = tuple(rng.choice(values[int(v_probe_size)]))
    kv_probe_size = int(k_probe_size) + int(v_probe_size)
    probe_idx = int(rng.choice(seq_len - kv_probe_size - kv_probe_size))
    probe_added: bool = False

    kv_map = {s: {} for s in range(1, k_motif_size + 1)}
    inputs: list[int] = []
    targets: list[int] = []
    keys_presented: dict = {}
    limit = seq_len - kv_probe_size - (k_motif_size + v_motif_size)
    while len(inputs) < limit:
        k_size = int(rng.choice(list(keys.keys()))) if is_training else k_motif_size
        v_size = int(rng.choice(list(values.keys())))

        if len(inputs) >= probe_idx and not probe_added:
            inputs.extend(k_probe)
            inputs.extend(v_probe)
            targets.extend([IGNORE_INDEX] * (int(k_probe_size) + len(v_probe)))
            kv_map[int(k_probe_size)][k_probe] = v_probe
            keys_presented[k_probe] = v_probe
            probe_added = True
            continue

        is_noise = rng.random() < config.frac_noise if config.frac_noise > 0 else False

        if is_noise:
            noise_size = k_size + v_size
            noise = rng.choice(noise_vocab, size=noise_size, replace=True)
            inputs.extend([int(t) for t in noise])
            targets.extend([IGNORE_INDEX] * noise_size)
            continue

        # A key/value pair.
        k = tuple(rng.choice(keys[k_size]))
        inputs.extend(int(c) for c in k)
        if k == k_probe:
            v = v_probe
            probe_added = True
        else:
            if k not in kv_map[k_size]:
                v = tuple(rng.choice(values[v_size]))
                kv_map[k_size][k] = v
            else:
                v = kv_map[k_size][k]
        inputs.extend(int(c) for c in v)

        targets.extend([IGNORE_INDEX] * k_size)
        if k not in keys_presented:
            targets.extend([IGNORE_INDEX] * len(v))
        else:
            if config.multi_query:
                targets.extend(int(c) for c in v)
            else:
                targets.extend([IGNORE_INDEX] * len(v))
        keys_presented[k] = v

    # Final key/value probe, plus (single-query) the copy prefix.
    if not config.multi_query:
        inputs.append(copy_prefix)
    inputs.extend(int(c) for c in k_probe)
    inputs.extend(int(c) for c in v_probe)
    if not config.multi_query:
        targets.append(IGNORE_INDEX)
    targets.extend([IGNORE_INDEX] * int(k_probe_size))
    targets.extend(int(c) for c in v_probe)

    inputs = _as_int_array(inputs)
    targets = _as_int_array(targets)

    # Left-pad to seq_len + 1 with the pad token.
    if len(inputs) < seq_len + 1:
        n_pad = seq_len + 1 - len(inputs)
        inputs = np.concatenate([np.full(n_pad, pad_token, dtype=np.int64), inputs])
        targets = np.concatenate(
            [np.full(n_pad, IGNORE_INDEX, dtype=np.int64), targets]
        )
    return inputs, targets


def fuzzy_recall_dataset(
    config: MadFuzzyConfig,
    num_examples: int,
    is_training: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """``num_examples`` instances generated sequentially from ``rng``, stacked.

    Args:
        config: The generator config.
        num_examples: Number of instances.
        is_training: Train-style (variable key sizes) or test-style.
        rng: The generator to advance; instances are drawn one after
            another from it.

    Returns:
        Tuple ``(inputs, targets)`` stacked over instances, shapes
        ``(num_examples, seq_len + 1)``.
    """
    pieces = [
        fuzzy_recall_instance(config, is_training, rng) for _ in range(num_examples)
    ]
    return (
        np.stack([p[0] for p in pieces]),
        np.stack([p[1] for p in pieces]),
    )


def mad_reference_split(
    config: MadFuzzyConfig,
    num_train: int,
    num_test: int,
    seed: int = 12345,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """MAD's train/test split: one generator, training first, then test.

    Args:
        config: The generator config.
        num_train: Number of train instances.
        num_test: Number of test instances.
        seed: Base seed of the single generator stream.

    Returns:
        Dict ``{"train": (inputs, targets), "test": (inputs, targets)}``,
        each stacked over instances.
    """
    rng = np.random.default_rng(seed)
    train = fuzzy_recall_dataset(config, num_train, True, rng)
    test = fuzzy_recall_dataset(config, num_test, False, rng)
    return {"train": train, "test": test}
