"""Zoology-format Multi-Query Associative Recall (MQAR) generator.

Reproduces the distribution of Zoology's ``multiquery_ar``
(https://github.com/HazyResearch/zoology, ``zoology/data/multiquery_ar.py``,
arXiv:2312.04927 App. E.1) in this repo's causal-LM label convention:

- keys are distinct tokens from ``[1, V/2)`` and values distinct tokens from
  ``[V/2, V)``, both drawn without replacement within an example;
- the ``D`` key-value pairs occupy the first ``2D`` positions, interleaved
  ``k_1 v_1 k_2 v_2 ... k_D v_D``;
- the remaining positions hold ``D`` queries at even offsets ``2 * gap`` after
  the context, with gaps drawn without replacement under a power law
  ``p(i) ~ a * (i + 1) ** (a - 1)`` (``a = power_a``, strongly favouring early
  slots, as in real-language bigram gaps);
- every other position (including the token *after* each query, where the
  answer is expected) is filler: uniform over the full vocabulary ``[0, V)``
  if ``random_non_queries`` else 0. Filler may coincide with a key or value;
  this is the standard benchmark and is kept for comparability.

Labels are **unshifted** with the repo convention
(`DualOptimizerModule.compute_loss` shifts internally, ``ignore_index=-100``):
``labels[p]`` is the token the model should predict from position ``p - 1``;
so the answer for the query at position ``q`` sits at ``labels[q + 1]``.

The generator is a pure function of ``(config, seed)``: it uses
``np.random.default_rng(seed)`` rather than the global numpy RNG. The random
streams therefore differ from Zoology's (which seeds the global RNG); the
distribution does not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
"""Label reserved for non-answer positions (repo convention)."""


@dataclass(frozen=True)
class MQARConfig:
    """Configuration for the MQAR generator.

    Args:
        vocab_size: vocabulary size ``V``. Keys come from ``[1, V/2)``,
            values from ``[V/2, V)``. Zoology uses 8192.
        seq_len: sequence length ``T`` (must be even).
        num_kv_pairs: the slice values: number of KV pairs ``D`` drawn
            uniformly per example. Zoology/Log-Linear mix 4..64 at T=256.
        power_a: power-law exponent for the query-gap distribution.
            ``power_a = 1`` is uniform; Zoology's default 0.01 strongly
            favours early query slots.
        random_non_queries: if True, non-query/non-context positions are
            filler drawn uniformly from the full vocabulary ``[0, V)``,
            else they are 0.
    """

    vocab_size: int = 8192
    seq_len: int = 256
    num_kv_pairs: tuple[int, ...] = (4, 8, 16, 32, 64)
    power_a: float = 0.01
    random_non_queries: bool = True

    def __post_init__(self) -> None:
        if self.seq_len % 2 != 0:
            raise ValueError(f"seq_len must be even, got {self.seq_len}")
        if self.vocab_size <= self.seq_len:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must exceed seq_len "
                f"({self.seq_len}) so keys/values cannot collide with filler"
            )
        if not self.num_kv_pairs:
            raise ValueError("num_kv_pairs must be non-empty")
        # Zoology's assertion: 4 * D <= T for every D (context 2D + queries 2D).
        for d in self.num_kv_pairs:
            if d < 1:
                raise ValueError(f"num_kv_pairs entries must be >= 1, got {d}")
            if 4 * d > self.seq_len:
                raise ValueError(
                    f"infeasible num_kv_pairs entry {d}: need 4 * D <= "
                    f"seq_len = {self.seq_len}"
                )
        if self.power_a <= 0:
            raise ValueError(f"power_a must be > 0, got {self.power_a}")

    @property
    def slices(self) -> tuple[int, ...]:
        """The per-example slice values ``D`` (sorted)."""
        return tuple(sorted(self.num_kv_pairs))


def query_gap_weights(power_a: float, space: int) -> np.ndarray:
    """Power-law weights over query gaps ``0 .. space - 1``.

    Follows Zoology: weights ``a * (i + 1) ** (a - 1)`` over ``i``, normalized
    to sum to 1 (Zoology writes ``a * arange(1, space + 1) ** (a - 1)``,
    which is the same vector shifted by one).

    Args:
        power_a: power-law exponent.
        space: number of query slots.

    Returns:
        Float64 array of shape ``(space,)`` summing to 1.
    """
    weights = power_a * np.arange(1, space + 1, dtype=np.float64) ** (power_a - 1)
    return weights / weights.sum()


def generate_mqar(
    config: MQARConfig,
    num_examples: int,
    seed: int,
    num_kv_pairs: int | None = None,
) -> dict[str, torch.Tensor]:
    """Generate a batch of Zoology-format MQAR examples.

    Args:
        config: task configuration.
        num_examples: number of examples ``N`` to generate.
        seed: seed for ``np.random.default_rng``. Pure and deterministic:
            the same ``(config, num_examples, seed, num_kv_pairs)`` always
            produces a bit-identical batch.
        num_kv_pairs: if given, every example uses exactly this many KV
            pairs (a fixed slice) instead of drawing one from
            ``config.num_kv_pairs``.

    Returns:
        Dict of stacked int64 tensors:

        - ``input_ids``: shape ``(N, T)`` token sequences.
        - ``labels``: shape ``(N, T)``, unshifted, ``IGNORE_INDEX`` everywhere
          except answer positions ``q_i + 1`` where the target is the value
          paired with the queried key.
        - ``slice_id``: shape ``(N,)``, the number of KV pairs ``D``.

    Raises:
        ValueError: if ``num_kv_pairs`` is infeasible for ``config.seq_len``.

    Shape:
        input_ids/labels: ``(num_examples, config.seq_len)``;
        slice_id: ``(num_examples,)``.
    """
    if num_examples < 1:
        raise ValueError(f"num_examples must be >= 1, got {num_examples}")
    if num_kv_pairs is not None:
        if num_kv_pairs < 1:
            raise ValueError(f"num_kv_pairs must be >= 1, got {num_kv_pairs}")
        if 4 * num_kv_pairs > config.seq_len:
            raise ValueError(
                f"infeasible num_kv_pairs {num_kv_pairs}: need 4 * D <= "
                f"seq_len = {config.seq_len}"
            )

    rng = np.random.default_rng(seed)
    seq_len, vocab = config.seq_len, config.vocab_size

    input_ids = np.zeros((num_examples, seq_len), dtype=np.int64)
    labels = np.full((num_examples, seq_len), IGNORE_INDEX, dtype=np.int64)
    slice_id = np.zeros(num_examples, dtype=np.int64)

    key_choices = np.arange(1, vocab // 2)
    value_choices = np.arange(vocab // 2, vocab)

    for n in range(num_examples):
        d = (
            num_kv_pairs
            if num_kv_pairs is not None
            else int(config.num_kv_pairs[rng.integers(0, len(config.num_kv_pairs))])
        )
        slice_id[n] = d

        # Step 2: distinct keys and values, paired by index.
        keys = rng.choice(key_choices, size=d, replace=False)
        values = rng.choice(value_choices, size=d, replace=False)

        x = input_ids[n]
        x[0 : 2 * d : 2] = keys
        x[1 : 2 * d : 2] = values

        # Step 3: query slots at even offsets after the context, gaps drawn
        # without replacement under the power law.
        space = (seq_len - 2 * d) // 2
        p = query_gap_weights(config.power_a, space)
        gaps = rng.choice(space, size=d, replace=False, p=p)
        query_positions = 2 * d + 2 * gaps
        x[query_positions] = keys

        # Step 4: unshifted labels; labels[q + 1] is predicted from q.
        # q + 1 <= T - 1 < T by construction (4 * D <= T, T even).
        labels[n, query_positions + 1] = values

        # Step 5: filler on every position not written in steps 2-3
        # (including each answer position q + 1).
        occupied = np.zeros(seq_len, dtype=bool)
        occupied[0 : 2 * d] = True
        occupied[query_positions] = True
        filler_mask = ~occupied
        if config.random_non_queries:
            x[filler_mask] = rng.integers(0, vocab, size=int(filler_mask.sum()))

    return {
        "input_ids": torch.from_numpy(input_ids),
        "labels": torch.from_numpy(labels),
        "slice_id": torch.from_numpy(slice_id),
    }
