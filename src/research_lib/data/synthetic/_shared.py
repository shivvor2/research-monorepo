"""Private helpers shared by the synthetic datamodules (streaming and
fixed-set): deterministic per-index seeding and the current distributed
world size / rank."""

from __future__ import annotations

import numpy as np
import torch


def batch_seed(train_seed: int, index: int) -> int:
    """Deterministic per-index seed: ``SeedSequence([train_seed, index])``.

    Args:
        train_seed: base seed of the train stream.
        index: the index (a batch index, or a fixed-set epoch).

    Returns:
        A usable int seed for ``np.random.default_rng`` (64-bit draw).
    """
    return int(
        np.random.SeedSequence([int(train_seed), int(index)]).generate_state(
            1, dtype=np.uint64
        )[0]
    )


def distributed_world_size_and_rank() -> tuple[int, int]:
    """World size and rank: ``torch.distributed`` when a process group is
    initialized, else ``(1, 0)``.

    Read fresh at each use so the current state is always seen and tests
    can simulate ranks.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    return 1, 0
