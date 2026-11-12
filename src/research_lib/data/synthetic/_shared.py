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


def distributed_reduce_device(tensor: torch.Tensor) -> torch.device:
    """The device a cross-rank reduction of ``tensor`` should run on.

    Under an NCCL process group, the current CUDA device: a rank that
    never saw a batch keeps its metric counters on the CPU (they are
    created there by ``reset``), and NCCL rejects CPU tensors.  For any
    other backend (gloo) or with no process group, the tensor's own
    device is left unchanged.

    Args:
        tensor: The tensor to reduce (its device is the fallback).

    Returns:
        The device the tensor should be moved to before the reduce.
    """
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
    ):
        return torch.device("cuda", torch.cuda.current_device())
    return tensor.device
