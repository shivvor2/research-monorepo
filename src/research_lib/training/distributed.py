"""Tiny distributed helpers for a training loss (or metric) that must not
depend on the device count.

The training module's loss formula is one expression for any device
count: it weights a rank's local mean by that rank's share of the global
count of scored positions, and that weight is built from the SUM all-reduce
and the world size defined here.  This is the one place that checks for a
process group, so the rest of the code reads ``world_size()`` /
``all_reduce_sum`` and never touches ``torch.distributed`` itself.
"""

from __future__ import annotations

import torch


def world_size() -> int:
    """The process-group world size, or 1 without an initialised group.

    Returns:
        ``torch.distributed.get_world_size()`` when a process group is
        initialised, else ``1`` (a single process behaves as one rank).
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """A SUM all-reduce of ``tensor`` across ranks, or ``tensor`` itself.

    When a process group with world size > 1 is initialised, reduces a
    **clone** of ``tensor`` (the input is left unchanged) with
    ``ReduceOp.SUM`` over the default group; otherwise returns ``tensor``
    itself, unchanged.  Every rank must call it the same number of times
    with the same shapes.

    Args:
        tensor: The tensor to reduce (any shape/dtype; int types work with
            both the gloo and NCCL backends).

    Returns:
        The reduced tensor: the elementwise sum over the ranks under a
        multi-rank group, else exactly the input.
    """
    if world_size() <= 1:
        return tensor
    clone = tensor.clone()
    torch.distributed.all_reduce(clone, op=torch.distributed.ReduceOp.SUM)
    return clone
