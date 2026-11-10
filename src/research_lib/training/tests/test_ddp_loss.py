"""
Test suite for the distributed (global-mean) training loss.

The loss a run optimises must not depend on how a batch is split across
ranks: under several devices the per-rank rows of a global batch hold
different numbers of scored positions, so the per-rank means DDP's
gradient averaging would combine are not the global mean.  The training
step all-reduces the count (SUM) and uses ``local_sum * world_size /
global_count`` as the backward loss, whose DDP gradient average is the
gradient of the global mean; the logged value is the global mean (the
sum is reduced as well).

* :func:`answer_loss_parts` returns the (sum, count) form whose
  quotient re-derives the mean (single-process);
* two real gloo processes holding different numbers of answer rows
  reproduce the single-process global-mean loss and its gradients.

Run with: pytest src/research_lib/training/tests/test_ddp_loss.py -v
"""

import multiprocessing
import socket

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from research_lib.training.losses import AnswerLogits, answer_loss_parts

_VOCAB = 6


def _free_port() -> int:
    """A TCP port that was free when it was picked (bounded race)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )


def _loss_worker(rank: int, world_size: int, port: int, queue) -> None:
    """One rank of the loss test: builds its rows on a parameter set that
    every rank and the single-process reference share (same seed and
    draw order), all-reduces the count, computes the backward loss and
    its gradients exactly as the training step's distributed branch
    would, and hands the pieces back through ``queue``."""
    _init_gloo(rank, world_size, port)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(4, _VOCAB, dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros(_VOCAB, dtype=torch.float64))
    # Ranks hold different numbers of answer rows (5 vs 3).
    n_rows = (5, 3)[rank]
    x = torch.randn(n_rows, 4, dtype=torch.float64)
    targets = torch.randint(0, _VOCAB, (n_rows,), dtype=torch.long)
    logits = x @ weight + bias
    answers = AnswerLogits(
        logits=logits,
        targets=targets,
        mask=torch.ones(n_rows, dtype=torch.bool),
    )
    local_sum, local_count = answer_loss_parts(answers)
    count = local_count.detach().clone()
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    global_count = int(count.item())
    loss = (
        local_sum * world_size / count
        if global_count > 0
        else torch.zeros_like(local_sum)
    )
    grad_weight, grad_bias = torch.autograd.grad(loss, (weight, bias))
    queue.put(
        (
            rank,
            float(loss.detach()),
            float(local_sum.detach()),
            int(local_count.item()),
            x.detach().clone(),
            targets.detach().clone(),
            grad_weight.detach().clone(),
            grad_bias.detach().clone(),
        )
    )
    dist.destroy_process_group()


def _spawn(worker, *args) -> list:
    """Run ``worker(rank, 2, port, queue, *args)`` in two spawn processes
    and return the queue contents in rank order."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    processes = [
        ctx.Process(target=worker, args=(rank, 2, port, queue, *args))
        for rank in (0, 1)
    ]
    for process in processes:
        process.start()
    results = [None, None]
    for _ in processes:
        entry = queue.get(timeout=120)
        results[entry[0]] = entry
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0, f"rank worker failed with exit {process.exitcode}"
    assert all(r is not None for r in results)
    return results


def test_answer_loss_parts_equals_the_mean_single_process():
    """The (sum, count) parts re-derive the mean in exact arithmetic, in
    fp64 and fp32."""
    torch.manual_seed(7)
    for dtype in (torch.float64, torch.float32):
        logits = torch.randn(11, _VOCAB, dtype=dtype, requires_grad=True)
        targets = torch.randint(0, _VOCAB, (11,), dtype=torch.long)
        answers = AnswerLogits(
            logits=logits,
            targets=targets,
            mask=torch.ones(11, dtype=torch.bool),
        )
        loss_sum, count = answer_loss_parts(answers)
        assert int(count.item()) == 11
        assert loss_sum.dtype == dtype
        mean = F.cross_entropy(logits, targets)
        assert torch.allclose(loss_sum / count, mean, atol=1e-5, rtol=1e-5)


@pytest.mark.slow
def test_global_mean_loss_and_gradients_two_gloo_processes():
    """Two real gloo processes with different answer-row counts: the
    backward loss the training step computes (``local_sum * W /
    global_count``), averaged over the ranks, equals the single-process
    global-mean loss on the concatenated rows, and the average of the
    ranks' parameter gradients equals the gradient of the global mean."""
    results = _spawn(_loss_worker)
    sums = [r[2] for r in results]
    counts = [r[3] for r in results]
    total = sum(sums) / sum(counts)  # the exact global mean (fp64 sums)

    # The single-process reference: the same shared parameter values
    # (every rank seeded identically) over the concatenated rows.
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    weight_p = torch.nn.Parameter(torch.randn(4, _VOCAB, dtype=torch.float64))
    bias_p = torch.nn.Parameter(torch.zeros(_VOCAB, dtype=torch.float64))
    x_full = torch.cat([r[4] for r in results], dim=0)
    targets_full = torch.cat([r[5] for r in results], dim=0)
    logits_full = x_full @ weight_p + bias_p
    ref = F.cross_entropy(logits_full, targets_full)
    grad_weight_full, grad_bias_full = torch.autograd.grad(ref, (weight_p, bias_p))

    # The quotient of the all-reduced sums is the global mean loss.
    assert torch.allclose(
        torch.tensor(total, dtype=torch.float64),
        ref,
        atol=1e-10,
        rtol=1e-10,
    ), "the reduced sum/count quotient is not the global mean"
    # ... and the per-rank backward losses average to it (they differ per
    # rank: 5 vs 3 rows).
    assert len({r[1] for r in results}) == 2
    assert torch.allclose(
        torch.tensor([r[1] for r in results], dtype=torch.float64).mean(),
        torch.tensor(total, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )

    # DDP averages the ranks' gradients; that average equals the gradient
    # of the global mean w.r.t. the shared parameters.
    for param_index, full_grad in zip((6, 7), (grad_weight_full, grad_bias_full)):
        averaged = (results[0][param_index] + results[1][param_index]) / 2
        assert torch.allclose(
            averaged, full_grad, atol=1e-10, rtol=1e-10
        ), "averaged rank gradients differ from the global-mean gradient"
    assert counts[0] != counts[1]  # the uneven-rows premise of the test
