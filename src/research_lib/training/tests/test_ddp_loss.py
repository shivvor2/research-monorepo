"""
Test suite for the device-count-independent (global-mean) training loss.

The loss a run optimises must not depend on how a batch is split across
ranks: under several devices the per-rank rows of a global batch hold
different numbers of scored positions, so the per-rank means DDP's
gradient averaging would combine are not the global mean.  ``compute_loss``
weights a rank's mean by its share ``W * c_r / C`` of the global count of
scored positions; at one device the weight is exactly 1.0, so
single-device runs are unchanged bit for bit.

* single-process (bitwise): for each of the three paths (answer head,
  full logits with answers, ``loss_on="all"``) ``compute_loss`` returns a
  tensor ``torch.equal`` to the local mean, with equal gradients;
* the distributed helpers with no process group: ``world_size() == 1``
  and ``all_reduce_sum`` is the identity;
* two real gloo processes with different answer-row counts: the
  ``compute_loss`` value is the global mean on every rank (training and
  eval mode alike); the rank-averaged parameter gradients equal the
  gradient of the global mean;
* two real gloo processes under ``loss_on="all"`` (equal counts): the
  per-rank weight is exactly 1.0 and the result is the mean of the
  per-rank means.

Run with: pytest src/research_lib/training/tests/test_ddp_loss.py -v
"""

import multiprocessing
import socket

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from research_lib.data.synthetic.mqar import MQARConfig, generate_mqar
from research_lib.training.configs import ScheduleConfig
from research_lib.training.distributed import all_reduce_sum, world_size
from research_lib.training.losses import AnswerLogits
from research_lib.training.modules.mgdn2_synthetic_recall import (
    MGDN2SyntheticRecallModule,
)
from research_lib.training.presets import build_weight_decay_configs
from research_lib.training.scheduling import WarmupStableDecaySchedule

_VOCAB = 6
_DTYPE = torch.float64


class _Stub(nn.Module):
    """A model exposing only the ``hidden_states`` / ``head`` methods the
    module's constructor checks for (``compute_loss`` never calls them)."""

    def hidden_states(self, input_ids):
        raise NotImplementedError

    def head(self, logits):
        raise NotImplementedError


def _schedule() -> ScheduleConfig:
    return ScheduleConfig(
        global_schedules=[
            WarmupStableDecaySchedule(
                param_name="lr",
                warmup_frac=0.0,
                cooldown_frac=1.0,
                min_value=0.0,
                max_value=1e-3,
                decay_type="cosine",
            )
        ]
    )


def _make_module(loss_on="answers") -> MGDN2SyntheticRecallModule:
    matrix_config, vector_config = build_weight_decay_configs(
        "grouped", weight_decay=0.1, betas=(0.9, 0.95)
    )
    return MGDN2SyntheticRecallModule(
        model=_Stub(),
        slice_values=(2, 4),
        loss_on=loss_on,
        answer_head=True,
        matrix_optimizer_config=matrix_config,
        vector_optimizer_config=vector_config,
        matrix_schedule_config=_schedule(),
        vector_schedule_config=_schedule(),
    )


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


def _spawn(worker, *args) -> list:
    """Run ``worker(rank, 2, port, queue, *args)`` in two spawn processes
    and return the queue contents in rank order.

    Spawn hygiene: the result loop marks the spawn completed only after
    every process's result was received; an incomplete loop terminates
    every still-alive process.  All processes are then joined, and any
    process still alive after that join is terminated and re-joined (a
    worker that sent its result but hangs in shutdown must not keep
    pytest waiting at exit), so the exit codes are only asserted after
    the cleanup.  A failed ``get`` is re-raised as an AssertionError after
    the cleanup; interrupts are not caught and propagate.
    """
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
    error: Exception | None = None
    completed = False
    try:
        for _ in processes:
            entry = queue.get(timeout=120)
            results[entry[0]] = entry
    except Exception as exc:
        # Only a failed get: turn it into the AssertionError below, after
        # the cleanup.  Interrupts (KeyboardInterrupt etc.) are not caught
        # and propagate unchanged.
        error = exc
    else:
        completed = True
    finally:
        if not completed:
            # A worker stuck in a collective (its get never fired) must
            # not keep pytest hanging at exit; a completed loop leaves
            # finished workers to exit on their own (terminating them
            # would corrupt their exit codes).
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join(timeout=30)
        # A worker that sent its result but hangs in shutdown must not keep
        # pytest waiting at exit, whether or not the loop completed; such a
        # process fails the exit-code assertion below with its exit code.
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)
        queue.close()
        queue.join_thread()
    if error is not None:
        raise AssertionError(f"a rank worker produced no result: {error!r}") from error
    # The exit codes are read only after every process was joined.
    for process in processes:
        assert process.exitcode == 0, f"rank worker failed with exit {process.exitcode}"
    assert all(r is not None for r in results)
    return results


# =============================================================================
# Single process (bitwise)
# =============================================================================


def _answers(n: int):
    logits = torch.randn(n, _VOCAB, dtype=_DTYPE, requires_grad=True)
    targets = torch.randint(0, _VOCAB, (n,), dtype=torch.long)
    return (
        AnswerLogits(
            logits=logits,
            targets=targets,
            mask=torch.ones(n, dtype=torch.bool),
        ),
        logits,
    )


def test_distributed_helpers_without_a_process_group():
    """``world_size()`` is 1 and ``all_reduce_sum`` is the identity."""
    assert world_size() == 1
    tensor = torch.zeros(3, dtype=_DTYPE)
    assert all_reduce_sum(tensor) is tensor


def test_compute_loss_is_bitwise_the_local_mean_single_process():
    """For each of the three paths, ``compute_loss`` returns a tensor
    ``torch.equal`` to the local mean, with ``torch.equal`` gradients."""
    # 1. Answer-head path.
    module = _make_module(loss_on="answers")
    answers, logits = _answers(11)
    mean = module._local_loss_and_count(answers, {})[0]
    loss = module.compute_loss(answers, {})
    assert torch.equal(loss, mean)
    assert loss.dtype == mean.dtype
    assert torch.equal(
        torch.autograd.grad(loss, logits)[0], torch.autograd.grad(mean, logits)[0]
    )

    # 2. Full logits under loss_on="answers" (the parent's ignore_index).
    module2 = _make_module(loss_on="answers")
    cfg = MQARConfig(vocab_size=64, seq_len=32, num_kv_pairs=(2, 4))
    batch = generate_mqar(cfg, num_examples=4, seed=0)
    logits2 = torch.randn(
        4, cfg.seq_len, cfg.vocab_size, dtype=_DTYPE, requires_grad=True
    )
    mean2 = module2._local_loss_and_count(logits2, batch)[0]
    loss2 = module2.compute_loss(logits2, batch)
    assert torch.equal(loss2, mean2)
    assert loss2.dtype == mean2.dtype
    assert torch.equal(
        torch.autograd.grad(loss2, logits2)[0],
        torch.autograd.grad(mean2, logits2)[0],
    )

    # 3. loss_on="all" (full-sequence CE).
    module3 = _make_module(loss_on="all")
    logits3 = torch.randn(
        4, cfg.seq_len, cfg.vocab_size, dtype=_DTYPE, requires_grad=True
    )
    mean3 = module3._local_loss_and_count(logits3, batch)[0]
    loss3 = module3.compute_loss(logits3, batch)
    assert torch.equal(loss3, mean3)
    assert loss3.dtype == mean3.dtype
    assert torch.equal(
        torch.autograd.grad(loss3, logits3)[0],
        torch.autograd.grad(mean3, logits3)[0],
    )


# =============================================================================
# Two gloo processes
# =============================================================================


def _loss_worker(rank: int, world_size: int, port: int, queue) -> None:
    """One rank of the loss test: builds its rows on a parameter set that
    every rank and the single-process reference share (same seed and
    draw order), computes the module's ``compute_loss`` (training and
    eval mode) and its parameter gradients, and hands the pieces back."""
    _init_gloo(rank, world_size, port)
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(4, _VOCAB, dtype=_DTYPE))
    bias = torch.nn.Parameter(torch.zeros(_VOCAB, dtype=_DTYPE))
    # Ranks hold different numbers of answer rows (5 vs 3).
    n_rows = (5, 3)[rank]
    x = torch.randn(n_rows, 4, dtype=_DTYPE)
    targets = torch.randint(0, _VOCAB, (n_rows,), dtype=torch.long)
    logits = x @ weight + bias
    answers = AnswerLogits(
        logits=logits,
        targets=targets,
        mask=torch.ones(n_rows, dtype=torch.bool),
    )
    module = _make_module(loss_on="answers")
    loss = module.compute_loss(answers, {})
    grad_weight, grad_bias = torch.autograd.grad(loss, (weight, bias))
    # Eval mode computes the same value (compute_loss is mode-independent).
    module.eval()
    loss_eval = module.compute_loss(answers, {})
    local_mean = F.cross_entropy(answers.logits, answers.targets)
    # Tensors cross the queue as plain lists (the sender exits right
    # after; a torch shared-memory tensor needs the sender alive to be
    # unpickled), rebuilt in the parent with the original dtypes.
    queue.put(
        (
            rank,
            float(loss.detach()),
            float(loss_eval.detach()),
            float(local_mean.detach()),
            int(answers.targets.numel()),
            x.detach().tolist(),
            targets.detach().tolist(),
            grad_weight.detach().tolist(),
            grad_bias.detach().tolist(),
        )
    )
    dist.destroy_process_group()


def _bf16_weight_worker(rank: int, world_size: int, port: int, queue) -> None:
    """One rank of the bf16-weight test: bf16 answer rows with counts
    above 2**11 (where a count cast to bf16 would round), the module's
    loss and its float64 weight."""
    _init_gloo(rank, world_size, port)
    torch.manual_seed(0)
    weight_param = torch.nn.Parameter(torch.randn(4, _VOCAB, dtype=torch.bfloat16))
    bias_param = torch.nn.Parameter(torch.zeros(_VOCAB, dtype=torch.bfloat16))
    # Counts above 2**11 (2049 vs 2050): in bf16 a count would round.
    n_rows = (2049, 2050)[rank]
    x = torch.randn(n_rows, 4, dtype=torch.bfloat16)
    targets = torch.randint(0, _VOCAB, (n_rows,), dtype=torch.long)
    logits = x @ weight_param + bias_param
    answers = AnswerLogits(
        logits=logits,
        targets=targets,
        mask=torch.ones(n_rows, dtype=torch.bool),
    )
    module = _make_module(loss_on="answers")
    loss = module.compute_loss(answers, {})
    count = module._local_loss_and_count(answers, {})[1]
    rank_weight = module._rank_weight(count)
    queue.put(
        (
            rank,
            n_rows,
            float(loss.detach()),
            int(count.item()),
            float(rank_weight.detach()),
            str(loss.dtype),
        )
    )
    dist.destroy_process_group()


def _all_worker(rank: int, world_size: int, port: int, queue) -> None:
    """One rank of the ``loss_on=\"all\"`` test (full-sequence CE, so every
    rank has the same count and the per-rank weight is exactly 1.0)."""
    _init_gloo(rank, world_size, port)
    torch.manual_seed(0)
    batch_size, seq_len = 4, 8
    input_ids = torch.randint(0, _VOCAB, (batch_size, seq_len), dtype=torch.long)
    logits = torch.randn(batch_size, seq_len, _VOCAB, dtype=_DTYPE, requires_grad=True)
    module = _make_module(loss_on="all")
    value = module.compute_loss(logits, {"input_ids": input_ids})
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    local_mean = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    queue.put((rank, float(value.detach()), float(local_mean.detach())))
    dist.destroy_process_group()


@pytest.mark.slow
def test_global_mean_loss_value_and_gradients_two_gloo_processes():
    """Two real gloo processes with different answer-row counts: each
    rank's ``compute_loss`` value (training and eval mode) is the global
    mean, both ranks read the identical value, and DDP's average of the
    ranks' parameter gradients is the gradient of the global mean."""
    results = _spawn(_loss_worker)
    global_mean = sum(r[3] * r[4] for r in results) / sum(
        r[4] for r in results
    )  # exact fp64
    values = [r[1] for r in results]
    # Both ranks reduce the same value: bit-identical, equal to the mean.
    assert values[0] == values[1]
    assert torch.allclose(
        torch.tensor(values[0], dtype=_DTYPE),
        torch.tensor(global_mean, dtype=_DTYPE),
        atol=1e-10,
        rtol=1e-10,
    )
    # Eval mode returns the same value.
    assert torch.allclose(
        torch.tensor([r[2] for r in results], dtype=_DTYPE),
        torch.tensor(global_mean, dtype=_DTYPE),
        atol=1e-10,
        rtol=1e-10,
    )

    # The rank-averaged gradients equal the gradient of the global mean on
    # the concatenated rows (the single-process reference uses the same
    # seed-drawn parameter values as the workers).
    torch.manual_seed(0)
    weight_p = torch.nn.Parameter(torch.randn(4, _VOCAB, dtype=_DTYPE))
    bias_p = torch.nn.Parameter(torch.zeros(_VOCAB, dtype=_DTYPE))
    x_full = torch.cat([torch.tensor(r[5], dtype=_DTYPE) for r in results], dim=0)
    targets_full = torch.cat(
        [torch.tensor(r[6], dtype=torch.long) for r in results], dim=0
    )
    ref = F.cross_entropy(x_full @ weight_p + bias_p, targets_full)
    grad_weight_full, grad_bias_full = torch.autograd.grad(ref, (weight_p, bias_p))
    for index, full_grad in zip((7, 8), (grad_weight_full, grad_bias_full)):
        averaged = (
            torch.tensor(results[0][index], dtype=_DTYPE)
            + torch.tensor(results[1][index], dtype=_DTYPE)
        ) / 2
        assert torch.allclose(
            averaged, full_grad, atol=1e-10, rtol=1e-10
        ), "averaged rank gradients differ from the global-mean gradient"
    assert results[0][4] != results[1][4]  # the uneven-rows premise


@pytest.mark.slow
def test_rank_weight_uses_integer_counts_two_gloo_processes():
    """Counts above 2**11 under a bf16 mean reduce as integers: the
    weight equals ``W * c_r / C`` to fp32 precision (a count rounded to
    bf16 would be off by a bf16 ulp), and the returned loss keeps the
    mean's (bf16) dtype."""
    results = _spawn(_bf16_weight_worker)
    total = sum(r[1] for r in results)
    for rank, n_rows, loss, count, rank_weight, dtype in results:
        reference = 2 * count / total
        assert rank_weight == pytest.approx(reference, abs=1e-6, rel=1e-6), rank
        assert dtype == "torch.bfloat16", rank
    assert total == 2049 + 2050


@pytest.mark.slow
def test_loss_on_all_weight_is_one_two_gloo_processes():
    """Under ``loss_on=\"all\"`` every rank scores every position, so each
    rank's weight is exactly 1.0 and the result is the mean of the
    per-rank means."""
    results = _spawn(_all_worker)
    combined = (results[0][2] + results[1][2]) / 2
    for rank, value, _ in results:
        assert torch.allclose(
            torch.tensor(value, dtype=_DTYPE),
            torch.tensor(combined, dtype=_DTYPE),
            atol=1e-10,
            rtol=1e-10,
        ), f"rank {rank} value is not the mean of the per-rank means"
