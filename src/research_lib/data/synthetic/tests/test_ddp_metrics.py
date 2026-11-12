"""
Test suite for the evaluation metrics under a two-process process group.

The evaluation metrics must read the same on several devices as on one:
their counts reduce across ranks (SUM all-reduce in ``compute``), so a
metric accumulated over a split of the batches equals the single-process
value on the same batches.  Under two real gloo processes this is
verified for:

* :class:`~research_lib.data.synthetic.RecallMetrics` (overall, per
  slice, per class),
* :class:`~research_lib.data.synthetic.MacroTokenAccuracy` (torchmetrics
  syncs its state across ranks itself),
* :class:`~research_lib.data.synthetic.TokenAccuracy` (custom counters),
* and a rank that saw NO evaluation batch (it still joins the collectives
  with zero counts).

Run with: pytest src/research_lib/data/synthetic/tests/test_ddp_metrics.py -v
"""

import multiprocessing
import socket

import pytest
import torch
import torch.distributed as dist

from research_lib.data.synthetic._shared import batch_seed
from research_lib.data.synthetic.mad_metrics import MacroTokenAccuracy, TokenAccuracy
from research_lib.data.synthetic.metrics import RecallMetrics
from research_lib.data.synthetic.mqar import MQARConfig, generate_mqar
from research_lib.data.synthetic.overwrite import OverwriteConfig, generate_overwrite

CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_B = 4
MQ_CFG = MQARConfig(vocab_size=128, seq_len=32, num_kv_pairs=(2, 4))
OW_CFG = OverwriteConfig(
    vocab_size=512,
    seq_len=128,
    num_keys=(4, 8),
    block_size=16,
    distant_min_blocks=4,
)
_VOCAB = 512  # the overwrite vocabulary (MQAR's 128 is a subset)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rank_batches(rank: int, mode: str) -> list[dict[str, torch.Tensor]]:
    """The batches this rank accumulates: batch ``i`` (``i`` in 0..3) is
    MQAR for even ``i`` and overwrite for odd; ranks always accumulate
    the same per-index batches (deterministic per seed), while ``mode``
    decides whether rank 1 also accumulates any (``split``: rank ``r``
    takes ``i % 2 == r``; ``empty``: rank 1 accumulates nothing)."""
    if mode == "empty" and rank == 1:
        return []
    if mode == "split":
        return [i for i in range(4) if i % 2 == rank]
    return list(range(4))  # mode == "all" (the single-process reference)


def _make_batch(index: int) -> dict[str, torch.Tensor]:
    """Batch ``index``: MQAR for even indices, overwrite otherwise, plus
    deterministic random logits over the vocab."""
    if index % 2 == 0:
        batch = generate_mqar(MQ_CFG, _B, seed=batch_seed(0, index))
    else:
        batch = generate_overwrite(OW_CFG, _B, seed=batch_seed(0, index))
    torch.manual_seed(1000 + index)
    batch["logits"] = torch.randn(
        _B, 32 if index % 2 == 0 else 128, _VOCAB, dtype=torch.float64
    )
    return batch


def _accumulate(
    batches: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Accumulate the batches into every metric and return the combined
    metric dict (the comparison surface).  No default dtype is set here:
    the batch logits are already have an explicit fp64 dtype and the
    metrics count integer positions, so a global dtype change is not
    needed (and would leak into the rest of the session)."""
    recall_mqar = RecallMetrics(tuple(MQ_CFG.slices))
    recall_ow = RecallMetrics(tuple(OW_CFG.slices), with_classes=True)
    macro = MacroTokenAccuracy(num_classes=_VOCAB)
    token = TokenAccuracy()
    for batch in batches:
        logits = torch.as_tensor(batch["logits"])
        del batch["logits"]
        if "key_class" in batch:
            recall_ow.update(logits, batch)
        else:
            recall_mqar.update(logits, batch)
        predictions = logits[:, :-1, :].argmax(-1)
        macro.update(predictions, batch)
        token.update(predictions, batch)
    out: dict[str, float] = {}
    for prefix, metrics in (
        ("mqar", recall_mqar),
        ("ow", recall_ow),
        ("macro", macro),
        ("token", token),
    ):
        for key, value in metrics.compute().items():
            out[f"{prefix}/{key}"] = value
    return out


def _metrics_worker(rank: int, world_size: int, port: int, queue, mode: str) -> None:
    """One rank of the metrics test: accumulates its share of the batches
    (deterministic per rank) and hands back its metric dict.  ``compute``
    reduces across ranks, so every rank's dict is the global reading."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    batches = [_make_batch(i) for i in _rank_batches(rank, mode)]
    values = _accumulate(batches)
    queue.put((rank, values))
    dist.destroy_process_group()


def _run_two(mode: str) -> list[dict[str, float]]:
    """Run the metrics worker on two spawn processes and return their
    metric dicts in rank order.

    Spawn hygiene: every started process is terminated if still alive,
    joined and the queue closed even when a worker fails or a ``get``
    times out, so a rank stuck in a collective cannot hang the test at
    exit; the exit codes are only asserted after the cleanup.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    processes = [
        ctx.Process(target=_metrics_worker, args=(rank, 2, port, queue, mode))
        for rank in (0, 1)
    ]
    for process in processes:
        process.start()
    results = [None, None]
    error: BaseException | None = None
    try:
        for _ in processes:
            rank, values = queue.get(timeout=120)
            results[rank] = values
    except Exception as exc:
        error = exc
    finally:
        if error is not None:
            # A worker stuck in a collective (its get never fired) must
            # not keep pytest hanging at exit; a finished worker exits on
            # its own, so it is not terminated (which would corrupt its
            # exit code).
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
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


def _assert_same(ranked: list[dict[str, float]], reference: dict[str, float]) -> None:
    for values in ranked:
        assert set(values) == set(reference)
        for key in reference:
            assert values[key] == pytest.approx(reference[key], abs=1e-9, rel=1e-9), key


@pytest.mark.slow
def test_metrics_equal_the_single_process_values_two_gloo_processes():
    """With the batches split across two real gloo processes (each rank
    holding only its half — MQAR on one rank, overwrite on the other),
    every metric's reduced value equals the single-process accumulation
    over the same batches: RecallMetrics (overall, per slice, per
    class), MacroTokenAccuracy and TokenAccuracy."""
    reference = _accumulate([_make_batch(i) for i in range(4)])
    # Sanity: the split really separates the metric families.
    assert "mqar/acc" in reference and "ow/acc/class=distant" in reference
    ranked = _run_two("split")
    _assert_same(ranked, reference)


@CUDA_ONLY
def test_recall_metrics_reduces_on_cuda_under_nccl(monkeypatch):
    """A fresh ``RecallMetrics`` with CPU counters (a rank that saw no
    batch) moves every clone to the current CUDA device before the
    all-reduces when the process group's backend is NCCL (NCCL rejects
    CPU tensors), and the emission flags land on the same device."""
    all_reduced: list[str] = []

    def fake_all_reduce(tensor, op=None):
        all_reduced.append(tensor.device.type)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "get_backend", lambda: torch.distributed.Backend.NCCL
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    metrics = RecallMetrics(slice_values=(4, 8))
    metrics.compute()  # no updates: every counter is still on the CPU

    assert all_reduced, "no all-reduce ran"
    assert all(device == "cuda" for device in all_reduced), all_reduced


@pytest.mark.slow
def test_metrics_with_an_empty_rank_two_gloo_processes():
    """A rank that saw no evaluation batch still joins the metrics'
    collectives (its counters were created at reset), and both ranks
    read the single-process values."""
    reference = _accumulate([_make_batch(i) for i in range(4)])
    ranked = _run_two("empty")  # rank 1 accumulates nothing
    _assert_same(ranked, reference)
