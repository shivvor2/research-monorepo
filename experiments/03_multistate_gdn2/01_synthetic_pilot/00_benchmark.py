"""
Timing benchmark for the synthetic recall pilot (the Phase-1 step).

Determines, on a given machine, before any protocol run:

- how long one training step takes for each core configuration
  (task x layout x gate x width), at batch sizes 64, 128 and 256;
- whether a run is overhead-bound (kernel launches, Python) or
  throughput-bound (FLOPs, memory bandwidth), from the scaling of the
  step time with the batch size and the GPU-busy fraction measured by
  the profiler;
- whether data generation keeps up with the step time;
- how much memory one run needs;
- how many independent runs per GPU maximize aggregate throughput;

and from those an upper-bound estimate of the grid's GPU-hours per
tier and per phase, the wall-clock at the recommended concurrency, and
the Phase 1 + Phase 2 total against the one-week budget.

Usage:
    python 00_benchmark.py                 # the full pass (GPU)

Outputs (under the launch directory; git-ignored):
    bench/<YYYY-MM-DD_HH-MM-SS>/bench.json
    bench/<YYYY-MM-DD_HH-MM-SS>/summary.md

Everything here reuses the builders of ``01_train.py`` (``build_model``,
``build_module``, ``build_task_config``, ``slice_values``,
``set_precision_flags``) via importlib, so the measured model and the
training step are the real ones. Each measurement seeds everything to 0,
both for the model init and the generated training stream. The step
loop is the training script's body: ``module.forward(**batch)``,
``compute_loss``, ``backward``, one step of every optimizer group,
``zero_grad``. The measured batches are pre-generated and moved to the
device one at a time, so data generation is excluded from the step time
and the device only ever holds the current batch, as in a real run. The
validation pass runs the module's validation path (loss plus
``RecallMetrics``) over the task's fixed validation set at batch size
256.

The concurrency workers are subprocesses of this same script
(``bench.mode=worker``) launched with the variant's config overrides,
so each worker composes the same config the training script would for
that variant; the parent aggregates their reported steps per second
and peak memory. Workers step for ``bench.concurrency_seconds`` after
``bench.warmup_steps`` warmup steps and print one ``BENCH_RESULT`` JSON
line on stdout.

On a machine without CUDA only the step timing, the data time and the
validation time are measured (a cheap smoke check); the GPU-only
readings (busy fraction, peak memory, concurrency) are skipped with a
note, and the grid estimate reports what is not measured.
"""

import hashlib
import importlib.util
import itertools
import json
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import hydra
import lightning as L
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# The training script is loaded by file (as the pilot tests do), so the
# benchmark shares its builders and the resolved-learning-rate logic
# verbatim.
_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "pilot_train_for_benchmark",
    Path(__file__).resolve().parent / "01_train.py",
)
_train = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(_train)

build_model = _train.build_model
build_module = _train.build_module
build_task_config = _train.build_task_config
load_frozen_protocol = _train.load_frozen_protocol
resolve_lr = _train.resolve_lr
set_precision_flags = _train.set_precision_flags
slice_values = _train.slice_values

from pilot_analysis.protocol import EXPECTED_RUNS  # noqa: E402  (sys.path)

from research_lib.data.synthetic import (  # noqa: E402
    RecallMetrics,
    SyntheticRecallDataModule,
)
from research_lib.training.losses import AnswerLogits  # noqa: E402

logger = logging.getLogger(__name__)

CONF_DIR = Path(__file__).resolve().parent / "conf"

# The measured grid's gate modes: the b_only/w_only/gdn runs of the
# protocol map onto these (see proxy_variant below).
MEASURED_GATES = ("kda", "full")


# =============================================================================
# Configuration helpers
# =============================================================================


def variant_config(
    cfg: DictConfig,
    task_file: str,
    layout: str,
    gate_mode: str,
    d_model: int,
    expand_v: float,
) -> DictConfig:
    """The resolved config of one measured variant.

    The benchmark composes variants in-process from the base config
    (config_name="bench" pulls the training config): each variant merges
    the task file and the model overrides the training script would
    receive on its command line. ``model.t_max`` is re-asserted from the
    task's ``max_seq_len`` (the base config interpolates it, and a merge
    cannot re-resolve the interpolation).

    Returns:
        A new config; the caller's ``cfg`` is not mutated.
    """
    task_cfg = OmegaConf.load(CONF_DIR / "task" / f"{task_file}.yaml")
    # A plain copy (non-struct): the composed config is struct-moded, and
    # the task files add keys the base task block does not have (e.g.
    # num_keys for overwrite), which a struct merge would refuse.
    variant = OmegaConf.create(cfg)
    OmegaConf.set_struct(variant, False)
    variant.task = task_cfg
    variant.model.layout = layout
    variant.model.gate_mode = gate_mode
    variant.model.d_model = int(d_model)
    variant.model.expand_v = float(expand_v)
    variant.model.t_max = int(variant.task.max_seq_len)
    return variant


def benchmark_lr(cfg: DictConfig) -> float:
    """The peak LR of every benchmark run.

    The frozen-protocol LR when one exists, ``0.003`` otherwise; the LR
    does not affect the timing (it only scales the AdamW update).
    """
    if cfg.optimizer.lr is not None:
        return float(cfg.optimizer.lr)
    try:
        return resolve_lr(cfg, load_frozen_protocol())
    except RuntimeError:
        return 0.003


def make_datamodule(
    cfg: DictConfig, batch_size: int, num_workers: int = 0
) -> SyntheticRecallDataModule:
    """The task's data module at one batch size; the train stream and the
    fixed validation/test sets are built lazily (no ``setup`` needed)."""
    return SyntheticRecallDataModule(
        task=str(cfg.task.name),
        task_config=build_task_config(cfg),
        batch_size=batch_size,
        num_val_per_slice=cfg.data.num_val_per_slice,
        num_test_per_slice=cfg.data.num_test_per_slice,
        train_seed=0,
        val_seed=cfg.task.val_seed,
        test_seed=cfg.task.test_seed,
        num_workers=num_workers,
    )


def move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """The batch on ``device`` (only its tensors)."""
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def sync(device: torch.device) -> None:
    """Block until the device is idle (a no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def optimizers_of(module: L.LightningModule) -> list[torch.optim.Optimizer]:
    """The module's real optimizer groups (the two AdamW groups).

    ``configure_optimizers`` reads ``self.trainer`` (for its partition
    summary) and raises when the module is unattached; the benchmark
    hands the module a bare trainer that only satisfies that check —
    nothing is ever fitted through it.
    """
    if getattr(module, "_trainer", None) is None:
        module.trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
    configured = module.configure_optimizers()
    if isinstance(configured, dict):
        configured = [configured]
    optimizers: list[torch.optim.Optimizer] = []
    for entry in configured:
        if isinstance(entry, torch.optim.Optimizer):
            optimizers.append(entry)
        elif isinstance(entry, dict) and isinstance(
            entry.get("optimizer"), torch.optim.Optimizer
        ):
            optimizers.append(entry["optimizer"])
    if not optimizers:
        raise RuntimeError("configure_optimizers returned no optimizer")
    return optimizers


def build_benchmark_module(cfg: DictConfig, lr: float) -> L.LightningModule:
    """The seeded training module on CPU (caller moves it to the device)."""
    model = build_model(cfg)
    return build_module(cfg, model, lr)


def pregenerate_batches(
    cfg: DictConfig, batch_size: int, count: int
) -> list[dict[str, torch.Tensor]]:
    """``count`` fresh training batches at one batch size, on CPU.

    Data generation is excluded from the step-time measurement: the
    batches are generated up front and only moved to the device during
    the loop.
    """
    dm = make_datamodule(cfg, batch_size, num_workers=0)
    batches: list[dict[str, torch.Tensor]] = []
    for batch in dm.train_dataloader():
        batches.append(batch)
        if len(batches) >= count:
            break
    return batches


def one_step(
    module: L.LightningModule,
    optimizers: Sequence[torch.optim.Optimizer],
    batch: Mapping[str, torch.Tensor],
) -> None:
    """One training step: forward, loss, backward, one optimizer step."""
    out = module.forward(**batch)
    loss = module.compute_loss(out, batch)
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


# =============================================================================
# Measurements
# =============================================================================


def measure_step_times(
    cfg: DictConfig,
    batch_size: int,
    warmup_steps: int,
    measure_steps: int,
    device: torch.device,
    lr: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Median ms per training step at one batch size, and peak memory.

    Builds the model and the module (``seed`` fixes the init and the
    training stream), pre-generates ``warmup_steps + measure_steps``
    batches (data generation excluded from the timing), steps with
    device synchronization around each measured step, and records
    ``torch.cuda.max_memory_allocated`` over the whole loop (the model,
    the optimizer state and one step's activations; the device holds
    only the current batch).

    Args:
        cfg: The variant's resolved config.
        batch_size: The training batch size.
        warmup_steps: Untimed steps before the measurement.
        measure_steps: Timed steps (the median is reported).
        device: The device to measure on.
        lr: The peak learning rate (does not affect the timing).
        seed: The run seed (model init and training stream).

    Returns:
        ``{"step_ms": float | None, "peak_memory_bytes": int | None,
        "oom": bool}``; ``step_ms`` is ``None`` and ``oom`` is true on
        a CUDA out-of-memory (peak memory is ``None`` without CUDA).
    """
    L.seed_everything(seed)
    module = build_benchmark_module(cfg, lr).to(device)
    optimizers = optimizers_of(module)
    batches = pregenerate_batches(cfg, batch_size, warmup_steps + measure_steps)
    result: dict[str, Any] = {"step_ms": None, "peak_memory_bytes": None, "oom": False}
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times: list[float] = []
        for step in range(warmup_steps + measure_steps):
            batch = move_batch(batches[step], device)
            sync(device)
            start = time.perf_counter()
            one_step(module, optimizers, batch)
            sync(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if step >= warmup_steps:
                times.append(elapsed_ms)
        result["step_ms"] = median(times)
        if device.type == "cuda":
            result["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM at batch size %d (variant %s)", batch_size, cfg.task.name)
        result["oom"] = True
        torch.cuda.empty_cache()
    return result


def measure_busy_fraction(
    cfg: DictConfig,
    batch_size: int,
    profile_steps: int,
    device: torch.device,
    lr: float,
) -> tuple[float, float] | None:
    """(GPU-busy fraction, CUDA kernels per step) at one batch size.

    Profiles ``profile_steps`` training steps with the PyTorch profiler
    (CPU and CUDA activities). Busy = (sum of CUDA kernel device time) /
    (wall time of those steps); kernels per step = profiled kernel
    launches / steps.

    Returns:
        ``None`` without CUDA (the reading is not defined there).
    """
    if device.type != "cuda":
        return None
    from torch.autograd.profiler import DeviceType
    from torch.profiler import ProfilerActivity, profile

    L.seed_everything(0)
    module = build_benchmark_module(cfg, lr).to(device)
    optimizers = optimizers_of(module)
    batches = pregenerate_batches(cfg, batch_size, profile_steps)
    wall_start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for step in range(profile_steps):
            one_step(module, optimizers, move_batch(batches[step], device))
            torch.cuda.synchronize(device)
    wall_us = (time.perf_counter() - wall_start) * 1e6
    # Only the CUDA kernel records count: host dispatch events carry a
    # device-time too (the profiler propagates it), and summing them
    # would double-count the kernels' work.
    kernel_events = [
        event for event in prof.key_averages() if event.device_type == DeviceType.CUDA
    ]
    device_us = sum(event.device_time_total for event in kernel_events)
    kernels = sum(event.count for event in kernel_events)
    busy = min(1.0, device_us / wall_us) if wall_us > 0 else 0.0
    return busy, kernels / profile_steps


def measure_data_time(cfg: DictConfig, batch_size: int, n_batches: int) -> float:
    """Mean ms to produce one training batch in one process.

    The task generator in one process (``num_workers=0``), timing the
    iteration itself (the batch is produced when the iterator advances).
    """
    dm = make_datamodule(cfg, batch_size, num_workers=0)
    loader = iter(dm.train_dataloader())
    times: list[float] = []
    for _ in range(n_batches):
        start = time.perf_counter()
        next(loader)
        times.append((time.perf_counter() - start) * 1000.0)
    return mean(times)


def measure_validation_time(
    cfg: DictConfig, batch_size: int, device: torch.device, lr: float
) -> tuple[float, float]:
    """(Seconds for one full validation pass, seconds to build the set).

    The pass runs the module's validation path — loss plus
    ``RecallMetrics`` — over the task's fixed validation set at
    ``batch_size``, with the set built before the clock starts.
    """
    L.seed_everything(0)
    module = build_benchmark_module(cfg, lr).to(device)
    dm = make_datamodule(cfg, batch_size, num_workers=0)
    build_start = time.perf_counter()
    loader = dm.val_dataloader()  # builds the fixed set on first access
    build_s = time.perf_counter() - build_start
    metrics = RecallMetrics(slice_values(cfg), with_classes=None)
    sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = module.forward(batch["input_ids"], labels=batch["labels"])
            module.compute_loss(out, batch)  # the validation loss path
            if isinstance(out, AnswerLogits):
                metrics.update_answers(out.logits, batch)
            else:
                metrics.update(out, batch)
    metrics.compute()  # the epoch-end aggregation
    sync(device)
    return time.perf_counter() - start, build_s


# =============================================================================
# Readings (pure, tested on hand-made numbers)
# =============================================================================


def scaling_ratio(step_ms: Mapping[int, float | None]) -> float | None:
    """t(256) / t(64): a value near 1 means the step does not grow with
    the batch size (overhead-bound); a value near 4 means it grows
    proportionally (throughput-bound)."""
    if step_ms.get(64) and step_ms.get(256):
        return step_ms[256] / step_ms[64]
    return None


def classify_reading(scaling: float | None, busy: float | None) -> str:
    """The overhead/throughput reading of one variant.

    Reading rules:
    - ``overhead-bound`` when the step time is nearly flat in the batch
      size (scaling < 2) and the GPU-busy fraction is low (< 0.5):
      kernel launches and Python dominate, several independent runs per
      GPU raise the throughput;
    - ``throughput-bound`` when the step time grows roughly with the
      batch size (scaling > 3) and the GPU is busy (> 0.8): one run per
      GPU is right;
    - ``mixed`` otherwise.
    Either number missing (no CUDA, or an OOM at batch size 256):
    ``"not measured"``.
    """
    if scaling is None or busy is None:
        return "not measured"
    if scaling < 2.0 and busy < 0.5:
        return "overhead-bound"
    if scaling > 3.0 and busy > 0.8:
        return "throughput-bound"
    return "mixed"


def data_bound(data_ms: float | None, step_ms: float | None, num_workers: int) -> bool:
    """True when data generation cannot keep up: the data time per batch
    exceeds the step time times the worker count (with one worker, the
    generation of batch k overlaps the step of batch k - 1)."""
    if num_workers <= 0 or data_ms is None or step_ms is None:
        return False
    return data_ms > step_ms * num_workers


def recommended_concurrency(aggregates: Sequence[float]) -> int:
    """The smallest N (1-based) whose aggregate throughput is within 10%
    of the best measured (``aggregates`` in increasing-N order)."""
    if not aggregates:
        return 1
    threshold = 0.9 * max(aggregates)
    for n, value in enumerate(aggregates, start=1):
        if value >= threshold:
            return n
    return len(aggregates)


# =============================================================================
# Concurrency (one worker per process, bench.mode=worker)
# =============================================================================


def worker_main(cfg: DictConfig) -> None:
    """One concurrency worker: steps from ``bench.start_at`` (a shared
    wall-clock start set by the parent) until ``bench.start_at +
    bench.concurrency_seconds`` — a shared end time, not its own
    duration — after ``bench.warmup_steps`` warmup steps, and prints one
    ``BENCH_RESULT`` JSON line on stdout.

    A worker that reaches the window after the shared start still
    measures until the shared end and reports ``"late": true``: its
    rate covers only part of the window, so the group's speedup may
    read high (the summary says so). Without ``start_at`` the worker
    falls back to measuring its own ``concurrency_seconds``.

    The worker is a subprocess of the benchmark itself, launched with
    the variant's config overrides; the parent aggregates the lines.
    The training batches stream from the task generator with the
    configured worker count, as in a real run.
    """
    set_precision_flags()  # fp32 without TF32, like the training script
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg.bench.batch_sizes[0])
    seconds = float(cfg.bench.concurrency_seconds)
    warmup_steps = int(cfg.bench.warmup_steps)
    start_at = cfg.bench.get("start_at", None)
    start_at = float(start_at) if start_at is not None else None
    lr = benchmark_lr(cfg)
    logger.info(
        "Benchmark worker: %s, %s, d=%s, B=%d, %.0f s after %d warmup steps%s",
        cfg.bench.task_file,
        cfg.model.layout,
        cfg.model.d_model,
        batch_size,
        seconds,
        warmup_steps,
        f", shared start {start_at:.0f}" if start_at is not None else "",
    )
    L.seed_everything(0)
    module = build_benchmark_module(cfg, lr).to(device)
    optimizers = optimizers_of(module)
    dm = make_datamodule(cfg, batch_size, num_workers=int(cfg.data.num_workers))
    batches = iter(dm.train_dataloader())
    result: dict[str, Any] = {
        "steps": 0,
        "elapsed_s": 0.0,
        "peak_memory_bytes": None,
        "oom": False,
        "late": False,
    }
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup_steps):
            one_step(module, optimizers, move_batch(next(batches), device))
        sync(device)
        if start_at is not None:
            if time.time() < start_at:
                while time.time() < start_at:
                    time.sleep(min(1.0, start_at - time.time()))
            else:
                result["late"] = True
            window_end = start_at + seconds
        else:
            window_end = time.time() + seconds
        start = time.perf_counter()
        while time.time() < window_end:
            one_step(module, optimizers, move_batch(next(batches), device))
            result["steps"] += 1
        sync(device)
        result["elapsed_s"] = time.perf_counter() - start
        if device.type == "cuda":
            result["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
    print(f"BENCH_RESULT {json.dumps(result)}", flush=True)


def parse_worker_result(output: str) -> dict[str, Any] | None:
    """The worker's ``BENCH_RESULT`` JSON line, or ``None``."""
    for line in output.splitlines():
        if line.startswith("BENCH_RESULT "):
            return json.loads(line[len("BENCH_RESULT ") :])
    return None


def summarize_group(
    worker_outputs: Sequence[tuple[int | str, dict[str, Any] | None]],
    first_aggregate: float | None = None,
) -> dict[str, Any]:
    """One group entry from the parsed worker results.

    A worker counts as failed on a non-zero exit, a missing
    ``BENCH_RESULT`` line or a timeout. The aggregate is the sum of the
    surviving workers' rates; the speedup needs **both** this group's
    aggregate and the first group's, and stays ``None`` otherwise (a
    failed group must never crash the benchmark). A worker that started
    its timed window late marks the group ``synchronized: False`` (its
    rate covers only part of the window, so the aggregate and the
    speedup may read high).

    Args:
        worker_outputs: Per worker: ``(returncode or "timeout", parsed
            BENCH_RESULT dict or None)``.
        first_aggregate: The N=1 group's aggregate throughput (the
            speedup denominator); ``None`` when the first group itself
            has no aggregate.

    Returns:
        The group entry: ``workers`` (rate, peak memory, late flag),
        ``aggregate_steps_per_s``, ``speedup``, ``synchronized``,
        ``oom`` and ``failed``.
    """
    group: dict[str, Any] = {
        "workers": [],
        "aggregate_steps_per_s": None,
        "speedup": None,
        "synchronized": True,
        "oom": False,
        "failed": 0,
    }
    rates: list[float] = []
    for status, result in worker_outputs:
        if status != 0 or result is None:
            group["failed"] += 1
            continue
        rate = result["steps"] / result["elapsed_s"] if result["elapsed_s"] > 0 else 0.0
        group["workers"].append(
            {
                "steps_per_s": rate,
                "peak_memory_bytes": result.get("peak_memory_bytes"),
                "late": bool(result.get("late", False)),
            }
        )
        if result.get("oom"):
            group["oom"] = True
        if result.get("late"):
            group["synchronized"] = False
        rates.append(rate)
    if rates:
        group["aggregate_steps_per_s"] = sum(rates)
    if first_aggregate and group["aggregate_steps_per_s"] is not None:
        group["speedup"] = group["aggregate_steps_per_s"] / first_aggregate
    return group


def worker_command(
    cfg: DictConfig,
    task_file: str,
    d_model: int,
    batch_size: int,
    seconds: int,
    start_at: float,
) -> list[str]:
    """One concurrency worker's config overrides.

    The parent's half of the shared-window protocol: the worker steps
    from ``start_at`` until ``start_at + seconds`` (a shared end time,
    not its own duration), so the N workers' timed windows overlap.

    Args:
        cfg: The resolved benchmark config (the batch size and window
            length come from it).
        task_file: The task-file label (also the Hydra task override).
        d_model: The model width.
        batch_size: The training batch size.
        seconds: The timed window's length.
        start_at: The shared window start (epoch seconds, wall clock).

    Returns:
        The override list to append to the subprocess command.
    """
    return [
        "bench.mode=worker",
        f"bench.task_file={task_file}",
        f"task={task_file}",
        "model.layout=ws",
        "model.gate_mode=full",
        f"model.d_model={d_model}",
        f"bench.batch_sizes=[{batch_size}]",
        f"bench.concurrency_seconds={seconds}",
        f"bench.start_at={start_at:.3f}",
    ]


def run_concurrency_groups(
    cfg: DictConfig, task_file: str, d_model: int, device: torch.device
) -> list[dict[str, Any]]:
    """Aggregate throughput of ``N`` concurrent workers, N in
    ``bench.concurrency``, for one variant (task_file, ws, full, d).

    One group per N. Every worker of a group steps inside the **same**
    timed window: the parent picks ``start_at = now + start_margin_s +
    15 s per worker`` (the margin covers the workers' start-up), passes
    it to the workers, and the workers step until the shared end. A
    worker that only joins the window after the shared start reports
    ``late`` and marks the group ``synchronized: false``. Each worker is
    a subprocess of this script (``bench.mode=worker``) that composes
    the variant's config itself; the parent parses the workers'
    ``BENCH_RESULT`` lines through :func:`summarize_group`. The N
    sequence stops at the first group whose workers OOM or fail.

    Returns:
        One entry per measured N; without CUDA a single entry with
        ``"skipped": "no CUDA"``.
    """
    if device.type != "cuda":
        return [{"n": 1, "skipped": "no CUDA"}]
    script = Path(__file__).resolve()
    batch_size = int(cfg.bench.batch_sizes[-1])
    seconds = int(cfg.bench.concurrency_seconds)
    groups: list[dict[str, Any]] = []
    for n in cfg.bench.concurrency:
        margin = float(cfg.bench.start_margin_s) + 15.0 * int(n)
        start_at = time.time() + margin
        command = [sys.executable, "-u", str(script)] + worker_command(
            cfg, task_file, d_model, batch_size, seconds, start_at
        )
        logger.info(
            "Concurrency group N=%d (%s, ws, full, d=%d): %d workers, "
            "shared window starting in %.0f s...",
            n,
            task_file,
            d_model,
            n,
            margin,
        )
        processes = [
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for _ in range(int(n))
        ]
        worker_outputs: list[tuple[int | str, dict[str, Any] | None]] = []
        for process in processes:
            output = process.communicate()[0]
            if process.returncode != 0:
                tail = "\n".join(output.splitlines()[-5:])
                logger.error(
                    "benchmark worker failed (exit %s):\n%s",
                    process.returncode,
                    tail,
                )
            worker_outputs.append((process.returncode, parse_worker_result(output)))
        first_aggregate = groups[0].get("aggregate_steps_per_s") if groups else None
        group = summarize_group(worker_outputs, first_aggregate)
        group["n"] = int(n)
        groups.append(group)
        if group["oom"] or group["failed"]:
            logger.warning("concurrency group N=%d: OOM or worker failure; stopping", n)
            break
    return groups


# =============================================================================
# Grid estimate (pure, tested on hand-made numbers)
# =============================================================================


@dataclass(frozen=True)
class VariantKey:
    """One measured cell of the timing grid."""

    task_file: str  # mqar | overwrite | overwrite_dk32 | mqar_longT
    layout: str  # single | ws | fs | ll
    gate_mode: str  # one of MEASURED_GATES
    d_model: int
    expand_v: float  # 1.0 in the core grid; the matched-bytes tier uses 10


@dataclass(frozen=True)
class RunSpec:
    """One run of a protocol tier (a sweep-grid job, resolved)."""

    tier: str
    task: str  # the sweep grid's task (mqar, overwrite, mqar_longT, ...)
    d_model: int  # the width the run will use (anchors set their own)
    layout: str
    gate_mode: str
    lambda_mode: str
    expand_v: float


@dataclass
class TierEstimate:
    """The grid estimate of one tier: per-run seconds and serial
    GPU-hours, from the measured timings."""

    tier: str
    runs: int
    measured: int
    per_run_s: float | None
    gpu_hours: float | None
    note: str = ""


def nearest_measured_d(d_model: int) -> int:
    """The nearest measured width (the grid measures 32 and 64; the
    16-wide anchors proxy onto 32)."""
    return 32 if d_model <= 32 else 64


def sweeper_values(value) -> list[str]:
    """Every value of a --multirun sweeper parameter, in order.

    ``"a,b,c"`` -> ``["a", "b", "c"]``; ``"range(0,5)"`` ->
    ``["0", ..., "4"]``; a single value passes through. The same
    expansion as Hydra's BasicSweeper (itertools.product over the
    parameters in file order).
    """
    value = str(value)
    if value.startswith("range("):
        inner = value[value.index("(") + 1 : value.rindex(")")]
        parts = [part.strip() for part in inner.split(",")]
        start, stop = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) > 2 else 1
        return [str(v) for v in range(start, stop, step)]
    return [part.strip() for part in value.split(",")]


def sweep_jobs(sweep_file: Path) -> list[dict[str, str]]:
    """The sweep's jobs in Hydra's order (the Cartesian product of the
    parameter values in file order)."""
    params = OmegaConf.select(
        OmegaConf.load(sweep_file), "hydra.sweeper.params", default={}
    )
    names = list(params.keys())
    grids = [sweeper_values(params[name]) for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*grids)]


def resolve_run(
    cfg: DictConfig,
    sweep_file: Path,
    job: dict[str, str],
    d_interpretation: int,
    tier: str,
) -> RunSpec:
    """One sweep-grid job resolved against the base config.

    Values the job does not set come from the benchmark's own composed
    config (the training config the sweep overlays, exactly as in the
    training script's multirun). The width of a run the sweep does not
    pin is the interpretation (the width the pilot will have calibrated
    to); the anchor jobs pin their own width through
    ``conf/anchor/*.yaml`` and ``calib_dmodel`` sweeps its own grid.

    Args:
        cfg: The benchmark's composed base config.
        sweep_file: The tier's sweep file (for the anchor overrides'
            directory).
        job: The expanded parameter dict of one sweep job.
        d_interpretation: The calibrated width the estimate assumes
            (32 or 64).
        tier: The tier name (the sweep files carry it).
    """
    task = str(job.get("task", cfg.task.name))
    layout = str(job.get("model.layout", cfg.model.layout))
    gate_mode = str(job.get("model.gate_mode", cfg.model.gate_mode))
    lambda_mode = str(job.get("model.lambda_mode", cfg.model.lambda_mode))
    expand_v = float(job.get("model.expand_v", cfg.model.expand_v))
    anchor = job.get("+anchor")
    if anchor is not None:
        anchor_cfg = OmegaConf.load(CONF_DIR / "anchor" / f"{anchor}.yaml")
        d_model = int(
            OmegaConf.select(anchor_cfg, "model.d_model", default=d_interpretation)
        )
    elif "model.d_model" in job:
        d_model = int(job["model.d_model"])
    else:
        d_model = int(d_interpretation)
    return RunSpec(
        tier=tier,
        task=task,
        d_model=d_model,
        layout=layout,
        gate_mode=gate_mode,
        lambda_mode=lambda_mode,
        expand_v=expand_v,
    )


def collect_runs(cfg: DictConfig, d_interpretation: int) -> list[RunSpec]:
    """Every protocol run, per tier, from the sweep grids.

    The tier list and the composition come from ``conf/sweep/*.yaml``;
    a count mismatch against the protocol's expected counts is logged
    (the protocol table is the gate that keeps the two in sync).
    """
    runs: list[RunSpec] = []
    for sweep_file in sorted((CONF_DIR / "sweep").glob("*.yaml")):
        sweep_cfg = OmegaConf.load(sweep_file)
        tier = str(OmegaConf.select(sweep_cfg, "experiment.tier", default=None))
        jobs = sweep_jobs(sweep_file)
        expected = EXPECTED_RUNS.get(tier)
        if expected is not None and len(jobs) != expected:
            logger.warning(
                "tier %s: the sweep grid expands to %d runs, the protocol "
                "expects %d",
                tier,
                len(jobs),
                expected,
            )
        runs.extend(
            resolve_run(cfg, sweep_file, job, d_interpretation, tier) for job in jobs
        )
    return runs


def proxy_variant(run: RunSpec) -> tuple[VariantKey | None, str]:
    """The measured cell whose timing proxies a run, and a label.

    Mapping (the measured grid is task files {mqar, overwrite,
    overwrite_dk32, mqar_longT} at widths {32, 64}, layouts {single, ws,
    fs, ll}, gates {kda, full}, lambda mlp, plus the expand_v=10 and the
    long-T flavours at batch size 256):

    - ``b_only`` / ``w_only`` use the ``full`` timing of the same layout
      (the gate ablations share the full gate's projections);
    - ``gdn`` (the anchors) uses the ``kda`` timing of the same layout
      at the nearest measured width (16 -> 32);
    - λ ``static`` / ``lin`` use the measured λ-``mlp`` timing: the mlp
      module is the biggest of the three, so a run with a smaller λ
      module only gets faster and the estimate stays an upper bound;
    - ``expand_v=10`` (matched bytes) uses the measured expand_v=10
      flavour;
    - the long-sequence tier uses the measured long-T flavour: its
      timing is dominated by the sequence length, not the layout or the
      gate;
    - every other run uses its own cell. ``overwrite`` at width 64 runs
      as ``overwrite_dk32`` (the task file the calibration would have
      switched to).

    Returns:
        ``(key, label)``; ``key`` is ``None`` when no measured cell
        exists (the grid cell was never measured).
    """
    if run.task == "mqar_longT":
        return (
            VariantKey(
                "mqar_longT", "ll", "full", nearest_measured_d(run.d_model), 1.0
            ),
            "long-T flavour",
        )
    gate = "full" if run.gate_mode in ("b_only", "w_only") else run.gate_mode
    gate = "kda" if gate == "gdn" else gate
    if gate not in MEASURED_GATES:
        return None, f"gate {run.gate_mode!r}"
    task_file = run.task
    if task_file == "overwrite" and run.d_model == 64:
        task_file = "overwrite_dk32"
    expand_v = 10.0 if run.expand_v != 1.0 else 1.0
    if expand_v == 10.0 and (run.layout, gate) != ("single", "full"):
        return None, "expand_v=10 outside (single, full)"
    if expand_v == 10.0 and task_file == "mqar_longT":
        return None, "expand_v=10 on the long sequence"
    return (
        VariantKey(
            task_file,
            run.layout,
            gate,
            nearest_measured_d(run.d_model),
            expand_v,
        ),
        "",
    )


def per_run_seconds(
    step_ms: float, val_s: float, max_steps: int, val_check_interval: int
) -> float:
    """Upper-bound time of one complete run, seconds.

    ``max_steps`` steps at the batch-256 step time, a validation pass
    every ``val_check_interval`` steps, and one final test pass
    (test ~ validation). An upper bound: it assumes no early stopping.
    """
    return (
        max_steps * step_ms / 1000.0 + (max_steps / val_check_interval) * val_s + val_s
    )


def estimate_grid(
    runs: Sequence[RunSpec],
    timings: Mapping[VariantKey, float],
    val_s: Mapping[tuple[str, int], float],
    max_steps: int,
    val_check_interval: int,
) -> list[TierEstimate]:
    """Serial GPU-hours per tier from the measured timings.

    Args:
        runs: Every protocol run (one entry per sweep job).
        timings: Batch-256 median step time in ms per measured cell
            (missing cell = not measured).
        val_s: Validation-pass seconds per (task file, width).
        max_steps: The training length (min = cfg.training.max_steps).
        val_check_interval: Validation cadence in steps.

    Returns:
        One :class:`TierEstimate` per tier, in the tiers' first-appearance
        order (the sweep order).
    """
    estimates: list[TierEstimate] = []
    for tier, tier_runs in itertools.groupby(runs, key=lambda run: run.tier):
        tier_runs = list(tier_runs)
        seconds: list[float] = []
        missing: list[str] = []
        for run in tier_runs:
            key, label = proxy_variant(run)
            step_ms = timings.get(key) if key is not None else None
            validation = (
                val_s.get((key.task_file, key.d_model)) if key is not None else None
            )
            if step_ms is None or validation is None:
                missing.append(
                    label or f"{key.task_file}/{key.layout}/{key.gate_mode}"
                    if key is not None
                    else run.gate_mode
                )
                continue
            seconds.append(
                per_run_seconds(step_ms, validation, max_steps, val_check_interval)
            )
        if seconds:
            estimates.append(
                TierEstimate(
                    tier=tier,
                    runs=len(tier_runs),
                    measured=len(seconds),
                    per_run_s=mean(seconds),
                    gpu_hours=sum(seconds) / 3600.0,
                )
            )
        else:
            reasons = ", ".join(sorted(set(missing))) or "the cell was never measured"
            estimates.append(
                TierEstimate(
                    tier=tier,
                    runs=len(tier_runs),
                    measured=0,
                    per_run_s=None,
                    gpu_hours=None,
                    note=f"not measured ({reasons})",
                )
            )
    return estimates


# =============================================================================
# Provenance (the bench.json header)
# =============================================================================


def _repo_root() -> Path | None:
    """The repository containing this checkout (worktree-safe)."""
    root = Path(__file__).resolve()
    for parent in root.parents:
        if (parent / ".git").exists():
            return parent
    return None


def git_info() -> dict[str, str | None]:
    """Commit hash and dirty flag of the checkout running the benchmark."""
    root = _repo_root()
    if root is None:
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError) as e:
        logger.warning("benchmark: git info unavailable: %s", e)
        return {"commit": None, "dirty": None}


def lock_hash() -> str | None:
    """SHA-256 of the repository's uv.lock (``None`` when missing)."""
    root = _repo_root()
    if root is None:
        return None
    try:
        return hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    except OSError:
        return None


def versions() -> dict[str, str | None]:
    """Python, torch and the pilot's other runtime versions."""
    import importlib.metadata

    def package(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "torch": package("torch"),
        "cuda_runtime": torch.version.cuda,
        "lightning": package("lightning"),
        "hydra_core": package("hydra-core"),
        "omegaconf": package("omegaconf"),
    }


def device_info() -> dict[str, Any]:
    """The GPU(s) the benchmark sees, ``count == 0`` on a CPU machine."""
    info: dict[str, Any] = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "name": None,
        "capability": None,
        "driver": None,
    }
    if not torch.cuda.is_available():
        return info
    info["name"] = torch.cuda.get_device_name(0)
    info["capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability(0))
    try:
        driver = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            .stdout.strip()
            .splitlines()
        )
        info["driver"] = driver[0].strip() if driver else None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        info["driver"] = None
    return info


# =============================================================================
# The benchmark run
# =============================================================================


def write_bench_json(results: Mapping[str, Any], out_dir: Path) -> None:
    """A snapshot of the results dict as ``out_dir/bench.json``, atomically.

    The benchmark writes a snapshot after every measured variant and
    after each stage, so a crash or a Ctrl-C can never discard the
    measurements already made: whatever finished is on disk, with
    ``complete: false`` until the final write.

    Args:
        results: The current results dict (``complete`` flips only in
            the final write).
        out_dir: The benchmark's output directory.
    """
    from research_lib.utils.io import atomic_write

    def write(file) -> None:
        file.write(json.dumps(results, indent=2))
        file.write("\n")

    atomic_write(out_dir / "bench.json", write, "w", encoding="utf-8")


def measure_all(cfg: DictConfig, device: torch.device, out_dir: Path) -> dict[str, Any]:
    """Run every measurement and the grid estimate; the results dict is
    both ``bench.json`` and the input of :func:`render_summary`.

    ``bench.json`` is written after every measured variant and after
    each stage (``complete: false`` until the final write), so an
    interrupted or failed benchmark keeps everything that was measured.
    The concurrency stage is protected: a failure there is recorded in
    the notes and the estimate and the summary still run. On success
    the summary is rendered and written, and the final ``bench.json``
    carries ``complete: true``.

    Args:
        cfg: The resolved benchmark config.
        device: The device to measure on.
        out_dir: The benchmark's output directory (``bench/<timestamp>``
            for a real run; any directory in the tests).
    """
    lr = benchmark_lr(cfg)
    has_cuda = device.type == "cuda"
    results: dict[str, Any] = {
        "complete": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "device": device_info(),
        "versions": versions(),
        "git": git_info(),
        "lock_hash": lock_hash(),
        "precision_flags": set_precision_flags(),
        "bench": {
            "d_models": list(cfg.bench.d_models),
            "tasks": list(cfg.bench.tasks),
            "layouts": list(cfg.bench.layouts),
            "gate_modes": list(cfg.bench.gate_modes),
            "extra_variants": [list(v) for v in cfg.bench.extra_variants],
            "batch_sizes": list(cfg.bench.batch_sizes),
            "warmup_steps": int(cfg.bench.warmup_steps),
            "measure_steps": int(cfg.bench.measure_steps),
            "profile_steps": int(cfg.bench.profile_steps),
            "data_batches": int(cfg.bench.data_batches),
            "concurrency": list(cfg.bench.concurrency),
            "concurrency_seconds": float(cfg.bench.concurrency_seconds),
        },
        "num_workers": int(cfg.data.num_workers),
        "lr": lr,
        "max_steps": int(cfg.training.max_steps),
        "val_check_interval": int(cfg.training.val_check_interval),
        "variants": [],
        "task_timings": [],
        "concurrency": [],
        "estimates": {},
        "notes": [],
    }

    variants_by_key: dict[VariantKey, dict[str, Any]] = {}
    task_timings: dict[tuple[str, int], dict[str, Any]] = {}

    def attach_task_timing(task_file: str, d_model: int) -> dict[str, Any]:
        key = (task_file, d_model)
        if key in task_timings:
            return task_timings[key]
        vcfg = variant_config(cfg, task_file, "ws", "full", d_model, 1.0)
        timing = measure_validation_time(
            vcfg, int(cfg.bench.batch_sizes[-1]), device, lr
        )
        data_ms = measure_data_time(
            vcfg, int(cfg.bench.batch_sizes[-1]), int(cfg.bench.data_batches)
        )
        task_timings[key] = {
            "task_file": task_file,
            "d_model": d_model,
            "data_ms": data_ms,
            "val_s": timing[0],
            "val_set_gen_s": timing[1],
        }
        results["task_timings"].append(task_timings[key])
        logger.info(
            "Task %s at d=%d: data %.1f ms/batch, validation %.1f s (+ %.1f s "
            "to build the set)",
            task_file,
            d_model,
            data_ms,
            timing[0],
            timing[1],
        )
        return task_timings[key]

    def measure_variant(
        variant_cfg: DictConfig, key: VariantKey, batch_sizes: Sequence[int]
    ) -> dict[str, Any]:
        """One variant at every given batch size (+ busy fraction at 256)."""
        step_ms: dict[int, float | None] = {}
        peak_memory: dict[int, int | None] = {}
        oom_at: int | None = None
        for batch_size in batch_sizes:
            m = measure_step_times(
                variant_cfg,
                int(batch_size),
                int(cfg.bench.warmup_steps),
                int(cfg.bench.measure_steps),
                device,
                lr,
            )
            step_ms[int(batch_size)] = m["step_ms"]
            peak_memory[int(batch_size)] = m["peak_memory_bytes"]
            if m["oom"]:
                oom_at = int(batch_size)
                break  # larger batch sizes can only OOM too
        busy = kernels = None
        if has_cuda and oom_at is None and step_ms.get(256) is not None:
            measured = measure_busy_fraction(
                variant_cfg, 256, int(cfg.bench.profile_steps), device, lr
            )
            if measured is not None:
                busy, kernels = measured
        task_timing = attach_task_timing(key.task_file, key.d_model)
        reading = classify_reading(scaling_ratio(step_ms), busy)
        warning = data_bound(
            task_timing["data_ms"],
            step_ms.get(256),
            int(cfg.data.num_workers),
        )
        row = {
            "task_file": key.task_file,
            "layout": key.layout,
            "gate_mode": key.gate_mode,
            "d_model": key.d_model,
            "expand_v": key.expand_v,
            "step_ms": {str(bs): step_ms[bs] for bs in batch_sizes},
            "peak_memory_bytes": {str(bs): peak_memory[bs] for bs in batch_sizes},
            "scaling_ratio": scaling_ratio(step_ms),
            "busy": busy,
            "kernels_per_step": kernels,
            "reading": reading,
            "data_bound_warning": warning,
            "oom_at": oom_at,
        }
        variants_by_key[key] = row
        results["variants"].append(row)
        write_bench_json(results, out_dir)
        logger.info(
            "Measured %s layout=%s gate=%s d=%d%s: %s ms at B=256, %s",
            key.task_file,
            key.layout,
            key.gate_mode,
            key.d_model,
            " ev=10" if key.expand_v != 1.0 else "",
            f"{step_ms.get(256):.1f}" if step_ms.get(256) is not None else "OOM",
            reading,
        )
        return row

    def task_file_for(task: str, d_model: int) -> str:
        """The task file a variant of ``task`` runs as at width ``d``
        (the overwrite task's file at 64 is overwrite_dk32)."""
        return "overwrite_dk32" if task == "overwrite" and d_model == 64 else task

    # 1. The core grid: task x layout x gate x width, at every batch size.
    # Without CUDA the full grid would take hours of CPU time; the step
    # timing of the first variant plus the data/validation times are the
    # smoke check, and the GPU-only readings are skipped below.
    core_variants = [
        (
            task_file_for(str(task), int(d_model)),
            str(layout),
            str(gate_mode),
            int(d_model),
        )
        for d_model in cfg.bench.d_models
        for task in cfg.bench.tasks
        for layout in cfg.bench.layouts
        for gate_mode in cfg.bench.gate_modes
    ]
    if not has_cuda:
        core_variants = core_variants[:1]
        results["notes"].append(
            f"CPU smoke mode: only the first variant ({core_variants[0][0]}, "
            f"{core_variants[0][1]}, {core_variants[0][2]}, d={core_variants[0][3]}) "
            f"is stepped; everything else reports not measured"
        )
    for task_file, layout, gate_mode, d_model in core_variants:
        key = VariantKey(task_file, layout, gate_mode, d_model, 1.0)
        variant_cfg = variant_config(cfg, task_file, layout, gate_mode, d_model, 1.0)
        measure_variant(variant_cfg, key, cfg.bench.batch_sizes)

    # 2. The extra variants at batch size 256: tier-5 matched bytes and
    # the long-T flavour (each per width; a variant without a task=
    # override wraps the core task files). Skipped in CPU smoke mode.
    if has_cuda:
        for extra in cfg.bench.extra_variants:
            overrides = dict(item.split("=", 1) for item in extra)
            for d_model in cfg.bench.d_models:
                if "task" in overrides:
                    task_files = [overrides["task"]]
                else:
                    task_files = [
                        task_file_for(str(task), int(d_model))
                        for task in cfg.bench.tasks
                    ]
                for task_file in task_files:
                    key = VariantKey(
                        task_file,
                        overrides.get("model.layout", str(cfg.model.layout)),
                        overrides.get("model.gate_mode", str(cfg.model.gate_mode)),
                        int(d_model),
                        float(overrides.get("model.expand_v", cfg.model.expand_v)),
                    )
                    if key in variants_by_key:
                        continue
                    variant_cfg = variant_config(
                        cfg,
                        task_file,
                        key.layout,
                        key.gate_mode,
                        key.d_model,
                        key.expand_v,
                    )
                    measure_variant(variant_cfg, key, [int(cfg.bench.batch_sizes[-1])])

    # 3. Concurrency: (task_file, ws, full) at each width — the variant
    # the recommended runs-per-GPU reading is anchored on. The stage is
    # protected: a failure must never discard the measurements already
    # made — it is recorded in the notes and the estimate and the
    # summary still run.
    for d_model in cfg.bench.d_models:
        task_file = task_file_for("overwrite", int(d_model))
        try:
            groups = run_concurrency_groups(cfg, task_file, int(d_model), device)
        except Exception as exc:  # noqa: BLE001 — recorded, not fatal
            logger.error("concurrency stage failed: %s", exc)
            results["notes"].append(f"concurrency stage failed: {exc}")
            groups = [{"n": 1, "skipped": "the concurrency stage failed"}]
        aggregate = [
            group["aggregate_steps_per_s"]
            for group in groups
            if group.get("aggregate_steps_per_s") is not None and not group["oom"]
        ]
        results["concurrency"].append(
            {
                "d_model": int(d_model),
                "task_file": task_file,
                "groups": groups,
                "recommended_runs_per_gpu": (
                    recommended_concurrency(aggregate) if aggregate else None
                ),
            }
        )
        write_bench_json(results, out_dir)

    # 4. The grid estimate for both width interpretations.
    max_steps = int(cfg.training.max_steps)
    val_check_interval = int(cfg.training.val_check_interval)
    step_ms_at_256 = {
        key: row["step_ms"].get("256")
        for key, row in variants_by_key.items()
        if row["step_ms"].get("256") is not None
    }
    val_s = {
        (entry["task_file"], entry["d_model"]): entry["val_s"]
        for entry in task_timings.values()
    }
    frozen = load_frozen_protocol()
    frozen_d = frozen.get("d_model", None)
    frozen_d = int(frozen_d) if isinstance(frozen_d, int) else frozen_d
    for d_interpretation in (32, 64):
        results["estimates"][str(d_interpretation)] = {
            "tiers": [
                {
                    "tier": estimate.tier,
                    "runs": estimate.runs,
                    "measured": estimate.measured,
                    "per_run_s": estimate.per_run_s,
                    "gpu_hours": estimate.gpu_hours,
                    "note": estimate.note,
                }
                for estimate in estimate_grid(
                    collect_runs(cfg, d_interpretation),
                    step_ms_at_256,
                    val_s,
                    max_steps,
                    val_check_interval,
                )
            ]
        }
    if frozen_d is not None:
        results["notes"].append(f"frozen protocol width: d_model={frozen_d}")
    write_bench_json(results, out_dir)

    # 5. Finalize: the summary is rendered and written, and the final
    # bench.json says the run is complete. Anything that raised before
    # here left the last snapshot on disk with "complete": false.
    results["complete"] = True
    summary = render_summary(results)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_bench_json(results, out_dir)
    return results


# =============================================================================
# The summary document (pure rendering of the results dict)
# =============================================================================

SUMMARY_EXPLANATION = """\
- **ms per step** is the median over `measure_steps` timed steps after
  `warmup_steps` untimed ones, at that batch size.
- **[t256/t64]** is the step time at B=256 over B=64: near 1 the step
  does not grow with the batch size (overhead-bound), near 4 it grows
  proportionally (throughput-bound).
- **busy** is the GPU-busy fraction: CUDA kernel device time over the
  wall time of `profile_steps` profiled steps.
- **peak MB** is `torch.cuda.max_memory_allocated` over the step loop
  (model + optimizer state + one step's activations).
- **reading**: `overhead-bound` (run several independent runs per GPU),
  `throughput-bound` (one run per GPU), `mixed`, or `not measured`
  (no CUDA / OOM at B=256).
- **data warning**: data time per batch exceeds the step time times
  `data.num_workers`, so generation cannot keep up.
"""


def render_summary(results: Mapping[str, Any]) -> str:
    """The benchmark's ``summary.md`` from the results dict."""
    import pandas as pd
    from pilot_analysis.render import markdown_table

    lines: list[str] = ["# Multi-state GDN-2 pilot: timing benchmark", ""]
    if results.get("complete", False):
        lines.append("**Status: complete.**")
    else:
        lines.append(
            "**Status: incomplete — this is a partial snapshot from a run "
            "that did not finish (`complete: false` in bench.json); the "
            "measurements below are what survived.**"
        )
    lines.append("")
    env = results["device"]
    versions = results["versions"]
    git = results["git"]

    def kv(key: str, value: Any, unit: str = "") -> str:
        if value is None:
            return f"- **{key}:** *not measured*"
        return f"- **{key}:** {value}{unit}"

    lines.append("## Environment")
    lines.extend(
        [
            f"Generated {results['generated_at']} by `{results['command']}`.",
            "",
            kv(
                "device",
                env["name"] or ("CPU (no CUDA)" if env["count"] == 0 else "?"),
                f" x {env['count']}",
            ),
            kv("driver", env["driver"]),
            kv("CUDA capability", env["capability"]),
            kv(
                "torch",
                versions.get("torch", versions.get("cuda_runtime")),
                (
                    f" (CUDA {versions.get('cuda_runtime')})"
                    if versions.get("cuda_runtime")
                    else ""
                ),
            ),
            kv("lightning", versions.get("lightning")),
            kv("hydra-core", versions.get("hydra_core")),
            kv("omegaconf", versions.get("omegaconf")),
            kv("python", versions.get("python")),
            kv(
                "git commit",
                git.get("commit"),
                f" (dirty={git.get('dirty')})" if git.get("commit") else "",
            ),
            kv(
                "uv.lock sha256",
                (
                    (results.get("lock_hash") or "")[:12]
                    if results.get("lock_hash")
                    else None
                ),
            ),
            kv("precision flags (TF32 off)", results.get("precision_flags")),
            kv("data.num_workers", results.get("num_workers")),
            "",
            "The benchmark is fp32 without TF32, like the protocol runs.",
            "",
        ]
    )

    lines.append("## Step time and GPU usage per variant")
    lines.extend(["", SUMMARY_EXPLANATION, ""])
    rows = []
    for row in results["variants"]:
        expand_v = row["expand_v"]
        step_ms = row["step_ms"]
        peak = row["peak_memory_bytes"]
        oom_at = row.get("oom_at")

        def ms_value(batch_size: int) -> str:
            if step_ms.get(str(batch_size)) is not None:
                return f"{step_ms[str(batch_size)]:.1f}"
            if oom_at is not None and batch_size == oom_at:
                return "OOM"
            return "-"

        def mb_value(batch_size: int) -> str:
            value = peak.get(str(batch_size))
            if value:
                return f"{int(value) // 2 ** 20}"
            if oom_at is not None and batch_size == oom_at:
                return "OOM"
            return "-"

        rows.append(
            {
                "task": row["task_file"],
                "layout": row["layout"],
                "gate": row["gate_mode"],
                "d": row["d_model"],
                "ev": f"{expand_v:g}" if expand_v != 1.0 else "",
                "ms@64": ms_value(64),
                "ms@128": ms_value(128),
                "ms@256": ms_value(256),
                "t256/t64": (
                    f"{row['scaling_ratio']:.2f}"
                    if row["scaling_ratio"] is not None
                    else "-"
                ),
                "peak MB": "/".join(mb_value(bs) for bs in (64, 128, 256)),
                "busy": f"{row['busy']:.2f}" if row["busy"] is not None else "-",
                "kernels/step": (
                    f"{row['kernels_per_step']:.0f}"
                    if row["kernels_per_step"] is not None
                    else "-"
                ),
                "reading": row["reading"],
                "note": "data-bound" if row["data_bound_warning"] else "",
            }
        )
    lines.append(markdown_table(pd.DataFrame(rows)))
    if not any(row["busy"] is not None for row in results["variants"]):
        lines.append(
            "\n_GPU-only readings (busy fraction, peak memory, kernels) are "
            "skipped: no CUDA device found._"
        )
    lines.append("")

    lines.append("## Data generation")
    lines.append("")
    lines.append(
        "Mean ms to produce one training batch of "
        f"{results['bench']['batch_sizes'][-1]} from the generator in one "
        "process (num_workers=0). The training script's worker count is "
        f"{results['num_workers']}."
    )
    lines.extend(["", markdown_table(pd.DataFrame(results["task_timings"])), ""])

    lines.append("## Validation")
    lines.append("")
    lines.append(
        "Seconds for one full validation pass over the task's fixed "
        "validation set at batch size "
        f"{results['bench']['batch_sizes'][-1]} through the module's "
        "validation path (loss plus RecallMetrics); `val_set_gen_s` is the "
        "one-time cost of building the fixed set (not charged to the runs)."
    )
    lines.extend(["", markdown_table(pd.DataFrame(results["task_timings"])), ""])

    lines.append("## Concurrency (runs per GPU)")
    lines.append("")
    if not results["concurrency"] or any(
        group.get("skipped")
        for entry in results["concurrency"]
        for group in entry["groups"]
    ):
        lines.append(
            "_Skipped: no CUDA device found._ Aggregate throughput of 1, 2 "
            "and 4 concurrent runs on one GPU needs a GPU."
        )
    else:
        for entry in results["concurrency"]:
            group_rows = []
            for group in entry["groups"]:
                group_rows.append(
                    {
                        "N": group["n"],
                        "steps/s (total)": (
                            f"{group['aggregate_steps_per_s']:.2f}"
                            if group.get("aggregate_steps_per_s") is not None
                            else ("OOM" if group["oom"] else "-")
                        ),
                        "speedup": (
                            f"{group['speedup']:.2f}"
                            if group.get("speedup") is not None
                            else "-"
                        ),
                        "sync": (
                            "no (late)" if group.get("synchronized") is False else "yes"
                        ),
                        "per-worker peak MB": "/".join(
                            (
                                f"{int(w.get('peak_memory_bytes') or 0) // 2**20}"
                                if w.get("peak_memory_bytes")
                                else "-"
                            )
                            for w in group["workers"]
                        ),
                        "failed": group["failed"],
                    }
                )
            recommended = entry["recommended_runs_per_gpu"]
            lines.append(
                f"**{entry['task_file']}, ws, full, d={entry['d_model']}** "
                f"(batch size {results['bench']['batch_sizes'][-1]}):"
            )
            lines.extend(["", markdown_table(pd.DataFrame(group_rows)), ""])
            if any(group.get("synchronized") is False for group in entry["groups"]):
                lines.append(
                    "_At least one worker started late: its rate covers only part "
                    "of the shared window, so the aggregate and the speedup may "
                    "read high — increase `bench.start_margin_s`._"
                )
                lines.append("")
            if recommended is None:
                lines.append("_No reading: every group failed or OOM-ed._")
            else:
                lines.append(
                    f"Recommended runs per GPU: **{recommended}** (the smallest "
                    "N whose aggregate throughput is within 10% of the best measured)."
                )
            lines.append("")
    lines.append("")

    lines.append("## Grid estimate (upper bound)")
    lines.append("")
    lines.append(
        "Per run: `max_steps x step_ms(B=256) + (max_steps / "
        "val_check_interval) x val_s + test_s` with test ~ validation. An "
        "upper bound: it assumes no early stopping. The two hour columns "
        "assume the pilot was calibrated to d=32 or d=64 (the sweep task "
        "files are interpreted at that width; the anchors keep their own). "
        "Rows whose cell was never measured (e.g. an OOM at B=256) carry a "
        "note in that column."
    )
    lines.append("")
    estimate_rows = []
    tiers_by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for interpretation, estimate in results["estimates"].items():
        for tier in estimate["tiers"]:
            tiers_by_name.setdefault(tier["tier"], {})[interpretation] = tier
    for tier_name, by_interpretation in tiers_by_name.items():
        row = {
            "tier": tier_name,
            "runs": next(iter(by_interpretation.values()))["runs"],
        }
        for interpretation, tier in by_interpretation.items():
            row[f"per-run s d={interpretation}"] = (
                f"{tier['per_run_s']:.0f}" if tier["per_run_s"] is not None else "-"
            )
            row[f"GPU-hours d={interpretation}"] = (
                f"{tier['gpu_hours']:.1f}" if tier["gpu_hours"] is not None else "-"
            )
        row["note"] = " / ".join(
            tier.get("note", "")
            for tier in by_interpretation.values()
            if tier.get("note")
        )
        estimate_rows.append(row)
    lines.append(markdown_table(pd.DataFrame(estimate_rows)))
    lines.append("")

    # Phase totals, serial GPU-hours, and the wall clock at the
    # recommended concurrency.
    phases = {
        "Phase 1 (calibration)": (
            "calib_dmodel",
            "calib_harness",
            "calib_lr",
            "calib_lr_anchors",
        ),
        "Phase 2 (core tier)": ("tier1_core",),
        "Phase 3 (remaining tiers)": (
            "tier2_gates",
            "tier3_frozen",
            "tier4_lambda",
            "tier5_matched_bytes",
            "tier6_anchors",
            "R1",
        ),
        "Phase 4 (long sequences)": ("tier_longT",),
    }
    by_tier = {
        tier["tier"]: tier
        for estimate in results["estimates"].values()
        for tier in estimate["tiers"]
    }
    recommended = {
        entry["d_model"]: entry["recommended_runs_per_gpu"]
        for entry in results["concurrency"]
        if entry.get("recommended_runs_per_gpu") is not None
    }
    phase_rows = []
    phase_serial: dict[str, int | None] = {}
    for phase, tiers in phases.items():
        hours: dict[str, float] = {}
        missing_tier: dict[str, str] = {}
        for interpretation in results["estimates"]:
            total = 0.0
            missing = None
            for tier_name in tiers:
                entry = by_tier.get(tier_name)
                if entry is None or entry["gpu_hours"] is None:
                    missing = tier_name
                    break
                total += entry["gpu_hours"]
            hours[interpretation] = total
            missing_tier[interpretation] = missing or ""
        phase_rows.append(
            {
                "phase": phase,
                "runs": sum(by_tier.get(t, {}).get("runs", 0) for t in tiers),
                **{
                    f"GPU-hours d={interp}": (
                        f"{hours[interp]:.0f}" if not missing_tier[interp] else "-"
                    )
                    for interp in results["estimates"]
                },
                "note": (
                    f"missing {next(v for v in missing_tier.values() if v)}"
                    if any(missing_tier.values())
                    else ""
                ),
            }
        )
        phase_serial[phase] = (
            {interp: hours[interp] for interp in results["estimates"]}
            if not any(missing_tier.values())
            else None
        )
    lines.append(markdown_table(pd.DataFrame(phase_rows)))
    lines.append("")

    frozen_d = None
    for note in results.get("notes", []):
        if note.startswith("frozen protocol width:"):
            frozen_d = int(note.split("=")[-1])
    for interpretation in results["estimates"]:
        d = int(interpretation)
        if frozen_d is not None and d != frozen_d:
            continue
        if (
            phase_serial["Phase 1 (calibration)"] is None
            or phase_serial["Phase 2 (core tier)"] is None
        ):
            continue
        total = (
            phase_serial["Phase 1 (calibration)"][interpretation]
            + phase_serial["Phase 2 (core tier)"][interpretation]
        )
        concurrency = recommended.get(d)
        if concurrency is None:
            verdict = "*not computable: no concurrency reading (no CUDA).*"
        else:
            wall_clock = total / concurrency
            verdict = (
                "**exceeds the one-week budget (168 h)** — rent"
                if wall_clock > 168.0
                else "**fits inside the one-week budget (168 h)** — run locally"
            )
            lines.append(
                f"**Phase 1 + Phase 2 at d={d}: {total:.0f} GPU-hours serial; "
                f"at the recommended {concurrency} runs per GPU: "
                f"{wall_clock:.0f} hours of wall clock — {verdict}.**"
            )
    lines.append("")

    cross_check_step = None
    cross_check_val = None
    for row in results["variants"]:
        if (
            row["task_file"] == "overwrite"
            and row["layout"] == "ws"
            and row["gate_mode"] == "full"
            and row["d_model"] in (32, 64)
            and row["step_ms"].get("256") is not None
        ):
            cross_check_step = row["step_ms"]["256"]
            cross_check_val = next(
                (
                    t["val_s"]
                    for t in results["task_timings"]
                    if t["task_file"] == "overwrite" and t["d_model"] == row["d_model"]
                ),
                None,
            )
            break
    if cross_check_step is not None and cross_check_val is not None:
        # The 1,000-step run adds two validation passes (steps 500 and
        # 1000), so its step rate is the benchmark's divided by
        # 1 + 2 * val_s / (1000 * step_s).
        expected_percent = 100.0 / (
            1.0 + 2.0 * cross_check_val / (1000.0 * cross_check_step / 1000.0)
        )
        cross_check_note = (
            f"The benchmark's (overwrite, ws, full) step time at B=256 is "
            f"{cross_check_step:.1f} ms. A 1,000-step run should reach about "
            f"{expected_percent:.0f}% of the benchmark's steps per second: "
            f"the two validation passes add {2.0 * cross_check_val:.0f} s over "
            f"{cross_check_step:.0f} s of stepping, plus the data loading and "
            f"the logging."
        )
    else:
        cross_check_note = (
            "The benchmark's (overwrite, ws, full) step time at B=256 was "
            "not measured (see the variant table); the difference is the "
            "data loading, the logging and the two validation passes."
        )
    lines.append(
        f"## Cross-check\n\n"
        f"Run one real run and compare its `timing.steps_per_second` "
        f"(which includes data loading, logging and the validation passes) "
        f"with the benchmark's step time.\n\n"
        f"```bash\n"
        f"python 01_train.py task=overwrite model.layout=ws "
        f"model.gate_mode=full training.max_steps=1000 experiment.tier=bench "
        f"optimizer.lr={results['lr']:g}\n"
        f"```\n\n"
        f"{cross_check_note}"
    )
    lines.append("")
    lines.append("## Notes and limitations")
    lines.extend(
        [
            "",
            "- The estimate is an upper bound: it assumes every run goes to "
            "max_steps (early stopping only shortens runs).",
            "- The task-level readings (data, validation) are measured once per "
            "(task, width) on the (ws, full) variant; the validation pass and "
            "the set size are identical for every variant of the same task.",
            f"- data.num_workers={results['num_workers']} on this machine; the "
            "data-bound readings assume that worker count.",
            "- The concurrency workers stream their training batches (as a real "
            "run does with DataLoader workers); the step-time measurement "
            "pre-generates them instead, so those two readings differ by the "
            "data time.",
        ]
    )
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Entry point
# =============================================================================


@hydra.main(version_base=None, config_path="conf", config_name="bench")
def main(cfg: DictConfig) -> None:
    """Run the benchmark and write bench/<timestamp>/{bench.json,summary.md}."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mode = str(cfg.bench.mode)
    if mode == "worker":
        worker_main(cfg)
        return
    if mode != "main":
        raise ValueError(f"bench.mode must be 'main' or 'worker', got {mode!r}")

    logger.info("Precision flags: %s", set_precision_flags())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning(
            "no CUDA device found: measuring only the step timing, the data "
            "time and the validation time (CPU smoke mode); the GPU-only "
            "readings are skipped"
        )
    out_dir = (
        Path(get_original_cwd())
        / "bench"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # measure_all writes bench.json as it goes (complete: false until the
    # final write) and finishes the summary; a crash or a Ctrl-C leaves
    # every measurement already made on disk.
    results = measure_all(cfg, device, out_dir)
    logger.info("Benchmark written to %s", out_dir)
    logger.info("Cross-check command:")
    logger.info(
        "  python 01_train.py task=overwrite model.layout=ws "
        "model.gate_mode=full training.max_steps=1000 experiment.tier=bench "
        "optimizer.lr=%g",
        results["lr"],
    )


if __name__ == "__main__":
    main()
