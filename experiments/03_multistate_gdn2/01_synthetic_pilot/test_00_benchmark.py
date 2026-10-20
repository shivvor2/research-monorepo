"""
Tests for the pilot timing benchmark.

Target: experiments/03_multistate_gdn2/01_synthetic_pilot/00_benchmark.py
Run with: pytest experiments/03_multistate_gdn2/01_synthetic_pilot/test_00_benchmark.py -v

The GPU-only measurements (busy fraction, memory, concurrency) are
exercised only through their pure readers; the step-timing measurement
runs on CPU on a tiny configuration.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SPEC = importlib.util.spec_from_file_location(
    "pilot_benchmark", Path(__file__).resolve().parent / "00_benchmark.py"
)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


def make_cfg(
    *,
    task="mqar",
    layout="ws",
    gate_mode="full",
    d_model=16,
    num_heads=1,
    num_kv_pairs=(2, 4),
    seq_len=32,
):
    """A tiny config on the test-size model (vocab 64, d 16, T 32)."""
    return OmegaConf.create(
        {
            "task": {
                "name": task,
                "vocab_size": 64,
                "seq_len": seq_len,
                "num_kv_pairs": list(num_kv_pairs),
                "power_a": 0.01,
                "random_non_queries": True,
                "max_seq_len": seq_len,
                "val_seed": 1000003,
                "test_seed": 2000003,
            },
            "model": {
                "vocab_size": 64,
                "d_model": d_model,
                "n_layer": 2,
                "layer_pattern": "zoology",
                "state_mixer": "none",
                "num_heads": num_heads,
                "head_dim": None,
                "expand_v": 1.0,
                "ff_mult": 4,
                "base_conv_kernel": 3,
                "gate_mode": gate_mode,
                "layout": layout,
                "lambda_mode": "mlp",
                "window": 4,
                "chunk_size": 16,
                "t_max": seq_len,
                "output_gate_activation": "silu",
                "mixer_mode": "chunk",
            },
            "optimizer": {
                "lr": 0.003,
                "weight_decay": 0.1,
                "betas": [0.9, 0.95],
                "decay_scheme": "grouped",
            },
            "training": {
                "loss_on": "answers",
                "answer_head": True,
                "grad_clip_val": None,
            },
            "data": {
                "num_workers": 0,
                "num_val_per_slice": 16,
                "num_test_per_slice": 16,
            },
            "experiment": {"tier": "bench"},
        }
    )


class TestStepTiming:
    """The step-timing function works on CPU (a cheap smoke check)."""

    def test_step_timing_returns_positive_times_on_cpu(self):
        cfg = make_cfg()
        result = bench.measure_step_times(
            cfg,
            batch_size=16,
            warmup_steps=2,
            measure_steps=3,
            device=torch.device("cpu"),
            lr=0.003,
        )
        assert result["oom"] is False
        assert result["step_ms"] is not None
        assert result["step_ms"] > 0
        assert result["peak_memory_bytes"] is None  # no CUDA: no memory reading

    def test_variant_config_applies_the_task_file(self):
        cfg = make_cfg()
        variant = bench.variant_config(cfg, "overwrite", "ws", "full", 32, 1.0)
        assert variant.task.name == "overwrite"
        assert variant.model.layout == "ws"
        assert variant.model.d_model == 32
        # t_max follows the task's max_seq_len, as in the training script.
        assert variant.model.t_max == variant.task.max_seq_len
        assert variant is not cfg  # the caller's config is not mutated


class TestReadings:
    """The pure readers on hand-made numbers (each branch)."""

    def test_classify_reading_each_branch(self):
        # Overhead-bound: step time nearly flat in B and the GPU mostly idle.
        assert bench.classify_reading(1.2, 0.3) == "overhead-bound"
        # Throughput-bound: step time grows with B and the GPU is busy.
        assert bench.classify_reading(3.5, 0.9) == "throughput-bound"
        # Mixed: one of the two signals is missing.
        assert bench.classify_reading(1.5, 0.9) == "mixed"
        assert bench.classify_reading(2.0, 0.4) == "mixed"
        assert bench.classify_reading(3.2, 0.5) == "mixed"
        # No CUDA (or an OOM at B=256): nothing to say.
        assert bench.classify_reading(1.5, None) == "not measured"
        assert bench.classify_reading(None, 0.9) == "not measured"

    def test_scaling_ratio(self):
        ratio = bench.scaling_ratio({64: 50.0, 128: 90.0, 256: 180.0})
        assert ratio == pytest.approx(3.6)
        assert bench.scaling_ratio({64: 50.0, 256: None}) is None
        assert bench.scaling_ratio({128: 90.0}) is None

    def test_data_bound_warning(self):
        # Data time above step time x workers: generation cannot keep up.
        assert bench.data_bound(50.0, 20.0, 2) is True
        assert bench.data_bound(30.0, 20.0, 2) is False
        assert bench.data_bound(50.0, None, 2) is False
        assert bench.data_bound(50.0, 20.0, 0) is False

    def test_recommended_concurrency(self):
        # The smallest measured N within 10% of the best aggregate, from
        # (n, aggregate) pairs — a position in the list is not an N.
        assert bench.recommended_concurrency([(1, 10.0), (2, 18.0), (4, 19.0)]) == 2
        assert bench.recommended_concurrency([(1, 10.0), (2, 10.5), (4, 10.9)]) == 1
        assert (
            bench.recommended_concurrency([(1, 5.0), (2, 9.0), (4, 10.0), (8, 9.4)])
            == 2
        )
        assert bench.recommended_concurrency([(1, 100.0)]) == 1
        assert bench.recommended_concurrency([]) == 1
        # A best at N=4 comes back as 4 (a before-N list only held
        # positions); near-ties resolve to the smaller measured N.
        assert bench.recommended_concurrency([(1, 10.0), (2, 10.2), (4, 19.0)]) == 4
        assert bench.recommended_concurrency([(1, 10.0), (2, 9.5)]) == 1
        assert bench.recommended_concurrency([(1, 10.0), (4, 10.5)]) == 1

    def test_eligible_groups_exclude_oom_failed_and_late(self):
        # Only a complete, synchronized, non-OOM group with a rate may
        # feed the recommendation: a partial failure aggregates fewer
        # workers than N, and a late worker's rate covers only part of
        # the shared window (its aggregate may read high).
        groups = [
            {
                "n": 1,
                "aggregate_steps_per_s": 10.0,
                "oom": False,
                "failed": 0,
                "synchronized": True,
            },
            {
                "n": 2,
                "aggregate_steps_per_s": 3.0,
                "oom": False,
                "failed": 1,
                "synchronized": True,
            },
            {
                "n": 2,
                "aggregate_steps_per_s": 8.0,
                "oom": False,
                "failed": 0,
                "synchronized": False,
            },
            {
                "n": 4,
                "aggregate_steps_per_s": 5.0,
                "oom": True,
                "failed": 0,
                "synchronized": True,
            },
            {
                "n": 4,
                "aggregate_steps_per_s": None,
                "oom": False,
                "failed": 4,
                "synchronized": True,
            },
        ]
        assert bench.eligible_groups(groups) == [(1, 10.0)]
        assert bench.recommended_concurrency(bench.eligible_groups(groups)) == 1


class TestGridEstimate:
    """The variant mapping and the per-tier hour figures."""

    def test_grid_estimate_includes_the_variant_mapping(self):
        timings = {
            bench.VariantKey("overwrite", "single", "kda", 32, 1.0): 100.0,
            bench.VariantKey("overwrite", "single", "full", 32, 1.0): 110.0,
            bench.VariantKey("overwrite", "ll", "full", 32, 1.0): 120.0,
            bench.VariantKey("overwrite", "single", "full", 32, 10.0): 250.0,
            bench.VariantKey("mqar", "single", "kda", 32, 1.0): 90.0,
            bench.VariantKey("mqar", "ll", "kda", 32, 1.0): 100.0,
            bench.VariantKey("mqar", "ll", "kda", 64, 1.0): 170.0,
            bench.VariantKey("mqar_longT", "ll", "full", 32, 1.0): 900.0,
        }
        val_s = {
            ("overwrite", 32): 1.0,
            ("mqar", 32): 0.9,
            ("mqar", 64): 1.5,
            ("mqar_longT", 32): 5.0,
        }
        runs = [
            # b_only -> full of the same layout.
            bench.RunSpec(
                "tier1_core", "overwrite", 32, "single", "b_only", "mlp", 1.0
            ),
            bench.RunSpec("tier1_core", "mqar", 32, "single", "kda", "mlp", 1.0),
            # w_only -> full of the same layout.
            bench.RunSpec("tier2_gates", "overwrite", 32, "ll", "w_only", "mlp", 1.0),
            # gdn anchors -> kda at the nearest measured width (16 -> 32).
            bench.RunSpec("tier6_anchors", "mqar", 16, "single", "gdn", "mlp", 1.0),
            bench.RunSpec("tier6_anchors", "mqar", 64, "ll", "gdn", "mlp", 1.0),
            # Matched bytes -> the expand_v=10 flavour.
            bench.RunSpec(
                "tier5_matched_bytes", "overwrite", 32, "single", "full", "mlp", 10.0
            ),
            # Long sequences -> the long-T flavour, whatever the layout/gate.
            bench.RunSpec("tier_longT", "mqar_longT", 32, "ws", "kda", "mlp", 1.0),
        ]
        estimates = {
            estimate.tier: estimate
            for estimate in bench.estimate_grid(runs, timings, val_s, 1000, 100)
        }

        # per_run = max_steps x step/1000 + (max_steps / interval) x val + test.
        assert estimates["tier1_core"].per_run_s == pytest.approx(
            (1000 * 110.0 / 1000 + 10 * 1.0 + 1.0 + 1000 * 90.0 / 1000 + 10 * 0.9 + 0.9)
            / 2
        )
        assert estimates["tier1_core"].gpu_hours == pytest.approx(
            (121.0 + 99.9) / 3600.0
        )
        assert estimates["tier1_core"].runs == 2
        assert estimates["tier1_core"].measured == 2

        assert estimates["tier2_gates"].gpu_hours == pytest.approx(
            (1000 * 120.0 / 1000 + 10 * 1.0 + 1.0) / 3600.0
        )
        # Both anchor runs: d=16 -> the d=32 kda cell, d=64 -> its own.
        assert estimates["tier6_anchors"].gpu_hours == pytest.approx(
            (99.9 + (1000 * 170.0 / 1000 + 10 * 1.5 + 1.5)) / 3600.0
        )
        assert estimates["tier5_matched_bytes"].gpu_hours == pytest.approx(
            (1000 * 250.0 / 1000 + 10 * 1.0 + 1.0) / 3600.0
        )
        assert estimates["tier_longT"].gpu_hours == pytest.approx(
            (1000 * 900.0 / 1000 + 10 * 5.0 + 5.0) / 3600.0
        )

    def test_grid_estimate_reports_unmeasured_cells(self):
        runs = [
            bench.RunSpec("tier1_core", "overwrite", 32, "single", "full", "mlp", 1.0),
            bench.RunSpec("tier1_core", "mqar", 32, "single", "kda", "mlp", 1.0),
        ]
        estimate = bench.estimate_grid(runs, {}, {}, 1000, 100)[0]
        assert estimate.measured == 0
        assert estimate.gpu_hours is None
        assert "not measured" in estimate.note

    def test_proxy_variant_rules(self):
        key, _ = bench.proxy_variant(
            bench.RunSpec("tier2_gates", "overwrite", 32, "ws", "b_only", "mlp", 1.0)
        )
        assert key == bench.VariantKey("overwrite", "ws", "full", 32, 1.0)
        key, _ = bench.proxy_variant(
            bench.RunSpec("tier6_anchors", "mqar", 16, "single", "gdn", "mlp", 1.0)
        )
        assert key == bench.VariantKey("mqar", "single", "kda", 32, 1.0)
        key, _ = bench.proxy_variant(
            bench.RunSpec("tier1_core", "mqar", 32, "single", "kda", "static", 1.0)
        )
        assert key == bench.VariantKey("mqar", "single", "kda", 32, 1.0)
        key, _ = bench.proxy_variant(
            bench.RunSpec("tier_longT", "mqar_longT", 64, "single", "kda", "mlp", 1.0)
        )
        assert key == bench.VariantKey("mqar_longT", "ll", "full", 64, 1.0)

    def test_sweep_jobs_expands_in_file_order(self):
        jobs = bench.sweep_jobs(
            Path(__file__).resolve().parent / "conf" / "sweep" / "tier3_frozen.yaml"
        )
        assert len(jobs) == 20  # 2 tasks x 2 gates x 5 seeds
        assert jobs[0]["task"] == "overwrite"
        assert jobs[19]["task"] == "mqar"
        seeds = [int(job["seed"]) for job in jobs[:10]]
        assert seeds == sorted(seeds)


def make_bench_cfg():
    """A tiny CPU bench config: one variant, one batch size, tiny eval
    sets — the CPU smoke grid of the resilience tests. The model vocab
    matches the task files the variant composition loads (8192), as in
    the real config."""
    cfg = make_cfg()
    cfg.model.vocab_size = 8192
    cfg.bench = OmegaConf.create(
        {
            "d_models": [16],
            "tasks": ["mqar"],
            "layouts": ["ws"],
            "gate_modes": ["full"],
            "extra_variants": [],
            "batch_sizes": [16],
            "warmup_steps": 1,
            "measure_steps": 2,
            "profile_steps": 2,
            "data_batches": 2,
            "concurrency": [1],
            "concurrency_seconds": 5,
            "mode": "main",
            "task_file": "mqar",
            "start_at": None,
            "start_margin_s": 60,
        }
    )
    cfg.data.num_val_per_slice = 4
    cfg.data.num_test_per_slice = 4
    cfg.training.max_steps = 10
    cfg.training.val_check_interval = 5
    return cfg


class TestConcurrencyResilience:
    """A group whose workers all fail records the failure instead of
    crashing the benchmark (the speedup needs both aggregates)."""

    def test_group_of_failed_workers_reports_failure_without_speedup(self):
        group = bench.summarize_group([(1, None), ("timeout", None)])
        assert group["failed"] == 2
        assert group["workers"] == []
        assert group["aggregate_steps_per_s"] is None
        assert group["speedup"] is None
        # Even when the first group has an aggregate, a failed group has
        # no speedup (the old code crashed with None / float here).
        group = bench.summarize_group([(1, None)], first_aggregate=10.0)
        assert group["speedup"] is None
        # And a healthy group against a healthy first group still does.
        group = bench.summarize_group(
            [
                (
                    0,
                    {
                        "steps": 10,
                        "elapsed_s": 2.0,
                        "peak_memory_bytes": None,
                        "oom": False,
                        "late": False,
                    },
                )
            ],
            first_aggregate=10.0,
        )
        assert group["failed"] == 0
        assert group["aggregate_steps_per_s"] == pytest.approx(5.0)
        assert group["speedup"] == pytest.approx(0.5)


class TestMeasurementResilience:
    """The benchmark writes bench.json as it goes, so a failure or an
    interrupt never discards the measurements already made."""

    def test_failed_concurrency_stage_is_recorded(self, tmp_path, monkeypatch):
        def boom(cfg, task_file, d_model, device):
            raise RuntimeError("simulated concurrency crash")

        monkeypatch.setattr(bench, "run_concurrency_groups", boom)
        bench.measure_all(make_bench_cfg(), torch.device("cpu"), Path(tmp_path))
        bench_json = Path(tmp_path) / "bench.json"
        summary = Path(tmp_path) / "summary.md"
        assert bench_json.exists()
        assert summary.exists()
        data = json.loads(bench_json.read_text())
        # The run finished: the crash hit one stage, was recorded, and
        # the estimate and the summary still ran.
        assert data["complete"] is True
        assert data["variants"]  # the measured variant survived
        assert any("concurrency stage failed" in note for note in data["notes"])

    def test_interruption_keeps_the_last_snapshot(self, tmp_path, monkeypatch):
        real_write = bench.write_bench_json
        calls = {"n": 0}

        def failing_write(results, out_dir):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("simulated interrupt")
            return real_write(results, out_dir)

        monkeypatch.setattr(bench, "write_bench_json", failing_write)
        with pytest.raises(RuntimeError):
            bench.measure_all(make_bench_cfg(), torch.device("cpu"), Path(tmp_path))
        # The last snapshot that reached the disk holds the measured
        # variant and says the run did not finish.
        data = json.loads((Path(tmp_path) / "bench.json").read_text())
        assert data["complete"] is False
        assert len(data["variants"]) == 1


class TestSynchronizedWindows:
    """The concurrency workers share one timed window: the parent picks
    a common start, every worker steps until the same end, and a worker
    that starts late says so (its rate covers only part of the window,
    so the group's speedup may read high)."""

    def test_worker_command_carries_the_shared_start(self):
        command = bench.worker_command(
            make_bench_cfg(), "overwrite", 32, 256, 60, start_at=1750000000.5
        )
        assert "bench.mode=worker" in command
        assert "task=overwrite" in command
        assert "bench.start_at=1750000000.500" in command

    def test_on_time_worker_uses_the_shared_window(self, capsys):
        cfg = make_bench_cfg()
        cfg.bench.concurrency_seconds = 1
        cfg.bench.start_at = time.time() + 5.0  # comfortable build margin
        bench.worker_main(cfg)
        result = bench.parse_worker_result(capsys.readouterr().out)
        assert result is not None
        assert result["late"] is False
        # The window ends at start_at + seconds, so the measured time is
        # the window length, not the worker's own duration.
        assert result["elapsed_s"] == pytest.approx(1.0, abs=0.20)
        assert result["steps"] >= 1

    def test_late_worker_reports_late(self, capsys):
        cfg = make_bench_cfg()
        cfg.bench.concurrency_seconds = 1.5
        cfg.bench.start_at = time.time() - 0.5  # the shared start went by
        bench.worker_main(cfg)
        result = bench.parse_worker_result(capsys.readouterr().out)
        assert result is not None
        assert result["late"] is True
        assert result["steps"] >= 1  # it still measures until the shared end

    def test_group_with_a_late_worker_is_not_synchronized(self):
        on_time = {
            "steps": 10,
            "elapsed_s": 2.0,
            "peak_memory_bytes": None,
            "oom": False,
            "late": False,
        }
        late = dict(on_time, late=True)
        group = bench.summarize_group([(0, on_time)])
        assert group["synchronized"] is True
        group = bench.summarize_group([(0, late)])
        assert group["synchronized"] is False
        group = bench.summarize_group([(0, on_time), (0, late)])
        assert group["synchronized"] is False
        assert group["workers"][1]["late"] is True


class TestWorkerLifecycle:
    """A hung worker times out, counts as failed, and is reaped — it can
    never block the benchmark or run on past its group."""

    def test_worker_timeout_is_reaped_and_counts_as_failed(self):
        outputs, processes = bench.run_worker_processes(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            n=1,
            timeout_s=1.0,
        )
        assert outputs == [("timeout", None)]
        group = bench.summarize_group(outputs)
        assert group["failed"] == 1
        assert group["aggregate_steps_per_s"] is None
        assert group["speedup"] is None
        # Every launched process was terminated and reaped.
        for process in processes:
            assert process.poll() is not None

    def test_timeout_logs_the_workers_partial_output(self, caplog):
        import logging

        with caplog.at_level(logging.ERROR, logger=bench.__name__):
            outputs, _ = bench.run_worker_processes(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import time; print('ALIVE'); time.sleep(60)",
                ],
                n=1,
                timeout_s=1.0,
            )
        assert outputs == [("timeout", None)]
        # The partial output is logged, so a hung worker is diagnosable.
        assert any("timed out" in record.message for record in caplog.records)
        assert any("ALIVE" in record.message for record in caplog.records)


class TestBusyFraction:
    """The busy reading uses the unprofiled step time as its
    denominator: the profiler's own CPU-tracing overhead would inflate
    its wall time, and the first steps carry one-time costs the warmup
    absorbs."""

    def test_busy_fraction_on_hand_made_numbers(self):
        # 5000 us of kernel time per step over a 10 ms unprofiled step.
        assert bench.busy_fraction(50_000.0, 10, 10.0) == pytest.approx(0.5)
        # Kernel time above the step time clips at 1.0.
        assert bench.busy_fraction(200_000.0, 10, 10.0) == 1.0
        assert bench.busy_fraction(0.0, 10, 10.0) == 0.0
        # Degenerate inputs stay 0 (never a division blow-up).
        assert bench.busy_fraction(50_000.0, 0, 10.0) == 0.0
        assert bench.busy_fraction(50_000.0, 10, 0.0) == 0.0


class TestParentMemoryRelease:
    """The parent releases its GPU memory before the concurrency workers
    launch: the caching allocator holds the measurement stage's freed
    blocks (the last module and optimizers, the profiler, the validation
    sets), and the workers each reserve 2+ GB — the parent's retained
    footprint can be what tips the card over."""

    def test_prepare_concurrency_empty_caches_when_cuda(self, monkeypatch):
        releases = {"n": 0}

        def fake_empty_cache():
            releases["n"] += 1

        monkeypatch.setattr(bench.torch.cuda, "empty_cache", fake_empty_cache)
        monkeypatch.setattr(
            bench.torch.cuda, "memory_reserved", lambda device=None: 1_234_567_890
        )
        reserved = bench.prepare_concurrency(torch.device("cuda"))
        assert reserved == 1_234_567_890
        assert releases["n"] == 1  # the cache was emptied once

    def test_prepare_concurrency_without_cuda_does_nothing(self, monkeypatch):
        releases = {"n": 0}

        def fake_empty_cache():
            releases["n"] += 1

        monkeypatch.setattr(bench.torch.cuda, "empty_cache", fake_empty_cache)
        assert bench.prepare_concurrency(torch.device("cpu")) is None
        assert releases["n"] == 0


class TestSummary:
    """summary.md has the expected sections; on CPU the GPU-only parts
    are skipped with a clear note."""

    def test_summary_sections_and_cpu_notes(self):
        results = {
            "complete": True,
            "generated_at": "2026-10-20T00:00:00",
            "command": "python 00_benchmark.py",
            "device": {
                "device": "cpu",
                "count": 0,
                "name": None,
                "capability": None,
                "driver": None,
            },
            "versions": {
                "python": "3.13",
                "torch": "2.9.1",
                "cuda_runtime": None,
                "lightning": "2.6.0",
                "hydra_core": "1.3.2",
                "omegaconf": "2.3.0",
            },
            "git": {"commit": "abc123", "dirty": False},
            "lock_hash": "deadbeef" * 4,
            "precision_flags": {
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
            },
            "num_workers": 2,
            "lr": 0.003,
            "max_steps": 20000,
            "val_check_interval": 500,
            "bench": {"batch_sizes": [64, 128, 256]},
            "variants": [
                {
                    "task_file": "mqar",
                    "layout": "ws",
                    "gate_mode": "full",
                    "d_model": 32,
                    "expand_v": 1.0,
                    "step_ms": {"64": 1.0, "128": 2.0, "256": 3.9},
                    "peak_memory_bytes": {"64": None, "128": None, "256": None},
                    "scaling_ratio": 3.9,
                    "busy": None,
                    "kernels_per_step": None,
                    "reading": "not measured",
                    "data_bound_warning": False,
                    "oom_at": None,
                }
            ],
            "task_timings": [
                {
                    "task_file": "mqar",
                    "d_model": 32,
                    "data_ms": 5.0,
                    "val_s": 1.0,
                    "val_set_gen_s": 0.5,
                }
            ],
            "parent_reserved_bytes_before_concurrency": 125_829_120,
            "concurrency": [
                {"d_model": 32, "groups": [{"n": 1, "skipped": "no CUDA"}]}
            ],
            "estimates": {
                "32": {"tiers": []},
                "64": {"tiers": []},
            },
            "notes": [],
        }
        summary = bench.render_summary(results)
        assert "**Status: complete.**" in summary
        assert "The parent's reserved CUDA memory" in summary
        assert "120 MB" in summary
        for section in (
            "## Environment",
            "## Step time and GPU usage per variant",
            "## Data generation",
            "## Validation",
            "## Concurrency (runs per GPU)",
            "## Grid estimate (upper bound)",
            "## Cross-check",
            "## Notes and limitations",
        ):
            assert section in summary, section
        # GPU-only readings are skipped with a clear note.
        assert "skipped: no CUDA device found" in summary
        assert "not measured" in summary


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
