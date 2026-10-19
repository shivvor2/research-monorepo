"""
Tests for the pilot timing benchmark.

Target: experiments/03_multistate_gdn2/01_synthetic_pilot/00_benchmark.py
Run with: pytest experiments/03_multistate_gdn2/01_synthetic_pilot/test_00_benchmark.py -v

The GPU-only measurements (busy fraction, memory, concurrency) are
exercised only through their pure readers; the step-timing measurement
runs on CPU on a tiny configuration.
"""

import importlib.util
import sys
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
        # The smallest N within 10% of the best aggregate.
        assert bench.recommended_concurrency([10.0, 18.0, 19.0]) == 2
        assert bench.recommended_concurrency([10.0, 10.5, 10.9]) == 1
        assert bench.recommended_concurrency([5.0, 9.0, 10.0, 9.4]) == 2
        assert bench.recommended_concurrency([100.0]) == 1
        assert bench.recommended_concurrency([]) == 1


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


class TestSummary:
    """summary.md has the expected sections; on CPU the GPU-only parts
    are skipped with a clear note."""

    def test_summary_sections_and_cpu_notes(self):
        results = {
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
