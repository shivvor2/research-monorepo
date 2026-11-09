"""This project's run-record flatten: one row per run for the analysis.

:func:`flatten` maps a parsed run record to the row columns the project
reads (layout, gate mode, lambda mode, width, learning rate, seed, tier,
task, per-slice and per-class metrics, best validation value and step).
The general loader
:func:`research_lib.training.records.load_records` builds the table.
Pilot-specific fix-ups (e.g. the first pilot's rebuilding of old
pooled-accuracy records) are passed in by the pilot as a wrapper around
:func:`flatten`, never built in here.
"""

from __future__ import annotations

from typing import Any, Mapping


def flatten(record: Mapping[str, Any]) -> dict[str, Any]:
    """One analysis row from a parsed run record.

    Args:
        record: The parsed run JSON (``config``, ``seed``, ``status``,
            ``lr_resolved``, ``test``, ``mixer_stats_final``, ``stopping``,
            ``timing``).

    Returns:
        The scalar config columns plus ``test/<key>`` and ``mixer/<key>``
        metric columns and the stopping/timing readings.
    """
    cfg = record.get("config", {}) or {}
    model = cfg.get("model", {}) or {}
    task_config = cfg.get("task", {}) or {}
    row: dict[str, Any] = {
        "run_id": record.get("run_id"),
        "task": task_config.get("name"),
        "layout": model.get("layout"),
        "gate_mode": model.get("gate_mode"),
        "lambda_mode": model.get("lambda_mode"),
        "d_model": model.get("d_model"),
        "num_heads": model.get("num_heads"),
        "head_dim": model.get("head_dim"),
        "expand_v": model.get("expand_v"),
        "layer_pattern": model.get("layer_pattern"),
        "state_mixer": model.get("state_mixer"),
        "mixer": model.get("mixer"),
        "harness": (cfg.get("harness") or {}).get("name"),
        "lr": (
            record["lr_resolved"]
            if "lr_resolved" in record
            else (cfg.get("optimizer") or {}).get("lr")
        ),
        # A record written before the resolved weight decay became a
        # top-level field falls back to the run's config value.  ``None``
        # is checked, not truthiness: ``0.0`` is a real selection.
        "weight_decay": (
            record.get("weight_decay")
            if record.get("weight_decay") is not None
            else (cfg.get("optimizer") or {}).get("weight_decay")
        ),
        "seq_len": task_config.get("seq_len"),
        "seed": record.get("seed"),
        "tier": (cfg.get("experiment") or {}).get("tier"),
        "phase": (cfg.get("experiment") or {}).get("phase"),
        "steps_completed": (record.get("timing") or {}).get("steps_completed"),
        "early_stopped": (record.get("stopping") or {}).get("early_stopped"),
        "best_value": (record.get("stopping") or {}).get("best_value"),
        "best_step": (record.get("stopping") or {}).get("best_step"),
        "monitored_first": (record.get("monitored") or {}).get("monitored_first"),
        "monitored_max": (record.get("monitored") or {}).get("monitored_max"),
        "monitored_last_half_min": (record.get("monitored") or {}).get(
            "monitored_last_half_min"
        ),
        "monitored_last_half_max": (record.get("monitored") or {}).get(
            "monitored_last_half_max"
        ),
        "status": record.get("status"),
        "decay_scheme": record.get("decay_scheme")
        or (cfg.get("optimizer") or {}).get("decay_scheme"),
    }
    for key, value in (record.get("test") or {}).items():
        row[f"test/{key}"] = value
    for key, value in (record.get("mixer_stats_final") or {}).items():
        row[f"mixer/{key}"] = value
    return row
