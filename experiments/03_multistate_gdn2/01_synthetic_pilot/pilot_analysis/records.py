"""Load run JSONs into a DataFrame and select tier rows."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_runs(runs_dir: Path) -> pd.DataFrame:
    """Load every run JSON under runs_dir into one row-per-run DataFrame.

    Flattens the run config into scalar columns (task, layout, gate_mode,
    lambda_mode, d_model, head_dim, layer_pattern, state_mixer, lr, seed,
    tier, phase, decay_scheme), stopping (early_stopped, best_value,
    best_step) and the test metrics into ``test/<key>`` columns.

    The ``lr`` column is the learning rate the run actually used: the
    record's top-level ``lr_resolved`` (the training script resolves a
    null ``optimizer.lr`` from the frozen table and stores the value it
    used), falling back to ``config.optimizer.lr`` only for records that
    predate ``lr_resolved``.
    """
    records = []
    for path in sorted(runs_dir.glob("**/*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                summary = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning("skipping unreadable JSON %s: %s", path, e)
            continue
        cfg = summary.get("config", {})
        model = cfg.get("model", {})
        task = cfg.get("task", {})
        record = {
            "run_id": summary.get("run_id"),
            "json_path": str(path),
            "task": task.get("name"),
            "layout": model.get("layout"),
            "gate_mode": model.get("gate_mode"),
            "lambda_mode": model.get("lambda_mode"),
            "d_model": model.get("d_model"),
            "num_heads": model.get("num_heads"),
            "head_dim": model.get("head_dim"),
            "layer_pattern": model.get("layer_pattern"),
            "state_mixer": model.get("state_mixer"),
            "lr": (
                summary["lr_resolved"]
                if "lr_resolved" in summary
                else (cfg.get("optimizer") or {}).get("lr")
            ),
            "seed": summary.get("seed"),
            "tier": (cfg.get("experiment") or {}).get("tier"),
            "phase": (cfg.get("experiment") or {}).get("phase"),
            "steps_completed": (summary.get("timing") or {}).get("steps_completed"),
            "early_stopped": (summary.get("stopping") or {}).get("early_stopped"),
            "best_value": (summary.get("stopping") or {}).get("best_value"),
            "best_step": (summary.get("stopping") or {}).get("best_step"),
            "status": summary.get("status"),
            "decay_scheme": summary.get("decay_scheme")
            or (cfg.get("optimizer") or {}).get("decay_scheme"),
        }
        for key, value in (summary.get("test") or {}).items():
            record[f"test/{key}"] = value
        for key, value in (summary.get("mixer_stats_final") or {}).items():
            record[f"mixer/{key}"] = value
        records.append(record)
    if not records:
        raise SystemExit(f"no run JSONs found under {runs_dir}")
    return pd.DataFrame(records)


def head_dim_of(row: pd.Series) -> float:
    """d_k of a run (head_dim, or d_model // num_heads when null)."""
    if row["head_dim"] is not None and not pd.isna(row["head_dim"]):
        return float(row["head_dim"])
    return float(row["d_model"]) / float(row["num_heads"])


def tier_rows(table: pd.DataFrame, tier: str) -> pd.DataFrame:
    """The rows of one protocol tier.

    Every statistic selects its rows by tier, never by excluding tiers:
    several tiers repeat core configurations (the matched-state tier, the
    lambda controls, R1 and the long-sequence tier), so to the per-seed
    DiD pairing those rows read as duplicated cells and would drop seeds.
    Calibration tiers (and smoke runs) never enter any statistic.

    Only **complete** rows are returned (``status == "test_done"``), the
    same rule as the missing-run report: a record that is anything else
    (a crashed run, or a test-only relaunch that failed but kept the
    earlier test metrics) never enters the statistics;
    :func:`completeness.incomplete_runs` reads the unfiltered table to
    list them.
    """
    if "tier" not in table.columns:
        return table.iloc[0:0]
    rows = table[table["tier"] == tier]
    if "status" in table.columns:
        rows = rows[rows["status"].fillna("unknown") == "test_done"]
    return rows
