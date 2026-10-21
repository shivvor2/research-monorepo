"""Load run JSONs into a DataFrame and select tier rows."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_RE_SLICE_KEY = re.compile(r"^(\d+)$")


def _slice_key_value(key: str, prefix: str) -> int | None:
    """Return the slice integer from a key like ``acc/slice=32``.

    Returns ``None`` for compound keys like ``acc/slice=32/class=distant``.
    """
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix) :]
    m = _RE_SLICE_KEY.match(rest)
    return int(m.group(1)) if m else None


def _weighted_mean_from_slices(
    metric_dict: dict[str, float], prefix: str, weights: dict[int, float]
) -> float | None:
    """Weighted mean of per-slice accuracies.

    Only exact ``<prefix><digits>`` keys are used; compound keys like
    ``acc/slice=32/class=distant`` or ``acc/slice=32/dist=0-16`` are
    ignored.

    Args:
        metric_dict: a metric section (e.g. ``summary["test"]``).
        prefix: the slice-key prefix, e.g. ``"acc/slice="``.
        weights: mapping from slice value to weight.

    Returns:
        The weighted mean, or ``None`` when no plain slice keys are present.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for key, value in metric_dict.items():
        slice_val = _slice_key_value(key, prefix)
        if slice_val is None:
            continue
        weight = weights.get(slice_val, 0.0)
        total_weight += weight
        weighted_sum += float(value) * weight
    if total_weight == 0.0:
        return None
    return weighted_sum / total_weight


def _normalize_overall_test_acc(record: dict) -> float | None:
    """Return the per-sequence-mean ``test/acc`` for a run record.

    If the record was written after the per-sequence accuracy change
    (it carries ``metrics.acc == "per_sequence_mean"``), the stored
    ``test/acc`` is already the right definition and is returned.

    Otherwise the per-sequence mean is recomputed from the stored
    per-slice ``test/acc/slice=...`` values, weighted by the number of
    sequences per slice taken from ``config.data.num_test_per_slice``.
    """
    metrics_def = (record.get("metrics") or {}).get("acc")
    test = record.get("test") or {}
    if metrics_def == "per_sequence_mean":
        return test.get("acc")
    cfg_data = (record.get("config", {})).get("data", {})
    num_per_slice = cfg_data.get("num_test_per_slice")
    if num_per_slice is None:
        return test.get("acc")
    # Build uniform weights when num_per_slice is a scalar.
    weights: dict[int, float] = {}
    for key in test:
        slice_val = _slice_key_value(key, "acc/slice=")
        if slice_val is not None:
            weights[slice_val] = float(num_per_slice)
    if not weights:
        return test.get("acc")
    recomputed = _weighted_mean_from_slices(test, "acc/slice=", weights)
    return recomputed if recomputed is not None else test.get("acc")


def _best_val_from_curves(record: dict) -> tuple[float | None, int | None, str]:
    """Best validation accuracy and its step, reading the curve CSV.

    Recomputes the per-sequence-mean overall ``val/acc`` at each
    validation step from the per-slice ``val/acc/slice=...`` columns in
    the CSV logger output, and returns the maximum and the step at which
    it occurs.

    Args:
        record: A run summary dict.

    Returns:
        ``(best_value, best_step, source)`` — ``source`` is ``"curve"``
        when the curves were read, ``"stopping"`` when the stored
        ``stopping.best_value`` was returned unchanged (new-style record),
        or ``"unavailable"`` when the curves could not be found.
    """
    metrics_def = (record.get("metrics") or {}).get("acc")
    stopping = record.get("stopping") or {}
    if metrics_def == "per_sequence_mean":
        return (
            stopping.get("best_value"),
            stopping.get("best_step"),
            "stopping",
        )
    # Old record: read curves and recompute.
    cfg = record.get("config", {})
    data_cfg = cfg.get("data", {})
    num_per_slice = data_cfg.get("num_val_per_slice")
    if num_per_slice is None:
        return stopping.get("best_value"), stopping.get("best_step"), "legacy_pooled"
    # Locate the curve CSV.
    json_path = Path(record.get("json_path", ""))
    if not json_path.exists():
        return None, None, "unavailable"
    run_id = record.get("run_id", "")
    # The canonical location: <runs_dir>/curves/<run_id>/version_0/metrics.csv
    curves_dir = json_path.parent / "curves"
    curves_path = curves_dir / run_id / "version_0" / "metrics.csv"
    if not curves_path.exists():
        # Fallback to the recorded logging.csv.save_dir (absolute or relative).
        csv_save_dir = (cfg.get("logging") or {}).get("csv", {}).get("save_dir", "")
        if csv_save_dir:
            launch_dir = json_path.parent.parent.parent  # runs_dir parent
            curves_path = (
                launch_dir / csv_save_dir / run_id / "version_0" / "metrics.csv"
            )
        if not curves_path.exists():
            return None, None, "unavailable"
    try:
        with curves_path.open(newline="") as f:
            reader = csv.DictReader(f)
            best_val = -float("inf")
            best_step = None
            for row in reader:
                step = row.get("step")
                if step is None or step == "":
                    continue
                weights: dict[int, float] = {}
                vals: dict[int, float] = {}
                for col, raw in row.items():
                    slice_val = _slice_key_value(col, "val/acc/slice=")
                    if slice_val is None or raw == "":
                        continue
                    weights[slice_val] = float(num_per_slice)
                    vals[slice_val] = float(raw)
                if not vals:
                    continue
                total_weight = sum(weights.values())
                step_acc = sum(vals[s] * weights[s] for s in vals) / total_weight
                if step_acc > best_val or best_step is None:
                    best_val = step_acc
                    best_step = int(step)
        if best_step is None:
            return None, None, "unavailable"
        return best_val, best_step, "curve"
    except (OSError, ValueError, KeyError) as e:
        logger.warning("Failed to read curves for %s: %s", run_id, e)
        return None, None, "unavailable"


def load_runs(runs_dir: Path) -> pd.DataFrame:
    """Load every run JSON under runs_dir into one row-per-run DataFrame.

    Flattens the run config into scalar columns (task, layout, gate_mode,
    lambda_mode, d_model, head_dim, layer_pattern, state_mixer, lr, seed,
    tier, phase, decay_scheme, expand_v), stopping (early_stopped,
    best_value, best_step, best_value_source) and the test metrics into
    ``test/<key>`` columns.

    The ``lr`` column is the learning rate the run actually used: the
    record's top-level ``lr_resolved`` (the training script resolves a
    null ``optimizer.lr`` from the frozen table and stores the value it
    used), falling back to ``config.optimizer.lr`` only for records that
    predate ``lr_resolved``.

    The ``test/acc`` value is normalised to the per-sequence-mean
    definition for every record, so downstream reports need not handle
    the old pooled definition.
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
            "expand_v": model.get("expand_v"),
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
        # Normalise overall test/acc to per-sequence mean.
        norm_test_acc = _normalize_overall_test_acc(summary)
        if norm_test_acc is not None:
            record["test/acc"] = norm_test_acc
        # Normalise best validation accuracy to per-sequence mean for old
        # records (reads the curve CSV when the record predates the
        # per-sequence change).
        best_val, best_step, source = _best_val_from_curves(summary)
        if best_val is not None:
            record["best_value"] = best_val
        if best_step is not None:
            record["best_step"] = best_step
        record["best_value_source"] = source
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
