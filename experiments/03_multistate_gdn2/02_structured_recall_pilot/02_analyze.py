"""Command-line entry point for the structured-recall pilot's reports.

Builds on :mod:`research_lib.training.record_table.load_records` (with
``common.records.flatten``) and the shared :mod:`common.analysis` pieces.
Each subcommand answers one question; the tier / screening-cell
constants live in the pilot's protocol.

Reports
-------
``calibration``
    The per-configuration LR selection and the single-state gap: for
    the gated-delta calibration tiers, per (task, layout, gate, lambda)
    the per-seed mean of the best validation value per LR, the selected
    LR (best mean, then fewer steps, then smaller LR), and for
    single-state the screening cell at each gate's selected LR with the
    ``full - kda`` gap. Attention runs get their own row per
    (task, layout) — they have no gate. A ``frozen_protocol_snippet.yaml``
    is written from the single-state selection.
``positive_control``
    The paired ``full - kda`` gap on each task's screening cell,
    single-state gated-delta runs in the tier-1 tier, over the selection
    seeds, with a percentile bootstrap CI. A duplicated (task, seed) row
    after filtering raises naming the run IDs.
``completeness``
    Per tier, expected vs observed done runs, plus tiers with runs but no
    expected count (via ``common.analysis.missing_runs``).

Usage:
    python 02_analyze.py calibration positive_control completeness
    python 02_analyze.py --runs-dir runs --out-dir analysis
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_PILOT = Path(__file__).resolve().parent
_PROJECT = _PILOT.parent
_SRC = _PROJECT.parents[1] / "src"
for _path in (_PILOT, _PROJECT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common.analysis import (  # noqa: E402
    missing_runs,
    paired_gap_report,
    per_seed_mean,
    select_lr,
)
from common.records import flatten  # noqa: E402
from structured_recall_analysis import protocol as _protocol  # noqa: E402

from research_lib.training.record_table import load_records  # noqa: E402

logger = logging.getLogger(__name__)


def _load(runs_dir: Path) -> pd.DataFrame:
    return load_records(runs_dir, flatten)


def _gated(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["mixer"].fillna("gated_delta") == "gated_delta"]


def _selection(table: pd.DataFrame) -> pd.DataFrame:
    return table[
        (table["status"].fillna("").astype(str) == "test_done")
        & table["best_value"].notna()
        & table["tier"].isin(_protocol.CALIBRATION_TIERS)
        & table["seed"].isin(_protocol.SELECTION_SEEDS)
    ]


def _lr_summary(grp: pd.DataFrame) -> pd.DataFrame:
    """One row per LR for a config group: per-seed means, then selection."""
    by = ["lr"]
    decay = grp["decay_scheme"].fillna("grouped")
    if decay.nunique() > 1:
        by.append("decay_scheme")
    per_lr = per_seed_mean(grp, by=by, selection_seeds=_protocol.SELECTION_SEEDS)
    if per_lr.empty:
        return per_lr
    renamed = per_lr.rename(
        columns={"mean_best_value": "mean_acc", "mean_best_step": "mean_steps"}
    )
    if "decay_scheme" in renamed.columns:
        renamed = renamed.set_index(["lr", "decay_scheme"])
    else:
        renamed = renamed.set_index("lr")
    return renamed


def _cell_mean(grp: pd.DataFrame, metric: str, lr: float) -> float | None:
    sub = grp[(grp["lr"] == lr) & grp[metric].notna()][metric]
    return float(sub.mean()) if len(sub) else None


def calibration(table: pd.DataFrame, out_dir: Path) -> None:
    """The LR reading (and the single-state gap on the screening cell)."""
    rows: list[dict] = []
    sel = _selection(table)
    gated_sel = _gated(sel)
    # --- gated-delta: single-state (the gap) and multi-state ---
    single = gated_sel[gated_sel["layout"] == "single"]
    multi = gated_sel[gated_sel["layout"] != "single"]
    for keys, grp in single.groupby(["task", "lambda_mode"]):
        task, lam = keys
        gaps = {}
        for gate in ("full", "kda"):
            by_gate = grp[grp["gate_mode"] == gate]
            if by_gate.empty:
                continue
            per_lr = _lr_summary(by_gate)
            if per_lr.empty:
                continue
            chosen = select_lr(per_lr)
            metric = _protocol.SCREENING_CELLS.get(task)
            cell = _cell_mean(by_gate, metric, chosen) if metric else None
            row = {
                "kind": "single",
                "task": task,
                "layout": "single",
                "gate_mode": gate,
                "lambda_mode": lam,
                "mixer": "gated_delta",
                "n_seeds": int(per_lr["n_seeds"].iloc[0]),
                "d_model": (
                    int(grp["d_model"].dropna().iloc[0])
                    if grp["d_model"].notna().any()
                    else None
                ),
                "selected_lr": chosen,
                "mean_best_value": (
                    float(per_lr.loc[chosen, "mean_acc"])
                    if chosen in per_lr.index and "mean_acc" in per_lr.columns
                    else None
                ),
                "cell": cell,
            }
            gaps[gate] = cell
            rows.append(row)
        # Attach the full - kda gap to the single-state rows of this cell.
        if (
            "full" in gaps
            and "kda" in gaps
            and gaps["full"] is not None
            and gaps["kda"] is not None
        ):
            gap = gaps["full"] - gaps["kda"]
            for row in rows:
                if (
                    row.get("kind") == "single"
                    and row["task"] == task
                    and row["lambda_mode"] == lam
                    and row["gate_mode"] in ("full", "kda")
                ):
                    row["gap_full_kda"] = gap

    for keys, grp in multi.groupby(["task", "layout", "gate_mode", "lambda_mode"]):
        task, layout, gate, lam = keys
        per_lr = _lr_summary(grp)
        if per_lr.empty:
            continue
        chosen = select_lr(per_lr)
        rows.append(
            {
                "kind": "multi",
                "task": task,
                "layout": layout,
                "gate_mode": gate,
                "lambda_mode": lam,
                "mixer": "gated_delta",
                "n_seeds": int(per_lr["n_seeds"].iloc[0]),
                "d_model": (
                    int(grp["d_model"].dropna().iloc[0])
                    if grp["d_model"].notna().any()
                    else None
                ),
                "selected_lr": chosen,
                "mean_best_value": (
                    float(per_lr.loc[chosen, "mean_acc"]) if not per_lr.empty else None
                ),
            }
        )
    # --- attention: one row per (task, layout); no gate ---
    attn = sel[sel["mixer"].fillna("gated_delta") == "attention"]
    for keys, grp in attn.groupby(["task", "layout"]):
        task, layout = keys
        per_lr = _lr_summary(grp)
        if per_lr.empty:
            continue
        chosen = select_lr(per_lr)
        rows.append(
            {
                "kind": "attention",
                "task": task,
                "layout": layout,
                "gate_mode": None,
                "lambda_mode": None,
                "mixer": "attention",
                "n_seeds": int(per_lr["n_seeds"].iloc[0]),
                "d_model": (
                    int(grp["d_model"].dropna().iloc[0])
                    if grp["d_model"].notna().any()
                    else None
                ),
                "selected_lr": chosen,
                "mean_best_value": (
                    float(per_lr.loc[chosen, "mean_acc"]) if not per_lr.empty else None
                ),
            }
        )

    df = pd.DataFrame(rows)
    (out_dir / "calibration.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    _frozen_snippet(rows, out_dir)


def _frozen_snippet(rows: list[dict], out_dir: Path) -> None:
    """A frozen-protocol LR-table snippet from the single-state selection.

    The gated-delta LRs come from the single-state rows (per gate); the
    attention LR is the first attention reading per task. This is a
    skeleton the calibration step can grow into the full ``lr_table``.
    """
    single = [r for r in rows if r.get("kind") == "single"]
    attention = [r for r in rows if r.get("kind") == "attention"]
    lines = ["lr_table:"]
    tasks = sorted({r["task"] for r in single} | {r["task"] for r in attention})
    for task in tasks:
        lines.append(f"  {task}:")
        gd = [r for r in single if r["task"] == task and r.get("selected_lr")]
        if gd:
            lines.append("    gated_delta:")
            by_gate = {r["gate_mode"]: r["selected_lr"] for r in gd}
            lines.append(f"      single:")
            for gate in ("full", "kda"):
                if gate in by_gate:
                    lines.append(f"        {gate}: {by_gate[gate]:g}")
        attn = [r for r in attention if r["task"] == task and r.get("selected_lr")]
        if attn:
            lines.append(f"    attention: {attn[0]['selected_lr']:g}")
    (out_dir / "frozen_protocol_snippet.yaml").write_text(
        "".join(line + "\n" for line in lines), encoding="utf-8"
    )


def positive_control(table: pd.DataFrame, out_dir: Path) -> None:
    """The paired gap over seeds on each task's screening cell, single-state
    gated-delta runs in the tier-1 tier."""
    subset = table[
        (table["tier"] == _protocol.POSITIVE_CONTROL_TIER)
        & (table["layout"] == "single")
        & (table["mixer"].fillna("gated_delta") == "gated_delta")
    ]
    if subset.empty:
        (out_dir / "positive_control.csv").write_text(
            pd.DataFrame().to_csv(index=False), encoding="utf-8"
        )
        return
    dup = subset.groupby(["task", "seed", "gate_mode"])["run_id"].nunique()
    dup = dup[dup > 1]
    if not dup.empty:
        run_ids = subset[
            subset.set_index(["task", "seed", "gate_mode"]).index.isin(dup.index)
        ]["run_id"].tolist()
        raise RuntimeError(
            "positive control has duplicate (task, seed) rows; refusing to drop "
            f"them silently. Runs: {sorted(set(run_ids))}"
        )
    rows = []
    for task, metric in _protocol.SCREENING_CELLS.items():
        if metric not in subset:
            continue
        by_task = subset[(subset["task"] == task) & subset[metric].notna()]
        report = paired_gap_report(by_task, metric)
        report.insert(0, "task", task)
        report.insert(1, "metric", metric)
        report["tier"] = _protocol.POSITIVE_CONTROL_TIER
        rows.append(report)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    (out_dir / "positive_control.csv").write_text(
        df.to_csv(index=False), encoding="utf-8"
    )


def completeness(table: pd.DataFrame, out_dir: Path) -> None:
    """Per tier, expected vs observed done runs (and unexpected tiers)."""
    df = missing_runs(table, dict(_protocol.EXPECTED_RUNS))
    (out_dir / "completeness.csv").write_text(df.to_csv(index=False), encoding="utf-8")


REPORTS = {
    "calibration": calibration,
    "positive_control": positive_control,
    "completeness": completeness,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="*",
        default=list(REPORTS),
        help=f"report names (default: all: {' '.join(REPORTS)})",
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="analysis")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _load(Path(args.runs_dir))
    logger.info("Loaded %d runs", len(table))
    for name in args.reports:
        if name not in REPORTS:
            raise SystemExit(
                f"unknown report {name!r}; available: {', '.join(REPORTS)}"
            )
        REPORTS[name](table, out_dir)
        logger.info("Wrote %s", name)


if __name__ == "__main__":
    main()
