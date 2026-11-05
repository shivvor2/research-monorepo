"""Command-line entry point for the structured-recall pilot's reports.

Builds on :mod:`research_lib.training.records.load_records` (with
``common.records.flatten``) and the shared :mod:`common.analysis` pieces;
each subcommand answers one question. Each task registers its screening
cell in ``structured_recall_analysis/protocol.py``.

Usage:
    python 02_analyze.py calibration positive_control completeness
    python 02_analyze.py --runs-dir runs --out-dir analysis
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

import common.records
from common.analysis import gap_column, paired_gap_report
from common.records import flatten
from research_lib.training.records import load_records

logger = logging.getLogger(__name__)


def _load(runs_dir: Path) -> pd.DataFrame:
    return load_records(runs_dir, flatten)


def calibration(table: pd.DataFrame, out_dir: Path) -> None:
    """The per-(task, layout, gate) best validation value and the gap
    ``full - kda`` at each selected point."""
    out = []
    for (task, layout), group in table.groupby(
        ["task", "layout"], dropna=False
    ):
        cell = None
        for metric in ("best_value", "test/acc/slice=64"):
            if metric in group.columns and group[metric].notna().any():
                cell = metric
                break
        if cell is None:
            continue
        row = {
            "task": task,
            "layout": layout,
            "best_full": _at(group, cell, "full"),
            "best_kda": _at(group, cell, "kda"),
        }
        row["gap_full_kda"] = (
            None if pd.isna(row["best_full"]) or pd.isna(row["best_kda"])
            else round(row["best_full"] - row["best_kda"], 4)
        )
        out.append(row)
    df = pd.DataFrame(out)
    (out_dir / "calibration.csv").write_text(df.to_csv(index=False), encoding="utf-8")


def _at(group: pd.DataFrame, metric: str, gate: str) -> float | None:
    sub = group[group["gate_mode"] == gate][metric]
    non_null = sub.dropna()
    return float(non_null.max()) if not non_null.empty else None


def positive_control(table: pd.DataFrame, out_dir: Path) -> None:
    """The paired gap over seeds on each task's screening cell."""
    from structured_recall_analysis.protocol import SCREENING_CELLS

    rows = []
    for task, metric in SCREENING_CELLS.items():
        single = table[table["layout"] == "single"]
        if metric not in single.columns:
            continue
        report = paired_gap_report(single, metric)
        report.insert(0, "task", task)
        rows.append(report)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    (out_dir / "positive_control.csv").write_text(
        df.to_csv(index=False), encoding="utf-8"
    )


def completeness(table: pd.DataFrame, out_dir: Path) -> None:
    """Missing runs per sweep (given the loaded records; expected counts
    come from the pilot protocol empty until the task PRs)."""
    from structured_recall_analysis.protocol import EXPECTED_RUNS

    counts = table.groupby("tier").size()
    df = pd.DataFrame(
        {"tier": list(counts.index), "local_done": list(counts.values)}
    )
    df["expected"] = df["tier"].map(EXPECTED_RUNS)
    df["missing"] = (df["expected"] - df["local_done"]).clip(lower=0)
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
            raise SystemExit(f"unknown report {name!r}; available: {', '.join(REPORTS)}")
        REPORTS[name](table, out_dir)
        logger.info("Wrote %s", name)


if __name__ == "__main__":
    main()