"""Command-line entry point for the structured-recall pilot's reports.

Builds on :mod:`research_lib.training.record_table.load_records` (with
``common.records.flatten``) and the shared :mod:`common.analysis` pieces.
Each subcommand answers one question; the task-specific readings and
screening cells live in the pilot's protocol.

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

from common.analysis import gap_column, paired_gap_report  # noqa: E402
from common.records import flatten  # noqa: E402

from research_lib.training.record_table import load_records  # noqa: E402

logger = logging.getLogger(__name__)

# The calibration tiers (single-state selection) come from the protocol.
from structured_recall_analysis import protocol as _protocol  # noqa: E402

lr_key = "lr"
CALIBRATION_TIERS = frozenset(_protocol.EXPECTED_RUNS) | frozenset({"calibration"})


def _load(runs_dir: Path) -> pd.DataFrame:
    return load_records(runs_dir, flatten)


def calibration(table: pd.DataFrame, out_dir: Path) -> None:
    """Per (task, layout, gate, lambda) the best validation value per LR,
    the selected point, and at each gate's point the screening cell with
    the ``full - kda`` gap for single-state rows (calibration tiers only)."""
    rows = []
    for keys, group in table.groupby(["task", "layout", "gate_mode", "lambda_mode"]):
        task, layout, gate, lam = keys
        if task not in _protocol.SCREENING_CELLS and layout != "single":
            continue
        if not set(group["tier"].dropna()).intersection(CALIBRATION_TIERS):
            continue
        numeric = group["best_value"].dropna()
        if numeric.empty:
            continue
        best = float(numeric.max())
        selected_lr = _lr_of(numeric, group)
        rows.append(
            {
                "task": task,
                "layout": layout,
                "gate_mode": gate,
                "lambda_mode": lam,
                "best_value": best,
                "selected_lr": selected_lr,
            }
        )
    df = pd.DataFrame(rows)
    # The gap column (full - kda, single-state, per the screening cell).
    partners = []
    for task, metric in _protocol.SCREENING_CELLS.items():
        if metric not in table:
            continue
        cells = table[table["task"] == task][
            ["layout", "gate_mode", "seed", metric]
        ].copy()
        cells["task"] = task
        gap = gap_column(cells, metric)
        cells["gap_full_kda"] = gap.reindex(cells.index).values
        partners.append(cells[["task", "layout", "gate_mode", metric, "gap_full_kda"]])
    if partners:
        gaps = pd.concat(partners, ignore_index=True).dropna(subset=["gap_full_kda"])
        df = pd.concat([df, gaps], ignore_index=True) if not df.empty else gaps
    (out_dir / "calibration.csv").write_text(df.to_csv(index=False), encoding="utf-8")


def _lr_of(best: pd.Series, group: pd.DataFrame) -> float | None:
    if "best_value" not in group:
        return None
    chosen = group.loc[best.index.max()]
    return float(chosen.get("lr")) if chosen.get("lr") is not None else None


def positive_control(table: pd.DataFrame, out_dir: Path) -> None:
    """The paired gap over seeds on each task's screening cell, single-state
    gated-delta runs in the task's tier, at the selected LR per gate."""
    rows = []
    for task, metric in _protocol.SCREENING_CELLS.items():
        if metric not in table:
            continue
        subset = table[
            (table["task"] == task)
            & (table["layout"] == "single")
            & (table["mixer"].fillna("gated_delta") == "gated_delta")
            & (table["tier"].isin(list(CALIBRATION_TIERS)))
        ]
        report = paired_gap_report(subset, metric)
        report.insert(0, "task", task)
        rows.append(report)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    (out_dir / "positive_control.csv").write_text(
        df.to_csv(index=False), encoding="utf-8"
    )


def completeness(table: pd.DataFrame, out_dir: Path) -> None:
    """Missing run IDs per sweep (the expected counts from the protocol)."""
    done = set(table["run_id"].dropna())
    missing = [r for r, _ in _protocol.EXPECTED_RUNS.items()]  # placeholder
    df = pd.DataFrame(
        {
            "run_id": missing,
            "expected": [_protocol.EXPECTED_RUNS.get(r, 0) for r in missing],
        }
    )
    df = df[~df["run_id"].isin(done)]
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
