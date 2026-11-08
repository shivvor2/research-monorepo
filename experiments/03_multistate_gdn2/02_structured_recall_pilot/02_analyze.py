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
    is written from the grouped-decay selection.
``positive_control``
    The paired ``full - kda`` gap on each task's screening cell,
    single-state gated-delta runs in the tier it declares, across the
    tier-1 seeds (no seed filter), with a percentile bootstrap CI. A
    duplicated (task, seed) row after filtering raises naming the run
    IDs. A task with a screening cell but no tiers entry is skipped with
    a log line.
``completeness``
    Per tier, expected vs observed done runs, plus tiers with runs but no
    expected count (via ``common.analysis.missing_runs``).

Tasks that declare a damage reading in the protocol get per-slice
``damage/slice={n}`` columns (``calibration``, at each single-state
gate's selected point and at each multi-state row's selected LR) and
per-gate ``damage_full/slice={n}`` / ``damage_kda/slice={n}`` columns
(``positive_control``): the mean of the high-class accuracy minus the
low-class accuracy over the cell's runs. A task with a screening
fallback reads the shorter slice when both gates fall below its floor.
An unreadable fallback (its column missing from the table, or a gate
with no fallback value at its selected point) leaves the gap
unreported: the calibration rows keep the primary cells without a gap
column and the positive control skips the task, both logged at warning
level.

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
from research_lib.training.run_control import RUN_DONE_STATUS  # noqa: E402

logger = logging.getLogger(__name__)


def _load(runs_dir: Path) -> pd.DataFrame:
    table = load_records(runs_dir, flatten)
    if table.empty:
        # No finished records: still carry the schema ``flatten`` produces
        # so the reports (column access, ``.get``) behave like on an
        # empty-but-typed table instead of raising ``AttributeError``.
        return pd.DataFrame(columns=list(flatten({}).keys()))
    return table


def _gated(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["mixer"].fillna("gated_delta") == "gated_delta"]


def _selection(table: pd.DataFrame) -> pd.DataFrame:
    return table[
        (table["status"].fillna("").astype(str) == RUN_DONE_STATUS)
        & table["best_value"].notna()
        & table["tier"].isin(_protocol.CALIBRATION_TIERS)
        & table["seed"].isin(_protocol.SELECTION_SEEDS)
    ]


def _lr_summary(grp: pd.DataFrame) -> pd.DataFrame:
    """One scalar-LR row per LR for a config group: per-seed means.

    The decay scheme is an *outer* cell attribute (:func:`calibration`
    groups by it), never part of the LR index, so ``select_lr`` stays a
    scalar-LR selection even when a cell has runs under two decay
    schemes.
    """
    per_lr = per_seed_mean(grp, by="lr", selection_seeds=_protocol.SELECTION_SEEDS)
    if per_lr.empty:
        return per_lr
    return per_lr.rename(
        columns={"mean_best_value": "mean_acc", "mean_best_step": "mean_steps"}
    ).set_index("lr")


def _n_seeds(per_lr: pd.DataFrame, chosen) -> int:
    """The selected LR's measured-seed count (0 when nothing was picked)."""
    if chosen is None or chosen not in per_lr.index:
        return 0
    return int(per_lr.loc[chosen, "n_seeds"])


def _cell_mean(grp: pd.DataFrame, metric: str, lr: float) -> float | None:
    sub = grp[(grp["lr"] == lr) & grp[metric].notna()][metric]
    return float(sub.mean()) if len(sub) else None


def _damage(rows: pd.DataFrame, task: str, n: int) -> float | None:
    """The task's damage reading for slice ``n``: the mean per-run
    ``high - low`` over ``rows`` where both columns are present, or
    ``None`` when the task has no reading or a column is missing.

    Args:
        rows: The runs to read (already filtered to one gate / LR).
        task: The task name (looks up the reading in the protocol).
        n: The slice value.

    Returns:
        The mean per-run ``high - low`` difference, or ``None``.
    """
    reading = _protocol.TASK_DAMAGE.get(task)
    if reading is None:
        return None
    high, low = reading.columns(n)
    if high not in rows.columns or low not in rows.columns:
        return None
    sub = rows[rows[high].notna() & rows[low].notna()]
    if sub.empty:
        return None
    return float((sub[high] - sub[low]).mean())


def calibration(table: pd.DataFrame, out_dir: Path) -> None:
    """The LR reading (and the single-state gap on the screening cell).

    The decay scheme is an outer cell attribute (``grouped`` when a run
    carries none), recorded on every row and never folded into the LR
    index, so a cell with runs under two decay schemes yields two rows
    with their own selected LR.
    """
    table = table.copy()
    table["decay_scheme"] = table.get("decay_scheme").fillna("grouped")
    rows: list[dict] = []
    sel = _selection(table)
    gated_sel = _gated(sel)
    # --- gated-delta: single-state (the gap) and multi-state ---
    single = gated_sel[gated_sel["layout"] == "single"]
    multi = gated_sel[gated_sel["layout"] != "single"]
    for keys, grp in single.groupby(["task", "decay_scheme", "lambda_mode"]):
        task, decay, lam = keys
        primary = _protocol.SCREENING_CELLS.get(task)
        gate_rows: dict[str, dict] = {}
        gate_lr: dict[str, float | None] = {}
        primary_cells: dict[str, float | None] = {}
        for gate in ("full", "kda"):
            by_gate = grp[grp["gate_mode"] == gate]
            if by_gate.empty:
                continue
            per_lr = _lr_summary(by_gate)
            if per_lr.empty:
                continue
            chosen = select_lr(per_lr)
            cell = _cell_mean(by_gate, primary, chosen) if primary else None
            row = {
                "kind": "single",
                "task": task,
                "layout": "single",
                "gate_mode": gate,
                "lambda_mode": lam,
                "decay_scheme": decay,
                "mixer": "gated_delta",
                "n_seeds": _n_seeds(per_lr, chosen),
                "d_model": (
                    int(grp["d_model"].dropna().iloc[0])
                    if grp["d_model"].notna().any()
                    else None
                ),
                "selected_lr": chosen,
                "mean_best_value": (
                    float(per_lr.loc[chosen, "mean_acc"])
                    if chosen is not None
                    and chosen in per_lr.index
                    and "mean_acc" in per_lr.columns
                    else None
                ),
                "cell": cell,
                "cell_metric": primary,
            }
            gate_rows[gate] = row
            gate_lr[gate] = chosen
            primary_cells[gate] = cell
            rows.append(row)
        # Apply the screening fallback: if both gates read below the floor
        # at the primary cell, read the shorter slice instead. The fallback
        # only fires when the primary gap is uninformative by protocol, so an
        # unreadable fallback leaves NO gap at all (the primary cells are
        # kept for inspection only).
        metric = _protocol.screening_cell(task, primary_cells) if primary else None
        unreadable_fallback = False
        if metric is not None and metric != primary:
            # Read the fallback cell for BOTH gates at their selected LRs
            # before touching the rows: applying it gate by gate could leave
            # one gate on the primary cell and the other on the fallback (or
            # None), mixing metrics.
            if metric not in grp.columns:
                logger.warning(
                    "screening fallback cell %r missing for %s; reporting "
                    "no gap",
                    metric,
                    task,
                )
                unreadable_fallback = True
            else:
                fallback_cells = {
                    gate: _cell_mean(
                        grp[grp["gate_mode"] == gate], metric, gate_lr[gate]
                    )
                    for gate in gate_rows
                }
                missing = [
                    gate for gate, cell in fallback_cells.items() if cell is None
                ]
                if missing:
                    logger.warning(
                        "screening fallback reading %r missing for %s (%s); "
                        "reporting no gap",
                        metric,
                        task,
                        ", ".join(sorted(missing)),
                    )
                    unreadable_fallback = True
                else:
                    logger.info("screening fallback for %s: reading %s", task, metric)
                    for gate, row in gate_rows.items():
                        row["cell"] = fallback_cells[gate]
                        row["cell_metric"] = metric
        # Per-slice damage at each gate's selected point.
        reading = _protocol.TASK_DAMAGE.get(task)
        if reading is not None:
            for gate, row in gate_rows.items():
                at_lr = grp[(grp["gate_mode"] == gate) & (grp["lr"] == gate_lr[gate])]
                for n in reading.slices:
                    row[f"damage/slice={n}"] = _damage(at_lr, task, n)
        # Attach the full - kda gap to the single-state rows of this cell --
        # never after an unreadable fallback, which leaves no valid screening
        # reading.
        cells = {gate: row["cell"] for gate, row in gate_rows.items()}
        if (
            not unreadable_fallback
            and "full" in cells
            and "kda" in cells
            and cells["full"] is not None
            and cells["kda"] is not None
        ):
            gap = cells["full"] - cells["kda"]
            for row in gate_rows.values():
                row["gap_full_kda"] = gap

    for keys, grp in multi.groupby(
        ["task", "decay_scheme", "layout", "gate_mode", "lambda_mode"]
    ):
        task, decay, layout, gate, lam = keys
        per_lr = _lr_summary(grp)
        if per_lr.empty:
            continue
        chosen = select_lr(per_lr)
        row = {
            "kind": "multi",
            "task": task,
            "layout": layout,
            "gate_mode": gate,
            "lambda_mode": lam,
            "decay_scheme": decay,
            "mixer": "gated_delta",
            "n_seeds": _n_seeds(per_lr, chosen),
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
        reading = _protocol.TASK_DAMAGE.get(task)
        if reading is not None:
            at_lr = grp[grp["lr"] == chosen]
            for n in reading.slices:
                row[f"damage/slice={n}"] = _damage(at_lr, task, n)
        rows.append(row)
    # --- attention: one row per (task, decay, layout); no gate ---
    attn = sel[sel["mixer"].fillna("gated_delta") == "attention"]
    for keys, grp in attn.groupby(["task", "decay_scheme", "layout"]):
        task, decay, layout = keys
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
                "decay_scheme": decay,
                "mixer": "attention",
                "n_seeds": _n_seeds(per_lr, chosen),
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
    """A frozen-protocol LR-table snippet from the grouped-decay selection.

    The gated-delta LRs come from the ``grouped``-decay rows, single and
    multi-state, in the shape :func:`run._frozen_lr` reads:
    ``lr_table.<task>.gated_delta.<layout>.<gate>``. The attention rate
    comes only from ``grouped``-decay attention rows and is a single
    flat value per task; a task with more than one such row (attention
    runs at different ``model.layout`` values) is an error rather than a
    silently-picked first. A skeleton the calibration step can grow into
    the full ``lr_table``.
    """
    gated = [
        r
        for r in rows
        if r.get("kind") in ("single", "multi")
        and r.get("decay_scheme") == "grouped"
        and r.get("selected_lr") is not None
    ]
    attention = [
        r
        for r in rows
        if r.get("kind") == "attention"
        and r.get("decay_scheme") == "grouped"
        and r.get("selected_lr") is not None
    ]
    lines = ["lr_table:"]
    tasks = sorted({r["task"] for r in gated} | {r["task"] for r in attention})
    for task in tasks:
        lines.append(f"  {task}:")
        by_layout: dict[str, dict[str, float]] = {}
        for r in gated:
            if r["task"] != task:
                continue
            layout = r.get("layout") or "single"
            by_layout.setdefault(layout, {})[r["gate_mode"]] = r["selected_lr"]
        if by_layout:
            lines.append("    gated_delta:")
            for layout in sorted(by_layout):
                lines.append(f"      {layout}:")
                for gate in ("full", "kda"):
                    if gate in by_layout[layout]:
                        lines.append(f"        {gate}: {by_layout[layout][gate]:g}")
        attn = [r for r in attention if r["task"] == task]
        if len(attn) > 1:
            raise RuntimeError(
                f"task {task} has more than one grouped attention reading "
                f"(attention runs at different model.layout values): "
                f"{sorted({r.get('layout') for r in attn})}; refusing to pick "
                f"the first"
            )
        if len(attn) == 1:
            lines.append(f"    attention: {attn[0]['selected_lr']:g}")
    (out_dir / "frozen_protocol_snippet.yaml").write_text(
        "".join(line + "\n" for line in lines), encoding="utf-8"
    )


def positive_control(table: pd.DataFrame, out_dir: Path) -> None:
    """The paired gap on each task's screening cell, single-state
    gated-delta runs in the tier the task declares."""
    base = table[
        (table["layout"] == "single")
        & (table["mixer"].fillna("gated_delta") == "gated_delta")
    ]
    rows = []
    for task, primary in _protocol.SCREENING_CELLS.items():
        tiers = _protocol.TASK_TIERS.get(task)
        if tiers is None:
            logger.info("skipping positive control for %s: no TASK_TIERS entry", task)
            continue
        if primary not in base.columns:
            logger.info(
                "skipping positive control for %s: screening cell %r not "
                "in the run table",
                task,
                primary,
            )
            continue
        tier_rows = base[
            (base["task"] == task) & (base["tier"] == tiers.positive_control)
        ]
        # Per-gate means at the primary cell, then the fallback rule.
        readings: dict[str, float | None] = {}
        for gate in ("full", "kda"):
            sub = tier_rows[tier_rows["gate_mode"] == gate][primary].dropna()
            readings[gate] = float(sub.mean()) if not sub.empty else None
        metric = _protocol.screening_cell(task, readings)
        if metric != primary and metric not in base.columns:
            logger.warning(
                "screening fallback cell %r missing for %s; skipping the task",
                metric,
                task,
            )
            continue
        if metric != primary:
            logger.info("screening fallback for %s: reading %s", task, metric)
        by_task = tier_rows[tier_rows[metric].notna()]
        dup = by_task.groupby(["seed", "gate_mode"])["run_id"].nunique()
        dup = dup[dup > 1]
        if not dup.empty:
            run_ids = by_task[
                by_task.set_index(["seed", "gate_mode"]).index.isin(dup.index)
            ]["run_id"].tolist()
            raise RuntimeError(
                "positive control has duplicate (task, seed) rows; refusing to "
                f"drop them silently. Runs: {sorted(set(run_ids))}"
            )
        report = paired_gap_report(by_task, metric)
        report.insert(0, "task", task)
        report.insert(1, "metric", metric)
        report["tier"] = tiers.positive_control
        # Per-slice damage per gate over the tier's runs.
        reading = _protocol.TASK_DAMAGE.get(task)
        if reading is not None:
            for gate in ("full", "kda"):
                gate_rows = by_task[by_task["gate_mode"] == gate]
                for n in reading.slices:
                    report[f"damage_{gate}/slice={n}"] = _damage(gate_rows, task, n)
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
