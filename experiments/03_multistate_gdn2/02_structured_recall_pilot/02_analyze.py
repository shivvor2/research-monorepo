"""Command-line entry point for the structured-recall pilot's reports.

Builds on :mod:`research_lib.training.record_table.load_records` (with
``common.records.flatten``) and the shared :mod:`common.analysis` pieces.
Each subcommand answers one question; the tier / screening-cell
constants live in the pilot's protocol.

Reports
-------
``calibration``
    The per-configuration tuning-point selection: for the gated-delta
    calibration tiers, per (task, seq_len, layout, gate, lambda) the
    per-seed mean of the best validation value per ``(lr, weight_decay)``
    point, the selected point (best mean, then fewer steps, then smaller
    LR / weight decay), and for single-state the screening cell at each
    gate's selected point with the ``full - kda`` gap. Attention runs get
    their own row per (task, seq_len, layout) — they have no gate. A
    ``frozen_protocol_snippet.yaml`` is written from the grouped-decay
    selection, limited to one sequence length per task.
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
gate's selected point and at each multi-state row's selected point) and
per-gate ``damage_full/slice={n}`` / ``damage_kda/slice={n}`` columns
(``positive_control``): the mean of the high-class accuracy minus the
low-class accuracy over the cell's runs. A task with a screening
fallback reads the shorter slice when both gates fall below its floor.
An unreadable fallback (its column missing from the table, or a gate
with no fallback value at its selected point) leaves the gap
unreported: the calibration rows keep the primary cells without a gap
column and the positive control skips the task, both logged at warning
level.

A task with a ``SettingBand`` (in :data:`~.protocol.TASK_SETTING_BANDS`)
gets best-validation ``band`` / ``near_edge`` columns per calibration
row, and its runs a look-stuck flag in ``safeguard.csv``. ``--seq-len
<task>=<n>`` names one sequence length per task for the reports; a task
whose rows span more than one length is refused without it, and an
override that matches no rows is an error rather than a silently-empty
report.

Usage:
    python 02_analyze.py calibration positive_control completeness
    python 02_analyze.py --runs-dir runs --out-dir analysis \
        --seq-len mad_fuzzy=512
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
    paired_gap_report,
    per_seed_mean,
    select_hyperparams,
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


def _hyper_summary(grp: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(lr, weight_decay)`` point for a config group.

    The decay scheme and sequence length are *outer* cell attributes
    (:func:`calibration` groups by them), never part of the tuning index,
    so the selection stays per-point even when a cell has runs under two
    decay schemes or two lengths.
    """
    per = per_seed_mean(
        grp, by=["lr", "weight_decay"], selection_seeds=_protocol.SELECTION_SEEDS
    )
    if per.empty:
        return per
    return per.rename(
        columns={"mean_best_value": "mean_acc", "mean_best_step": "mean_steps"}
    )


def _n_seeds(per: pd.DataFrame, lr: float, weight_decay: float) -> int:
    """The selected point's measured-seed count (0 when nothing was picked)."""
    if per.empty:
        return 0
    sub = per[(per["lr"] == lr) & (per["weight_decay"] == weight_decay)]
    return int(sub["n_seeds"].iloc[0]) if len(sub) else 0


def _cell_mean(
    grp: pd.DataFrame, metric: str, lr: float, weight_decay: float
) -> float | None:
    """The mean of ``metric`` over ``grp`` at the selected point."""
    sub = grp[
        (grp["lr"] == lr) & (grp["weight_decay"] == weight_decay) & grp[metric].notna()
    ][metric]
    return float(sub.mean()) if len(sub) else None


def _selected_mean(per: pd.DataFrame, lr: float, weight_decay: float) -> float | None:
    """The selected point's mean best validation value."""
    if per.empty:
        return None
    sub = per[(per["lr"] == lr) & (per["weight_decay"] == weight_decay)]
    return float(sub["mean_acc"].iloc[0]) if len(sub) else None


def _damage(rows: pd.DataFrame, task: str, n: int) -> float | None:
    """The task's damage reading for slice ``n``: the mean per-run
    ``high - low`` over ``rows`` where both columns are present, or
    ``None`` when the task has no reading or a column is missing.

    Args:
        rows: The runs to read (already filtered to one gate / point).
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


def _band(value: float | None, band: _protocol.SettingBand) -> str | None:
    """The band of a best validation value under ``band``."""
    if value is None:
        return None
    if value < band.floor:
        return "below_floor"
    if value <= band.ceiling:
        return "in_band"
    return "above_ceiling"


def _near_edge(value: float | None, band: _protocol.SettingBand) -> bool | None:
    """Whether a best validation value lies within ``band.margin`` of the
    band's floor or ceiling (a reading to re-check)."""
    if value is None:
        return None
    return bool(
        abs(value - band.floor) < band.margin or abs(value - band.ceiling) < band.margin
    )


def _own_task_tiers(sel: pd.DataFrame) -> pd.DataFrame:
    """Keep only selected rows whose tier belongs to their own task's
    calibration tiers; log any excluded run IDs (a mismatched row would
    otherwise form a spurious calibration group)."""
    if sel.empty:
        return sel
    keep: list[str] = []
    exclude: list[str] = []
    for _, r in sel.iterrows():
        task = str(r.get("task"))
        tiers = _protocol.TASK_TIERS.get(task)
        if tiers is not None and str(r.get("tier")) in tiers.calibration:
            keep.append(str(r.get("run_id")))
        else:
            exclude.append(str(r.get("run_id")))
    if exclude:
        logger.warning(
            "calibration: excluding runs whose tier is not in their task's "
            "calibration tiers: %s",
            sorted(exclude),
        )
    return sel[sel["run_id"].isin(keep)]


def _validate_seq_len_overrides(
    table: pd.DataFrame, seq_len_overrides: dict[str, int]
) -> None:
    """Refuse a ``--seq-len`` override that names no task in the table or a
    length with no rows for that task (an ignored override must not pass)."""
    if not seq_len_overrides:
        return
    present_tasks = (
        set(table["task"].dropna().astype(str)) if "task" in table.columns else set()
    )
    for task, length in seq_len_overrides.items():
        if task not in present_tasks:
            raise RuntimeError(
                f"--seq-len names task {task!r}, which has no rows in the run "
                "table; refusing an ignored override"
            )
        if "seq_len" in table.columns:
            lengths = set(table[table["task"] == task]["seq_len"].dropna().astype(int))
            if length not in lengths:
                raise RuntimeError(
                    f"--seq-len {task}={length} matches no rows; readings "
                    f"exist at {sorted(lengths)}"
                )


def _looks_stuck(row: pd.Series) -> bool | None:
    """A per-run look-stuck flag from the recorded monitored metric.

    ``True`` when the validation metric never exceeds twice its first
    value (``max <= 2 * first``), **or** every last-half value stays
    within 0.02 of the first value (``last_half_max - first <= 0.02``
    and ``first - last_half_min <= 0.02``).  ``None`` when the run has no
    recorded monitored history.  The report only prints the flag; it
    decides nothing.
    """

    def _nan(value: object) -> bool:
        return value is None or (isinstance(value, float) and value != value)

    first = row.get("monitored_first")
    max_value = row.get("monitored_max")
    last_half_max = row.get("monitored_last_half_max")
    last_half_min = row.get("monitored_last_half_min")
    if any(_nan(v) for v in (first, max_value, last_half_max, last_half_min)):
        return None
    never_doubles = max_value <= 2 * first
    flat_tail = last_half_max - first <= 0.02 and first - last_half_min <= 0.02
    return bool(never_doubles or flat_tail)


def calibration(
    table: pd.DataFrame, out_dir: Path, seq_len_overrides: dict[str, int] | None = None
) -> None:
    """The tuning-point reading (and the single-state gap on the screening
    cell).

    The sequence length and the decay scheme are outer cell attributes
    (``seq_len`` from the run's task config, ``grouped`` when a run
    carries none), recorded on every row and never folded into the tuning
    index: a task with calibration runs at more than one length yields one
    set of rows per length, and selection happens inside each length only.
    """
    table = table.copy()
    table["decay_scheme"] = table.get("decay_scheme").fillna("grouped")
    seq = table.get("seq_len")
    table["seq_len"] = 0 if seq is None else seq.fillna(0).astype(int)
    _validate_seq_len_overrides(table, seq_len_overrides or {})
    rows: list[dict] = []
    sel = _own_task_tiers(_selection(table))
    gated_sel = _gated(sel)
    attn = sel[sel["mixer"].fillna("gated_delta") == "attention"]
    # --- Refuse a silent drop: a selected gated/attention run must name a
    # weight decay and a learning rate (from the record or the config), or
    # the (lr, weight_decay) grouping would drop it silently.
    for label, chosen in (("gated", gated_sel), ("attention", attn)):
        for column, name in (("weight_decay", "weight decay"), ("lr", "learning rate")):
            missing = chosen[chosen[column].isna()]["run_id"].dropna()
            if not missing.empty:
                raise RuntimeError(
                    f"calibration: {label} runs have no {name} (neither the "
                    f"record field nor config.optimizer.{column}); refusing a "
                    f"silent drop. Runs: {sorted(missing.tolist())}"
                )
    # --- Refuse a task's band rows recorded on another monitor key: their
    # best validation value would be classified under the wrong metric.
    for task, band in _protocol.TASK_SETTING_BANDS.items():
        band_rows = sel[sel["task"] == task]
        if "monitor_key" not in band_rows.columns:
            continue
        has = band_rows["monitor_key"].notna()
        wrong = band_rows[has & (band_rows["monitor_key"] != band.metric)]["run_id"]
        if not wrong.empty:
            raise RuntimeError(
                f"calibration: task {task!r} has a band on {band.metric}, but "
                f"runs were monitored on another key. Runs: "
                f"{sorted(wrong.tolist())}"
            )
    # --- gated-delta: single-state (the gap) and multi-state ---
    single = gated_sel[gated_sel["layout"] == "single"]
    multi = gated_sel[gated_sel["layout"] != "single"]
    for keys, grp in single.groupby(["task", "seq_len", "decay_scheme", "lambda_mode"]):
        task, seq_len, decay, lam = keys
        primary = _protocol.SCREENING_CELLS.get(task)
        gate_rows: dict[str, dict] = {}
        gate_point: dict[str, tuple[float | None, float | None]] = {}
        primary_cells: dict[str, float | None] = {}
        for gate in ("full", "kda"):
            by_gate = grp[grp["gate_mode"] == gate]
            if by_gate.empty:
                continue
            per = _hyper_summary(by_gate)
            if per.empty:
                continue
            chosen_lr, chosen_wd = select_hyperparams(per)
            cell = (
                _cell_mean(by_gate, primary, chosen_lr, chosen_wd) if primary else None
            )
            row = {
                "kind": "single",
                "task": task,
                "seq_len": int(seq_len),
                "layout": "single",
                "gate_mode": gate,
                "lambda_mode": lam,
                "decay_scheme": decay,
                "mixer": "gated_delta",
                "n_seeds": _n_seeds(per, chosen_lr, chosen_wd),
                "d_model": (
                    int(grp["d_model"].dropna().iloc[0])
                    if grp["d_model"].notna().any()
                    else None
                ),
                "selected_lr": chosen_lr,
                "selected_weight_decay": chosen_wd,
                "mean_best_value": _selected_mean(per, chosen_lr, chosen_wd),
                "cell": cell,
                "cell_metric": primary,
            }
            band = _protocol.TASK_SETTING_BANDS.get(task)
            if band is not None:
                row["band"] = _band(row["mean_best_value"], band)
                row["near_edge"] = _near_edge(row["mean_best_value"], band)
            gate_rows[gate] = row
            gate_point[gate] = (chosen_lr, chosen_wd)
            primary_cells[gate] = cell
            rows.append(row)
        # The screening fallback: if both gates read below the floor at the
        # primary cell, read the shorter slice instead, at each gate's
        # selected point.  An unreadable fallback leaves NO gap (the
        # primary cells are kept for inspection only).
        metric = _protocol.screening_cell(task, primary_cells) if primary else None
        unreadable_fallback = False
        if metric is not None and metric != primary:
            if metric not in grp.columns:
                logger.warning(
                    "screening fallback cell %r missing for %s; reporting no gap",
                    metric,
                    task,
                )
                unreadable_fallback = True
            else:
                fallback_cells = {
                    gate: _cell_mean(
                        grp[grp["gate_mode"] == gate], metric, *gate_point[gate]
                    )
                    for gate in gate_rows
                }
                missing_list = [
                    gate for gate, cell in fallback_cells.items() if cell is None
                ]
                if missing_list:
                    logger.warning(
                        "screening fallback reading %r missing for %s (%s); "
                        "reporting no gap",
                        metric,
                        task,
                        ", ".join(sorted(missing_list)),
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
                at_point = grp[
                    (grp["gate_mode"] == gate)
                    & (grp["lr"] == gate_point[gate][0])
                    & (grp["weight_decay"] == gate_point[gate][1])
                ]
                for n in reading.slices:
                    row[f"damage/slice={n}"] = _damage(at_point, task, n)
        # Attach the full - kda gap to these single-state rows -- never after
        # an unreadable fallback, which leaves no valid screening reading.
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
        ["task", "seq_len", "decay_scheme", "layout", "gate_mode", "lambda_mode"]
    ):
        task, seq_len, decay, layout, gate, lam = keys
        per = _hyper_summary(grp)
        if per.empty:
            continue
        chosen_lr, chosen_wd = select_hyperparams(per)
        row = {
            "kind": "multi",
            "task": task,
            "seq_len": int(seq_len),
            "layout": layout,
            "gate_mode": gate,
            "lambda_mode": lam,
            "decay_scheme": decay,
            "mixer": "gated_delta",
            "n_seeds": _n_seeds(per, chosen_lr, chosen_wd),
            "d_model": (
                int(grp["d_model"].dropna().iloc[0])
                if grp["d_model"].notna().any()
                else None
            ),
            "selected_lr": chosen_lr,
            "selected_weight_decay": chosen_wd,
            "mean_best_value": _selected_mean(per, chosen_lr, chosen_wd),
        }
        reading = _protocol.TASK_DAMAGE.get(task)
        if reading is not None:
            at_point = grp[
                (grp["lr"] == chosen_lr) & (grp["weight_decay"] == chosen_wd)
            ]
            for n in reading.slices:
                row[f"damage/slice={n}"] = _damage(at_point, task, n)
        rows.append(row)
    # --- attention: one row per (task, seq_len, decay, layout); no gate ---
    for keys, grp in attn.groupby(["task", "seq_len", "decay_scheme", "layout"]):
        task, seq_len, decay, layout = keys
        per = _hyper_summary(grp)
        if per.empty:
            continue
        chosen_lr, chosen_wd = select_hyperparams(per)
        rows.append(
            {
                "kind": "attention",
                "task": task,
                "seq_len": int(seq_len),
                "layout": layout,
                "gate_mode": None,
                "lambda_mode": None,
                "decay_scheme": decay,
                "mixer": "attention",
                "n_seeds": _n_seeds(per, chosen_lr, chosen_wd),
                "d_model": (
                    int(grp["d_model"].dropna().iloc[0])
                    if grp["d_model"].notna().any()
                    else None
                ),
                "selected_lr": chosen_lr,
                "selected_weight_decay": chosen_wd,
                "mean_best_value": _selected_mean(per, chosen_lr, chosen_wd),
            }
        )

    df = pd.DataFrame(rows)
    (out_dir / "calibration.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    _stuck_report(sel, out_dir)
    _frozen_snippet(rows, out_dir, seq_len_overrides or {})


def _stuck_report(sel: pd.DataFrame, out_dir: Path) -> None:
    """Per-run look-stuck flags for the calibration-reading tasks.

    Emits one row per run that has a recorded monitored history, flagged
    by :func:`_looks_stuck`.  The report prints the flags only; it
    decides nothing.
    """
    flagged = []
    for _, r in sel.iterrows():
        if str(r.get("task")) not in _protocol.TASK_STUCK_CHECK:
            continue
        flag = _looks_stuck(r)
        if flag is None:
            continue
        flagged.append(
            {
                "run_id": r.get("run_id"),
                "task": r.get("task"),
                "tier": r.get("tier"),
                "layout": r.get("layout"),
                "gate_mode": r.get("gate_mode"),
                "seed": r.get("seed"),
                "looks_stuck": flag,
            }
        )
    (out_dir / "safeguard.csv").write_text(
        pd.DataFrame(flagged).to_csv(index=False), encoding="utf-8"
    )


def _scalar_value(row: dict, key: str) -> float | None:
    """The row's selected value for ``key`` (``None`` when absent / NaN)."""
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return None


def _frozen_snippet(
    rows: list[dict], out_dir: Path, seq_len_overrides: dict[str, int]
) -> None:
    """A frozen-protocol snippet from the grouped-decay selection.

    Emits ``lr_table`` and ``weight_decay_table``, each in the shape
    :func:`run._frozen_value` reads: ``<table>.<task>.gated_delta.<layout>
    .<gate>`` for the gated-delta cells and ``<table>.<task>.attention``
    (a single flat value per task) for the attention control.  The
    sequence length is picked once per task over the combined (gated and
    attention) rows, and for each task that carries a frozen setting the
    snippet also writes ``task_settings.<task>`` with the length its tables
    were taken from (so the length and the tables cannot disagree).  Without
    ``seq_len_overrides`` a task with rows at more than one length is
    refused; an override matching no rows is an error.
    """
    entries = [
        r
        for r in rows
        if r.get("decay_scheme") == "grouped"
        and (
            _scalar_value(r, "selected_lr") is not None
            or _scalar_value(r, "selected_weight_decay") is not None
        )
    ]
    chosen = _chosen_lengths(entries, "length", seq_len_overrides)
    tables = "".join(
        _frozen_table_section(rows, name, chosen) for name in ("lr", "weight_decay")
    )
    blocks = [tables.rstrip("\n")]
    task_names = sorted({r["task"] for r in entries} & set(_protocol.FROZEN_TASK_KEYS))
    if task_names:
        block = ["task_settings:"]
        for task in task_names:
            block.append(f"  {task}:")
            for key in _protocol.FROZEN_TASK_KEYS[task]:
                block.append(f"    {key}: {chosen[task]}")
        blocks.append("\n".join(block))
    (out_dir / "frozen_protocol_snippet.yaml").write_text(
        "\n\n".join(blocks) + "\n", encoding="utf-8"
    )


def _chosen_lengths(
    entries: list[dict], scalar: str, seq_len_overrides: dict[str, int]
) -> dict[str, int]:
    """One frozen length per task over ``entries`` (gated + attention).

    A task named in ``seq_len_overrides`` freezes at that length (an
    override with no matching row is an error naming the lengths that do
    have readings); otherwise a task is frozen at its single distinct
    length, and a task whose rows span more than one length is an error
    naming them (the frozen snippet must not silently mix lengths).
    """
    by_length: dict[str, set] = {}
    for r in entries:
        by_length.setdefault(r["task"], set()).add(r.get("seq_len"))
    chosen: dict[str, int] = {}
    for task, lengths in by_length.items():
        if task in seq_len_overrides:
            override = int(seq_len_overrides[task])
            if override not in lengths:
                raise RuntimeError(
                    f"task {task!r} has no {scalar} rows at the frozen "
                    f"length {override} (readings exist at {sorted(lengths)}); "
                    "refusing a silently-empty frozen table"
                )
            chosen[task] = override
        elif len(lengths) > 1:
            raise RuntimeError(
                f"task {task!r} has {scalar} rows at more than one sequence "
                f"length ({sorted(lengths)}); pass --seq-len {task}=<n> to "
                "freeze one"
            )
        else:
            chosen[task] = next(iter(lengths))
    return chosen


def _frozen_table_section(rows: list[dict], scalar: str, chosen: dict[str, int]) -> str:
    """One frozen table's YAML block (``lr`` or ``weight_decay``), limited to
    the frozen length ``chosen`` per task. Two grouped rows at the same
    (task, layout, gate) with different values are refused (identical values
    may coexist)."""
    key = "selected_lr" if scalar == "lr" else "selected_weight_decay"
    gated = []
    attention = []
    for r in rows:
        if r.get("decay_scheme") != "grouped":
            continue
        value = _scalar_value(r, key)
        if value is None:
            continue
        if r.get("kind") == "attention":
            attention.append({**r, "_selected": value})
        elif r.get("kind") in ("single", "multi"):
            gated.append({**r, "_selected": value})
    gated = [r for r in gated if r.get("seq_len") == chosen.get(r["task"])]
    attention = [r for r in attention if r.get("seq_len") == chosen.get(r["task"])]
    table_name = f"{scalar}_table"
    lines = [f"{table_name}:"]
    tasks = sorted({r["task"] for r in gated} | {r["task"] for r in attention})
    for task in tasks:
        lines.append(f"  {task}:")
        by_layout: dict[str, dict[str, float]] = {}
        for r in gated:
            if r["task"] != task:
                continue
            layout = r.get("layout") or "single"
            gate = r["gate_mode"]
            existing = by_layout.get(layout, {}).get(gate)
            if existing is not None and existing != r["_selected"]:
                raise RuntimeError(
                    f"task {task!r} has conflicting frozen {scalar} values at "
                    f"({layout}, {gate}): {existing!r} and {r['_selected']!r}; "
                    "refusing to overwrite one with the other"
                )
            by_layout.setdefault(layout, {})[gate] = r["_selected"]
        if by_layout:
            lines.append("    gated_delta:")
            for layout in sorted(by_layout):
                lines.append(f"      {layout}:")
                for gate in ("full", "kda"):
                    if gate in by_layout[layout]:
                        lines.append(f"        {gate}: {by_layout[layout][gate]:g}")
        attn = [a for a in attention if a["task"] == task]
        if len(attn) > 1:
            raise RuntimeError(
                f"task {task} has more than one grouped attention reading at "
                f"the frozen length: {sorted({a.get('layout') for a in attn})}; "
                "refusing to pick the first"
            )
        if len(attn) == 1:
            lines.append(f"    attention: {attn[0]['_selected']:g}")
    return "".join(line + "\n" for line in lines)


def _one_length(
    tier_rows: pd.DataFrame, task: str, seq_len_overrides: dict[str, int]
) -> pd.DataFrame:
    """Limit a task's tier-1 rows to one sequence length.

    Rows sharing one length are kept as-is; rows spanning more than one
    length use the override when given and otherwise raise naming the
    lengths (so a mixed-length tier would surface as a wrong-looking
    duplicate-row error no longer).
    """
    lengths = (
        set(tier_rows["seq_len"].dropna().astype(int))
        if "seq_len" in tier_rows
        else set()
    )
    if len(lengths) <= 1:
        return tier_rows
    override = seq_len_overrides.get(task)
    if override is not None:
        return tier_rows[tier_rows["seq_len"] == int(override)]
    raise RuntimeError(
        f"task {task!r}: positive-control tier-1 runs span more than one "
        f"sequence length ({sorted(lengths)}); pass --seq-len {task}=<n> to "
        "pick one"
    )


def positive_control(
    table: pd.DataFrame, out_dir: Path, seq_len_overrides: dict[str, int] | None = None
) -> None:
    """The paired gap on each task's screening cell, single-state
    gated-delta runs in the tier the task declares."""
    overrides = seq_len_overrides or {}
    _validate_seq_len_overrides(table, overrides)
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
        # A registered task may have the metric column absent (e.g. before
        # any of its runs exist): fill so the mask is all-False and the
        # report stays empty rather than raising on a missing column.
        if primary not in base.columns:
            base = base.copy()
            base[primary] = float("nan")
        tier_rows = base[
            (base["task"] == task) & (base["tier"] == tiers.positive_control)
        ]
        if tier_rows.empty:
            continue
        tier_rows = _one_length(tier_rows, task, overrides)
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
        if by_task.empty:
            continue
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


def _order_lengths(lengths: set) -> list:
    """Lengths in ascending order, the missing length (``pd.NA``) last."""
    ordered = sorted(l for l in lengths if l is not pd.NA)
    if any(l is pd.NA for l in lengths):
        ordered.append(pd.NA)
    return ordered


def _completeness_rows(
    table: pd.DataFrame, expected: dict[str, int], overrides: dict[str, int]
) -> list[dict]:
    """Per-``(tier, seq_len)`` expected / observed done counts.

    The length rerun reuses the same tier, so completeness reads one cell
    per length: the tier's expected count applies to each length present,
    and an override limits a task to the given length (other lengths of
    that task are ignored). Runs without a length (a missing ``seq_len``
    column value, or an absent column at all) are counted in their own
    ``(tier, <NA>)`` row rather than dropped, and that row's ``seq_len`` is
    written empty. A tier with no done runs gets a single row with the
    expected count.
    """
    table = table.copy()
    if "seq_len" in table.columns:
        table["_seq"] = pd.to_numeric(table["seq_len"], errors="coerce").astype("Int64")
    else:
        table["_seq"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
    keep = pd.Series(True, index=table.index)
    for task, length in overrides.items():
        # Numeric comparison (never str, which would float-truncate): a row
        # at the task's own other length, or without a length, is dropped;
        # rows of other tasks are untouched.
        own = table["task"] == task
        at_length = table["_seq"].fillna(-1) == int(length)
        keep &= ~(own & ~at_length)
    table = table[keep]
    done = table[table["status"].fillna("").astype(str) == RUN_DONE_STATUS]
    observed: dict[tuple[str, object], int] = {}
    for (tier, length), grp in done.groupby(["tier", "_seq"], dropna=False):
        observed[(str(tier), length)] = int(grp["run_id"].nunique())
    present_lengths: dict[str, set] = {}
    for tier_s, length in observed:
        present_lengths.setdefault(tier_s, set()).add(length)
    rows: list[dict] = []
    for tier_s, exp in expected.items():
        lengths = present_lengths.get(tier_s)
        if not lengths:
            rows.append(
                {
                    "tier": tier_s,
                    "seq_len": "",
                    "expected": exp,
                    "observed": 0,
                    "missing": exp,
                    "known": True,
                }
            )
            continue
        for length in _order_lengths(lengths):
            obs = observed.get((tier_s, length), 0)
            rows.append(
                {
                    "tier": tier_s,
                    "seq_len": "" if length is pd.NA else length,
                    "expected": exp,
                    "observed": obs,
                    "missing": max(0, exp - obs),
                    "known": True,
                }
            )
    # Unexpected tiers present in the done runs (a tier with runs but no
    # expected count) are reported per length.
    for tier_s in sorted(present_lengths):
        if tier_s in expected:
            continue
        for length in _order_lengths(present_lengths[tier_s]):
            rows.append(
                {
                    "tier": tier_s,
                    "seq_len": "" if length is pd.NA else length,
                    "expected": 0,
                    "observed": observed.get((tier_s, length), 0),
                    "missing": 0,
                    "known": False,
                }
            )
    return rows


def completeness(
    table: pd.DataFrame,
    out_dir: Path,
    seq_len_overrides: dict[str, int] | None = None,
) -> None:
    """Per ``(tier, seq_len)`` expected vs observed done runs (and unexpected
    tiers). ``seq_len_overrides`` limits a task to the given length."""
    overrides = seq_len_overrides or {}
    _validate_seq_len_overrides(table, overrides)
    rows = _completeness_rows(table, dict(_protocol.EXPECTED_RUNS), overrides)
    df = pd.DataFrame(rows)
    (out_dir / "completeness.csv").write_text(df.to_csv(index=False), encoding="utf-8")


def _parse_seq_len_overrides(values: list[str] | None) -> dict[str, int]:
    """Parse ``--seq-len <task>=<n>`` pairs into ``{task: length}``."""
    overrides: dict[str, int] = {}
    for item in values or ():
        if "=" not in item:
            raise SystemExit(f"--seq-len expects <task>=<n>, got {item!r}")
        task, _, raw = item.partition("=")
        overrides[task] = int(raw)
    return overrides


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
    parser.add_argument(
        "--seq-len",
        action="append",
        metavar="TASK=N",
        help="freeze task TASK's snippet at sequence length N "
        "(a task with rows at more than one length is refused without this)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_len_overrides = _parse_seq_len_overrides(args.seq_len)
    table = _load(Path(args.runs_dir))
    logger.info("Loaded %d runs", len(table))
    for name in args.reports:
        if name not in REPORTS:
            raise SystemExit(
                f"unknown report {name!r}; available: {', '.join(REPORTS)}"
            )
        REPORTS[name](table, out_dir, seq_len_overrides)
        logger.info("Wrote %s", name)


if __name__ == "__main__":
    main()
