"""The calibration report: the pre-registered readings of Phase 1.

Each calibration sweep ends with a choice that is written by hand into
``conf/frozen_protocol.yaml`` and ``conf/config.yaml`` and committed.
This report computes each reading from the calibration run records; it
never edits configuration files.

Common rules for every reading:

- only completed runs count (``status == "test_done"``), as everywhere;
- every reading uses the **best validation accuracy during the run**
  (``stopping.best_value``), averaged over the selection seeds
  (100, 101).

The readings:

- ``d_model``: the smallest candidate whose mean accuracy lies in the
  pre-registered window; the fallback when none does.
- harness: the mean accuracy per layer pattern, next to the published
  Gated DeltaNet numbers at the chosen width. The choice itself is a
  judgment (keep the default unless it lands far outside the published
  range), so the report prints the numbers and marks no verdict.
- learning rates: per width and core configuration, the LR with the
  best mean accuracy; ties go to the fewer mean steps to the best
  value, then to the smaller LR. The anchors get their own
  per-(d_model, layout) table.
- a ``frozen_protocol.yaml`` snippet filled from the readings, with
  ``null`` for the fields whose step has not run.
- missing runs per calibration tier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import (
    CALIBRATION_TIERS,
    DMODEL_ACC_RANGE,
    DMODEL_FALLBACK,
    LOG_LINEAR_STD_2,
    LOG_LINEAR_TABLE_2,
    SELECTION_SEEDS,
    TIER_CALIB_DMODEL,
    TIER_CALIB_HARNESS,
    TIER_CALIB_LR,
    TIER_CALIB_LR_ANCHORS,
)
from pilot_analysis.records import tier_rows
from pilot_analysis.render import markdown_table

from .completeness import missing_runs

TITLE = "Calibration readings"

INTRO = """Every reading uses the best validation accuracy during the run,
averaged over the selection seeds {seeds}; only completed runs count.
The report computes the readings; the choices are written into
``conf/frozen_protocol.yaml`` and ``conf/config.yaml`` by hand and
committed.""".format(
    seeds=", ".join(str(s) for s in SELECTION_SEEDS)
)

# Tie tolerance for "equal means": values come from JSON, so equal means
# are exactly equal; the epsilon only guards summation-order noise.
_TIE_ATOL = 1e-12


def _selection_rows(table: pd.DataFrame, tier: str) -> pd.DataFrame:
    """The tier's completed rows, restricted to the selection seeds."""
    rows = tier_rows(table, tier)
    if "seed" in rows.columns:
        rows = rows[rows["seed"].isin(SELECTION_SEEDS)]
    return rows


def _per_seed_mean(rows: pd.DataFrame, by: str) -> pd.DataFrame:
    """One row per ``by`` value: per-seed best values, then the mean.

    The empty case (no completed calibration runs) returns the same
    columns with no rows.
    """
    columns = [by, "n_seeds", *(f"seed={seed}" for seed in SELECTION_SEEDS), "mean"]
    if rows.empty or by not in rows.columns:
        return pd.DataFrame(columns=columns)
    records = []
    for value, group in rows.groupby(by, dropna=False):
        record = {by: value, "n_seeds": int(group["best_value"].notna().sum())}
        for seed in SELECTION_SEEDS:
            cell = group[group["seed"] == seed]["best_value"]
            record[f"seed={seed}"] = float(cell.iloc[0]) if len(cell) == 1 else None
        values = group["best_value"].dropna()
        record["mean"] = float(values.mean()) if len(values) else None
        records.append(record)
    return pd.DataFrame(records)


def select_d_model(means: pd.Series) -> tuple[int | None, str | None]:
    """The width reading: the smallest d whose mean lies in the window.

    Args:
        means: Mean accuracy per d_model (NaN means "not measured").

    Returns:
        tuple: ``(d_model, reading)`` — ``(None, None)`` when no d_model
        was measured at all; ``(fallback, ...)`` with the fallback
        reading when no measured mean lies in the window; else the
        smallest d_model in the window.
    """
    means = means.dropna()
    if means.empty:
        return None, None
    lo, hi = DMODEL_ACC_RANGE
    in_range = means[(means >= lo) & (means <= hi)]
    if in_range.empty:
        return (
            DMODEL_FALLBACK,
            f"no measured d_model has its mean accuracy in [{lo}, {hi}]; "
            f"fallback d_model={DMODEL_FALLBACK}",
        )
    d = int(in_range.index.min())
    return d, f"d_model={d}: the smallest d whose mean accuracy lies in [{lo}, {hi}]"


def select_lr(per_lr: pd.DataFrame) -> float | None:
    """The LR reading for one configuration: best mean accuracy, then
    fewer mean steps to the best value, then the smaller LR.

    Args:
        per_lr: One row per learning rate (indexed by ``lr``), with the
            columns ``mean_acc`` (mean best validation accuracy) and
            ``mean_steps`` (mean steps to the best value; NaN = never
            measured).

    Returns:
        The selected learning rate, or ``None`` when nothing was
        measured.
    """
    per_lr = per_lr[per_lr["mean_acc"].notna()]
    if per_lr.empty:
        return None
    best_acc = per_lr["mean_acc"].max()
    tied = per_lr[np.isclose(per_lr["mean_acc"], best_acc, rtol=0.0, atol=_TIE_ATOL)]
    best_steps = tied["mean_steps"].min()  # skips NaN
    if not np.isnan(best_steps):
        tied = tied[
            np.isclose(tied["mean_steps"], best_steps, rtol=0.0, atol=_TIE_ATOL)
        ]
    return float(tied.index.min())


def _lr_reading(rows: pd.DataFrame, cell_cols: list[str]) -> pd.DataFrame:
    """Per learning rate: mean best accuracy and steps, with the selection.

    One row per (configuration, lr): the cell's identifying columns,
    the LR, the number of selection seeds, the mean best validation
    accuracy, the mean steps to the best value, and whether this LR is
    the selected one for the configuration.
    """
    records = []
    columns = [
        *cell_cols,
        "lr",
        "n_seeds",
        "mean_best_acc",
        "mean_best_step",
        "selected",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    # Rows without an LR carry no grid point: they are never part of a
    # reading (a run whose LR was inherited or overridden elsewhere
    # must not appear in a selection table).
    if "lr" not in rows.columns:
        return pd.DataFrame(columns=columns)
    rows = rows[rows["lr"].notna()]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    for cell, group in rows.groupby(cell_cols, dropna=False):
        cell_values = dict(zip(cell_cols, [cell] if len(cell_cols) == 1 else cell))
        per_lr = (
            group.groupby("lr")
            .agg(
                mean_acc=("best_value", "mean"),
                mean_steps=("best_step", "mean"),
                n_seeds=("best_value", "count"),
            )
            .sort_index()
        )
        selected = select_lr(per_lr)
        for lr, row in per_lr.iterrows():
            records.append(
                {
                    **cell_values,
                    "lr": lr,
                    "n_seeds": int(row["n_seeds"]),
                    "mean_best_acc": float(row["mean_acc"]),
                    "mean_best_step": (
                        None if pd.isna(row["mean_steps"]) else float(row["mean_steps"])
                    ),
                    "selected": bool(lr == selected),
                }
            )
    return pd.DataFrame(records)


def _fmt_lr(value) -> str:
    """A learning rate as YAML: null, or %g (0.001, 0.003, 0.01)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "null"
    return f"{float(value):g}"


def frozen_snippet(
    d_model: int | None,
    lr_width: int | None,
    lr_cells: pd.DataFrame,
    anchor_cells: pd.DataFrame,
) -> str:
    """A ``conf/frozen_protocol.yaml`` snippet filled from the readings.

    Fields whose calibration step has not run stay ``null``. The harness
    block is left ``null`` on purpose: its reading is a judgment (keep
    the default unless far outside the published range), made when the
    snippet is pasted. The ``lr_table`` takes only the rows at ``lr_width``
    (the width the snippet uses), so learning rates measured at another
    width never fill it.

    Args:
        d_model: The width reading (``None`` = the step has not run).
        lr_width: The width whose ``lr_cells`` rows fill the table —
            the width reading, else the single width measured (``None``
            = no width to use: the entries stay ``null``).
        lr_cells: The ``calib_lr`` reading table (one row per
            (d_model, task, layout, gate, lr), with a ``selected``
            column).
        anchor_cells: The ``calib_lr_anchors`` reading table (one row
            per (d_model, layout, lr), with a ``selected`` column).

    Returns:
        str: The YAML text, ready to paste.
    """
    selected_lr: dict[tuple, float] = {}
    if not lr_cells.empty and lr_width is not None:
        at_width = lr_cells[lr_cells["d_model"] == lr_width]
        picked = at_width[at_width["selected"]]
        for _, row in picked.iterrows():
            selected_lr[(row["task"], row["layout"], row["gate_mode"])] = float(
                row["lr"]
            )
    selected_anchor: dict[tuple, float] = {}
    if not anchor_cells.empty:
        picked = anchor_cells[anchor_cells["selected"]]
        for _, row in picked.iterrows():
            selected_anchor[(int(row["d_model"]), row["layout"])] = float(row["lr"])

    def _entry(task, layout, gate):
        return _fmt_lr(selected_lr.get((task, layout, gate)))

    lines = []
    lines.append(f"d_model: {d_model if d_model is not None else 'null'}")
    lines.append("")
    lines.append("# The harness choice is a judgment call (keep the default")
    lines.append("# unless the harness table lands far outside the published")
    lines.append("# range); fill it in when pasting.")
    lines.append("harness:")
    lines.append("  layer_pattern: null")
    lines.append("  state_mixer: null")
    lines.append("")
    lines.append("lr_table:")
    for task in ("mqar", "overwrite"):
        lines.append(f"  {task}:")
        for layout in ("single", "ws", "fs", "ll"):
            entries = ", ".join(
                f"{gate}: {_entry(task, layout, gate)}"
                for gate in ("full", "b_only", "w_only", "kda", "gdn")
            )
            lines.append(f"    {layout}: {{{entries}}}")
    lines.append("")
    lines.append("anchor_lr:")
    for d in (16, 32, 64):
        entries = ", ".join(
            f"{layout}: {_fmt_lr(selected_anchor.get((d, layout)))}"
            for layout in ("single", "ll")
        )
        lines.append(f"  {d}: {{{entries}}}")
    return "\n".join(lines) + "\n"


def calibration_missing(table: pd.DataFrame) -> pd.DataFrame:
    """Missing runs per calibration tier (expected counts)."""
    frame = missing_runs(table)
    if frame.empty:
        return frame
    return frame[frame["tier"].isin(CALIBRATION_TIERS)]


def run(table: pd.DataFrame) -> dict:
    """Every calibration reading from the run table.

    Returns:
        dict: ``dmodel`` (table), ``dmodel_choice`` (d, reading),
        ``harness`` (table), ``lr`` (table), ``lr_anchors`` (table),
        ``missing`` (table), ``snippet`` (YAML text),
        ``lr_width_note`` (a one-line warning when the learning rates
        span several widths and no width reading exists to pick one).
    """
    dmodel_rows = _selection_rows(table, TIER_CALIB_DMODEL)
    dmodel = _per_seed_mean(dmodel_rows, "d_model").sort_values("d_model")
    means = pd.Series(dtype=float)
    if not dmodel.empty:
        means = dmodel.set_index("d_model")["mean"]
    d_choice, d_reading = select_d_model(means)

    harness_rows = _selection_rows(table, TIER_CALIB_HARNESS)
    harness = _per_seed_mean(harness_rows, "layer_pattern")
    # The published reference is read at the chosen width: the width
    # reading when it exists, else the width the harness rows ran at
    # (when they all share one).
    chosen_d = d_choice
    if chosen_d is None and not harness_rows.empty:
        widths = harness_rows["d_model"].dropna().unique()
        if len(widths) == 1:
            chosen_d = int(widths[0])
    if not harness.empty:
        harness["d_model"] = chosen_d
        harness["log_linear_acc"] = (
            LOG_LINEAR_TABLE_2.get(int(chosen_d), {}).get("single")
            if chosen_d is not None
            else None
        )
        harness["log_linear_std"] = (
            LOG_LINEAR_STD_2.get(int(chosen_d), {}).get("single")
            if chosen_d is not None
            else None
        )

    lr_rows = _selection_rows(table, TIER_CALIB_LR)
    lr = _lr_reading(lr_rows, ["d_model", "task", "layout", "gate_mode"])
    anchor_rows = _selection_rows(table, TIER_CALIB_LR_ANCHORS)
    lr_anchors = _lr_reading(anchor_rows, ["d_model", "layout"])
    missing = calibration_missing(table)
    # The snippet's lr_table uses one width: the width reading when it
    # exists; otherwise the single width the records ran at. Several
    # widths with no reading leave the table null — the widths cannot
    # be averaged, and the report says which step is missing.
    lr_widths = sorted(int(w) for w in lr["d_model"].dropna().unique())
    snippet_width = d_choice
    lr_width_note = None
    if snippet_width is None and len(lr_widths) == 1:
        snippet_width = lr_widths[0]
    elif snippet_width is None and len(lr_widths) > 1:
        lr_width_note = (
            f"calib_lr records found at widths {lr_widths}; the width "
            "step (calib_dmodel) must be recorded before the "
            "learning-rate table can be filled"
        )
    snippet = frozen_snippet(d_choice, snippet_width, lr, lr_anchors)
    return {
        "dmodel": dmodel,
        "dmodel_choice": (d_choice, d_reading),
        "harness": harness,
        "lr": lr,
        "lr_anchors": lr_anchors,
        "missing": missing,
        "snippet": snippet,
        "lr_width_note": lr_width_note,
    }


def _verdict_lines(readings: dict) -> list[str]:
    lines = []
    d_choice, d_reading = readings["dmodel_choice"]
    if d_reading is not None:
        lines.append(f"Reading: {d_reading}.")
    else:
        lines.append("Reading: no completed calibration runs for the width step yet.")
    return lines


def render(readings: dict) -> str:
    """The calibration section: tables, readings and the paste snippet."""
    lo, hi = DMODEL_ACC_RANGE
    lines = [f"## {TITLE}", ""]
    lines.append(INTRO.replace("\n", " "))
    lines.append("")
    lines.append(
        f"### d_model (the smallest d whose mean accuracy lies in "
        f"[{lo}, {hi}]; fallback {DMODEL_FALLBACK})"
    )
    lines.append("")
    lines.append(markdown_table(readings["dmodel"]))
    lines.append("")
    for line in _verdict_lines(readings):
        lines.append(line)
    lines.append("")
    lines.append(
        "### Harness (single-state gdn on MQAR; keep the default unless "
        "far outside the published range)"
    )
    lines.append("")
    lines.append(markdown_table(readings["harness"]))
    lines.append("")
    lines.append(
        "The harness choice is a judgment: keep the default "
        "(``layer_pattern=zoology``, ``state_mixer=none``) unless its mean "
        "lands far outside the published range; the report marks no verdict."
    )
    lines.append("")
    lines.append(
        "### Learning rates (per width and core configuration; best mean "
        "accuracy; ties: fewer steps to the best value, then the smaller "
        "LR)"
    )
    lines.append("")
    lines.append(markdown_table(readings["lr"]))
    lines.append("")
    if readings.get("lr_width_note"):
        lines.append(f"{readings['lr_width_note']}.")
        lines.append("")
    lines.append("### Anchor learning rates (per d_model and layout)")
    lines.append("")
    lines.append(markdown_table(readings["lr_anchors"]))
    lines.append("")
    lines.append("### Missing runs (calibration tiers)")
    lines.append("")
    lines.append(markdown_table(readings["missing"]))
    lines.append("")
    lines.append("### frozen_protocol.yaml snippet")
    lines.append("")
    lines.append("```yaml")
    lines.append(readings["snippet"].rstrip("\n"))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def report(table: pd.DataFrame, options: ReportOptions) -> Report:
    readings = run(table)
    return Report(
        name="calibration",
        title=TITLE,
        tables={
            "calibration_dmodel": readings["dmodel"],
            "calibration_harness": readings["harness"],
            "calibration_lr": readings["lr"],
            "calibration_lr_anchors": readings["lr_anchors"],
        },
        markdown=render(readings),
    )
