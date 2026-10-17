"""
Pilot analysis: Delta_int, bootstrap CIs and the pre-registered readings.

Reads the run JSONs written by 01_train.py (``runs/**/*.json``) and
writes CSVs plus ``summary.md`` under ``analysis/``:

- the primary endpoint (protocol §6): Delta_int^LL and Delta_int^WS on
  overwrite, class ``distant``, K = 2 d_k, with bootstrap CIs and the
  pre-registered decision-rule reading (§9);
- the same for every other (task, slice, class) cell that has the four
  core cells;
- predictions 1-4 (tiers 2-4) with the same bootstrap;
- the anchor table next to Log-Linear Table 2 (tier 6);
- accuracy and old-value leakage by the write-distance bin, per layout
  and gate mode (core tier);
- lambda and erase-strength statistics per slot;
- the list of missing runs per tier.

Usage:
    python 02_analyze.py                      # runs/ -> analysis/
    python 02_analyze.py --runs-dir runs --out-dir analysis
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from research_lib.utils.stats import bootstrap_ci, did_per_seed

logger = logging.getLogger(__name__)

# Log-Linear Table 2 (Gated DeltaNet / w Log-Linear), for the anchor table.
LOG_LINEAR_TABLE_2 = {
    16: {"single": 38.4, "ll": 40.0},
    32: {"single": 79.0, "ll": 84.4},
    64: {"single": 99.0, "ll": 99.0},  # ">= 99"
}
LOG_LINEAR_STD_2 = {
    16: {"single": 1.0, "ll": 1.4},
    32: {"single": 2.1, "ll": 1.2},
    64: {"single": None, "ll": None},
}

# Expected run counts per tier (protocol §7), for the missing-run report.
EXPECTED_RUNS = {
    "calib_dmodel": 4,
    "calib_harness": 4,
    "calib_lr": 72,
    "calib_lr_anchors": 36,
    "tier1_core": 120,
    "tier2_gates": 60,
    "tier3_frozen": 20,
    "tier4_lambda": 40,
    "tier5_matched_bytes": 10,
    "tier6_anchors": 30,
    "tier_longT": 60,
    "R1": 30,
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_runs(runs_dir: Path) -> pd.DataFrame:
    """Load every run JSON under runs_dir into one row-per-run DataFrame.

    Flattens the run config into scalar columns (task, layout, gate_mode,
    lambda_mode, d_model, head_dim, seed, tier, phase, decay_scheme) and
    the test metrics into ``test/<key>`` columns.
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
            "seed": summary.get("seed"),
            "tier": (cfg.get("experiment") or {}).get("tier"),
            "phase": (cfg.get("experiment") or {}).get("phase"),
            "steps_completed": (summary.get("timing") or {}).get("steps_completed"),
            "best_value": (summary.get("stopping") or {}).get("best_value"),
            "best_step": (summary.get("stopping") or {}).get("best_step"),
            "status": summary.get("status"),
            # The weight-decay scheme (grouped | uniform). Prefer the
            # top-level field recorded by the run JSON; fall back to the
            # config (None = recorded before the scheme switch = grouped).
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


TIER_CORE = "tier1_core"
TIER_R1 = "R1"
TIER_LONG_T = "tier_longT"


def tier_rows(table: pd.DataFrame, tier: str) -> pd.DataFrame:
    """The rows of one protocol tier.

    Every statistic selects its rows by tier, never by excluding tiers:
    several tiers repeat core configurations (the matched-state tier, the
    λ controls, R1 and the long-sequence tier), so to the per-seed DiD
    pairing those rows read as duplicated cells and would drop seeds.
    Calibration tiers (and smoke runs) never enter any statistic.
    """
    if "tier" not in table.columns:
        return table.iloc[0:0]
    return table[table["tier"] == tier]


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #


def decision_rule(low: float, high: float, primary: bool) -> str:
    """The pre-registered reading of a Delta_int CI (protocol §9)."""
    if low > 0:
        return "CI entirely above 0: supports the hypothesis at pilot scale"
    if high < 0:
        return (
            "CI entirely below 0: transport makes selective erase redundant; "
            "informative, report it"
        )
    if primary:
        return (
            "CI includes 0 on the distant, K=2d_k cell: evidence AGAINST the "
            "mechanism (this cell stresses (a)+(b)+(c))"
        )
    return "CI includes 0: uninformative about the hypothesis on this cell"


def paired_diff(
    table: pd.DataFrame, metric: str, layout: str, gate_mode: str
) -> pd.Series:
    """Per-seed paired difference ``(layout-gate) - (single-gate)``."""
    rows = table[(table["gate_mode"] == gate_mode)]
    out = {}
    for seed in sorted(rows["seed"].dropna().unique()):
        cell_x = rows[(rows["layout"] == layout) & (rows["seed"] == seed)][metric]
        cell_s = rows[(rows["layout"] == "single") & (rows["seed"] == seed)][metric]
        if len(cell_x) == 1 and len(cell_s) == 1:
            out[int(seed)] = float(cell_x.iloc[0]) - float(cell_s.iloc[0])
    return pd.Series(out, dtype=float)


def did_table_for(
    table: pd.DataFrame, metric: str, layouts=("ws", "ll", "fs")
) -> pd.DataFrame:
    """Delta_int per layout for one metric, with bootstrap CIs."""
    records = []
    for layout in layouts:
        sub = table[table[metric].notna()]
        try:
            deltas = did_per_seed(sub, metric, layout)
        except ValueError:
            continue
        if deltas.empty:
            continue
        mean, low, high = bootstrap_ci(deltas.values)
        records.append(
            {
                "layout": layout,
                "metric": metric,
                "n_seeds": int(deltas.size),
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "reading": decision_rule(low, high, primary=False),
            }
        )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def _primary_records(
    overwrite: pd.DataFrame, n_resamples: int, alpha: float
) -> list[dict]:
    """Δ_int records at the primary endpoint for one tier's overwrite rows."""
    records = []
    for (d_model, num_heads, head_dim), group in overwrite.groupby(
        ["d_model", "num_heads", "head_dim"], dropna=False
    ):
        dk = head_dim if not pd.isna(head_dim) else d_model / num_heads
        key = f"test/acc/slice={int(2 * dk)}/class=distant"
        if key not in group.columns or group[key].notna().sum() == 0:
            logger.warning(
                "primary endpoint metric %s not present (d_model=%s); skipping",
                key,
                d_model,
            )
            continue
        for layout in ("ll", "ws"):
            sub = group[group[key].notna()]
            deltas = did_per_seed(sub, key, layout)
            if deltas.empty:
                continue
            mean, low, high = bootstrap_ci(
                deltas.values, n_resamples=n_resamples, alpha=alpha
            )
            records.append(
                {
                    "layout": layout,
                    "metric": key,
                    "d_model": d_model,
                    "n_seeds": int(deltas.size),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "reading": decision_rule(low, high, primary=True),
                }
            )
    return records


def analyze_primary(
    table: pd.DataFrame, n_resamples: int, alpha: float
) -> pd.DataFrame:
    """Δ_int at the primary endpoint, core tier rows only."""
    overwrite = tier_rows(table, TIER_CORE)
    overwrite = overwrite[overwrite["task"] == "overwrite"]
    return pd.DataFrame(_primary_records(overwrite, n_resamples, alpha))


def analyze_r1(table: pd.DataFrame, n_resamples: int, alpha: float) -> pd.DataFrame:
    """Δ_int at the primary endpoint under uniform weight decay (tier R1).

    Reported next to the main result: how much the weight-decay exemption
    of the vector group (λ, ``A_log``, ``dt_bias``, …) matters to the
    interaction. R1 rows never enter the pre-registered statistic.
    """
    overwrite = tier_rows(table, TIER_R1)
    overwrite = overwrite[overwrite["task"] == "overwrite"]
    return pd.DataFrame(_primary_records(overwrite, n_resamples, alpha))


def analyze_long_t(table: pd.DataFrame, n_resamples: int, alpha: float) -> pd.DataFrame:
    """Δ_int on the long-sequence tier, per accuracy cell.

    The long-sequence tier repeats the core grid on the long, sparse MQAR
    variant; its own table, never mixed into the T=256 statistic.
    """
    records = []
    long_t = tier_rows(table, TIER_LONG_T)
    if long_t.empty:
        return pd.DataFrame()
    acc_columns = [
        c
        for c in long_t.columns
        if c == "test/acc" or (c.startswith("test/acc/slice=") and "/class=" not in c)
    ]
    for column in acc_columns:
        sub = long_t[long_t[column].notna()]
        if sub.empty:
            continue
        for layout in ("ws", "ll"):
            deltas = did_per_seed(sub, column, layout)
            if deltas.empty:
                continue
            mean, low, high = bootstrap_ci(
                deltas.values, n_resamples=n_resamples, alpha=alpha
            )
            records.append(
                {
                    "layout": layout,
                    "metric": column,
                    "n_seeds": int(deltas.size),
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "reading": decision_rule(low, high, primary=False),
                }
            )
    return pd.DataFrame(records)


def analyze_all_cells(table: pd.DataFrame) -> pd.DataFrame:
    """Delta_int for every measured cell, core tier rows only."""
    frames = []
    acc_columns = [c for c in table.columns if c.startswith("test/acc")]
    core = tier_rows(table, TIER_CORE)
    core = core[core["gate_mode"].isin(["full", "kda"])]
    for task in core["task"].dropna().unique():
        sub = core[core["task"] == task]
        for column in acc_columns:
            if sub[column].notna().sum() == 0:
                continue
            result = did_table_for(sub, column)
            if not result.empty:
                result.insert(0, "task", task)
                frames.append(result)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def analyze_predictions(
    table: pd.DataFrame, n_resamples: int, alpha: float
) -> pd.DataFrame:
    """Predictions 1-4, each computed on its own tier, joined to the core
    tier rows wherever it needs the core cells (P1: gate-decomposition
    tier + core; P2/P3: frozen-control tier + core ``ws``; P4: lambda tier
    + core ``single`` rows).
    """
    records = []

    def _add(name, layout, gate, values, extra=None):
        if len(values) == 0:
            return
        mean, low, high = bootstrap_ci(
            np.asarray(values, dtype=float), n_resamples=n_resamples, alpha=alpha
        )
        record = {
            "prediction": name,
            "layout": layout,
            "gate_mode": gate,
            "n_seeds": len(values),
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
        }
        if extra:
            record.update(extra)
        records.append(record)

    # ---- P1: b_only recovers most of Delta_int, w_only little.
    # Effect: (layout, b/w_only) - (single, b/w_only) on the
    # gate-decomposition tier; reference: (layout, full) - (single, full)
    # on the core tier; joined by seed.
    core = tier_rows(table, TIER_CORE)
    core_ow = core[core["task"] == "overwrite"]
    t2 = tier_rows(table, "tier2_gates")
    t2_ow = t2[t2["task"] == "overwrite"]
    distant = [
        c
        for c in table.columns
        if c.startswith("test/acc/slice=") and c.endswith("/class=distant")
    ]
    for column in distant:
        for layout in ("ws", "ll"):
            reference = paired_diff(core_ow, column, layout, "full")
            for gate in ("b_only", "w_only"):
                effect = paired_diff(t2_ow, column, layout, gate)
                if effect.empty or reference.empty:
                    continue
                common = effect.index.intersection(reference.index)
                if len(common) == 0:
                    continue
                recovered = (
                    (effect[common] / reference[common])
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                _add(
                    f"P1 recovered fraction ({column})",
                    layout,
                    gate,
                    recovered.values,
                )

    # ---- P2: transported >> frozen on cross-block overwrite, with FS's
    # errors concentrated on the stale value. ws rows from the core tier,
    # fs rows from the frozen-control tier, paired by seed.
    leak = [
        c
        for c in table.columns
        if c.startswith("test/leak/slice=") and c.endswith("/class=distant")
    ]
    t3 = tier_rows(table, "tier3_frozen")
    t3_ow = t3[t3["task"] == "overwrite"]
    core_ow = core[core["task"] == "overwrite"]

    def _ws_fs_diff(ws_rows: pd.DataFrame, fs_rows: pd.DataFrame, column: str) -> list:
        values = []
        seeds = sorted(set(ws_rows["seed"].dropna()) & set(fs_rows["seed"].dropna()))
        for seed in seeds:
            cell_ws = ws_rows[(ws_rows["seed"] == seed)][column]
            cell_fs = fs_rows[(fs_rows["seed"] == seed)][column]
            if len(cell_ws) == 1 and len(cell_fs) == 1:
                values.append(float(cell_ws.iloc[0]) - float(cell_fs.iloc[0]))
        return values

    for gate in ("full", "kda"):
        ws_rows = core_ow[(core_ow["gate_mode"] == gate) & (core_ow["layout"] == "ws")]
        fs_rows = t3_ow[(t3_ow["gate_mode"] == gate) & (t3_ow["layout"] == "fs")]
        for column in distant + leak:
            _add(
                f"P2 ws - fs ({column})",
                "ws-fs",
                gate,
                _ws_fs_diff(ws_rows, fs_rows, column),
            )

    # ---- P3: frozen >= transported on never-overwritten keys, with a
    # smaller gap under GDN-2 gates than under KDA gates. Core-tier ws
    # rows against frozen-control fs rows on single-class accuracy.
    single_cols = [
        c
        for c in table.columns
        if c.startswith("test/acc/slice=") and c.endswith("/class=single")
    ]
    for gate in ("full", "kda"):
        ws_rows = core_ow[(core_ow["gate_mode"] == gate) & (core_ow["layout"] == "ws")]
        fs_rows = t3_ow[(t3_ow["gate_mode"] == gate) & (t3_ow["layout"] == "fs")]
        for column in single_cols:
            _add(
                f"P3 ws - fs ({column})",
                "ws-fs",
                gate,
                _ws_fs_diff(ws_rows, fs_rows, column),
            )

    # ---- P4: lambda-static gives no gain over single-state GDN-2 once the
    # number of keys exceeds d_k. Lambda-control tier (static) rows against
    # the core tier's single-state `full` rows, paired by seed.
    t4 = tier_rows(table, "tier4_lambda")
    t4_ow = t4[t4["task"] == "overwrite"]
    base = core_ow[(core_ow["layout"] == "single") & (core_ow["gate_mode"] == "full")]
    for layout in ("ws", "ll"):
        static = t4_ow[(t4_ow["lambda_mode"] == "static") & (t4_ow["layout"] == layout)]
        for column in distant:
            values = []
            for seed in sorted(
                set(static["seed"].dropna()) & set(base["seed"].dropna())
            ):
                cell_x = static[
                    (static["seed"] == seed) & (static["gate_mode"] == "full")
                ][column]
                cell_s = base[(base["seed"] == seed)][column]
                if len(cell_x) == 1 and len(cell_s) == 1:
                    values.append(float(cell_x.iloc[0]) - float(cell_s.iloc[0]))
            _add(f"P4 static - single-full ({column})", layout, "full", values)
    return pd.DataFrame(records)


def analyze_anchors(table: pd.DataFrame) -> pd.DataFrame:
    """Tier 6 anchor table next to Log-Linear Table 2."""
    anchors = table[table["tier"] == "tier6_anchors"]
    if anchors.empty:
        return pd.DataFrame()
    records = []
    for (d_model, layout), group in anchors.groupby(["d_model", "layout"]):
        acc = group["test/acc"].dropna()
        record = {
            "d_model": d_model,
            "layout": layout,
            "n_seeds": int(acc.size),
            "acc_mean": float(acc.mean()) if acc.size else None,
            "acc_std": float(acc.std()) if acc.size > 1 else None,
            "log_linear_acc": LOG_LINEAR_TABLE_2.get(int(d_model), {}).get(layout),
            "log_linear_std": LOG_LINEAR_STD_2.get(int(d_model), {}).get(layout),
        }
        records.append(record)
    return pd.DataFrame(records).sort_values(["d_model", "layout"])


def analyze_mixer_stats(table: pd.DataFrame) -> pd.DataFrame:
    """Lambda / erase statistics per slot, averaged per (layout, gate_mode)."""
    mixer_cols = [c for c in table.columns if c.startswith("mixer/")]
    if not mixer_cols:
        return pd.DataFrame()
    rows = table[table["layout"] != "single"]
    records = []
    for (layout, gate_mode), group in rows.groupby(["layout", "gate_mode"]):
        record = {"layout": layout, "gate_mode": gate_mode, "n_runs": len(group)}
        for column in sorted(mixer_cols):
            values = group[column].dropna()
            if values.empty:
                continue
            record[column] = float(values.mean())
        records.append(record)
    return pd.DataFrame(records)


def analyze_missing_runs(table: pd.DataFrame) -> pd.DataFrame:
    """Missing runs per tier against the protocol's expected counts.

    Only complete runs (``status == "test_done"``) count as observed; a
    run whose status is anything else is listed by
    :func:`analyze_incomplete` instead.
    """
    records = []
    if "status" in table.columns:
        table = table[table["status"].fillna("unknown") == "test_done"]
    observed = table.groupby("tier")["run_id"].count()
    for tier, expected in EXPECTED_RUNS.items():
        got = int(observed.get(tier, 0))
        records.append(
            {
                "tier": tier,
                "expected": expected,
                "observed": got,
                "missing": expected - got,
            }
        )
    for tier, got in observed.items():
        if tier not in EXPECTED_RUNS:
            records.append(
                {"tier": tier, "expected": None, "observed": got, "missing": None}
            )
    return pd.DataFrame(records)


def analyze_incomplete(table: pd.DataFrame) -> pd.DataFrame:
    """Runs whose status is not 'test_done', with their status."""
    if "status" not in table.columns:
        return pd.DataFrame()
    incomplete = table[table["status"].fillna("unknown") != "test_done"]
    if incomplete.empty:
        return pd.DataFrame()
    return incomplete[
        ["run_id", "task", "tier", "layout", "gate_mode", "seed", "status"]
    ].sort_values(["tier", "run_id"])


def analyze_distance(table: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and leakage by write-distance bin, per layout and gate.

    Core tier rows only. The run JSONs carry the per-bin metrics from
    ``RecallMetrics`` (``test/acc/dist={lo}-{hi}`` and
    ``test/leak/dist={lo}-{hi}``); this table averages them over seeds per
    (layout, gate mode), so the distance response of each row of the grid
    is readable next to the class-binned numbers.
    """
    core = tier_rows(table, TIER_CORE)
    core = core[core["task"] == "overwrite"]
    if core.empty:
        return pd.DataFrame()
    dist_columns = sorted(
        c
        for c in core.columns
        if c.startswith("test/acc/dist=") or c.startswith("test/leak/dist=")
    )
    records = []
    for (layout, gate_mode), group in core.groupby(["layout", "gate_mode"]):
        for column in dist_columns:
            values = group[column].dropna()
            if values.empty:
                continue
            records.append(
                {
                    "layout": layout,
                    "gate_mode": gate_mode,
                    "metric": column,
                    "n_seeds": int(values.size),
                    "mean": float(values.mean()),
                }
            )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Summary rendering
# --------------------------------------------------------------------------- #


def _fmt(value, digits=4):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Render a DataFrame as a GitHub markdown table (no extra deps)."""
    if frame is None or frame.empty:
        return "_none_"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], digits) for c in columns) + " |")
    return "\n".join(lines)


def render_summary(
    primary: pd.DataFrame,
    r1: pd.DataFrame,
    long_t: pd.DataFrame,
    cells: pd.DataFrame,
    predictions: pd.DataFrame,
    distance: pd.DataFrame,
    anchors: pd.DataFrame,
    missing: pd.DataFrame,
    incomplete: pd.DataFrame,
) -> str:
    lines = ["# Synthetic pilot analysis", ""]
    lines.append("## Primary endpoint (core tier, overwrite, distant, K = 2 d_k)")
    lines.append("")
    if primary.empty:
        lines.append("_No core-tier runs with the primary endpoint present._")
    else:
        lines.append(_markdown_table(primary))
        lines.append("")
    lines.append("### Under uniform weight decay (tier R1)")
    lines.append("")
    if r1.empty:
        lines.append("_No R1 runs present._")
    else:
        lines.append(_markdown_table(r1))
    lines.append("")
    lines.append("## Delta_int on every measured cell (core tier)")
    lines.append("")
    lines.append(_markdown_table(cells))
    lines.append("")
    lines.append("## Predictions 1-4 (each on its own tier)")
    lines.append("")
    lines.append(_markdown_table(predictions))
    lines.append("")
    lines.append("## Write distance (core tier)")
    lines.append("")
    lines.append(_markdown_table(distance))
    lines.append("")
    lines.append("## Long-sequence tier (Delta_int)")
    lines.append("")
    lines.append(_markdown_table(long_t))
    lines.append("")
    lines.append("## Anchors next to Log-Linear Table 2")
    lines.append("")
    lines.append(_markdown_table(anchors))
    lines.append("")
    lines.append("## Missing runs per tier")
    lines.append("")
    lines.append(_markdown_table(missing))
    lines.append("")
    lines.append("## Incomplete runs (status is not test_done)")
    lines.append("")
    lines.append(_markdown_table(incomplete))
    lines.append("")
    lines.append("## Notes and limitations")
    lines.append("")
    lines.append(
        "- Missing keys in a metrics dict mean 'not measured' (empty cells "
        "are omitted by RecallMetrics), not zero."
    )
    lines.append(
        "- Every statistic above selects its rows by tier: several tiers "
        "repeat core configurations (matched bytes, the lambda controls, "
        "R1, the long-sequence tier), so filtering by configuration alone "
        "would create duplicated cells. Calibration tiers and smoke runs "
        "never enter any statistic."
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir", default="runs", help="run JSON root (default: runs)"
    )
    parser.add_argument("--out-dir", default="analysis", help="output directory")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_runs(runs_dir)
    logger.info("Loaded %d runs from %s", len(table), runs_dir)

    primary = analyze_primary(table, args.n_resamples, args.alpha)
    r1 = analyze_r1(table, args.n_resamples, args.alpha)
    long_t = analyze_long_t(table, args.n_resamples, args.alpha)
    cells = analyze_all_cells(table)
    predictions = analyze_predictions(table, args.n_resamples, args.alpha)
    distance = analyze_distance(table)
    anchors = analyze_anchors(table)
    mixer = analyze_mixer_stats(table)
    missing = analyze_missing_runs(table)
    incomplete = analyze_incomplete(table)

    primary.to_csv(out_dir / "primary_endpoint.csv", index=False)
    r1.to_csv(out_dir / "r1_primary_endpoint.csv", index=False)
    long_t.to_csv(out_dir / "long_t_did.csv", index=False)
    cells.to_csv(out_dir / "did_all_cells.csv", index=False)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    distance.to_csv(out_dir / "write_distance.csv", index=False)
    anchors.to_csv(out_dir / "anchors.csv", index=False)
    mixer.to_csv(out_dir / "mixer_stats.csv", index=False)
    missing.to_csv(out_dir / "missing_runs.csv", index=False)
    incomplete.to_csv(out_dir / "incomplete_runs.csv", index=False)
    summary = render_summary(
        primary, r1, long_t, cells, predictions, distance, anchors, missing, incomplete
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    logger.info("Wrote analysis to %s", out_dir)


if __name__ == "__main__":
    main()
