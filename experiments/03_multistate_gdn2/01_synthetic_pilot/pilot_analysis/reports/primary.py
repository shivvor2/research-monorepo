"""Primary endpoint report: Delta_int on overwrite, distant, K = 2 d_k."""

from __future__ import annotations

import logging

import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import PRIMARY_CLASS, PRIMARY_SLICE_MULTIPLIER, TIER_CORE
from pilot_analysis.records import tier_rows
from pilot_analysis.render import markdown_table
from pilot_analysis.statistics import bootstrap_ci, decision_rule, did

logger = logging.getLogger(__name__)

TITLE = "Primary endpoint (core tier, overwrite, distant, K = 2 d_k)"


def primary_records(
    overwrite: pd.DataFrame, n_resamples: int, alpha: float
) -> list[dict]:
    """Delta_int records at the primary endpoint for one tier's overwrite rows."""
    records = []
    for (d_model, num_heads, head_dim), group in overwrite.groupby(
        ["d_model", "num_heads", "head_dim"], dropna=False
    ):
        dk = head_dim if not pd.isna(head_dim) else d_model / num_heads
        key = (
            f"test/acc/slice={int(PRIMARY_SLICE_MULTIPLIER * dk)}/class={PRIMARY_CLASS}"
        )
        if key not in group.columns or group[key].notna().sum() == 0:
            logger.warning(
                "primary endpoint metric %s not present (d_model=%s); skipping",
                key,
                d_model,
            )
            continue
        for layout in ("ll", "ws"):
            sub = group[group[key].notna()]
            deltas = did(sub, key, layout)
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


def run(table: pd.DataFrame, n_resamples: int, alpha: float) -> pd.DataFrame:
    """Delta_int at the primary endpoint, core tier rows only."""
    overwrite = tier_rows(table, TIER_CORE)
    overwrite = overwrite[overwrite["task"] == "overwrite"]
    return pd.DataFrame(primary_records(overwrite, n_resamples, alpha))


def render(frame: pd.DataFrame) -> str:
    lines = [f"## {TITLE}", ""]
    if frame.empty:
        lines.append("_No core-tier runs with the primary endpoint present._")
        return "\n".join(lines) + "\n"
    lines.append(markdown_table(frame))
    return "\n".join(lines) + "\n\n"


def report(table: pd.DataFrame, options: ReportOptions) -> Report:
    frame = run(table, options.n_resamples, options.alpha)
    return Report(
        name="primary",
        title=TITLE,
        tables={"primary_endpoint": frame},
        markdown=render(frame),
    )
