"""Markdown rendering: tables, report sections, the summary document."""

from __future__ import annotations

import numpy as np
import pandas as pd

SUMMARY_TITLE = "# Synthetic pilot analysis"

SUMMARY_NOTES = """## Notes and limitations

- Missing keys in a metrics dict mean 'not measured' (empty cells are omitted by RecallMetrics), not zero.
- Every statistic above selects its rows by tier: several tiers repeat core configurations (matched bytes, the lambda controls, R1, the long-sequence tier), so filtering by configuration alone would create duplicated cells. Calibration tiers and smoke runs never enter any statistic.
"""


def _fmt(value, digits=4):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
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


def markdown_section(title: str, frame: pd.DataFrame, level: str = "##") -> str:
    """A summary section: header, blank line, table, blank line.

    The empty case renders the table placeholder (``_none_``), so the
    section is always present and always ends with a blank line — the
    sections concatenate into a summary document without glue.
    """
    return f"{level} {title}\n\n{markdown_table(frame)}\n\n"
