"""Missing runs per tier and incomplete runs."""

from __future__ import annotations

import pandas as pd
from pilot_analysis.base import Report, ReportOptions
from pilot_analysis.protocol import EXPECTED_RUNS
from pilot_analysis.render import markdown_section

TITLE = "Missing runs per tier"


def missing_runs(table: pd.DataFrame) -> pd.DataFrame:
    """Missing runs per tier against the protocol's expected counts.

    Only complete runs (``status == "test_done"``) count as observed; a
    run whose status is anything else is listed by
    :func:`incomplete_runs` instead.
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


def incomplete_runs(table: pd.DataFrame) -> pd.DataFrame:
    """Runs whose status is not 'test_done', with their status."""
    if "status" not in table.columns:
        return pd.DataFrame()
    incomplete = table[table["status"].fillna("unknown") != "test_done"]
    if incomplete.empty:
        return pd.DataFrame()
    return incomplete[
        ["run_id", "task", "tier", "layout", "gate_mode", "seed", "status"]
    ].sort_values(["tier", "run_id"])


def report(table: pd.DataFrame, options: ReportOptions) -> Report:
    missing = missing_runs(table)
    incomplete = incomplete_runs(table)
    markdown = markdown_section("Missing runs per tier", missing) + markdown_section(
        "Incomplete runs (status is not test_done)", incomplete
    )
    return Report(
        name="completeness",
        title=TITLE,
        tables={"missing_runs": missing, "incomplete_runs": incomplete},
        markdown=markdown,
    )
