"""The pilot analysis package: one report per question, one module each.

The protocol's constants live in :mod:`pilot_analysis.protocol` (one
source of truth shared with the training script), the run-record loading
in :mod:`pilot_analysis.records`, the shared statistics in
:mod:`pilot_analysis.statistics` and the markdown rendering in
:mod:`pilot_analysis.render`. Every report is a pure function of the
loaded run table; ``02_analyze.py`` is the command-line entry point
that writes each report's tables (CSV) and section (markdown).

The registry order is the summary order.
"""

from pilot_analysis.base import Report, ReportFn, ReportOptions
from pilot_analysis.reports import (
    all_cells,
    anchors,
    calibration,
    completeness,
    distance,
    frozen,
    gates,
    lambda_controls,
    long_t,
    matched_bytes,
    mixer,
    primary,
    robustness,
)

__all__ = [
    "REPORTS",
    "REPORT_TITLES",
    "SUMMARY_EXCLUDES",
    "Report",
    "ReportFn",
    "ReportOptions",
]

# One entry per report, in summary order. `calibration` is registered but
# excluded from `all`: it is a Phase-1 reading, not a tier summary.
REPORTS: dict[str, ReportFn] = {
    "primary": primary.report,
    "robustness": robustness.report,
    "all_cells": all_cells.report,
    "gates": gates.report,
    "frozen": frozen.report,
    "lambda_controls": lambda_controls.report,
    "distance": distance.report,
    "long_t": long_t.report,
    "anchors": anchors.report,
    "matched_bytes": matched_bytes.report,
    "mixer": mixer.report,
    "completeness": completeness.report,
    "calibration": calibration.report,
}

REPORT_TITLES: dict[str, str] = {
    "primary": primary.TITLE,
    "robustness": robustness.TITLE,
    "all_cells": all_cells.TITLE,
    "gates": gates.TITLE,
    "frozen": frozen.TITLE,
    "lambda_controls": lambda_controls.TITLE,
    "distance": distance.TITLE,
    "long_t": long_t.TITLE,
    "anchors": anchors.TITLE,
    "matched_bytes": matched_bytes.TITLE,
    "mixer": mixer.TITLE,
    "completeness": completeness.TITLE,
    "calibration": calibration.TITLE,
}

SUMMARY_EXCLUDES: tuple[str, ...] = ("calibration",)
"""Reports registered but never run by ``all``."""
