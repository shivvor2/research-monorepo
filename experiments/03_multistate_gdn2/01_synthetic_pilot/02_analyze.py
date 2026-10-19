"""Command-line entry point for the pilot analysis reports.

Each report answers one question (the primary endpoint, one
prediction, the anchors, calibration, ...) and is a pure function of
the loaded run records; this script loads the runs and writes each
requested report's tables as CSVs and its section as markdown.

Usage:
    python 02_analyze.py                                  # every report
    python 02_analyze.py primary gates calibration        # a selection
    python 02_analyze.py --list                           # report names
    python 02_analyze.py --runs-dir runs --out-dir analysis

Outputs (under ``--out-dir``, default ``analysis/``):

- ``<name>.md`` and ``<stem>.csv`` per report;
- ``summary.md`` for ``all``: every report's section concatenated, in
  registry order. The ``calibration`` report is registered but never
  part of ``all`` — it is a Phase-1 reading, run on its own.
"""

import argparse
import logging
from pathlib import Path

from pilot_analysis import (
    REPORT_TITLES,
    REPORTS,
    SUMMARY_EXCLUDES,
    ReportOptions,
    records,
    render,
)

logger = logging.getLogger(__name__)


def expand_names(requested: list[str]) -> tuple[list[str], bool]:
    """Resolve the requested report names to an ordered, deduplicated list.

    ``all`` expands to every report except the summary excludes
    (``calibration``); other names are kept in their given order.

    Raises:
        SystemExit: Via argparse-style error listing the valid names.
    """
    if "all" in requested:
        return [n for n in REPORTS if n not in SUMMARY_EXCLUDES], True
    names = []
    for name in requested:
        if name not in REPORTS:
            valid = ", ".join(["all", *REPORTS])
            raise SystemExit(f"unknown report {name!r}; available: {valid}")
        if name not in names:
            names.append(name)
    return names, False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "reports",
        nargs="*",
        help="report names (default: all, which excludes calibration)",
    )
    parser.add_argument("--runs-dir", default="runs", help="run JSON root")
    parser.add_argument("--out-dir", default="analysis", help="output directory")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--list", action="store_true", help="print the report names and exit"
    )
    args = parser.parse_args()

    if args.list:
        for name, title in REPORT_TITLES.items():
            print(f"{name}: {title}")
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    names, is_all = expand_names(args.reports or ["all"])

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = records.load_runs(runs_dir)
    logger.info("Loaded %d runs from %s", len(table), runs_dir)

    options = ReportOptions(n_resamples=args.n_resamples, alpha=args.alpha)
    sections: list[str] = []
    for name in names:
        report = REPORTS[name](table, options)
        (out_dir / f"{name}.md").write_text(report.markdown, encoding="utf-8")
        for stem, frame in report.tables.items():
            frame.to_csv(out_dir / f"{stem}.csv", index=False)
        if is_all:
            sections.append(report.markdown)
        logger.info("Wrote %s (report %s)", out_dir, name)

    if is_all:
        summary = (
            render.SUMMARY_TITLE + "\n\n" + "".join(sections) + render.SUMMARY_NOTES
        )
        (out_dir / "summary.md").write_text(summary, encoding="utf-8")
        logger.info("Wrote %s", out_dir / "summary.md")


if __name__ == "__main__":
    main()
