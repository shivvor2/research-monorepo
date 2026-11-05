"""The structured-recall pilot's report modules.

Each report is a pure function of the loaded run table built on the
shared :mod:`common.analysis` pieces. The report registry is filled in by
the task PRs (each task registers its screening cell in the pilot
protocol and any task-specific reading here).
"""

REPORTS: dict[str, callable] = {}
"""name -> report(table, options) registry; filled by task PRs."""

__all__ = ["REPORTS"]