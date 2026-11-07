"""The structured-recall pilot's report modules.

Each report is a pure function of the loaded run table built on the
shared :mod:`common.analysis` pieces. The registry maps a report name to
its reader; a task needs its own reading it registers here, and each
task's screening cell lives in the pilot protocol.
"""

REPORTS: dict[str, callable] = {}
"""name -> report(table, options) registry."""

__all__ = ["REPORTS"]
