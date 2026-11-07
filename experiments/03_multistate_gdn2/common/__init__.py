"""Shared project package for the multi-state GDN-2 pilots.

Both pilots of this project import from ``common``: the launch helpers
(:mod:`common.launch`), the run-ID builder and identity exclusion set
(:mod:`common.run_id`), the run layout and artifact contract
(:mod:`common.layout`), the thin ``03_pull.py`` entry point
(:mod:`common.pull`), the record flatten (:mod:`common.records`), the
shared report pieces (:mod:`common.analysis`), the shared protocol
constants (:mod:`common.protocol`) and the task / harness registries
(:mod:`common.registries`). It is shared by this project's pilots only;
general machinery lives in ``research_lib`` and each pilot keeps its own
protocol.

A pilot's scripts add this folder (or its parent) to ``sys.path``, the
same way they add the library's ``src``; its ``conftest.py`` does the
same for pytest.
"""

from . import (
    analysis,
    launch,
    layout,
    protocol,
    pull,
    records,
    registries,
    run_id,
    run_lifecycle,
)

__all__ = [
    "analysis",
    "layout",
    "launch",
    "protocol",
    "pull",
    "records",
    "registries",
    "run_id",
    "run_lifecycle",
]
