"""Compatibility shim: the artifact contract lives in
:mod:`research_lib.training.run_records` and :mod:`common.layout`.

The name is kept so this pilot's scripts and their tests keep importing as
they always have; nothing pilot-specific is defined here. New pilots
import ``ARTIFACT_TYPE`` / ``artifact_name`` from ``research_lib`` and the
project's ``record_metadata`` from ``common.layout``.
"""

from __future__ import annotations

# The project's metadata (identity + provenance + task/tier), so the pull
# sees the same fields it always did.
from common.layout import record_metadata as run_record_metadata  # noqa: E402

from research_lib.training.run_records import ARTIFACT_TYPE, artifact_name

__all__ = ["ARTIFACT_TYPE", "artifact_name", "run_record_metadata"]
