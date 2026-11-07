"""The structured-recall pilot's protocol constants.

The shared grid (selection seeds, gate modes, layouts, lambda modes) is
imported from :mod:`common.protocol`. What is specific to this pilot
lives here: its screening cells (the metric key each task's report reads),
its expected run counts and its decision thresholds. A task registers its
screening cell here when it is added.
"""

from __future__ import annotations

from common.protocol import (  # noqa: F401  (re-exported for convenience)
    GATE_MODES,
    LAMBDA_MODES,
    LAYOUTS,
    SELECTION_SEEDS,
    TIER_NONE,
)

# The screening cell the reports read, per task: the metric key (a
# ``test/<key>`` column) that single-state calibration reports the gap
# and positive-control on. Filled in by the task PRs.
SCREENING_CELLS: dict[str, str] = {
    "mqar": "test/acc/slice=64",
}
"""The screening metric key per registered task."""

# Expected run counts per tier: filled in by the task PRs' sweep files.
EXPECTED_RUNS: dict[str, int] = {}

# The tiers whose finished runs are recognised as done by the skip check
# (tooling tiers like smoke/debug always run). Filled in as the task PRs
# add their sweeps.
PROTOCOL_TIERS: frozenset[str] = frozenset(EXPECTED_RUNS) | frozenset({"calibration"})
