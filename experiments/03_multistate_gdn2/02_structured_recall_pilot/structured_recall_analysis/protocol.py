"""The structured-recall pilot's protocol constants.

The shared grid (gate modes, layouts, lambda modes, tier-seed
convention) is imported from :mod:`common.protocol`. What is specific to
this pilot lives here: its tier names, its calibration selection seeds,
its screening cells (the metric key each task's report reads) and its
expected run counts. A task registers its screening cell here when it is
added.
"""

from __future__ import annotations

from common.protocol import (  # noqa: F401  (re-exported for convenience)
    GATE_MODES,
    LAMBDA_MODES,
    LAYOUTS,
    TIER_NONE,
)

# --------------------------------------------------------------------------- #
# Tier names
# --------------------------------------------------------------------------- #

TIER_CALIB_LR = "calib_lr"
"""The single-state calibration tier (learning-rate selection)."""

TIER_CALIB_WS = "calib_ws"
"""The multi-state calibration tier (the layout sweep that reads the
interaction, in the same family as ``calib_lr``)."""

TIER_CORE = "tier1_core"
"""The tier-1 tier, whose single-state gated-delta runs are the
positive control."""

POSITIVE_CONTROL_TIER = TIER_CORE
"""The tier the positive-control report reads (derived, not repeated)."""

# --------------------------------------------------------------------------- #
# Selection seeds
# --------------------------------------------------------------------------- #

SELECTION_SEEDS: tuple[int, ...] = tuple(range(10))
"""The seeds the calibration selection and the positive control average
over (this pilot uses seeds 0-9, unlike the first pilot's 100,101)."""

# --------------------------------------------------------------------------- #
# Screening cells and tiers
# --------------------------------------------------------------------------- #

# The screening cell the reports read, per task: the metric key (a
# ``test/<key>`` column) that single-state calibration reports the gap
# and positive-control on. Each registered task declares its cell here.
SCREENING_CELLS: dict[str, str] = {
    "mqar": "test/acc/slice=64",
}
"""The screening metric key per registered task."""

CALIBRATION_TIERS: frozenset[str] = frozenset({TIER_CALIB_LR, TIER_CALIB_WS})
"""The tiers whose completed runs the calibration report reads."""

# Expected run counts per tier: a task's sweep files add its expected
# runs here.
EXPECTED_RUNS: dict[str, int] = {}

# The tiers whose finished runs are recognised as done by the skip check
# and the completeness report (tooling tiers like smoke/debug always run).
PROTOCOL_TIERS: frozenset[str] = frozenset({TIER_CALIB_LR, TIER_CALIB_WS, TIER_CORE})
