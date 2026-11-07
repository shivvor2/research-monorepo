"""The structured-recall pilot's protocol constants.

The shared grid (gate modes, layouts, lambda modes, tier-seed
convention, the selection seeds) is imported from :mod:`common.protocol`.
What is specific to this pilot lives here: each task's tiers (the
calibration tiers and the positive-control tier it registers), its
screening cells (the metric key each task's report reads) and its
expected run counts. A task adds one :class:`TaskTiers` entry and its
sweep tiers' counts here when it is added.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.protocol import (  # noqa: F401  (re-exported for convenience)
    GATE_MODES,
    LAMBDA_MODES,
    LAYOUTS,
    SELECTION_SEEDS,
    TIER_NONE,
)

# --------------------------------------------------------------------------- #
# Per-task tiers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskTiers:
    """The tiers one task registers.

    Attributes:
        calibration: The task's calibration tiers, whose completed runs
            the calibration report reads (the selection seeds 100 and
            101 run these).
        positive_control: The tier whose single-state gated-delta cells
            the positive-control report reads (tier-1 seeds, no seed
            filter).
    """

    calibration: tuple[str, ...]
    positive_control: str


# Each task adds one entry (its task config names the same tiers in
# ``conf/sweep/``). MQAR is smoke and debug only, so it registers
# nothing here; entity-field registers its calibration tiers and its
# positive-control tier (the tier-1 single-state cells).
TASK_TIERS: dict[str, TaskTiers] = {
    "entity_field": TaskTiers(
        calibration=("calib_ef_single", "calib_ef_multi"),
        positive_control="tier1_ef_single",
    ),
}

# Expected done-count per tier, added alongside a task's tiers
# (12 = 1 layout x 2 gates x 3 LRs x 2 seeds; 24 = 2 layouts x 2 gates
# x 3 LRs x 2 seeds; 20 = 1 layout x 2 gates x seeds 0-9; 40 = 2
# layouts x 2 gates x seeds 0-9).
EXPECTED_RUNS: dict[str, int] = {
    "calib_ef_single": 12,
    "calib_ef_multi": 24,
    "tier1_ef_single": 20,
    "tier1_ef_multi": 40,
}

# --------------------------------------------------------------------------- #
# Derived tier sets
# --------------------------------------------------------------------------- #

CALIBRATION_TIERS = frozenset(
    tier for tt in TASK_TIERS.values() for tier in tt.calibration
)
"""The tiers whose completed runs the calibration report reads."""

PROTOCOL_TIERS = (
    frozenset(EXPECTED_RUNS)
    | CALIBRATION_TIERS
    | frozenset(tt.positive_control for tt in TASK_TIERS.values())
)
"""The tiers recognised as protocol (as opposed to tooling smoke/debug):
the expected-count tiers plus every registered calibration and
positive-control tier."""

# --------------------------------------------------------------------------- #
# Screening cells
# --------------------------------------------------------------------------- #

SCREENING_CELLS: dict[str, str] = {
    "mqar": "test/acc/slice=64",
    "entity_field": "test/acc/slice=32/class=sibling",
}
"""The screening metric key per registered task (a ``test/<key>``
column). A task with a screening cell but no :data:`TASK_TIERS` entry is
skipped by the reports with a log line, not an error."""

# Single-step fallback for the entity-field screening cell: if both gates
# read the sibling accuracy below this floor at the longest slice, the
# reports read the next-shorter slice instead.
SIBLING_FLOOR = 0.15
"""Below this, both gates at the longest slice fall back to the shorter
slice's sibling cell."""

SCREENING_FALLBACK_CELL = "test/acc/slice=16/class=sibling"
"""The fallback screening cell when both gates read below
:data:`SIBLING_FLOOR` at the longest slice."""

# A task's damage reading: the erase's unwanted spread. For entity-field:
# the sibling (written before an update) accuracy subtracted from the
# untouched accuracy, at a gate's selected point.
TASK_DAMAGE_PAIRS: dict[str, tuple[str, str]] = {
    "entity_field": (
        "test/acc/class=untouched",
        "test/acc/class=sibling",
    ),
}
"""The high-minus-low metric pair defining a task's damage reading, per
registered task. The reports print the reading (``high - low``) per gate
when both columns are present."""
