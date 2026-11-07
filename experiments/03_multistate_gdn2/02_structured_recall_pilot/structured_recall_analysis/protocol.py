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
# nothing here.
TASK_TIERS: dict[str, TaskTiers] = {}

# Expected done-count per tier, added alongside a task's tiers.
EXPECTED_RUNS: dict[str, int] = {}

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
}
"""The screening metric key per registered task (a ``test/<key>``
column). A task with a screening cell but no :data:`TASK_TIERS` entry is
skipped by the reports with a log line, not an error."""
