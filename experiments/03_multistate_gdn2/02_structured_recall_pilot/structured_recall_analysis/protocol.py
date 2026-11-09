"""The structured-recall pilot's protocol constants.

The shared grid (gate modes, layouts, lambda modes, tier-seed
convention, the selection seeds) is imported from :mod:`common.protocol`.
What is specific to this pilot lives here: each task's tiers (the
calibration tiers and the positive-control tier it registers), its
screening cells (the metric key each task's report reads, and any
fallback cell), and its expected run counts. A task adds one
:class:`TaskTiers` entry and its sweep tiers' counts here when it is
added.
"""

from __future__ import annotations

from collections.abc import Mapping
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
# nothing here; entity-field and MAD fuzzy register their calibration
# and positive-control tiers.
TASK_TIERS: dict[str, TaskTiers] = {
    "entity_field": TaskTiers(
        calibration=("calib_ef_single", "calib_ef_multi", "calib_ef_attention"),
        positive_control="tier1_ef_single",
    ),
    "mad_fuzzy": TaskTiers(
        calibration=(
            "calib_mad_single",
            "calib_mad_multi",
            "calib_mad_attention",
        ),
        positive_control="tier1_mad_single",
    ),
}

# Expected done-count per tier, added alongside a task's tiers
# (12 = 1 layout x 2 gates x 3 LRs x 2 seeds; 24 = 2 layouts x 2 gates
# x 3 LRs x 2 seeds; 20 = 1 layout x 2 gates x seeds 0-9; 40 = 2
# layouts x 2 gates x seeds 0-9; 6 = 3 LRs x 2 seeds, no gate for the
# attention control).
EXPECTED_RUNS: dict[str, int] = {
    "calib_ef_single": 12,
    "calib_ef_multi": 24,
    "calib_ef_attention": 6,
    "tier1_ef_single": 20,
    "tier1_ef_multi": 40,
    "calib_mad_single": 12,
    "calib_mad_multi": 24,
    "calib_mad_attention": 6,
    "tier1_mad_single": 20,
    "tier1_mad_multi": 40,
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
    "mad_fuzzy": "test/acc_mad",
}
"""The screening metric key per registered task (a ``test/<key>``
column). A task with a screening cell but no :data:`TASK_TIERS` entry is
skipped by the reports with a log line, not an error."""

SCREENING_FALLBACKS: dict[str, tuple[str, float]] = {
    # Both gates below 0.15 at the longest slice: read the next-shorter slice.
    "entity_field": ("test/acc/slice=16/class=sibling", 0.15),
}
"""Per task, the fallback screening cell and the floor below which both
gates at the primary cell trigger it."""


def screening_cell(task: str, readings: Mapping[str, float | None]) -> str:
    """The screening cell to read for ``task``.

    Returns the task's fallback cell only when it has one, both gates'
    readings are present, and both are below the floor; otherwise the
    primary cell from :data:`SCREENING_CELLS`.

    Args:
        task: The task name.
        readings: The per-gate readings at the primary cell (``full`` and
            ``kda``), ``None`` where unmeasured.

    Returns:
        The metric key to read.
    """
    primary = SCREENING_CELLS[task]
    fallback = SCREENING_FALLBACKS.get(task)
    if fallback is None:
        return primary
    cell, floor = fallback
    full = readings.get("full")
    kda = readings.get("kda")
    if full is None or kda is None:
        return primary
    if full < floor and kda < floor:
        return cell
    return primary


@dataclass(frozen=True)
class DamageReading:
    """A task's damage reading: the high-class accuracy minus the
    low-class accuracy, per slice.

    Attributes:
        high_class: The class name whose accuracy is the minuend.
        low_class: The class name whose accuracy is the subtrahend.
        slices: The slice values to read (the shortest slice is omitted
            when it has no high-class answers).
    """

    high_class: str
    low_class: str
    slices: tuple[int, ...]

    def columns(self, n: int) -> tuple[str, str]:
        """The two ``test/acc/slice={n}/class={c}`` column names for
        slice ``n``."""
        return (
            f"test/acc/slice={n}/class={self.high_class}",
            f"test/acc/slice={n}/class={self.low_class}",
        )


TASK_DAMAGE: dict[str, DamageReading] = {
    # The shortest slice (N = 8) has no untouched pairs: every entity is
    # updated, so it is omitted.
    "entity_field": DamageReading("untouched", "sibling", slices=(16, 32)),
}
"""The damage reading per registered task. The reports print the reading
(``high - low``) per slice and gate when both columns are present."""

# --------------------------------------------------------------------------- #
# Per-task length-setting band
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SettingBand:
    """A task's length-setting band: the interval a best validation reading
    should sit in, and a margin near either edge that flags a re-check.

    Below the floor the setting moves down, above the ceiling it moves up,
    and a reading within ``margin`` of either edge is flagged for a
    re-check.  The report only prints these; it decides nothing.
    """

    floor: float
    ceiling: float
    margin: float


TASK_SETTING_BANDS: dict[str, SettingBand] = {
    # The MAD length rule reads a best validation ``acc_mad`` and moves
    # the length if it sits outside the band [0.15, 0.75].
    "mad_fuzzy": SettingBand(floor=0.15, ceiling=0.75, margin=0.05),
}
"""The tasks whose best validation reading drives a length-setting band,
keyed by task.  A task with no entry gets no band columns."""

TASK_STUCK_CHECK: frozenset[str] = frozenset({"mad_fuzzy"})
"""The tasks whose runs get a per-run look-stuck flag in the calibration
reading (a task-level registration, not a name check)."""
