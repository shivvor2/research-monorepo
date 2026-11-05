"""Shared protocol constants of this project's pilots.

Everything the project's pilots share about the experimental grid lives
here: the selection seeds, the grid's gate modes, layouts and lambda
modes, and the tier-seed convention. Tier-specific constants (expected
run counts, published anchors, decision thresholds, screening cells)
belong to each pilot's own protocol, not here.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Selection and seeds
# --------------------------------------------------------------------------- #

SELECTION_SEEDS: tuple[int, ...] = (100, 101)
"""The seeds used for calibration selection (the reading a report picks
the operating point from)."""

# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #

GATE_MODES: tuple[str, ...] = ("kda", "full")
"""The gate modes of the grid: the two contrast levels every pilot
screens single-state."""

LAYOUTS: tuple[str, ...] = ("single", "ws", "ll")
"""The multi-state layouts of the grid (single-state is the reference); a
pilot may add a frozen-control layout of its own."""

LAMBDA_MODES: tuple[str, ...] = ("mlp", "static", "lin")
"""The lambda mixing modes of the grid."""

# --------------------------------------------------------------------------- #
# Tier-seed convention
# --------------------------------------------------------------------------- #

TIER_NONE = "none"
"""The group name for tier-less runs (e.g. the smoke config)."""

DEFAULT_SEED_RANGE = range(10)
"""The default per-cell seeds of a tier, 0-9."""


def tier_of_config(config: dict) -> str:
    """The tier of a resolved config dict (``TIER_NONE`` when tier-less).

    Args:
        config: A resolved config dict (``experiment.tier``).

    Returns:
        The tier name, or ``TIER_NONE`` when absent or empty.
    """
    tier = ((config.get("experiment") or {}).get("tier")) or TIER_NONE
    return str(tier) if tier else TIER_NONE


def group_of_metadata(metadata: dict) -> str:
    """The group of an artifact's metadata: its tier (``TIER_NONE`` when
    unset)."""
    return str(metadata.get("tier")) if metadata.get("tier") else TIER_NONE