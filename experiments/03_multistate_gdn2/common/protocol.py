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
"""The group name for a tier-less run (e.g. the smoke config)."""


def group_name(value) -> str:
    """The group name of an unset value: ``TIER_NONE`` for a null or empty
    tier. The one place a missing group becomes ``"none"``; metadata,
    records and the W&B group all go through it, so they agree."""
    return str(value) if value else TIER_NONE


def tier_of_config(config: dict) -> str:
    """The tier of a resolved config dict (``TIER_NONE`` when tier-less).

    Args:
        config: A resolved config dict (``experiment.tier``).

    Returns:
        The tier name, normalised through :func:`group_name`.
    """
    return group_name((config.get("experiment") or {}).get("tier"))


def group_of_metadata(metadata: dict) -> str:
    """The group of an artifact's metadata: its tier, normalised through
    :func:`group_name`."""
    return group_name(metadata.get("tier"))
