"""Utility helpers (secrets, paired contrasts, bootstrap intervals)."""

from .stats import Factor, bootstrap_ci, interaction_contrast, paired_contrast

__all__ = [
    "Factor",
    "bootstrap_ci",
    "interaction_contrast",
    "paired_contrast",
]
