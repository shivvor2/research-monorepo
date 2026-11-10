"""Fixed column schemas for the structured-recall pilot's reports.

Every report's columns are decided by the code and the protocol tables,
never by which runs happen to exist: each :func:`~.builder` below
returns an ordered tuple built from fixed base lists plus groups derived
from the protocol (the damage slices, the setting bands). A report with
no rows still writes a header carrying exactly these columns, so a
reader sees the same schema before and after a task's runs exist.

A task that registers in an existing protocol table (a damage reading, a
setting band) needs no change here — its columns already exist. A new
*kind* of reading (a column group the rows carry that no builder
returns) means a new group in the matching builder; the writer's check
that no unregistered column is written makes a forgotten registration
fail in tests rather than drop or appear silently. Column names are
shared across tasks (two tasks with damage at slice 16 use the same
column), so the width grows with the kinds of reading, not with the
number of tasks.
"""

from __future__ import annotations

from common.analysis import PAIRED_GAP_COLUMNS

from . import protocol

# The keys every calibration row carries, in order. A row that belongs
# to a group without a given cell (an attention row on a gap, a
# multi-state row on a screening cell) holds an empty cell for it.
_CALIBRATION_BASE: tuple[str, ...] = (
    "kind",
    "task",
    "seq_len",
    "layout",
    "gate_mode",
    "lambda_mode",
    "decay_scheme",
    "mixer",
    "n_seeds",
    "d_model",
    "selected_lr",
    "selected_weight_decay",
    "mean_best_value",
    "cell",
    "cell_metric",
    "gap_full_kda",
)


def _damage_slices() -> list[int]:
    """The sorted union of the damage slice values across the protocol's
    readings (``[]`` when no task registers a reading)."""
    return sorted(
        {n for reading in protocol.TASK_DAMAGE.values() for n in reading.slices}
    )


def calibration_columns() -> tuple[str, ...]:
    """The calibration report's columns, in order.

    The tuning-point keys, then one ``damage/slice={n}`` column per
    slice in the protocol's damage readings, then the ``band`` /
    ``near_edge`` columns when any task registers a setting band.
    """
    columns = list(_CALIBRATION_BASE)
    columns.extend(f"damage/slice={n}" for n in _damage_slices())
    if protocol.TASK_SETTING_BANDS:
        columns.extend(("band", "near_edge"))
    return tuple(columns)


def positive_control_columns() -> tuple[str, ...]:
    """The positive-control report's columns, in order.

    ``task, seq_len, metric, tier``, then the paired-gap columns from
    :data:`common.analysis.PAIRED_GAP_COLUMNS`, then per-gate per-slice
    ``damage_{gate}/slice={n}`` columns for ``gate`` in (``full``,
    ``kda``).
    """
    columns = ["task", "seq_len", "metric", "tier", *PAIRED_GAP_COLUMNS]
    for gate in ("full", "kda"):
        columns.extend(f"damage_{gate}/slice={n}" for n in _damage_slices())
    return tuple(columns)


def safeguard_columns() -> tuple[str, ...]:
    """The safeguard report's columns, in order: the per-run identity, the
    run's sequence length, and the look-stuck flag."""
    return (
        "run_id",
        "task",
        "seq_len",
        "tier",
        "layout",
        "gate_mode",
        "seed",
        "looks_stuck",
    )


def completeness_columns() -> tuple[str, ...]:
    """The completeness report's columns, in order: per ``(tier, seq_len)``
    the expected and observed done runs, the missing count, and whether
    the tier is a declared one."""
    return ("tier", "seq_len", "expected", "observed", "missing", "known")
