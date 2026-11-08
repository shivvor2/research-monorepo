"""Shared without-replacement sampling helpers for the synthetic tasks.

Both the repeated-key overwrite and entity-field generators draw distinct
token ids by redrawing whole rows until they are distinct (rejection),
which preserves the uniform law over ordered distinct draws. The
construction-time checks bound the per-round success probability so a
valid configuration never stalls; these helpers are the shared redraw cap
and the exact log-probability those checks use.
"""

from __future__ import annotations

import math

import numpy as np

_MAX_SAMPLING_ROUNDS = 10_000
"""Hard cap on redraw rounds of :func:`_distinct_ints` before a
RuntimeError. The configuration-time sampler bounds keep valid configs
far below this; a hit means a caller bypassed them (e.g. an impossible
draw)."""

_SAMPLER_LOG_FLOOR = math.log(0.01)
"""log of the per-round success floor (1%) for the without-replacement
samplers; the expected number of redraws per row is then at most ~100."""


def _sampler_log_prob(n: int, d: int) -> float:
    """Exact log-probability that ``d`` uniform draws from ``n`` ids (with
    replacement) come out distinct: ``log(n! / ((n - d)! n^d))``, computed
    via log-gammas so it never overflows. Requires ``1 <= d <= n`` (the
    ``n >= d`` feasibility check runs first).
    """
    return math.lgamma(n + 1) - math.lgamma(n - d + 1) - d * math.log(n)


def _check_sampler_floor(
    n: int,
    d: int,
    subject: str,
    sampler: str,
    formula: str,
    hint: str,
) -> None:
    """Raise when a without-replacement draw would stall the redraw loop.

    The per-round success probability of drawing ``d`` distinct ids from
    an ``n``-id pool must clear the 1% floor; otherwise the rejection loop
    is expected to redraw a row too often (or, when the probability
    underflows to zero, never succeed).

    Args:
        n: The number of ids in the pool.
        d: The number of distinct ids drawn per row.
        subject: The leading description, e.g. ``"num_keys entry 16"``.
        sampler: The pool name used in the message (``"key"``,
            ``"value"`` or ``"entity"``).
        formula: The exact probability expression to show before the
            numeric value (``""`` to omit it).
        hint: The remedy appended to the message.

    Raises:
        ValueError: If the probability is below the floor. When it
            underflows to ``0`` the expected redraws are reported as
            "effectively unbounded" instead of dividing by zero.
    """
    log_prob = _sampler_log_prob(n, d)
    if log_prob >= _SAMPLER_LOG_FLOOR:
        return
    prob = math.exp(log_prob)
    prob_clause = f"{formula} = {prob:.3%}" if formula else f"{prob:.3%}"
    if prob > 0.0:
        redraws = f"~{1.0 / prob:.0f} per row"
    else:
        redraws = "effectively unbounded per row"
    raise ValueError(
        f"{subject} would stall the {sampler} sampler: drawing {d} distinct "
        f"ids from the {n}-id {sampler} pool succeeds with exact probability "
        f"{prob_clause}, below the 1% floor (expected redraws {redraws}); "
        f"{hint}"
    )


def _distinct_ints(
    rng: np.random.Generator, low: int, high: int, shape: tuple[int, int]
) -> np.ndarray:
    """Uniform draws without replacement, per row.

    Each row of the result holds ``shape[1]`` distinct values from
    ``[low, high)``. Rows are redrawn (rejection) until distinct, which
    preserves the uniform law over ordered distinct rows.

    Args:
        rng: numpy Generator.
        low: inclusive lower bound.
        high: exclusive upper bound.
        shape: ``(rows, draws)``.

    Returns:
        Int64 array of ``shape`` whose rows hold distinct values.

    Shape:
        ``(shape)``.
    """
    out = np.empty(shape, dtype=np.int64)
    pending = np.arange(shape[0])
    rounds = 0
    while pending.size:
        rounds += 1
        if rounds > _MAX_SAMPLING_ROUNDS:
            raise RuntimeError(
                f"_distinct_ints: could not draw {shape[1]} distinct values "
                f"from the range [{low}, {high}) in {_MAX_SAMPLING_ROUNDS} "
                f"rounds"
            )
        out[pending] = rng.integers(low, high, size=(pending.size, shape[1]))
        ordered = np.sort(out[pending], axis=1)
        duplicate = np.any(ordered[:, 1:] == ordered[:, :-1], axis=1)
        pending = pending[duplicate]
    return out
