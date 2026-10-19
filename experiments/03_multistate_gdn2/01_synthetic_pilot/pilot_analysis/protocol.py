"""Protocol constants: tier names, expected counts, calibration parameters.

One source of truth for both the training script (the frozen-choice
gate and the skip-completed check) and the analysis reports: a tier
added to one cannot silently skip the other.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Tier names
# --------------------------------------------------------------------------- #

TIER_CALIB_DMODEL = "calib_dmodel"
TIER_CALIB_HARNESS = "calib_harness"
TIER_CALIB_LR = "calib_lr"
TIER_CALIB_LR_ANCHORS = "calib_lr_anchors"
TIER_CORE = "tier1_core"
TIER_GATES = "tier2_gates"
TIER_FROZEN = "tier3_frozen"
TIER_LAMBDA = "tier4_lambda"
TIER_MATCHED_BYTES = "tier5_matched_bytes"
TIER_ANCHORS = "tier6_anchors"
TIER_R1 = "R1"
TIER_LONG_T = "tier_longT"

# --------------------------------------------------------------------------- #
# Calibration parameters
# --------------------------------------------------------------------------- #

SELECTION_SEEDS = (100, 101)
LR_GRID = (0.001, 0.003, 0.01)
D_MODEL_CANDIDATES = (32, 64)
# The d_model reading: the smallest candidate whose mean validation
# accuracy lands inside this interval; 32 is the fallback.
DMODEL_ACC_RANGE = (0.5, 0.95)
DMODEL_FALLBACK = 32

# Primary endpoint: the largest key count is twice the head dimension,
# so the metric column is ``test/acc/slice=<2*dk>/class=distant``.
PRIMARY_SLICE_MULTIPLIER = 2
PRIMARY_CLASS = "distant"

# --------------------------------------------------------------------------- #
# Published anchor numbers (Log-Linear Table 2, Gated DeltaNet), for the
# anchor report and the harness reading.
# --------------------------------------------------------------------------- #

LOG_LINEAR_TABLE_2 = {
    16: {"single": 38.4, "ll": 40.0},
    32: {"single": 79.0, "ll": 84.4},
    64: {"single": 99.0, "ll": 99.0},  # ">= 99"
}
LOG_LINEAR_STD_2 = {
    16: {"single": 1.0, "ll": 1.4},
    32: {"single": 2.1, "ll": 1.2},
    64: {"single": None, "ll": None},
}

# --------------------------------------------------------------------------- #
# Expected run counts per tier (the missing-run report)
# --------------------------------------------------------------------------- #

EXPECTED_RUNS: dict[str, int] = {
    TIER_CALIB_DMODEL: 4,
    TIER_CALIB_HARNESS: 4,
    TIER_CALIB_LR: 72,
    TIER_CALIB_LR_ANCHORS: 36,
    TIER_CORE: 120,
    TIER_GATES: 60,
    TIER_FROZEN: 20,
    TIER_LAMBDA: 40,
    TIER_MATCHED_BYTES: 10,
    TIER_ANCHORS: 30,
    TIER_LONG_T: 60,
    TIER_R1: 30,
}

CALIBRATION_TIERS: frozenset[str] = frozenset(
    {
        TIER_CALIB_DMODEL,
        TIER_CALIB_HARNESS,
        TIER_CALIB_LR,
        TIER_CALIB_LR_ANCHORS,
    }
)

PROTOCOL_TIERS: frozenset[str] = CALIBRATION_TIERS | frozenset(EXPECTED_RUNS.keys())
