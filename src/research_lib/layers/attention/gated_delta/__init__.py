"""Gated delta-rule token mixers (KDA / GDN-2 / GDN), reference implementation.

The single-state reference layer plus its functional core, shared
projections and cache, and the multi-state layouts: windowed
uniform (WS), frozen control (FS) and Fenwick/log-linear (LL), with the
λ mixer.
"""

from research_lib.layers.attention.gated_delta.cache import GatedDeltaCache
from research_lib.layers.attention.gated_delta.functional import (
    ChunkLocals,
    GateMode,
    apply_gate_mode,
    chunk_local_terms,
    chunk_scan,
    recurrent_scan,
    transition,
)
from research_lib.layers.attention.gated_delta.layouts import (
    Layout,
    ll_chunk_merge,
    ll_level,
    ll_level_matrix,
    ll_token_merge,
    lssb,
    num_slots,
    ws_shift,
)
from research_lib.layers.attention.gated_delta.mixing import LambdaMixer, LambdaMode
from research_lib.layers.attention.gated_delta.multistate import (
    MultistateGatedDelta,
    multistate_chunk_scan,
    multistate_recurrent_scan,
)
from research_lib.layers.attention.gated_delta.projections import (
    GatedDeltaProjections,
    HeadRMSNorm,
)
from research_lib.layers.attention.gated_delta.reference import GatedDeltaReference

__all__ = [
    "ChunkLocals",
    "GateMode",
    "GatedDeltaCache",
    "GatedDeltaProjections",
    "GatedDeltaReference",
    "HeadRMSNorm",
    "LambdaMode",
    "LambdaMixer",
    "Layout",
    "MultistateGatedDelta",
    "apply_gate_mode",
    "chunk_local_terms",
    "chunk_scan",
    "ll_chunk_merge",
    "ll_level",
    "ll_level_matrix",
    "ll_token_merge",
    "lssb",
    "multistate_chunk_scan",
    "multistate_recurrent_scan",
    "num_slots",
    "recurrent_scan",
    "transition",
    "ws_shift",
]
