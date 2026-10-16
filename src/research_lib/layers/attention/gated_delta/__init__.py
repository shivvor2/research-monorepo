"""Gated delta-rule token mixers (KDA / GDN-2 / GDN), reference implementation.

The single-state reference layer (PR 1) plus its functional core, shared
projections and cache.  The multi-state layouts (PR 2) extend the same
subpackage.
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
    "apply_gate_mode",
    "chunk_local_terms",
    "chunk_scan",
    "recurrent_scan",
    "transition",
]
