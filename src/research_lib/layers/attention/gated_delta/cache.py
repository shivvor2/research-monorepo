"""Cache object for gated delta-rule layers.

Carries everything needed to continue a sequence: the recurrent state
(fp32), the short-conv caches for ``q``/``k``/``v`` (the classic decode bug
is forgetting these), and the absolute token position (needed by the
multi-state layouts, whose merge schedule depends on it).
"""

from dataclasses import dataclass

import torch


@dataclass
class GatedDeltaCache:
    """Inference/decode cache for a gated delta-rule layer.

    Attributes:
        state (Tensor): Recurrent state of shape ``(B, H, d_k, d_v)`` for the
            single-state layer.  The multi-state layer uses
            ``(B, H, L, d_k, d_v)`` with one bucket per level.  Always fp32
            (fp64 in the fp64 test suite).
        conv_state (tuple[Tensor, Tensor, Tensor] | None): Short-conv caches
            for ``q``, ``k``, ``v``, each of shape
            ``(B, H * d, K - 1)`` (pre-conv inputs), or ``None`` if the layer
            was built with ``use_short_conv=False``.
        position (int): Absolute number of tokens consumed so far (0 for a
            fresh cache).  The multi-state merge schedule depends on it.
    """

    state: torch.Tensor
    conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    position: int = 0
