"""Single-state gated delta-rule reference layer.

**Reference implementation** (naive PyTorch + autograd, no Triton, no
hand-written backward).  Use for correctness baselines and small synthetic
experiments, not for production-scale training — for that use fla's kernels
(``fla.ops``).  Chunk size is fixed per call, no automatic switching, and
precision-sensitive operations are pinned per the policy in
``functional.py``.

One class covers GDN-2 (``gate_mode="full"``), KDA (``"kda"``), Gated
DeltaNet (``"gdn"``) and the GDN-2 Table 5 gate ablations (``"b_only"``,
``"w_only"``): the gate treatment is a runtime scalarization of the *same*
projections, so parameter counts and ``state_dict`` keys are identical
across modes.

Run tests with::

    pytest src/research_lib/layers/attention/gated_delta/tests/test_reference.py -v
"""

import logging

import torch
import torch.nn as nn

from research_lib.layers.attention.gated_delta.cache import GatedDeltaCache
from research_lib.layers.attention.gated_delta.functional import (
    GateMode,
    apply_gate_mode,
    chunk_scan,
    recurrent_scan,
)
from research_lib.layers.attention.gated_delta.projections import GatedDeltaProjections

logger = logging.getLogger(__name__)


class GatedDeltaReference(nn.Module):
    r"""Single-state gated delta-rule token mixer.

    Implements the recurrence (per batch element and head, ``q``, ``k``
    L2-normalized):

    .. math::

        S_t = (I - k_t (b_t \odot k_t)^T)\, \mathrm{Diag}(e^{g_t})\, S_{t-1}
              + k_t (w_t \odot v_t)^T,
        \qquad o_t = \mathrm{scale} \cdot S_t^T q_t

    with ``b`` the channel-wise erase gate, ``w`` the channel-wise write
    gate and ``g`` the channel-wise log-decay (GDN-2, arXiv:2605.22791).
    Setting ``b = w = \beta\mathbf{1}`` recovers KDA exactly; additionally
    scalarizing the decay recovers Gated DeltaNet.  See the module docstring
    of :mod:`~research_lib.layers.attention.gated_delta.functional` for the
    locked conventions (read-after-write; decay, then erase, then write).

    Args:
        hidden_size (int): Model dimension :math:`d`.
        num_heads (int): Number of heads ``H`` (the pilot uses
            :math:`H_v = H`).
        head_dim (int): Key head dimension :math:`d_k`.
        expand_v (float): :math:`d_v = d_k \cdot \mathrm{expand\_v}`.
            Defaults to 1.0.
        gate_mode (str | GateMode): ``"full"`` (GDN-2, default),
            ``"b_only"``, ``"w_only"``, ``"kda"``, ``"gdn"``.
        mode (str): Forward path: ``"chunk"`` (default) or ``"recurrent"``;
            overridable per call.  No automatic switching.
        chunk_size (int): Chunk length ``C``. Defaults to 16.
        use_short_conv (bool): Short causal conv (k=4, SiLU) on ``q``/``k``/
            ``v``. Defaults to ``True``.
        conv_size (int): Short-conv kernel size. Defaults to 4.
        output_gate_activation (str): ``"silu"`` (paper, default) or
            ``"sigmoid"`` (fla parity).
        low_rank_gates (bool): Low-rank log-decay projection. Defaults to
            ``False``.
        scale (float | None): Output scale; ``None`` means
            :math:`d_k^{-1/2}` (fla/paper kernels).  The per-head output
            RMSNorm cancels any constant scale, so this only matters for
            raw-functional comparisons.
        norm_eps (float): Epsilon for the per-head output RMSNorm. Defaults
            to 1e-5.
        init_scheme (str): ``"paper"`` (default).
        compute_dtype (torch.dtype | None): Overrides the precision policy
            of the recurrence core (chunked and recurrent scans).  ``None``
            (the default) keeps the core in at least fp32 (fp64 modules stay
            fp64), as in the GDN-2 paper and fla's kernels; an explicit
            dtype (e.g. ``torch.bfloat16``) is meant for degradation
            experiments.
        device: Optional torch device.
        dtype: Optional torch dtype.

    Shape:
        - Input: :math:`(B, T, d)`
        - Output: :math:`(B, T, d)`

    Examples::

        >>> layer = GatedDeltaReference(hidden_size=64, num_heads=4, head_dim=16)
        >>> y, cache = layer(torch.randn(2, 32, 64), use_cache=True)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        expand_v: float = 1.0,
        gate_mode: str = "full",
        mode: str = "chunk",
        chunk_size: int = 16,
        use_short_conv: bool = True,
        conv_size: int = 4,
        output_gate_activation: str = "silu",
        low_rank_gates: bool = False,
        scale: float | None = None,
        norm_eps: float = 1e-5,
        init_scheme: str = "paper",
        compute_dtype: torch.dtype | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if mode not in ("chunk", "recurrent"):
            raise ValueError(f"mode must be 'chunk' or 'recurrent', got {mode!r}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_k_dim = head_dim
        self.gate_mode = GateMode(gate_mode)
        self.mode = mode
        self.chunk_size = chunk_size
        self.scale = 1.0 / (head_dim**0.5) if scale is None else float(scale)
        self.compute_dtype = compute_dtype
        self.projections = GatedDeltaProjections(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            expand_v=expand_v,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            output_gate_activation=output_gate_activation,
            low_rank_gates=low_rank_gates,
            norm_eps=norm_eps,
            init_scheme=init_scheme,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: GatedDeltaCache | None = None,
        use_cache: bool = False,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, GatedDeltaCache | None]:
        """Run the layer.

        Args:
            x (Tensor): Input of shape :math:`(B, T, d)`.
            cache (GatedDeltaCache | None): Cache from a previous call
                (initial state + conv caches + position), or ``None``.
            use_cache (bool): If ``True``, return the updated cache.
            mode (str | None): Overrides ``self.mode`` for this call
                (``"chunk"`` or ``"recurrent"``).

        Returns:
            tuple: ``(y, new_cache)``; ``new_cache`` is ``None`` unless
            ``use_cache=True``.
        """
        path = self.mode if mode is None else mode
        if path not in ("chunk", "recurrent"):
            raise ValueError(f"mode must be 'chunk' or 'recurrent', got {path!r}")
        initial_state = None
        if cache is not None:
            expected = (
                x.shape[0],
                self.num_heads,
                self.head_k_dim,
                self.projections.head_v_dim,
            )
            if cache.state.shape != expected:
                raise ValueError(
                    f"cache.state must be (B,H,dk,dv) = {expected}, "
                    f"got {tuple(cache.state.shape)}"
                )
            initial_state = cache.state
        q, k, v, b, w, g, conv_cache = self.projections.project(
            x,
            conv_cache=cache.conv_state if cache is not None else None,
            use_cache=use_cache,
        )
        b, w, g = apply_gate_mode(b, w, g, self.gate_mode)
        if path == "chunk":
            o, S = chunk_scan(
                q,
                k,
                v,
                g,
                b,
                w,
                initial_state=initial_state,
                chunk_size=self.chunk_size,
                scale=self.scale,
                compute_dtype=self.compute_dtype,
            )
        else:
            o, S = recurrent_scan(
                q,
                k,
                v,
                g,
                b,
                w,
                initial_state=initial_state,
                scale=self.scale,
                compute_dtype=self.compute_dtype,
            )
        y = self.projections.output(o, x)
        new_cache = None
        if use_cache:
            new_cache = GatedDeltaCache(
                state=S,
                conv_state=conv_cache,
                position=((cache.position if cache is not None else 0) + x.shape[1]),
            )
        return y, new_cache
