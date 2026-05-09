"""
Recurrent depth primitives for weight-shared iterative refinement.

This module provides:
    * :class:`SelfAttentionWrapper` - adapts ``(q,k,v)`` attention for single-input blocks
    * :class:`LTIInjection` - stable recurrent state update (spectral radius < 1)
    * :class:`ACTHalting` - adaptive computation time halting mechanism
    * :func:`loop_index_embedding` - sinusoidal depth-position signal
    * :class:`RecurrentTransformerBlock` - single standard block with optional DepthFiLM
    * :class:`RecurrentBlock` - the *shell* that loops a core block T times

All components are config-free and accept explicit constructor arguments so
that they can be reused across different model families without introducing
opaque dataclass dependencies.

Reusability
-----------
The :class:`RecurrentTransformerBlock` and :class:`RecurrentBlock` are
designed to wrap **arbitrary** attention and feedforward modules without
requiring those modules to conform to a specific keyword-argument
signature.

Kwargs are forwarded selectively: each sub-layer receives only the keyword
arguments that its ``forward`` method actually accepts (determined by
:func:`inspect.signature` at construction time).  This means you can plug
in ``nn.MultiheadAttention``, a custom RoPE-aware GQA module, or any other
attention implementation without modification.
"""

import inspect
import math
from typing import Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from research_lib.layers.depth_film import DepthFiLM

# ---------------------------------------------------------------------------
# Signature introspection
# ---------------------------------------------------------------------------


def _accepted_kwargs(module: nn.Module) -> Set[str]:
    """Return the set of keyword argument names that *module.forward* accepts.

    If the signature contains ``**kwargs`` (a VAR_KEYWORD parameter), returns
    a sentinel set that contains *every* string (i.e. ``"anything" in result``
    is always ``True``), since the module will accept any keyword argument.

    This is used by :class:`RecurrentTransformerBlock` to forward only the
    kwargs that each sub-layer actually understands, enabling plug-and-play
    composition with heterogeneous attention and FFN implementations.
    """
    sig = inspect.signature(module.forward)
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            # Module accepts **kwargs — it will take anything.
            return _UniversalSet()
    return {
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


class _UniversalSet(set):
    """A set-like object where ``x in self`` is always ``True``."""

    def __contains__(self, item: object) -> bool:
        return True


def _filter_kwargs(accepted: Set[str], kwargs: dict) -> dict:
    """Return only the entries of *kwargs* whose keys are in *accepted*."""
    return {k: v for k, v in kwargs.items() if k in accepted}


# ---------------------------------------------------------------------------
# SelfAttentionWrapper
# ---------------------------------------------------------------------------


class SelfAttentionWrapper(nn.Module):
    r"""Wrap an attention primitive that expects ``(q, k, v, ...)`` into a
    block-compatible module that accepts a single tensor ``x``.

    The wrapper calls the underlying attention with :math:`q=k=v=x`.
    Only keyword arguments that the wrapped module's ``forward`` actually
    accepts are forwarded; all others are silently dropped.  This lets the
    wrapper sit inside a :class:`RecurrentTransformerBlock` that passes a
    superset of kwargs (``freqs_cis``, ``attn_mask``, ``kv_cache``, etc.)
    without requiring the wrapped module to understand all of them.

    Args:
        attention_module (nn.Module): Attention implementation to wrap.
            Must be callable as ``module(query, key, value, **kwargs)``.

    Shape:
        - Input: :math:`(B, T, C)` — a single sequence tensor.
        - Output: same shape.

    Examples::

        >>> from torch.nn import MultiheadAttention
        >>> # MHA only accepts key_padding_mask, attn_mask, etc. — not
        >>> # freqs_cis or kv_cache.  The wrapper filters kwargs automatically.
        >>> wrapped = SelfAttentionWrapper(
        ...     MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        ... )
        >>> x = torch.randn(2, 128, 512)
        >>> out = wrapped(x)  # triggers self-attention internally

    .. note::
        If the wrapped module returns a tuple (e.g. ``nn.MultiheadAttention``
        returns ``(output, weights)``), only the first element is returned.
    """

    def __init__(self, attention_module: nn.Module):
        super().__init__()
        self.attn = attention_module
        self._accepted = _accepted_kwargs(attention_module)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Apply self-attention with ``q=k=v=x``.

        Args:
            x (Tensor): Input of shape :math:`(B, T, C)`.
            **kwargs: Extra arguments; only those accepted by the wrapped
                module are forwarded, others are silently dropped.

        Returns:
            Tensor: Attention output, same shape as ``x``.
        """
        filtered = _filter_kwargs(self._accepted, kwargs)
        out = self.attn(x, x, x, **filtered)
        # Handle modules that return tuples (e.g. nn.MultiheadAttention).
        if isinstance(out, tuple):
            out = out[0]
        return out


# ---------------------------------------------------------------------------
# LTIInjection
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    r"""Linear Time-Invariant (LTI) hidden-state injection.

    Implements a stable recurrent update for a hidden state :math:`h_t`
    inside a looped transformer block:

    .. math::
        h_{t+1} = A \odot h_t + B \odot e + \text{transformer\_out}

    where :math:`e` is a frozen encoded input (prelude output) and
    :math:`\odot` denotes element-wise multiplication.  The diagonal
    state matrix :math:`A` is guaranteed to have spectral radius
    :math:`\rho(A) < 1` *by construction* via the log-parameterization:

    .. math::
        A = \exp\!\bigl(-\exp(\log\Delta t + \log A)\bigr)

    This ensures the recurrent dynamics remain contractive regardless of
    the learned parameter values, preventing hidden-state explosion when
    the loop depth is large.

    Args:
        dim (int): Hidden dimension (channel count) of the state vector.

    Shape:
        - ``h``, ``e``, ``transformer_out``: each :math:`(B, T, C)`.
        - Output: same shape.

    Examples::

        >>> inject = LTIInjection(dim=512)
        >>> h = torch.randn(2, 128, 512)
        >>> e = torch.randn(2, 128, 512)
        >>> out = torch.randn(2, 128, 512)
        >>> h_next = inject(h, e, out)

    References:
        * Prairie et al., "Parcae: Parallel Recurrent Deep Equilibrium
          Models", 2026 (stable log-space discretization).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        r"""Compute the discretized diagonal state matrix with
        :math:`\rho(A)<1`.

        Returns:
            Tensor: 1-D tensor of shape ``(dim,)`` with values in ``(0,1)``.
        """
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        r"""Single LTI update step.

        Args:
            h (Tensor): Current hidden state :math:`h_t`,
                shape :math:`(B, T, C)`.
            e (Tensor): Frozen encoded input (prelude output),
                shape :math:`(B, T, C)`.
            transformer_out (Tensor): Output of the core block at this
                loop step, shape :math:`(B, T, C)`.

        Returns:
            Tensor: Updated state :math:`h_{t+1}`, same shape as inputs.
        """
        A = self.get_A()
        return A * h + self.B * e + transformer_out


# ---------------------------------------------------------------------------
# ACT Halting
# ---------------------------------------------------------------------------


class ACTHalting(nn.Module):
    r"""Adaptive Computation Time halting mechanism (Graves, 2016).

    Learns a per-position halting probability at each loop iteration.
    Positions where the hidden state has converged (high cumulative halting
    probability) stop accumulating updates, while positions still being
    refined continue.  This lets easy tokens halt early and hard tokens
    receive more computation, all within the same batch.

    Args:
        dim (int): Hidden state dimension.

    Examples::

        >>> act = ACTHalting(dim=512)
        >>> h = torch.randn(2, 128, 512)
        >>> p = act(h)  # shape (2, 128), values in (0, 1)

    References:
        * Graves, "Adaptive Computation Time for Recurrent Neural
          Networks", arXiv:1603.08983, 2016.
        * Dehghani et al., "Universal Transformers", arXiv:1807.03819,
          2018.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        r"""Predict per-position halting probability.

        Args:
            h (Tensor): Hidden state of shape :math:`(B, T, C)`.

        Returns:
            Tensor: Halting probabilities, shape :math:`(B, T)`, values
            in :math:`(0, 1)`.
        """
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Loop-index embedding
# ---------------------------------------------------------------------------


def loop_index_embedding(
    h: torch.Tensor,
    loop_t: int,
    loop_dim: Optional[int] = None,
    theta: float = 10000.0,
) -> torch.Tensor:
    r"""Sinusoidal depth-position embedding for recurrent loops.

    Analogous to Rotary Position Embedding (RoPE) applied over *depth*
    rather than sequence position.  A sinusoidal bias is added to the
    first ``loop_dim`` channels of ``h`` so that the shared block
    weights can distinguish which loop iteration they are executing.

    For an even ``loop_dim`` the embedding is:

    .. math::
        \text{emb}[j] &= \sin\!\bigl(t \cdot \theta_j\bigr),\quad j\in[0,\tfrac{D}{2}) \\
        \text{emb}[j] &= \cos\!\bigl(t \cdot \theta_{j-D/2}\bigr),\quad j\in[\tfrac{D}{2},D)

    where :math:`\theta_j = \theta^{-2j/D}`.

    Args:
        h (Tensor): Hidden state of shape :math:`(B, T, C)`.
        loop_t (int): Current loop index :math:`t`.
        loop_dim (int, optional): Number of leading channels to perturb.
            If ``None``, defaults to ``max(2, h.size(-1) // 8)``.
            Clamped to a minimum of ``2`` to ensure at least one
            sin/cos pair is produced.
        theta (float, optional): Base frequency. Default: ``10000.0``.

    Returns:
        Tensor: A new tensor with the depth bias added to the first
        ``loop_dim`` channels (``h`` is not modified in-place).

    Shape:
        - Input: :math:`(B, T, C)`.
        - Output: same shape.

    Examples::

        >>> h = torch.zeros(2, 4, 512)
        >>> h = loop_index_embedding(h, loop_t=3)

    References:
        * Su et al., "RoFormer: Enhanced Transformer with Rotary Position
          Embedding", arXiv:2104.09864, 2021.
    """
    loop_dim = loop_dim or max(2, h.shape[-1] // 8)
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    full[:loop_dim] = emb
    return h + full.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# RecurrentTransformerBlock
# ---------------------------------------------------------------------------


class RecurrentTransformerBlock(nn.Module):
    r"""A single Transformer block whose sub-layers can be *depth-conditioned*.

    This is a standard pre-normalisation residual block:

    .. code-block:: text

         x --> RMSNorm --> Attention --> [DepthFiLM] --> + -->
         ↑                                               (residual)
         x --> RMSNorm --> FFN --> [DepthFiLM] --> + --> out
         ↑                                               (residual)

    The crucial difference from a vanilla Transformer block is the optional
    :class:`DepthFiLM` insertion **after** each sub-layer computation but
    **before** the residual addition.  The FiLM layer receives the
    current loop index :math:`t` and re-scales the sub-layer output so
    that the same shared weights behave differently at different
    recurrent depths.

    **Kwargs forwarding** — The block does *not* require the wrapped
    attention or feedforward modules to accept any particular set of
    keyword arguments.  At construction time it inspects each module's
    ``forward`` signature and records which kwargs it accepts.  During
    the forward pass only matching kwargs are forwarded; the rest are
    silently dropped.  This makes the block fully reusable across
    different attention implementations (``nn.MultiheadAttention``,
    custom RoPE-GQA, Flash Attention wrappers, etc.) without any
    modification to those modules.

    .. important::
        The wrapped ``attention`` and ``feedforward`` modules require
        **zero modification**.  DepthFiLM operates purely on their output
        tensors.

    Args:
        dim (int): Model hidden dimension :math:`C`.
        attention (nn.Module): Attention sub-layer.  Must accept at least
            a positional argument ``x`` (or ``q, k, v`` if wrapped via
            :class:`SelfAttentionWrapper`).  Any subset of the standard
            kwargs (``freqs_cis``, ``attn_mask``, ``kv_cache``,
            ``cache_key``) will be forwarded if the module accepts them.
        feedforward (nn.Module): FFN / MLP sub-layer.  Accepts ``(x)``
            and returns a tensor of the same shape.
        max_loop_iters (int): Maximum loop depth used to size the FiLM
            embedding table.
        film_hidden (int, optional): FiLM generator hidden dimension.
            Default: ``64``.
        dropout (float, optional): Dropout probability on residual
            branches. Default: ``0.0``.
        use_film (bool, optional): If ``False``, FiLM layers are replaced
            by identity mappings (useful for prelude / coda blocks).
            Default: ``True``.
        norm_eps (float, optional): Epsilon for RMSNorm. Default:
            ``1e-6``.

    Shape:
        - Input: :math:`(B, T, C)`.
        - Output: :math:`(B, T, C)`.

    Examples::

        >>> block = RecurrentTransformerBlock(
        ...     dim=512,
        ...     attention=SelfAttentionWrapper(
        ...         nn.MultiheadAttention(512, 8, batch_first=True)
        ... ),
        ...     feedforward=nn.Sequential(
        ...         nn.Linear(512, 2048), nn.GELU(), nn.Linear(2048, 512)
        ...     ),
        ...     max_loop_iters=8,
        ... )
        >>> x = torch.randn(2, 128, 512)
        >>> out = block(x, freqs_cis=None, loop_t=3)

    .. note::
        ``loop_t`` is consumed by this block (for FiLM conditioning) and
        is **never** forwarded to sub-layers.  It is ignored when
        ``use_film=False``.
    """

    def __init__(
        self,
        dim: int,
        attention: nn.Module,
        feedforward: nn.Module,
        max_loop_iters: int,
        film_hidden: int = 64,
        dropout: float = 0.0,
        use_film: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.use_film = use_film
        self.attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.attn = attention
        self.mlp = feedforward
        self.resid_drop = nn.Dropout(dropout)

        # Introspect sub-layer signatures once at construction time.
        self._attn_accepted = _accepted_kwargs(attention)
        self._mlp_accepted = _accepted_kwargs(feedforward)

        if use_film:
            self.film_attn = DepthFiLM(dim, max_loop_iters, film_hidden)
            self.film_mlp = DepthFiLM(dim, max_loop_iters, film_hidden)
        else:
            self.film_attn = None
            self.film_mlp = None

    def forward(
        self,
        x: torch.Tensor,
        loop_t: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        r"""Execute one forward pass through the block.

        DepthFiLM is applied to the attention and FFN outputs when
        :attr:`use_film` is ``True``, using ``loop_t`` as the depth
        index.

        Args:
            x (Tensor): Input of shape :math:`(B, T, C)`.
            loop_t (int, optional): Current loop iteration index.
                Default: ``0``.
            **kwargs: Arbitrary keyword arguments forwarded selectively
                to each sub-layer based on its ``forward`` signature.
                Common examples include:

                - ``freqs_cis`` — precomputed RoPE frequencies.
                - ``attn_mask`` — attention mask tensor.
                - ``kv_cache`` — key-value cache dict.
                - ``cache_key`` — unique string for cache slotting.

                Arguments not accepted by a sub-layer are silently
                dropped; no ``TypeError`` is raised.

        Returns:
            Tensor: Output of shape :math:`(B, T, C)`.
        """
        # ---- Attention branch ----
        h = self.attn_norm(x)
        attn_kw = _filter_kwargs(self._attn_accepted, kwargs)
        h = self.attn(h, **attn_kw)
        # Handle modules that return tuples (e.g. nn.MultiheadAttention).
        if isinstance(h, tuple):
            h = h[0]
        if self.film_attn is not None:
            h = self.film_attn(h, loop_t)
        x = x + self.resid_drop(h)

        # ---- FFN branch ----
        h = self.ffn_norm(x)
        mlp_kw = _filter_kwargs(self._mlp_accepted, kwargs)
        h = self.mlp(h, **mlp_kw)
        if isinstance(h, tuple):
            h = h[0]
        if self.film_mlp is not None:
            h = self.film_mlp(h, loop_t)
        x = x + self.resid_drop(h)
        return x


# ---------------------------------------------------------------------------
# RecurrentBlock (the shell)
# ---------------------------------------------------------------------------


class RecurrentBlock(nn.Module):
    r"""Recurrent shell that loops a core block up to ``max_loop_iters`` times.

    This module forms the **recurrent depth** abstraction.  It takes a
    single reusable block (usually a :class:`RecurrentTransformerBlock`)
    and iteratively refines a hidden state by feeding the block's output
    back into itself.  At every loop step the block receives:

    1. A sinusoidal *depth-position* signal (via
       :func:`loop_index_embedding`) added to the hidden state.
    2. A frozen encoded input :math:`e` (typically the prelude output)
       mixed in via RMSNorm.
    3. The current loop index :math:`t` forwarded to internal
       :class:`DepthFiLM` layers.

    The state transition is stabilised by :class:`LTIInjection` to
    guarantee contractive dynamics regardless of loop depth.

    An :class:`ACTHalting` mechanism produces a per-position weighted sum
    of hidden states across iterations.  Positions that converge early
    stop accumulating updates; positions still being refined continue to
    receive computation.  This enables variable compute per token within
    a single batch.

    The number of loop iterations can be overridden at **inference**
    time via the ``n_loops`` argument (depth extrapolation).  During
    training the default ``max_loop_iters`` is used.

    Args:
        dim (int): Hidden dimension.
        core_block (nn.Module): The block to loop.  Must accept ``x`` as
            the first positional argument and ``loop_t`` as a keyword
            argument.  All other keyword arguments (e.g. ``freqs_cis``,
            ``attn_mask``, ``kv_cache``, ``cache_key``) are forwarded
            verbatim from the shell's ``forward`` call.
        max_loop_iters (int): Default number of loop iterations.
        act_threshold (float, optional): Cumulative halting probability
            at which a position stops receiving updates.  Default:
            ``0.99``.
        use_lti (bool, optional): If ``True``, apply :class:`LTIInjection`
            to stabilise the recurrent state transition.  Default: ``True``.
        loop_dim_fraction (int, optional): Denominator used to compute the
            loop-index sinusoid dimension: ``loop_dim = dim //
            loop_dim_fraction``.  Default: ``8``.
        loop_theta (float, optional): Base frequency for the sinusoidal
            depth embedding.  Default: ``10000.0``.
        norm_eps (float, optional): Epsilon for the internal RMSNorm.
            Default: ``1e-6``.

    Shape:
        - ``h``: :math:`(B, T, C)` — initial hidden state from prelude.
        - ``e``: :math:`(B, T, C)` — frozen encoded input (prelude output).
        - Output: :math:`(B, T, C)` — ACT-weighted sum of hidden states.

    Examples::

        >>> core = RecurrentTransformerBlock(
        ...     dim=512,
        ...     attention=SelfAttentionWrapper(
        ...         nn.MultiheadAttention(512, 8, batch_first=True)
        ...     ),
        ...     feedforward=nn.Sequential(
        ...         nn.Linear(512, 2048),
        ...         nn.GELU(),
        ...         nn.Linear(2048, 512),
        ...     ),
        ...     max_loop_iters=4,
        ... )
        >>> shell = RecurrentBlock(
        ...     dim=512,
        ...     core_block=core,
        ...     max_loop_iters=4,
        ... )
        >>> h = torch.randn(2, 128, 512)
        >>> e = h.clone()
        >>> out = shell(h, e, n_loops=4)

    .. warning::
        Depth extrapolation (``n_loops > max_loop_iters`` at inference)
        is supported only up to the clamping behaviour of the internal
        :class:`DepthFiLM` embedding table.  Beyond ``max_loop_iters-1``
        the table reuses its last entry; the loop-index embedding
        (sinusoidal) can extrapolate indefinitely.
    """

    def __init__(
        self,
        dim: int,
        core_block: nn.Module,
        max_loop_iters: int,
        act_threshold: float = 0.99,
        use_lti: bool = True,
        loop_dim_fraction: int = 8,
        loop_theta: float = 10000.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.max_loop_iters = max_loop_iters
        self.act_threshold = act_threshold
        self.core_block = core_block
        self.use_lti = use_lti
        self.norm = nn.RMSNorm(dim, eps=norm_eps)
        self.loop_dim = dim // loop_dim_fraction
        self.loop_theta = loop_theta

        self.act = ACTHalting(dim)

        if use_lti:
            self.injection = LTIInjection(dim)
        else:
            self.injection = None

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Loop the core block up to ``n_loops`` times with ACT halting.

        The output is a weighted sum of hidden states across iterations,
        where the per-position weights come from the ACT mechanism.
        Positions whose cumulative halting probability exceeds
        :attr:`act_threshold` stop accumulating updates.

        The number of iterations is controlled by ``n_loops``; when
        ``None`` the constructor default ``max_loop_iters`` is used.
        This enables *depth extrapolation* at inference time: you may
        train with, for example, 4 loops and decode with 8 by passing
        ``n_loops=8``.

        Args:
            h (Tensor): Initial hidden state, shape :math:`(B, T, C)`.
            e (Tensor): Frozen encoded input (prelude output), same shape.
            n_loops (int, optional): Number of iterations.  If ``None``,
                defaults to :attr:`max_loop_iters`.
            kv_cache (dict, optional): Key-value cache forwarded to core
                block.  When present, early exit is disabled so that
                every loop depth populates its cache slot.
            **kwargs: Additional keyword arguments forwarded to the core
                block (e.g. ``freqs_cis``, ``attn_mask``).

        Returns:
            Tensor: ACT-weighted sum of hidden states, shape
            :math:`(B, T, C)`.
        """
        n_loops = n_loops if n_loops is not None else self.max_loop_iters
        B, T, D = h.shape

        # kv_cache should never appear in kwargs
        # Since Someone could subclass or wrap the call differently,
        # apply this fix defensively
        kwargs.pop("kv_cache", None)

        # ACT bookkeeping
        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            # 1. Inject loop-depth positional signal into hidden state
            h_loop = loop_index_embedding(h, t, self.loop_dim, self.loop_theta)

            # 2. Mix with frozen prelude input and normalise
            combined = self.norm(h_loop + e)

            # 3. Run the core block (internal FiLM sees loop_t=t)
            cache_key = f"recurrent_loop_{t}"
            trans_out = self.core_block(
                combined,
                loop_t=t,
                kv_cache=kv_cache,
                cache_key=cache_key,
                **kwargs,
            )

            # 4. Stabilised state update
            if self.injection is not None:
                h = self.injection(h, e, trans_out)
            else:
                h = trans_out

            # 5. ACT halting
            p = self.act(h)  # (B, T)
            still_running = ~halted

            # ACT remainder trick: once cumulative_p + p crosses
            # threshold, assign the remaining probability mass as the
            # final weight.  Gate by still_running so halted positions
            # contribute exactly once (on the halting step) and zero
            # thereafter.
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.act_threshold)

            # Short-circuit only when there is no KV cache to keep
            # consistent.  With a cache, every loop depth must run so
            # later decode steps find populated keys at every cache_key.
            if halted.all() and kv_cache is None:
                break

        return h_out
