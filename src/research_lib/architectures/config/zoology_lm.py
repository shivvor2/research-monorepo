"""Configuration dataclass for the Zoology-style two-layer language model."""

from dataclasses import dataclass


@dataclass
class ZoologyLMConfig:
    r"""Configuration for :class:`~research_lib.architectures.zoology_lm.ZoologyLM`.

    The pilot harness (``06_experiment_protocol.md`` §2): a two-layer
    causal LM whose only variable part is the gated-delta mixer, so
    every row of the experiment grid shares one class.

    Args:
        vocab_size: Vocabulary size ``V`` (Zoology: 8192).
        d_model: Model dimension :math:`d`.
        n_layer: Number of pre-norm residual blocks (2 in the pilot).
        layer_pattern: ``"zoology"`` — layer 0 is
            :class:`~research_lib.layers.base_conv.BaseConv`, layer 1 the
            gated-delta mixer (Zoology's ``Hybrid([BaseConv, mixer])`` picks
            by ``layer_idx % 2``); ``"mixer_x2"`` — the gated-delta mixer in
            every block (the calibration alternative).
        state_mixer: ``"none"`` (identity) or ``"mlp"`` (adds
            ``x = x + mlp(norm2(x))`` to every block).
        ff_mult: MLP hidden multiplier (``hidden = ff_mult * d_model``),
            used only when ``state_mixer == "mlp"``.
        base_conv_kernel: Kernel width of the BaseConv layer-0 mixer.
        norm_eps: Epsilon of every RMSNorm.
        num_heads: Gated-delta head count ``H``.
        head_dim: Key head dimension :math:`d_k`; ``None`` means
            ``d_model // num_heads``.
        expand_v: :math:`d_v = d_k \cdot \mathrm{expand\_v}`.
        gate_mode: GDN-2 / Table 5 ablations / KDA / GDN
            (``full`` / ``b_only`` / ``w_only`` / ``kda`` / ``gdn``).
        layout: Multi-state layout (``single`` / ``ws`` / ``fs`` / ``ll``).
        lambda_mode: λ parameterization (``static`` / ``lin`` / ``mlp``);
            ignored for ``layout="single"``.
        window: WS/FS window ``W`` (slots = ``window + 2``).
        chunk_size: Chunk length ``C`` (WS/FS block size ``B = C``).
        t_max: Longest sequence the model is ever run on (sets the ``ll``
            slot count, ``bitlen(t_max) + 1``).
        output_gate_activation: ``"silu"`` (paper) or ``"sigmoid"`` (fla).
        mixer_mode: Gated-delta forward path (``"chunk"`` / ``"recurrent"``).
        collect_stats: Collect λ / erase statistics during forward passes.

    Raises:
        ValueError: invalid ``layer_pattern``, ``state_mixer`` or
            ``mixer_mode`` values, ``d_model`` not divisible by
            ``num_heads`` when ``head_dim is None``, or ``t_max < 1``.
    """

    vocab_size: int = 8192
    d_model: int = 32
    n_layer: int = 2
    layer_pattern: str = "zoology"
    state_mixer: str = "none"
    ff_mult: int = 4
    base_conv_kernel: int = 3
    norm_eps: float = 1e-5
    # gated-delta mixer
    num_heads: int = 2
    head_dim: int | None = None
    expand_v: float = 1.0
    gate_mode: str = "full"
    layout: str = "single"
    lambda_mode: str = "mlp"
    window: int = 8
    chunk_size: int = 16
    t_max: int = 256
    output_gate_activation: str = "silu"
    mixer_mode: str = "chunk"
    collect_stats: bool = False

    def __post_init__(self) -> None:
        if self.layer_pattern not in ("zoology", "mixer_x2"):
            raise ValueError(
                f"layer_pattern must be 'zoology' or 'mixer_x2', "
                f"got {self.layer_pattern!r}"
            )
        if self.state_mixer not in ("none", "mlp"):
            raise ValueError(
                f"state_mixer must be 'none' or 'mlp', got {self.state_mixer!r}"
            )
        if self.mixer_mode not in ("chunk", "recurrent"):
            raise ValueError(
                f"mixer_mode must be 'chunk' or 'recurrent', got {self.mixer_mode!r}"
            )
        if self.head_dim is None and self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads "
                f"({self.num_heads}) when head_dim is None"
            )
        if self.t_max < 1:
            raise ValueError(f"t_max must be >= 1, got {self.t_max}")
        if self.n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {self.n_layer}")
