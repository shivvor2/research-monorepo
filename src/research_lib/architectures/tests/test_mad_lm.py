"""
Tests for :class:`~research_lib.architectures.mad_lm.MadLM`.

Covers: the backbone structure (block order, LayerNorm placement, head
with bias, no position embedding), forward shapes and causality for every
layout and for an attention mixer, parameter-count equivalence across the
gate modes and the λ-only difference across layouts, the MAD backbone
initialisation (embedding std, zeroed head bias, mixer init kept), the
two weight-decay groups, and the equivalence of the answer-position head
with the full head on a MAD training batch in fp64.

Run with: pytest src/research_lib/architectures/tests/test_mad_lm.py -v
"""

import torch
import torch.nn.functional as F
from torch import nn

from research_lib.layers.attention import CausalSelfAttention, MultistateGatedDelta
from research_lib.training.losses import answer_cross_entropy, answer_position_logits
from research_lib.training.param_utils import partition_parameters_with_names

from ..config.mad_lm import MadLMConfig
from ..mad_lm import MadLM

# The same vector-side patterns the project's module builder uses: the
# token embedding, the untied head, every norm, the gated-delta decay
# parameters and λ, convolution filters, and every bias.
VECTOR_TARGET_MODULES = [
    "embedding",
    "lm_head",
    r".*norm.*",
    "A_log",
    "dt_bias",
    "lambda_mixer",
    "conv",
    r".*\.bias$",
]


def make_mixer(layout="single", gate="full", lam="mlp", seed=0):
    if seed is not None:
        torch.manual_seed(seed)
    return MultistateGatedDelta(
        hidden_size=32,
        num_heads=2,
        head_dim=16,
        expand_v=1.0,
        gate_mode=gate,
        layout=layout,
        lambda_mode=lam,
        window=8,
        t_max=64,
        mode="chunk",
        chunk_size=16,
        output_gate_activation="silu",
    )


def make_model(layout="single", gate="full", lam="mlp", mixers=None, **cfg_kwargs):
    cfg = MadLMConfig(vocab_size=16, d_model=32, **cfg_kwargs)
    mixers = (
        mixers
        if mixers is not None
        else [make_mixer(layout, gate, lam) for _ in range(2)]
    )
    return MadLM(cfg, mixers)


def total_params(module):
    return sum(p.numel() for p in module.parameters())


def lambda_params(module):
    return sum(
        p.numel() for name, p in module.named_parameters() if "lambda_mixer" in name
    )


class TestStructure:
    def test_block_order_and_norms(self):
        model = make_model(layout="ws")
        blk = model.blocks[0]
        assert isinstance(blk.norm1, nn.LayerNorm)
        assert isinstance(blk.norm2, nn.LayerNorm)
        assert isinstance(blk.swiglu, nn.Module)
        assert isinstance(blk.mixer, MultistateGatedDelta)
        assert isinstance(model.norm_f, nn.LayerNorm)
        assert isinstance(model.lm_head, nn.Linear)
        assert model.lm_head.bias is not None

    def test_no_position_embedding(self):
        model = make_model()
        for name, _ in model.named_parameters():
            assert "position" not in name and "pos_emb" not in name

    def test_forward_shapes(self):
        model = make_model()
        logits = model(torch.zeros(2, 8, dtype=torch.long))
        assert logits.shape == (2, 8, 16)

    def test_forward_shapes_and_causality_attention_mixer(self):
        model = make_model(mixers=[CausalSelfAttention(32, 2, rotary_dim=16)])
        x = torch.randint(0, 16, (2, 12))
        logits = model(x)
        assert logits.shape == (2, 12, 16)
        # Causality: the token at the last position must not change the
        # earlier positions' logits.
        x_altered = x.clone()
        x_altered[:, -1] = (x_altered[:, -1] + 1) % 16
        assert torch.equal(logits[:, :-1], model(x_altered)[:, :-1])

    @torch.no_grad()
    def test_causality_every_layout(self):
        for layout in ("single", "ws", "ll"):
            model = make_model(layout=layout)
            x = torch.randint(0, 16, (1, 16))
            base = model(x)
            x_altered = x.clone()
            x_altered[:, -1] = (x_altered[:, -1] + 1) % 16
            assert torch.equal(base[:, :-1], model(x_altered)[:, :-1]), layout


class TestParameterCounts:
    def test_gate_modes_equal(self):
        counts = {
            gate: total_params(make_model(gate=gate))
            for gate in ("full", "b_only", "w_only", "kda", "gdn")
        }
        assert len(set(counts.values())) == 1

    def test_layouts_differ_only_by_lambda(self):
        base_non_lambda = None
        seen = set()
        for layout in ("single", "ws", "fs", "ll"):
            model = make_model(layout=layout)
            nl = total_params(model) - lambda_params(model)
            if base_non_lambda is None:
                base_non_lambda = nl
            assert nl == base_non_lambda, layout
            seen.add(total_params(model))
        # The λ-free single-state has the fewest parameters, and the
        # layouts with λ strictly add parameters.
        single = total_params(make_model(layout="single"))
        ws = total_params(make_model(layout="ws"))
        assert ws > single


class TestInit:
    def test_embedding_std_and_zeroed_head_bias(self):
        model = make_model()
        std = model.embedding.weight.std().item()
        assert 0.01 <= std <= 0.06
        assert torch.all(model.lm_head.bias == 0)

    def test_mixer_keeps_its_own_init(self):
        # Two independently (same-seed) built layers are identical, and
        # building one INTO a MadLM leaves it unchanged.
        a = make_mixer(layout="ws", seed=7)
        b = make_mixer(layout="ws", seed=7)
        model = make_model(mixers=[b])
        model_a = {n: p for n, p in a.named_parameters()}
        for n, p in model.blocks[0].mixer.named_parameters():
            assert torch.equal(p, model_a[n]), n


class TestWeightDecayGroups:
    def test_partitioning(self):
        model = make_model(layout="ws")
        vector_params, matrix_params = partition_parameters_with_names(
            model, VECTOR_TARGET_MODULES
        )
        vector_names = {name for name, _ in vector_params}
        matrix_names = {name for name, _ in matrix_params}
        assert vector_names | matrix_names == {n for n, _ in model.named_parameters()}
        assert not (vector_names & matrix_names)
        for name, param in model.named_parameters():
            is_lambda = "lambda_mixer" in name
            is_vector_kind = (
                "A_log" in name
                or "dt_bias" in name
                or "lambda_mixer" in name
                or "norm" in name
                or name.endswith(".bias")
                or name.startswith(("embedding.", "lm_head."))
                or (param.dim() != 2 and not is_lambda and "conv" in name)
            )
            if is_vector_kind:
                assert name in vector_names, name
            elif param.dim() == 2:
                assert name in matrix_names, name
        assert any("lambda_mixer" in n and ".weight" in n for n in vector_names)


class TestAnswerHeadEquivalence:
    def test_loss_and_gradients_match_full_head(self):
        torch.manual_seed(0)
        model = make_model(layout="ws").double()
        input_ids = torch.randint(0, 16, (2, 12))
        labels = input_ids  # training: every position is a target

        grads_a = self._grads(model, input_ids, labels, answer=True)
        grads_b = self._grads(model, input_ids, labels, answer=False)
        assert grads_a["loss"] == grads_b["loss"]
        for key in grads_a["grads"]:
            assert torch.allclose(
                grads_a["grads"][key], grads_b["grads"][key], atol=1e-10
            ), key

    def _grads(self, model, input_ids, labels, answer):
        params = {n: p for n, p in model.named_parameters()}
        for p in params.values():
            p.grad = None
        if answer:
            hidden = model.hidden_states(input_ids)
            ans = answer_position_logits(hidden, model.head, labels)
            loss = answer_cross_entropy(ans)
        else:
            full = model(input_ids)
            loss = F.cross_entropy(
                full[..., :-1, :].reshape(-1, model.config.vocab_size),
                labels[..., 1:].reshape(-1),
            )
        loss.backward()
        return {
            "loss": float(loss.detach()),
            "grads": {n: p.grad.clone() for n, p in params.items()},
        }
