"""
Comprehensive test suite for NanoMythos architecture.

Target: src/research_lib/architectures/nano_mythos.py
Run with: pytest src/research_lib/architectures/tests/test_nano_mythos.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from research_lib.architectures.config import NanoMythosConfig
from research_lib.architectures.nano_mythos import (
    AttentionType,
    NanoMythosAttnRes,
    _build_recurrent_block,
    _KDASublayer,
    _MHASublayer,
    _MLPSublayer,
    build_attention_pattern,
)

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TestNanoMythos:
    """Container for shared fixtures."""

    @pytest.fixture
    def tiny_config(self):
        """Minimal configuration for fast CPU-safe testing."""
        return NanoMythosConfig(
            vocab_size=100,
            block_size=32,
            n_embd=64,
            n_head=4,
            ff_dim=128,
            bias=True,
            dropout=0.0,
            padding_idx=0,
            n_prelude_blocks=1,
            n_loop_blocks=1,
            n_coda_blocks=1,
            linear_to_full_ratio=1,
            max_loop_iters=2,
            act_threshold=0.99,
            use_lti=True,
            loop_dim_fraction=8,
            loop_theta=10000.0,
            film_hidden=16,
            attnres_block_size=2,
            kda_head_dim=32,
            kda_expand_v=1.0,
            kda_use_short_conv=True,
            kda_conv_size=4,
            kda_mode="chunk",
        )

    @pytest.fixture
    def model(self, tiny_config):
        torch.manual_seed(42)
        return NanoMythosAttnRes(tiny_config)


# ---------------------------------------------------------------------------
# 1. build_attention_pattern
# ---------------------------------------------------------------------------


class TestBuildAttentionPattern:
    """Unit tests for attention-pattern generator."""

    def test_docstring_example_6_2(self):
        assert build_attention_pattern(6, 2) == [
            AttentionType.KDA,
            AttentionType.KDA,
            AttentionType.MHA,
            AttentionType.KDA,
            AttentionType.KDA,
            AttentionType.MHA,
        ]

    def test_docstring_example_5_2(self):
        assert build_attention_pattern(5, 2) == [
            AttentionType.KDA,
            AttentionType.KDA,
            AttentionType.MHA,
            AttentionType.KDA,
            AttentionType.MHA,
        ]

    def test_docstring_example_4_1(self):
        assert build_attention_pattern(4, 1) == [
            AttentionType.KDA,
            AttentionType.MHA,
            AttentionType.KDA,
            AttentionType.MHA,
        ]

    def test_zero_blocks(self):
        assert build_attention_pattern(0, 2) == []

    def test_all_mha_ratio_zero(self):
        assert build_attention_pattern(3, 0) == [
            AttentionType.MHA,
            AttentionType.MHA,
            AttentionType.MHA,
        ]

    def test_incomplete_period_forces_final_mha(self):
        # period = 3; 7 blocks => [KDA,KDA,MHA,KDA,KDA,MHA,KDA] -> last forced to MHA
        pattern = build_attention_pattern(7, 2)
        assert pattern[-1] is AttentionType.MHA
        assert pattern.count(AttentionType.MHA) >= 2


# ---------------------------------------------------------------------------
# 2. Sublayers
# ---------------------------------------------------------------------------


class TestSublayers(TestNanoMythos):
    """Tests for KDASublayer, MHASublayer, and MLPSublayer."""

    def test_mha_sublayer_forward_shape(self, tiny_config):
        layer = _MHASublayer(tiny_config)
        h = torch.randn(2, 8, tiny_config.n_embd)
        out = layer(h)
        assert out.shape == (2, 8, tiny_config.n_embd)
        assert not torch.isnan(out).any()

    def test_mha_sublayer_uses_causal_mask(self, tiny_config):
        """MHASublayer must respect causal masking via is_causal=True."""
        layer = _MHASublayer(tiny_config)
        layer.eval()
        h = torch.randn(1, 10, tiny_config.n_embd)
        with torch.no_grad():
            out1 = layer(h)
            h_mod = h.clone()
            h_mod[:, -1, :] += 10.0
            out2 = layer(h_mod)
        # Causal: positions 0..T-2 should be unchanged
        assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)
        # Last position should change
        assert not torch.allclose(out1[:, -1, :], out2[:, -1, :])

    def test_mlp_sublayer_forward_shape(self, tiny_config):
        layer = _MLPSublayer(tiny_config)
        h = torch.randn(2, 8, tiny_config.n_embd)
        out = layer(h)
        assert out.shape == (2, 8, tiny_config.n_embd)
        assert not torch.isnan(out).any()

    def test_mlp_sublayer_no_residual(self, tiny_config):
        """MLPSublayer is residual-free: zero input -> non-zero output."""
        layer = _MLPSublayer(tiny_config)
        h = torch.zeros(1, 4, tiny_config.n_embd)
        out = layer(h)
        # FeedForward has bias + activation, so output should not be zero
        assert out.abs().sum() > 0

    def test_kda_sublayer_init(self, tiny_config):
        layer = _KDASublayer(tiny_config, layer_idx=0)
        assert isinstance(layer.norm, nn.RMSNorm)
        assert hasattr(layer.attn, "forward")

    @CUDA_ONLY
    def test_kda_sublayer_forward_shape(self, tiny_config):
        layer = _KDASublayer(tiny_config, layer_idx=0).cuda()
        h = torch.randn(2, 8, tiny_config.n_embd, device="cuda")
        out = layer(h)
        assert out.shape == (2, 8, tiny_config.n_embd)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# 3. Recurrent block builder
# ---------------------------------------------------------------------------


class TestRecurrentBlockBuilder(TestNanoMythos):
    """Tests for _build_recurrent_block factory."""

    def test_build_mha_block_init(self, tiny_config):
        block = _build_recurrent_block(tiny_config, AttentionType.MHA, layer_idx=0)
        assert hasattr(block, "core_block")
        assert hasattr(block, "injection")
        assert block.use_lti == tiny_config.use_lti

    def test_build_kda_block_init(self, tiny_config):
        block = _build_recurrent_block(tiny_config, AttentionType.KDA, layer_idx=0)
        assert hasattr(block, "core_block")
        assert block.use_lti == tiny_config.use_lti

    def test_mha_block_forward_cpu(self, tiny_config):
        block = _build_recurrent_block(tiny_config, AttentionType.MHA, layer_idx=0)
        h = torch.randn(2, 4, tiny_config.n_embd)
        e = torch.randn(2, 4, tiny_config.n_embd)
        out = block(h, e, n_loops=2)
        assert out.shape == (2, 4, tiny_config.n_embd)
        assert not torch.isnan(out).any()

    @CUDA_ONLY
    def test_kda_block_forward_cuda(self, tiny_config):
        block = _build_recurrent_block(
            tiny_config, AttentionType.KDA, layer_idx=0
        ).cuda()
        h = torch.randn(2, 4, tiny_config.n_embd, device="cuda")
        e = torch.randn(2, 4, tiny_config.n_embd, device="cuda")
        out = block(h, e, n_loops=2)
        assert out.shape == (2, 4, tiny_config.n_embd)
        assert not torch.isnan(out).any()

    def test_mha_block_loop_count_override(self, tiny_config):
        block = _build_recurrent_block(tiny_config, AttentionType.MHA, layer_idx=0)
        h = torch.randn(1, 2, tiny_config.n_embd)
        e = torch.randn(1, 2, tiny_config.n_embd)
        # Disable LTI so we can compare outputs more directly
        block.use_lti = False
        block.injection = None
        out_1 = block(h, e, n_loops=1)
        out_3 = block(h, e, n_loops=3)
        assert out_1.shape == out_3.shape
        # More loops should generally produce different outputs
        assert not torch.equal(out_1, out_3)


# ---------------------------------------------------------------------------
# 4. NanoMythosAttnRes — Initialization
# ---------------------------------------------------------------------------


class TestNanoMythosAttnResInit(TestNanoMythos):
    """CPU-safe initialization and structural tests."""

    def test_initialization(self, model, tiny_config):
        assert isinstance(model, NanoMythosAttnRes)
        assert model.embedding.num_embeddings == tiny_config.vocab_size
        assert model.embedding.embedding_dim == tiny_config.n_embd
        assert model.output.in_features == tiny_config.n_embd
        assert model.output.out_features == tiny_config.vocab_size

    def test_total_params(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert total_params > 10_000  # rough lower bound for even tiny config

    def test_pseudo_queries_shape_and_init(self, model, tiny_config):
        n_depth = tiny_config.n_attnres_depth_layers
        assert model.pseudo_queries.shape == (n_depth, tiny_config.n_embd)
        assert torch.all(model.pseudo_queries == 0.0)

    def test_attnres_depth_layer_count(self, tiny_config):
        expected = (
            tiny_config.n_prelude_blocks * 2
            + tiny_config.n_loop_blocks
            + tiny_config.n_coda_blocks * 2
        )
        assert tiny_config.n_attnres_depth_layers == expected

    def test_effective_attnres_block_size_explicit(self, tiny_config):
        tiny_config.attnres_block_size = 7
        assert tiny_config.effective_attnres_block_size == 7

    def test_effective_attnres_block_size_auto(self, tiny_config):
        tiny_config.attnres_block_size = None
        # 1*2 + 1 + 1*2 = 5 -> max(1, 5//8) = 1
        assert tiny_config.effective_attnres_block_size == 1

    def test_layer_counts_match_config(self, model, tiny_config):
        assert len(model.prelude_layers) == tiny_config.n_prelude_blocks * 2
        assert len(model.loop_blocks) == tiny_config.n_loop_blocks
        assert len(model.coda_layers) == tiny_config.n_coda_blocks * 2

    def test_initialization_empty_prelude(self, tiny_config):
        tiny_config.n_prelude_blocks = 0
        model = NanoMythosAttnRes(tiny_config)
        assert len(model.prelude_layers) == 0
        assert len(model.loop_blocks) == tiny_config.n_loop_blocks

    def test_initialization_empty_loop(self, tiny_config):
        tiny_config.n_loop_blocks = 0
        model = NanoMythosAttnRes(tiny_config)
        assert len(model.loop_blocks) == 0
        assert len(model.prelude_layers) == tiny_config.n_prelude_blocks * 2

    def test_initialization_empty_coda(self, tiny_config):
        tiny_config.n_coda_blocks = 0
        model = NanoMythosAttnRes(tiny_config)
        assert len(model.coda_layers) == 0

    def test_logit_clipping_present(self, model):
        from research_lib.layers.logit_clipping import TanhSoftCapping

        assert isinstance(model.logit_clipping, TanhSoftCapping)

    def test_norm_f_is_rmsnorm(self, model):
        assert isinstance(model.norm_f, nn.RMSNorm)


# ---------------------------------------------------------------------------
# 5. NanoMythosAttnRes — Forward Pass
# ---------------------------------------------------------------------------


class TestNanoMythosAttnResForward(TestNanoMythos):
    """Forward-pass tests requiring CUDA."""

    def _make_cuda_model(self, tiny_config):
        """Helper: build model on CUDA (float32)."""
        torch.manual_seed(42)
        return NanoMythosAttnRes(tiny_config).cuda()

    @CUDA_ONLY
    def test_forward_pass_float32(self, tiny_config):
        model = self._make_cuda_model(tiny_config)
        batch_size, seq_len = 4, tiny_config.block_size
        x = torch.randint(
            0, tiny_config.vocab_size, (batch_size, seq_len), device="cuda"
        )
        output = model(x)
        assert output.shape == (batch_size, seq_len, tiny_config.vocab_size)
        assert not torch.isnan(output).any(), "Output contains NaNs"
        assert not torch.isinf(output).any(), "Output contains Infs"

    @CUDA_ONLY
    def test_forward_pass_amp_bf16(self, tiny_config):
        """bfloat16 via AMP works because flash_attn_res casts its own I/O.

        Casting the *model* to bf16 breaks torch.utils.checkpoint (the
        sublayer tensors inside the checkpointed lambda revert to fp32).
        AMP keeps weights in fp32 and lets the individual layers cast,
        which matches how the model is trained in practice.
        """
        model = self._make_cuda_model(tiny_config)
        x = torch.randint(0, tiny_config.vocab_size, (2, 8), device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(x)
        assert output.shape == (2, 8, tiny_config.vocab_size)
        # AMP bfloat16 forward keeps the logits in bfloat16 inside the
        # autocast region; the output head may be cast back.
        assert not torch.isnan(output).any()

    @CUDA_ONLY
    def test_variable_sequence_length(self, tiny_config):
        model = self._make_cuda_model(tiny_config)
        x = torch.randint(0, tiny_config.vocab_size, (2, 10), device="cuda")
        output = model(x)
        assert output.shape == (2, 10, tiny_config.vocab_size)

    @CUDA_ONLY
    def test_n_loops_override(self, tiny_config):
        model = self._make_cuda_model(tiny_config)
        model.eval()
        x = torch.randint(0, tiny_config.vocab_size, (1, 8), device="cuda")
        with torch.no_grad():
            out_default = model(x)
            out_1 = model(x, n_loops=1)
            out_5 = model(x, n_loops=5)
        assert not torch.equal(out_default, out_1)
        assert not torch.equal(out_default, out_5)

    @CUDA_ONLY
    def test_causal_masking(self, tiny_config):
        """Prediction at t must not depend on tokens > t."""
        model = self._make_cuda_model(tiny_config)
        model.eval()
        B, T = 2, 10
        x = torch.randint(0, tiny_config.vocab_size, (B, T), device="cuda")
        with torch.no_grad():
            out1 = model(x)
            x_mod = x.clone()
            x_mod[:, -1] = (x_mod[:, -1] + 1) % tiny_config.vocab_size
            out2 = model(x_mod)

        assert torch.allclose(
            out1[:, :-1, :], out2[:, :-1, :], atol=1e-5
        ), "Causality violation: changing future token affected past predictions."
        assert not torch.allclose(
            out1[:, -1, :], out2[:, -1, :], atol=1e-5
        ), "Sanity check failed: changing input did not change output."

    @CUDA_ONLY
    def test_key_padding_mask_semantics(self, tiny_config):
        """Padded tokens must not affect logits of non-padded tokens.

        We create two inputs that differ only at padded positions, pass an
        explicit key_padding_mask (True=pad), and assert that logits at
        non-padded positions are identical. This exercises both the KDA
        attention_mask inversion and the MHA key_padding_mask plumbing.
        """
        model = self._make_cuda_model(tiny_config)
        model.eval()
        B, T = 2, 12
        # Positions 0-5 are real, positions 6-11 are padded
        real_len = 6
        x1 = torch.randint(0, tiny_config.vocab_size, (B, T), device="cuda")
        x2 = x1.clone()
        # Change token IDs only at padded positions
        x2[:, real_len:] = (x2[:, real_len:] + 1) % tiny_config.vocab_size

        # PyTorch convention: True = pad, False = real token
        key_padding_mask = torch.zeros((B, T), dtype=torch.bool, device="cuda")
        key_padding_mask[:, real_len:] = True

        with torch.no_grad():
            out1 = model(x1, key_padding_mask=key_padding_mask)
            out2 = model(x2, key_padding_mask=key_padding_mask)

        # Non-padded positions should see identical logits regardless of
        # what token sits at padded positions, because the mask prevents
        # attention to those keys in both MHA and KDA paths.
        assert torch.allclose(
            out1[:, :real_len, :],
            out2[:, :real_len, :],
            atol=1e-5,
        ), "key_padding_mask failed: padded tokens leaked into non-padded logits."

        # Sanity check: padded positions themselves can differ because their
        # own input embeddings changed (even if they cannot attend to padded
        # keys, their query embedding is different).
        assert not torch.allclose(
            out1[:, real_len:, :],
            out2[:, real_len:, :],
            atol=1e-5,
        ), "Sanity check: changing padded tokens had no effect anywhere."

    @CUDA_ONLY
    def test_logit_clipping(self, tiny_config):
        model = self._make_cuda_model(tiny_config)
        with torch.no_grad():
            model.output.weight.data *= 10.0
        x = torch.randint(0, tiny_config.vocab_size, (1, 5), device="cuda")
        logits = model(x)
        cap = model.logit_clipping.soft_cap_value
        assert (
            logits.abs().max() < cap + 1.0
        ), f"Logit soft capping failed: max abs logit {logits.abs().max().item()}"

    @CUDA_ONLY
    def test_padding_idx_behavior(self, tiny_config):
        model = self._make_cuda_model(tiny_config)
        x = torch.zeros((1, 5), dtype=torch.long, device="cuda")
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert model.embedding.weight.grad is not None
        if tiny_config.padding_idx is not None:
            assert (
                model.embedding.weight.grad[tiny_config.padding_idx].sum() == 0.0
            ), "Padding index received non-zero gradient."


# ---------------------------------------------------------------------------
# 6. NanoMythosAttnRes — Training / Optimization
# ---------------------------------------------------------------------------


class TestNanoMythosTraining(TestNanoMythos):
    """Gradient and convergence tests requiring CUDA."""

    def _make_cuda_model(self, tiny_config):
        torch.manual_seed(42)
        return NanoMythosAttnRes(tiny_config).cuda()

    @CUDA_ONLY
    def test_gradient_propagation_float32(self, tiny_config):
        model = self._make_cuda_model(tiny_config)
        model.train()
        x = torch.randint(0, tiny_config.vocab_size, (2, 8), device="cuda")
        targets = torch.randint(0, tiny_config.vocab_size, (2, 8), device="cuda")

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, tiny_config.vocab_size), targets.view(-1)
        )
        loss.backward()

        assert model.embedding.weight.grad is not None
        assert model.output.weight.grad is not None
        recurrent_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.loop_blocks.parameters()
        )
        assert recurrent_has_grad, "No gradients in loop blocks."
        assert model.prelude_layers[0].norm.weight.grad is not None

    @CUDA_ONLY
    def test_overfit_single_batch_float32(self, tiny_config):
        model = self._make_cuda_model(tiny_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model.train()

        x = torch.randint(0, tiny_config.vocab_size, (4, 8), device="cuda")
        y = x.clone()

        initial_loss = None
        for step in range(50):
            optimizer.zero_grad()
            logits = model(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert (
            final_loss < initial_loss
        ), f"Model failed to learn: init {initial_loss:.4f} -> final {final_loss:.4f}"
        assert (
            final_loss < 2.0
        ), f"Model failed to converge on trivial task: {final_loss:.4f}"

    @CUDA_ONLY
    def test_gradient_propagation_amp_bf16(self, tiny_config):
        model = self._make_cuda_model(tiny_config)
        model.train()
        x = torch.randint(0, tiny_config.vocab_size, (2, 8), device="cuda")
        targets = torch.randint(0, tiny_config.vocab_size, (2, 8), device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, tiny_config.vocab_size).float(), targets.view(-1)
        )
        loss.backward()
        assert model.embedding.weight.grad is not None
        assert model.output.weight.grad is not None


# ---------------------------------------------------------------------------
# 7. System Tests
# ---------------------------------------------------------------------------


class TestNanoMythosSystem(TestNanoMythos):
    """GPU and compilation compatibility."""

    def _make_cuda_model(self, tiny_config):
        torch.manual_seed(42)
        return NanoMythosAttnRes(tiny_config).cuda()

    @CUDA_ONLY
    def test_cuda_compatibility_float32(self, tiny_config):
        model = self._make_cuda_model(tiny_config)
        x = torch.randint(0, tiny_config.vocab_size, (2, 8), device="cuda")
        output = model(x)
        assert output.device.type == "cuda"
        assert output.shape == (2, 8, tiny_config.vocab_size)

    @CUDA_ONLY
    def test_torch_compile_float32(self, tiny_config):
        torch._dynamo.reset()
        model = self._make_cuda_model(tiny_config)
        model = torch.compile(model)
        x = torch.randint(0, tiny_config.vocab_size, (2, 8), device="cuda")
        try:
            output = model(x)
            assert output.shape == (2, 8, tiny_config.vocab_size)
        except Exception as e:
            pytest.fail(f"torch.compile failed: {e}")
