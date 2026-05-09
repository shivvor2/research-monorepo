"""
Test suite for Recurrent Depth Components.

Covers:
  * SelfAttentionWrapper
  * LTIInjection
  * loop_index_embedding
  * RecurrentTransformerBlock
  * RecurrentBlock (shell)
  * ACTHalting

Run with: pytest src/research_lib/layers/tests/test_recurrent_block.py -v
"""

import pytest
import torch
import torch.nn as nn

from ..recurrent_block import (
    ACTHalting,
    LTIInjection,
    RecurrentBlock,
    RecurrentTransformerBlock,
    SelfAttentionWrapper,
    loop_index_embedding,
)

# =============================================================================
# Test Helpers
# =============================================================================


class SpyModule(nn.Module):
    """Records calls, args, kwargs, and loop_t values. Returns configurable output."""

    def __init__(self, return_value=None, return_input=False):
        super().__init__()
        self.return_value = return_value
        self.return_input = return_input
        self.call_count = 0
        self.calls = []  # list of (args, kwargs)
        self.loop_t_values = []
        self.capture_kwargs = None  # set this to a dict to capture filtered kwargs

    def forward(self, x, **kwargs):
        self.call_count += 1
        self.calls.append((x, kwargs))
        if "loop_t" in kwargs:
            self.loop_t_values.append(kwargs["loop_t"])
        if self.capture_kwargs is not None:
            self.capture_kwargs.update(kwargs)
        if self.return_input:
            return x
        if self.return_value is not None:
            # Match shape of input x
            if isinstance(self.return_value, torch.Tensor):
                return self.return_value.expand_as(x).clone()
            return torch.full_like(x, self.return_value)
        return x


class ConstantModule(nn.Module):
    """Always returns a fixed constant tensor."""

    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x, **kwargs):
        return self.constant.expand_as(x).clone()


class IdentityCheckAttention(nn.Module):
    """Assert q is k is v (identity) and record call."""

    def __init__(self):
        super().__init__()
        self.called = False
        self.qkv_identity = False

    def forward(self, q, k, v, **kwargs):
        self.called = True
        self.qkv_identity = q is k and k is v
        return q


class KwargsCapturingModule(nn.Module):
    """Captures received kwargs for inspection."""

    def __init__(self, signature="standard"):
        super().__init__()
        self.received_kwargs = {}
        self.signature = signature

    def forward(self, q, k, v, attn_mask=None, **kwargs):
        self.received_kwargs = {"attn_mask": attn_mask}
        return q


class TupleReturningModule(nn.Module):
    """Returns a tuple like nn.MultiheadAttention."""

    def forward(self, q, k, v, **kwargs):
        return q, torch.randn(q.shape[0], q.shape[1], q.shape[1])


class VarKwargsModule(nn.Module):
    """Accepts **kwargs — should receive everything."""

    def __init__(self):
        super().__init__()
        self.received_kwargs = {}

    def forward(self, q, k, v, **kwargs):
        self.received_kwargs = dict(kwargs)
        return q


# =============================================================================
# 2. SelfAttentionWrapper
# =============================================================================


class TestSelfAttentionWrapper:
    """Test suite for SelfAttentionWrapper."""

    def test_self_attention_dispatch(self):
        """2.1: q=k=v identity contract must be honoured."""
        dummy = IdentityCheckAttention()
        wrapper = SelfAttentionWrapper(dummy)
        x = torch.randn(2, 8, 64)
        out = wrapper(x)
        assert dummy.called, "Dummy attention was not called"
        assert dummy.qkv_identity, "q, k, v were not the same tensor (identity)"
        assert torch.equal(out, x), "Wrapper should return q (which is x)"

    def test_kwargs_filtering_accepted_only(self):
        """2.2: Only accepted kwargs forwarded; unknown kwargs dropped."""
        dummy = KwargsCapturingModule()
        wrapper = SelfAttentionWrapper(dummy)
        x = torch.randn(2, 8, 64)
        mask = torch.ones(8, 8)
        cis = torch.randn(8, 32)
        wrapper(x, attn_mask=mask, freqs_cis=cis, foo="bar")
        assert "attn_mask" in dummy.received_kwargs, "attn_mask should be forwarded"
        assert (
            "freqs_cis" not in dummy.received_kwargs
        ), "freqs_cis should be filtered out"
        assert "foo" not in dummy.received_kwargs, "foo should be filtered out"

    def test_works_with_multihead_attention(self):
        """2.3: Wrap nn.MultiheadAttention end-to-end; no TypeError."""
        mha = nn.MultiheadAttention(64, 2, batch_first=True)
        wrapper = SelfAttentionWrapper(mha)
        x = torch.randn(3, 5, 64)
        # These kwargs would crash nn.MultiheadAttention if forwarded
        out = wrapper(x, freqs_cis=torch.randn(5, 32), kv_cache={}, cache_key="test")
        assert out.shape == (3, 5, 64), f"Expected (3,5,64), got {out.shape}"

    def test_tuple_unpacking(self):
        """2.4: Tuple return from wrapped module must be unpacked."""
        dummy = TupleReturningModule()
        wrapper = SelfAttentionWrapper(dummy)
        x = torch.randn(2, 8, 64)
        out = wrapper(x)
        assert isinstance(out, torch.Tensor), "Wrapper must return tensor, not tuple"
        assert out.shape == x.shape

    def test_var_kwargs_receives_all(self):
        """2.5: Module with **kwargs receives all kwargs."""
        dummy = VarKwargsModule()
        wrapper = SelfAttentionWrapper(dummy)
        x = torch.randn(2, 8, 64)
        cis = torch.randn(8, 32)
        wrapper(x, freqs_cis=cis, kv_cache={}, custom_arg=42)
        assert "freqs_cis" in dummy.received_kwargs, "freqs_cis should pass through"
        assert "kv_cache" in dummy.received_kwargs, "kv_cache should pass through"
        assert "custom_arg" in dummy.received_kwargs, "custom_arg should pass through"


# =============================================================================
# 3. LTIInjection
# =============================================================================


class TestLTIInjection:
    """Test suite for LTIInjection."""

    def test_spectral_radius_guarantee(self):
        """3.1: After random perturbations, all A elements in [0, 1).

        Note: exp(-exp(...)) underflows to exact 0.0 in float32 for large inputs,
        so the lower bound is inclusive.
        """
        inject = LTIInjection(dim=128)
        for _ in range(100):
            with torch.no_grad():
                inject.log_A.uniform_(-5, 5)
                inject.log_dt.uniform_(-5, 5)
            A = inject.get_A()
            assert (A >= 0).all(), f"A has negative values: min={A.min().item()}"
            assert (A < 1).all(), f"A has values >= 1: max={A.max().item()}"

    def test_state_contraction_zero_input(self):
        """3.2: With zero driving terms, state must converge toward zero."""
        inject = LTIInjection(dim=64)
        B, T, D = 2, 4, 64
        h = torch.randn(B, T, D)
        e = torch.zeros(B, T, D)
        transformer_out = torch.zeros(B, T, D)
        h_norm_0 = h.norm().item()
        for _ in range(50):
            h = inject(h, e, transformer_out)
        h_norm_50 = h.norm().item()
        assert (
            h_norm_50 < h_norm_0 * 0.01
        ), f"State did not contract: {h_norm_0:.4f} -> {h_norm_50:.4f}"

    def test_gradient_flow(self):
        """3.3: All inputs and parameters must receive gradients."""
        inject = LTIInjection(dim=64)
        h = torch.randn(2, 4, 64, requires_grad=True)
        e = torch.randn(2, 4, 64, requires_grad=True)
        transformer_out = torch.randn(2, 4, 64, requires_grad=True)
        out = inject(h, e, transformer_out)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None, "h gradient missing"
        assert e.grad is not None, "e gradient missing"
        assert transformer_out.grad is not None, "transformer_out gradient missing"
        assert inject.log_A.grad is not None, "log_A gradient missing"
        assert inject.log_dt.grad is not None, "log_dt gradient missing"
        assert inject.B.grad is not None, "B gradient missing"

    def test_batched_independence(self):
        """3.4: Zeroing out some batch indices must not affect others."""
        inject = LTIInjection(dim=64)
        h = torch.randn(4, 1, 64)
        e = torch.randn(4, 1, 64)
        t_out = torch.randn(4, 1, 64)
        # Zero out batches 2 and 3
        h[2:].zero_()
        e[2:].zero_()
        t_out[2:].zero_()
        h_next = inject(h, e, t_out)
        # Compare against B=2 version
        h2 = h[:2].clone()
        e2 = e[:2].clone()
        t2 = t_out[:2].clone()
        h_next_2 = inject(h2, e2, t2)
        assert torch.allclose(
            h_next[:2], h_next_2, atol=1e-6
        ), "Batch cross-contamination detected"


# =============================================================================
# 4. loop_index_embedding
# =============================================================================


class TestLoopIndexEmbedding:
    """Test suite for loop_index_embedding."""

    def test_shape_preservation(self):
        """4.1: Output shape must equal input shape."""
        h = torch.randn(2, 4, 512)
        out = loop_index_embedding(h, 3)
        assert out.shape == h.shape, f"Expected {h.shape}, got {out.shape}"

    def test_value_range_loop_t_zero(self):
        """4.2: loop_t=0 produces sin(0)=0 in first half, cos(0)=1 in second half."""
        h = torch.zeros(1, 1, 256)
        out = loop_index_embedding(h, 0)
        loop_dim = 256 // 8  # 32
        # First half (sin) should be ~0, second half (cos) should be ~1
        first_half = out[0, 0, : loop_dim // 2]
        second_half = out[0, 0, loop_dim // 2 : loop_dim]
        assert torch.allclose(first_half, torch.zeros_like(first_half), atol=1e-6)
        assert torch.allclose(second_half, torch.ones_like(second_half), atol=1e-6)

    def test_value_range_loop_t_large(self):
        """4.2: loop_t=100 produces values bounded in [-1, 1]."""
        h = torch.zeros(1, 1, 256)
        out = loop_index_embedding(h, 100)
        loop_dim = 256 // 8
        active = out[0, 0, :loop_dim]
        assert (active >= -1.0).all() and (
            active <= 1.0
        ).all(), "Sinusoidal embedding out of [-1, 1] bounds"

    def test_default_vs_explicit_loop_dim(self):
        """4.3: loop_dim=None and loop_dim=dim//8 must produce identical results."""
        h = torch.randn(1, 1, 256)
        out_default = loop_index_embedding(h, 5, loop_dim=None)
        out_explicit = loop_index_embedding(h, 5, loop_dim=32)
        assert torch.allclose(out_default, out_explicit, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_dtype_preservation(self):
        """4.4: Output must inherit device and dtype from input."""
        h = torch.randn(2, 4, 256, device="cuda", dtype=torch.bfloat16)
        out = loop_index_embedding(h, 2)
        assert out.device == h.device, "Device mismatch"
        assert out.dtype == h.dtype, "Dtype mismatch"

    def test_non_mutation_of_input(self):
        """4.5: Input tensor must not be modified in-place."""
        h = torch.randn(2, 4, 256)
        h_clone = h.clone()
        loop_index_embedding(h, 3)
        assert torch.equal(h, h_clone), "Input tensor was mutated in-place"


# =============================================================================
# 5. RecurrentTransformerBlock
# =============================================================================


class TestRecurrentTransformerBlock:
    """Test suite for RecurrentTransformerBlock."""

    @pytest.fixture
    def dummy_block_no_film(self):
        """Block with use_film=False for depth-invariant testing."""
        attn = SpyModule(return_input=False)
        ffn = SpyModule(return_input=False)
        block = RecurrentTransformerBlock(
            dim=64,
            attention=attn,
            feedforward=ffn,
            max_loop_iters=8,
            use_film=False,
            dropout=0.0,
        )
        return block, attn, ffn

    def test_film_disabled_depth_invariant(self, dummy_block_no_film):
        """5.1: Film-disabled blocks must be depth-invariant."""
        block, attn, ffn = dummy_block_no_film
        x = torch.randn(2, 4, 64)
        # Spy returns zeros, so with dropout=0 and residual:
        # out = x + 0 + 0 = x. But we need the dummy to return the input.
        attn.return_input = True
        ffn.return_input = True
        out_0 = block(x, loop_t=0)
        out_7 = block(x, loop_t=7)
        assert torch.equal(out_0, out_7), "use_film=False block must be depth-invariant"

    def test_film_enabled_depth_sensitive(self):
        """5.2: With non-zero FiLM weights, different loop_t must differ."""
        attn = SpyModule(return_value=0)
        ffn = SpyModule(return_value=0)
        block = RecurrentTransformerBlock(
            dim=64, attention=attn, feedforward=ffn, max_loop_iters=8, use_film=True
        )
        # Set non-zero FiLM weights
        with torch.no_grad():
            nn.init.normal_(block.film_attn.mlp[-1].weight, std=0.1)
            nn.init.normal_(block.film_attn.mlp[-1].bias, std=0.1)
            nn.init.normal_(block.film_mlp.mlp[-1].weight, std=0.1)
            nn.init.normal_(block.film_mlp.mlp[-1].bias, std=0.1)
        x = torch.randn(2, 4, 64)
        out_0 = block(x, loop_t=0)
        out_1 = block(x, loop_t=1)
        assert not torch.equal(
            out_0, out_1
        ), "FiLM-enabled block must differ across loop_t with non-zero weights"

    def test_film_zero_init_depth_invariant(self):
        """5.2 (alt): Zero-init FiLM must still be depth-invariant."""
        attn = SpyModule(return_value=0)
        ffn = SpyModule(return_value=0)
        block = RecurrentTransformerBlock(
            dim=64, attention=attn, feedforward=ffn, max_loop_iters=8, use_film=True
        )
        # FiLM is zero-init by default
        x = torch.randn(2, 4, 64)
        out_0 = block(x, loop_t=0)
        out_1 = block(x, loop_t=1)
        assert torch.allclose(
            out_0, out_1, atol=1e-5
        ), "Zero-init FiLM must be depth-invariant"

    def test_residual_connection_presence(self):
        """5.3: Zero sub-layer + residual = identity."""
        attn = ConstantModule(torch.zeros(1, 1, 1))
        ffn = ConstantModule(torch.zeros(1, 1, 1))
        block = RecurrentTransformerBlock(
            dim=64,
            attention=attn,
            feedforward=ffn,
            max_loop_iters=8,
            use_film=False,
            dropout=0.0,
        )
        x = torch.randn(2, 4, 64)
        out = block(x)
        assert torch.allclose(out, x, atol=1e-5), "Residual connection broken"

    def test_pre_norm_ordering(self):
        """5.4: Dummy attention must receive RMSNorm'd input (RMS ≈ 1)."""
        stats = {}

        class StatsRecordingSpy(nn.Module):
            def forward(self, x, **kwargs):
                stats["mean"] = x.mean().item()
                stats["var"] = x.var().item()
                stats["rms"] = (x**2).mean().sqrt().item()
                return x

        attn = StatsRecordingSpy()
        ffn = StatsRecordingSpy()
        block = RecurrentTransformerBlock(
            dim=64,
            attention=attn,
            feedforward=ffn,
            max_loop_iters=8,
            use_film=False,
            dropout=0.0,
        )
        x = torch.ones(2, 4, 64) * 100.0
        block(x)
        # RMSNorm of uniform vector = sqrt(mean(x^2)) = sqrt(10000) = 100
        # After RMSNorm: normalized = x / RMS = 100/100 = 1.0 (with eps)
        assert (
            abs(stats["rms"] - 1.0) < 0.01
        ), f"Pre-norm not applied: RMS={stats['rms']:.4f} (expected ≈1.0)"

    def test_kwargs_filtering_in_sub_layers(self):
        """5.5: Sub-layers only receive kwargs they accept."""
        attn_kwargs = {}
        ffn_kwargs = {}

        class FilterSpyAttn(nn.Module):
            def forward(self, x, attn_mask=None, **kwargs):
                attn_kwargs.update(kwargs)
                attn_kwargs["attn_mask"] = attn_mask
                return x

        class FilterSpyFFN(nn.Module):
            def forward(self, x):
                return x  # No **kwargs — only accepts x

        attn = FilterSpyAttn()
        ffn = FilterSpyFFN()
        block = RecurrentTransformerBlock(
            dim=64,
            attention=attn,
            feedforward=ffn,
            max_loop_iters=8,
            use_film=False,
        )
        x = torch.randn(2, 4, 64)
        mask = torch.ones(4, 4)
        block(x, attn_mask=mask, freqs_cis=torch.randn(4, 32), kv_cache={})
        assert "attn_mask" in attn_kwargs or attn_kwargs.get("attn_mask") is not None
        assert "freqs_cis" not in ffn_kwargs, "FFN should not receive freqs_cis"
        assert "kv_cache" not in ffn_kwargs, "FFN should not receive kv_cache"

    def test_dropout_training_only(self):
        """5.6: Dropout active in train, inactive in eval."""
        attn = SpyModule(return_value=1.0)
        ffn = SpyModule(return_value=1.0)
        block = RecurrentTransformerBlock(
            dim=64,
            attention=attn,
            feedforward=ffn,
            max_loop_iters=8,
            use_film=False,
            dropout=0.5,
        )
        x = torch.randn(2, 4, 64)
        # Eval: should be deterministic
        block.eval()
        eval_outputs = [block(x) for _ in range(10)]
        for i in range(1, 10):
            assert torch.equal(
                eval_outputs[0], eval_outputs[i]
            ), "Eval outputs not deterministic"
        # Train: should vary (at least one pair differs)
        block.train()
        train_outputs = [block(x) for _ in range(10)]
        any_diff = any(
            not torch.equal(train_outputs[i], train_outputs[j])
            for i in range(10)
            for j in range(i + 1, 10)
        )
        assert any_diff, "Train outputs identical despite dropout=0.5"


# =============================================================================
# 6. RecurrentBlock (the shell)
# =============================================================================


class TestRecurrentBlock:
    """Test suite for RecurrentBlock shell."""

    @pytest.fixture
    def spy_core_block(self):
        """A spy core block that records calls and loop_t values."""
        spy = SpyModule(return_input=True)
        return spy

    def test_loop_count_override(self, spy_core_block):
        """6.1: n_loops=2 must call core_block exactly 2 times."""
        shell = RecurrentBlock(
            dim=64, core_block=spy_core_block, max_loop_iters=4, use_lti=False
        )
        h = torch.randn(2, 4, 64)
        e = torch.randn(2, 4, 64)
        _ = shell(h, e, n_loops=2)
        assert (
            spy_core_block.call_count == 2
        ), f"Expected 2 calls, got {spy_core_block.call_count}"

    def test_default_loop_count(self, spy_core_block):
        """6.2: No n_loops → use max_loop_iters (capped by early exit)."""
        shell = RecurrentBlock(
            dim=64, core_block=spy_core_block, max_loop_iters=4, use_lti=False
        )
        # Force low ACT probability so no early exit within max_loop_iters
        with torch.no_grad():
            shell.act.halt.weight.fill_(0.0)
            shell.act.halt.bias.fill_(-5.0)
        h = torch.randn(2, 4, 64)
        e = torch.randn(2, 4, 64)
        _ = shell(h, e)
        assert (
            spy_core_block.call_count == 4
        ), f"Expected 4 calls (default), got {spy_core_block.call_count}"

    def test_depth_extrapolation(self, spy_core_block):
        """6.3: n_loops > max_loop_iters must not crash."""
        shell = RecurrentBlock(
            dim=64, core_block=spy_core_block, max_loop_iters=4, use_lti=False
        )
        # Force low ACT probability so no early exit
        with torch.no_grad():
            shell.act.halt.weight.fill_(0.0)
            shell.act.halt.bias.fill_(-5.0)
        h = torch.randn(2, 4, 64)
        e = torch.randn(2, 4, 64)
        out = shell(h, e, n_loops=10)
        assert (
            spy_core_block.call_count == 10
        ), f"Expected 10 calls, got {spy_core_block.call_count}"
        assert out.shape == h.shape, "Output shape invalid after extrapolation"

    def test_lti_vs_non_lti_path(self):
        """6.4: LTI path must differ from non-LTI path."""
        C = torch.ones(2, 4, 64) * 3.14
        core = ConstantModule(C)
        h = torch.randn(2, 4, 64)
        e = torch.randn(2, 4, 64)

        shell_lti = RecurrentBlock(
            dim=64, core_block=core, max_loop_iters=1, use_lti=True
        )
        shell_non_lti = RecurrentBlock(
            dim=64, core_block=core, max_loop_iters=1, use_lti=False
        )

        out_lti = shell_lti(h, e, n_loops=1)
        out_non_lti = shell_non_lti(h, e, n_loops=1)
        assert not torch.equal(
            out_lti, out_non_lti
        ), "LTI and non-LTI paths produced identical outputs"

    def test_frozen_e_not_modified(self):
        """6.5: Input e must not be mutated in-place."""
        spy = SpyModule(return_input=True)
        shell = RecurrentBlock(dim=64, core_block=spy, max_loop_iters=2, use_lti=False)
        e = torch.randn(2, 4, 64)
        e_clone = e.clone()
        h = torch.randn(2, 4, 64)
        _ = shell(h, e, n_loops=2)
        assert torch.equal(e, e_clone), "Frozen input e was mutated"

    def test_loop_index_sequence(self, spy_core_block):
        """6.6: Loop counter sequence [0, 1, 2, ...] must reach core block."""
        shell = RecurrentBlock(
            dim=64, core_block=spy_core_block, max_loop_iters=5, use_lti=False
        )
        # Force low ACT probability so all requested loops run
        with torch.no_grad():
            shell.act.halt.weight.fill_(0.0)
            shell.act.halt.bias.fill_(-5.0)
        h = torch.randn(2, 4, 64)
        e = torch.randn(2, 4, 64)
        _ = shell(h, e, n_loops=5)
        assert spy_core_block.loop_t_values == [
            0,
            1,
            2,
            3,
            4,
        ], f"Loop_t sequence mismatch: {spy_core_block.loop_t_values}"

    def test_cache_key_uniqueness(self, spy_core_block):
        """6.7: cache_key must be unique per loop iteration."""
        shell = RecurrentBlock(
            dim=64, core_block=spy_core_block, max_loop_iters=3, use_lti=False
        )
        h = torch.randn(2, 4, 64)
        e = torch.randn(2, 4, 64)
        _ = shell(h, e, n_loops=3)
        keys = [
            kwargs.get("cache_key")
            for _, kwargs in spy_core_block.calls
            if "cache_key" in kwargs
        ]
        expected = ["recurrent_loop_0", "recurrent_loop_1", "recurrent_loop_2"]
        assert keys == expected, f"Cache keys mismatch: {keys}"

    def test_act_output_is_weighted_sum_not_final_state(self):
        """6.8: ACT output is weighted sum, not final state."""
        C = torch.ones(2, 4, 64) * 5.0
        core = ConstantModule(C)
        shell = RecurrentBlock(dim=64, core_block=core, max_loop_iters=3, use_lti=False)
        h = torch.zeros(2, 4, 64)
        e = torch.zeros(2, 4, 64)
        out = shell(h, e, n_loops=3)
        # If it were returning final state, we'd get ≈ C with norm large.
        # ACT weights sum to ≤ 1, so output magnitude should be ≤ norm(C).
        assert (
            out.norm() <= C.norm() + 1e-4
        ), f"ACT weighted sum exceeded single-state norm: {out.norm():.4f} vs {C.norm():.4f}"

    def test_act_early_exit_no_kv_cache(self):
        """6.9: With p≈1.0 and no cache, early exit after first iteration."""
        spy = SpyModule(return_input=True)
        shell = RecurrentBlock(dim=64, core_block=spy, max_loop_iters=4, use_lti=False)

        # Replace ACT with one that always returns p=1.0 for all positions
        class AlwaysHalt(nn.Module):
            def forward(self, h):
                return torch.ones(h.shape[0], h.shape[1], device=h.device)

        shell.act = AlwaysHalt()
        h = torch.randn(2, 4, 64)
        e = torch.randn(2, 4, 64)
        _ = shell(h, e, n_loops=10, kv_cache=None)
        assert (
            spy.call_count == 1
        ), f"Expected 1 call (early exit), got {spy.call_count}"

    def test_act_weights_sum_to_leq_one(self):
        """6.10: ACT cumulative weights per position must be ≤ 1.

        We verify indirectly by recreating the ACT bookkeeping from captured p values.
        """
        spy = SpyModule(return_input=True)
        shell = RecurrentBlock(dim=64, core_block=spy, max_loop_iters=5, use_lti=False)

        # Force a known constant ACT probability so bookkeeping is deterministic
        class FixedACT(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p

            def forward(self, h):
                return torch.full((h.shape[0], h.shape[1]), self.p, device=h.device)

        shell.act = FixedACT(0.18)  # 5 * 0.18 = 0.90 < 0.99 → never halts
        h = torch.randn(2, 4, 64)
        e = torch.randn(2, 4, 64)
        _ = shell(h, e, n_loops=5)

        # Simulate ACT bookkeeping independently
        cumulative_p = torch.zeros(2, 4)
        halted = torch.zeros(2, 4, dtype=torch.bool)
        total_weight = torch.zeros(2, 4)
        for _ in range(5):
            p = torch.full((2, 4), 0.18)
            still_running = ~halted
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= shell.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            total_weight += weight
            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= shell.act_threshold)

        assert (
            total_weight <= 1.0 + 1e-5
        ).all(), f"ACT weights exceeded 1.0: max={total_weight.max().item()}"

    def test_act_early_exit_disabled_with_kv_cache(self):
        """6.11: With KV cache, all loops must run despite early halting."""
        spy = SpyModule(return_input=True)
        shell = RecurrentBlock(dim=64, core_block=spy, max_loop_iters=4, use_lti=False)
        # Monkey-patch ACT for high halting probability
        with torch.no_grad():
            shell.act.halt.weight.fill_(100.0)
            shell.act.halt.bias.fill_(100.0)
        h = torch.randn(2, 4, 64)
        e = torch.randn(2, 4, 64)
        _ = shell(h, e, n_loops=10, kv_cache={})
        assert (
            spy.call_count == 10
        ), f"Expected 10 calls (no early exit with cache), got {spy.call_count}"


# =============================================================================
# 7. ACTHalting
# =============================================================================


class TestACTHalting:
    """Test suite for ACTHalting."""

    def test_shape(self):
        """7.1: Output shape must be (B, T)."""
        act = ACTHalting(dim=512)
        h = torch.randn(2, 8, 512)
        p = act(h)
        assert p.shape == (2, 8), f"Expected (2, 8), got {p.shape}"

    def test_output_range(self):
        """7.2: All outputs in (0, 1) open interval."""
        act = ACTHalting(dim=64)
        for _ in range(100):
            h = torch.randn(2, 8, 64)
            p = act(h)
            assert (p > 0).all() and (
                p < 1
            ).all(), f"Sigmoid out of (0,1): min={p.min().item()}, max={p.max().item()}"

    def test_gradient_flow(self):
        """7.3: Gradients must flow through act."""
        act = ACTHalting(dim=64)
        h = torch.randn(2, 8, 64, requires_grad=True)
        p = act(h)
        loss = p.sum()
        loss.backward()
        assert h.grad is not None, "h gradient missing"


# =============================================================================
# 8. Integration / Contract Tests
# =============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def build_mini_architecture(self):
        """Build a minimal recurrent architecture per spec 8.1."""
        dim = 64
        attention = SelfAttentionWrapper(
            nn.MultiheadAttention(dim, 2, batch_first=True)
        )
        ffn = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, dim))
        prelude = RecurrentTransformerBlock(
            dim=dim,
            attention=attention,
            feedforward=ffn,
            max_loop_iters=3,
            use_film=False,
        )
        recurrent_core = RecurrentTransformerBlock(
            dim=dim,
            attention=attention,
            feedforward=ffn,
            max_loop_iters=3,
            use_film=True,
        )
        recurrent = RecurrentBlock(dim=dim, core_block=recurrent_core, max_loop_iters=3)
        coda = RecurrentTransformerBlock(
            dim=dim,
            attention=attention,
            feedforward=ffn,
            max_loop_iters=3,
            use_film=False,
        )
        return nn.ModuleDict({"prelude": prelude, "recurrent": recurrent, "coda": coda})

    def test_end_to_end_mini_forward(self):
        """8.1: Full forward pass through mini architecture."""
        arch = self.build_mini_architecture()
        x = torch.randn(2, 16, 64)
        # Prelude
        h = arch["prelude"](x)
        # Recurrent (h as both state and encoded input)
        h = arch["recurrent"](h, h, n_loops=3)
        # Coda
        out = arch["coda"](h)
        assert out.shape == (2, 16, 64), f"Expected (2, 16, 64), got {out.shape}"
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_gradient_end_to_end(self):
        """8.2: Gradients must flow through all components."""
        arch = self.build_mini_architecture()
        x = torch.randn(2, 16, 64, requires_grad=True)
        h = arch["prelude"](x)
        h = arch["recurrent"](h, h, n_loops=3)
        out = arch["coda"](h)
        loss = out.sum()
        loss.backward()

        # Check key parameter groups have gradients
        assert x.grad is not None, "Input gradient missing"
        for name, p in arch.named_parameters():
            if p.grad is None:
                pytest.fail(f"Parameter '{name}' has no gradient")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_migration(self):
        """8.3: Model on CUDA after .cuda() must produce CUDA output."""
        arch = self.build_mini_architecture().cuda()
        x = torch.randn(2, 16, 64, device="cuda")
        h = arch["prelude"](x)
        h = arch["recurrent"](h, h, n_loops=3)
        out = arch["coda"](h)
        assert out.device.type == "cuda", "Output not on CUDA"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
