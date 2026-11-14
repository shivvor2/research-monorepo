"""
Tests for CausalSelfAttention: output shape, position causality, and the
end-to-end causality of a ZoologyLM built over attention mixers.

Target: src/research_lib/layers/attention/causal_self_attention.py
Run with: pytest src/research_lib/layers/attention/tests/test_causal_self_attention.py -v
"""

import torch
import torch.nn as nn

from research_lib.architectures.config.zoology_lm import ZoologyLMConfig
from research_lib.architectures.zoology_lm import ZoologyLM
from research_lib.layers.attention import CausalSelfAttention
from research_lib.layers.attention.rotatory_mha import RotaryMultiheadAttention


class TestOutputShape:
    def test_shape_is_preserved(self):
        attn = CausalSelfAttention(32, 2)
        x = torch.randn(3, 16, 32)
        assert attn(x).shape == x.shape

    def test_rotary_dim_defaults_to_head_width(self):
        attn = CausalSelfAttention(32, 2)
        assert attn.rotary_dim is None or attn.rotary_dim == 16


class TestCausality:
    def _attn(self):
        torch.manual_seed(0)
        return CausalSelfAttention(32, 2)

    def test_perturbing_a_position_leaves_earlier_outputs_unchanged(self):
        attn = self._attn()
        x = torch.randn(2, 12, 32)
        y = attn(x)
        x2 = x.clone()
        x2[:, 8, :] += 5.0  # perturb position 8
        y2 = attn(x2)
        # Outputs for positions < 8 are unaffected by the later perturbation.
        torch.testing.assert_close(y[:, :8, :], y2[:, :8, :], atol=1e-4, rtol=1e-4)
        # ... and the perturbed position itself (and later) changes.
        assert not torch.allclose(y[:, 8:, :], y2[:, 8:, :], atol=1e-4)

    def test_causal_shape_agrees_with_reference(self):
        attn = self._attn()
        x = torch.randn(2, 8, 32)
        out = attn(x)
        assert out.shape == (2, 8, 32)


class TestZoologyLMWithAttention:
    def test_end_to_end_causality(self):
        torch.manual_seed(1)
        mixers = [CausalSelfAttention(16, 2) for _ in range(2)]
        model = ZoologyLM(ZoologyLMConfig(vocab_size=64, d_model=16), mixers)
        ids = torch.randint(0, 64, (2, 10))
        logits = model(ids)
        ids2 = ids.clone()
        ids2[:, 6] = (ids2[:, 6] + 5) % 64
        logits2 = model(ids2)
        # Answers (logits) for positions < 6 are unaffected by the exchange
        # at position 6 (causal: nothing after position t feeds position t).
        torch.testing.assert_close(
            logits[:, :6, :], logits2[:, :6, :], atol=1e-5, rtol=1e-5
        )


class TestOddRotaryRejected:
    def test_odd_resolved_rotary_dimension_raises(self):
        import pytest

        # A head width of 5 (d_model 10, 2 heads) is odd: the rotary
        # dimension resolves to it and cannot be paired into sin/cos.
        with pytest.raises(ValueError, match="rotary dimension must be even"):
            CausalSelfAttention(10, 2)

    def test_explicit_odd_rotary_dimension_raises(self):
        import pytest

        with pytest.raises(ValueError, match="rotary dimension must be even"):
            CausalSelfAttention(32, 2, rotary_dim=3)


class TestValidation:
    def test_nonpositive_d_model_is_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="d_model must be positive"):
            CausalSelfAttention(0, 2)

    def test_rotary_dim_above_head_width_is_rejected(self):
        import pytest

        # Head width is 32 // 2 = 16; a rotary dimension of 24 rotates
        # beyond the head.
        with pytest.raises(ValueError, match="cannot exceed the head width"):
            CausalSelfAttention(32, 2, rotary_dim=24)

    def test_rotary_dim_of_two_is_rejected(self):
        import pytest

        # rotary-embedding-torch 0.9.1 divides by a zero stride for dim 2.
        with pytest.raises(ValueError, match="rotary dimension of 2"):
            CausalSelfAttention(4, 2)

    def test_explicit_rotary_dim_of_two_is_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="rotary dimension of 2"):
            CausalSelfAttention(32, 2, rotary_dim=2)

    def test_full_width_rotary_dim_is_accepted(self):
        # A rotary dimension exactly the head width is fine.
        attn = CausalSelfAttention(32, 2, rotary_dim=16)
        x = torch.randn(2, 8, 32)
        assert attn(x).shape == (2, 8, 32)


class TestQkNorm:
    """The query/key-norm switch: the default is unchanged (an RMSNorm on
    the query and key projections), ``None`` removes it entirely and agrees
    with a literal scaled-dot-product attention, and the underlying layer is
    unchanged."""

    def test_default_applies_rmsnorm_and_matches_explicit(self):
        # The default must be the same RMSNorm as before the switch existed:
        # for a fixed seed the default builds bit-identically to an explicit
        # ``qk_norm=nn.RMSNorm``, with the same state_dict keys.
        torch.manual_seed(0)
        attn = CausalSelfAttention(32, 2)
        torch.manual_seed(0)
        explicit = CausalSelfAttention(32, 2, qk_norm=nn.RMSNorm)
        assert attn.attention.qk_norm_str == "RMSNorm"
        keys = set(explicit.attention.state_dict())
        # the norm is present and elementwise-affine on the head width
        assert set(attn.attention.state_dict()) == keys
        assert "attention.q_norm.weight" in attn.state_dict()
        assert "attention.k_norm.weight" in attn.state_dict()
        assert attn.attention.q_norm.weight.shape == (16,)
        torch.manual_seed(0)
        x = torch.randn(2, 8, 32)
        torch.testing.assert_close(attn(x), explicit(x), atol=0.0, rtol=0.0)

    def test_default_attention_submodule_matches_the_underlying_layer(self):
        # The default wrapper should leave the attention submodule exactly as
        # a bare RotaryMultiheadAttention with the same arguments (nothing
        # added or moved); equal weights (same seed) give equal output.
        torch.manual_seed(0)
        attn = CausalSelfAttention(32, 2, rotary_dim=16)
        torch.manual_seed(0)
        bare = RotaryMultiheadAttention(embed_dim=32, num_heads=2, rotary_dim=16)
        assert set(attn.attention.state_dict()) == set(bare.state_dict())
        for key in bare.state_dict():
            torch.testing.assert_close(
                attn.attention.state_dict()[key], bare.state_dict()[key], atol=0, rtol=0
            )

    def test_qk_norm_none_removes_norm_parameters(self):
        attn = CausalSelfAttention(32, 2, qk_norm=None)
        assert attn.attention.qk_norm_str == "None"
        keys = set(attn.state_dict())
        assert not any(
            k in keys for k in ("attention.q_norm.weight", "attention.k_norm.weight")
        )

    def test_qk_norm_none_equals_literal_scaled_dot_product(self):
        # With qk_norm=None the attention is exactly the raw projections,
        # rotated, run through scaled-dot-product attention and out-projected
        # (no query/key normalisation anywhere).
        torch.manual_seed(0)
        attn = CausalSelfAttention(32, 2, qk_norm=None)
        torch.manual_seed(0)
        x = torch.randn(2, 16, 32)
        y = attn(x)

        mha = attn.attention
        B, T, _ = x.shape
        H, D = mha.num_heads, mha.head_dim
        qkv = torch.nn.functional.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        qkv = qkv.view(B, T, 3, H, D)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = mha._apply_rotary(q, k, seq_dim=-2, offset=0)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).contiguous().view(B, T, mha.embed_dim)
        o = torch.nn.functional.linear(o, mha.out_proj.weight, mha.out_proj.bias)
        torch.testing.assert_close(y, o, atol=1e-5, rtol=1e-5)
