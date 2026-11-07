"""
Tests for CausalSelfAttention: output shape, position causality, and the
end-to-end causality of a ZoologyLM built over attention mixers.

Target: src/research_lib/layers/attention/causal_self_attention.py
Run with: pytest src/research_lib/layers/attention/tests/test_causal_self_attention.py -v
"""

import torch

from research_lib.architectures.config.zoology_lm import ZoologyLMConfig
from research_lib.architectures.zoology_lm import ZoologyLM
from research_lib.layers.attention import CausalSelfAttention


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
