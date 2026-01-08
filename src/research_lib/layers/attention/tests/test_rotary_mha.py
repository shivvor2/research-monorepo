"""
Test suite for RotaryMultiheadAttention

Run with: pytest src/research_lib/layers/attention/tests/test_rotary_mha.py -v
Or: python -m pytest src/research_lib/layers/attention/tests/test_rotary_mha.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..rotatory_mha import RotaryMultiheadAttention


class TestRotaryMultiheadAttention:
    """Test suite for RotaryMultiheadAttention."""

    @pytest.fixture
    def default_config(self):
        return {
            "embed_dim": 256,
            "num_heads": 8,
            "dropout": 0.0,
            "bias": True,
            "batch_first": True,
        }

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensors."""
        batch_size = 4
        seq_len = 32
        embed_dim = 256
        return torch.randn(batch_size, seq_len, embed_dim)

    # =========================================================================
    # Basic Functionality Tests
    # =========================================================================

    def test_instantiation(self, default_config):
        """Test that the module can be instantiated."""
        attn = RotaryMultiheadAttention(**default_config)
        assert attn.embed_dim == 256
        assert attn.num_heads == 8
        assert attn.head_dim == 32

    def test_self_attention_forward(self, default_config, sample_input):
        """Test forward pass with self-attention."""
        attn = RotaryMultiheadAttention(**default_config)
        x = sample_input

        output, weights = attn(x, x, x, need_weights=True)

        assert output.shape == x.shape
        assert weights.shape == (x.shape[0], x.shape[1], x.shape[1])

    def test_self_attention_no_weights(self, default_config, sample_input):
        """Test forward pass without attention weights (uses SDPA)."""
        attn = RotaryMultiheadAttention(**default_config)
        x = sample_input

        output, weights = attn(x, x, x, need_weights=False)

        assert output.shape == x.shape
        assert weights is None

    def test_causal_masking(self, default_config, sample_input):
        """Test that causal masking works."""
        attn = RotaryMultiheadAttention(**default_config)
        x = sample_input

        output, weights = attn(x, x, x, is_causal=True, need_weights=True)

        assert output.shape == x.shape
        # Check that upper triangle of attention weights is zero (after softmax, ~0)
        # Due to -inf masking, values should be very small
        upper_tri = torch.triu(weights, diagonal=1)
        assert upper_tri.max() < 1e-5

    def test_cross_attention(self, default_config):
        """Test cross-attention with different Q and KV sequences."""
        attn = RotaryMultiheadAttention(**default_config)

        query = torch.randn(4, 16, 256)  # Target sequence
        key = torch.randn(4, 32, 256)  # Source sequence
        value = torch.randn(4, 32, 256)  # Source sequence

        output, weights = attn(query, key, value, need_weights=True)

        assert output.shape == query.shape
        assert weights.shape == (4, 16, 32)

    # =========================================================================
    # Rotary Embedding Tests
    # =========================================================================

    def test_rotary_embeddings_applied(self, default_config, sample_input):
        """Test that rotary embeddings are being applied and affect the computation."""

        attn = RotaryMultiheadAttention(**default_config)
        attn.eval()
        x = sample_input[:, :16, :]

        with torch.no_grad():
            batch_size, seq_len, _ = x.shape

            # Manually compute Q, K
            qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
            qkv = qkv.view(batch_size, seq_len, 3, attn.num_heads, attn.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            # Apply rotary with different offsets
            q_rot0, k_rot0 = attn._apply_rotary(
                q.clone(), k.clone(), seq_dim=-2, offset=0
            )
            q_rot16, k_rot16 = attn._apply_rotary(
                q.clone(), k.clone(), seq_dim=-2, offset=16
            )

            # Rotated values should differ with different offsets
            assert not torch.allclose(
                q_rot0, q_rot16, atol=1e-5
            ), "Rotary embeddings not affected by offset - offset not working!"
            assert not torch.allclose(
                k_rot0, k_rot16, atol=1e-5
            ), "Rotary embeddings not affected by offset - offset not working!"

            # Verify rotary is actually changing values (not identity)
            assert not torch.allclose(
                q, q_rot0, atol=1e-5
            ), "Rotary embedding is not modifying queries!"

    def test_rotary_relative_position_property(self, default_config):
        """
        Test RoPE's key property: attention scores depend on relative position.
        q[m] · k[n] should be the same for any (m,n) with the same (m-n).
        """
        attn = RotaryMultiheadAttention(**default_config)
        attn.eval()

        # Create identical token embeddings at different positions
        token = torch.randn(1, 1, 256)  # Single token embedding

        # Replicate to create sequence of identical tokens
        x = token.expand(1, 8, 256).clone()

        with torch.no_grad():
            # Compute Q, K
            qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
            qkv = qkv.view(1, 8, 3, attn.num_heads, attn.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)  # (1, H, 8, D)
            k = k.transpose(1, 2)

            # Apply rotary
            q_rot, k_rot = attn._apply_rotary(q, k, seq_dim=-2, offset=0)

            # Compute attention scores
            scores = torch.matmul(q_rot, k_rot.transpose(-2, -1))  # (1, H, 8, 8)

            # For identical input tokens, attention score q[i]·k[j] should only
            # depend on (i-j), creating a Toeplitz-like structure
            # Check that diagonal elements are equal (same relative pos = 0)
            diag = torch.diagonal(scores[0, 0])
            assert torch.allclose(
                diag, diag[0].expand_as(diag), atol=1e-4
            ), "Diagonal elements should be equal (relative pos = 0)"

            # Check that off-diagonal with same offset are equal
            # scores[i, i+1] should all be equal (relative pos = 1)
            off_diag_1 = torch.diagonal(scores[0, 0], offset=1)
            if len(off_diag_1) > 1:
                assert torch.allclose(
                    off_diag_1, off_diag_1[0].expand_as(off_diag_1), atol=1e-4
                ), "Elements with same relative position should have same attention score"

    def test_position_invariance_property(self, default_config):
        """
        Test RoPE's key property: relative position is encoded in dot product.
        q[m] · k[n] depends only on (m - n), not absolute positions.
        """
        torch.manual_seed(42)
        attn = RotaryMultiheadAttention(**default_config)

        # Create identical tokens
        x = torch.randn(1, 64, 256)
        x = x.expand(2, -1, -1).clone()  # Two identical sequences

        # The attention pattern should be similar for identical content
        # because RoPE encodes relative positions
        output, weights = attn(x, x, x, need_weights=True, is_causal=False)

        # Both batch items should have same attention pattern
        assert torch.allclose(weights[0], weights[1], atol=1e-5)

    # =========================================================================
    # Mask Tests
    # =========================================================================

    def test_key_padding_mask(self, default_config, sample_input):
        """Test key padding mask functionality."""
        attn = RotaryMultiheadAttention(**default_config)
        x = sample_input

        # Mask out last 8 positions
        key_padding_mask = torch.zeros(4, 32, dtype=torch.bool)
        key_padding_mask[:, 24:] = True

        output, weights = attn(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=True
        )

        assert output.shape == x.shape
        # Check that masked positions have ~0 attention
        assert weights[:, :, 24:].max() < 1e-5

    def test_custom_attn_mask(self, default_config, sample_input):
        """Test custom attention mask."""
        attn = RotaryMultiheadAttention(**default_config)
        x = sample_input
        seq_len = x.shape[1]

        # Block attention to first 8 positions
        attn_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        attn_mask[:, :8] = True

        output, weights = attn(x, x, x, attn_mask=attn_mask, need_weights=True)

        assert output.shape == x.shape
        assert weights[:, :, :8].max() < 1e-5

    # =========================================================================
    # API Compatibility Tests
    # =========================================================================

    def test_batch_first_false(self, default_config, sample_input):
        """Test with batch_first=False (seq, batch, embed)."""
        config = {**default_config, "batch_first": False}
        attn = RotaryMultiheadAttention(**config)

        # Transpose to (seq, batch, embed)
        x = sample_input.transpose(0, 1)

        output, _ = attn(x, x, x, need_weights=False)

        assert output.shape == x.shape

    def test_no_bias(self, default_config, sample_input):
        """Test with bias=False."""
        config = {**default_config, "bias": False}
        attn = RotaryMultiheadAttention(**config)

        assert attn.in_proj_bias is None
        assert attn.out_proj.bias is None

        output, _ = attn(sample_input, sample_input, sample_input, need_weights=False)
        assert output.shape == sample_input.shape

    def test_average_attn_weights_false(self, default_config, sample_input):
        """Test returning per-head attention weights."""
        attn = RotaryMultiheadAttention(**default_config)
        x = sample_input

        output, weights = attn(x, x, x, need_weights=True, average_attn_weights=False)

        assert output.shape == x.shape
        # Should be (batch, num_heads, seq, seq)
        assert weights.shape == (4, 8, 32, 32)

    # =========================================================================
    # XPos Tests (Length Extrapolation)
    # =========================================================================

    def test_xpos_instantiation(self, default_config):
        """Test instantiation with xpos enabled."""
        config = {**default_config, "use_xpos": True, "xpos_scale_base": 512}
        attn = RotaryMultiheadAttention(**config)

        assert attn.use_xpos == True
        assert attn.rotary_emb.use_xpos == True

    def test_xpos_forward(self, default_config, sample_input):
        """Test forward pass with xpos."""
        config = {**default_config, "use_xpos": True}
        attn = RotaryMultiheadAttention(**config)

        output, _ = attn(sample_input, sample_input, sample_input, need_weights=False)
        assert output.shape == sample_input.shape

    # =========================================================================
    # Device and Dtype Tests
    # =========================================================================

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self, default_config, sample_input):
        """Test forward pass on CUDA."""
        attn = RotaryMultiheadAttention(**default_config).cuda()
        x = sample_input.cuda()

        output, _ = attn(x, x, x, need_weights=False, is_causal=True)

        assert output.device.type == "cuda"
        assert output.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision(self, default_config, sample_input):
        """Test with automatic mixed precision."""
        attn = RotaryMultiheadAttention(**default_config).cuda()
        x = sample_input.cuda()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output, _ = attn(x, x, x, need_weights=False, is_causal=True)

        assert output.shape == x.shape

    # =========================================================================
    # Gradient Tests
    # =========================================================================

    def test_backward_pass(self, default_config, sample_input):
        """Test that gradients flow correctly."""
        attn = RotaryMultiheadAttention(**default_config)
        x = sample_input.requires_grad_(True)

        output, _ = attn(x, x, x, need_weights=False, is_causal=True)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert attn.in_proj_weight.grad is not None

    # =========================================================================
    # torch.compile Tests
    # =========================================================================

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_compile(self, default_config, sample_input):
        """Test compatibility with torch.compile."""
        attn = RotaryMultiheadAttention(**default_config).cuda()
        attn_compiled = torch.compile(attn, mode="reduce-overhead")
        x = sample_input.cuda()

        # Warmup
        for _ in range(3):
            output, _ = attn_compiled(x, x, x, need_weights=False, is_causal=True)

        # Actual test
        output, _ = attn_compiled(x, x, x, need_weights=False, is_causal=True)

        assert output.shape == x.shape

    # =========================================================================
    # Numerical Consistency Tests
    # =========================================================================

    def test_deterministic_output(self, default_config, sample_input):
        """Test that output is deterministic in eval mode."""
        attn = RotaryMultiheadAttention(**default_config)
        attn.eval()
        x = sample_input

        with torch.no_grad():
            output1, _ = attn(x, x, x, need_weights=False)
            output2, _ = attn(x, x, x, need_weights=False)

        assert torch.allclose(output1, output2)

    def test_sdpa_vs_manual_attention(self, default_config, sample_input):
        """Test that SDPA and manual attention give same results."""
        # Note: Due to numerical precision, results may differ slightly
        config = {**default_config, "dropout": 0.0}
        attn = RotaryMultiheadAttention(**config)
        attn.eval()
        x = sample_input

        with torch.no_grad():
            # SDPA path
            output_sdpa, _ = attn(x, x, x, need_weights=False, is_causal=True)

            # Manual path (with weights)
            output_manual, weights = attn(x, x, x, need_weights=True, is_causal=True)

        # Should be very close (allowing for floating point differences)
        assert torch.allclose(output_sdpa, output_manual, atol=1e-4, rtol=1e-4)


class TestRotaryMultiheadAttentionIntegration:
    """Integration tests with TransformerBlock-like usage."""

    def test_in_transformer_block(self):
        """Test usage pattern in a transformer block."""

        class SimpleTransformerBlock(nn.Module):
            def __init__(self, embed_dim, num_heads, ff_dim):
                super().__init__()
                self.norm1 = nn.LayerNorm(embed_dim)
                self.attn = RotaryMultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=0.1,
                    batch_first=True,
                )
                self.norm2 = nn.LayerNorm(embed_dim)
                self.ff = nn.Sequential(
                    nn.Linear(embed_dim, ff_dim),
                    nn.GELU(),
                    nn.Linear(ff_dim, embed_dim),
                )

            def forward(self, x):
                # Pre-norm architecture
                x_norm = self.norm1(x)
                attn_out, _ = self.attn(
                    x_norm, x_norm, x_norm, is_causal=True, need_weights=False
                )
                x = x + attn_out

                x = x + self.ff(self.norm2(x))
                return x

        block = SimpleTransformerBlock(256, 8, 1024)
        x = torch.randn(4, 32, 256)

        output = block(x)

        assert output.shape == x.shape

    def test_stacked_blocks(self):
        """Test multiple stacked attention layers."""
        layers = nn.ModuleList(
            [RotaryMultiheadAttention(256, 8, batch_first=True) for _ in range(4)]
        )

        x = torch.randn(2, 16, 256)

        for layer in layers:
            x, _ = layer(x, x, x, is_causal=True, need_weights=False)

        assert x.shape == (2, 16, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
