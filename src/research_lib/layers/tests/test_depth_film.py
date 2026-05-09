"""
Test suite for DepthFiLM (Depth-conditioned Feature-wise Linear Modulation).

Run with: pytest src/research_lib/layers/tests/test_depth_film.py -v
"""

import pytest
import torch

from ..depth_film import DepthFiLM


class TestDepthFiLM:
    """Test suite for DepthFiLM."""

    # =====================================================================
    # Shape Invariants
    # =====================================================================

    def test_shape_invariant(self):
        """Forward pass must preserve spatial dimensions. FiLM is per-channel."""
        film = DepthFiLM(dim=512, max_loops=8, hidden=64)
        x = torch.randn(2, 4, 512)
        out = film(x, loop_t=0)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    # =====================================================================
    # Identity at Initialization
    # =====================================================================

    def test_identity_at_init(self):
        """At init, scale≈0, shift≈0 → output must equal input (identity)."""
        film = DepthFiLM(dim=256, max_loops=8, hidden=32)
        x = torch.randn(2, 4, 256)
        out = film(x, loop_t=0)
        assert torch.allclose(
            out, x, atol=1e-5
        ), "DepthFiLM output should equal input at zero-init"

    # =====================================================================
    # Depth Extrapolation (Clamping)
    # =====================================================================

    def test_depth_extrapolation_clamping(self):
        """loop_t beyond max_loops-1 must safely clamp to last entry."""
        film = DepthFiLM(dim=64, max_loops=4, hidden=16)
        x = torch.randn(2, 4, 64)
        out_t3 = film(x, loop_t=3)
        out_t7 = film(x, loop_t=7)
        assert torch.equal(
            out_t3, out_t7
        ), "loop_t=7 (clamped to 3) must equal loop_t=3"

    # =====================================================================
    # Different Scales Per Depth
    # =====================================================================

    def test_different_scales_per_depth(self):
        """With non-zero final MLP weights, loop_t=0 and loop_t=1 must differ."""
        film = DepthFiLM(dim=64, max_loops=4, hidden=16)
        # Manually set non-zero weights on the final MLP layer
        with torch.no_grad():
            film.mlp[-1].weight.normal_(mean=0.0, std=0.1)
            film.mlp[-1].bias.uniform_(-0.1, 0.1)
        x = torch.randn(2, 4, 64)
        out_0 = film(x, loop_t=0)
        out_1 = film(x, loop_t=1)
        assert not torch.equal(
            out_0, out_1
        ), "Outputs at different loop_t must differ with non-zero MLP weights"

    # =====================================================================
    # Parameter Count Sanity
    # =====================================================================

    def test_parameter_count(self):
        """Exact parameter count for dim=512, max_loops=8, hidden=64."""
        dim = 512
        max_loops = 8
        hidden = 64
        film = DepthFiLM(dim=dim, max_loops=max_loops, hidden=hidden)

        # Embedding: max_loops * hidden = 8 * 64 = 512
        # Linear 1: hidden * hidden + hidden = 64 * 64 + 64 = 4160
        # Linear 2 (final): hidden * (dim * 2) + (dim * 2) = 64 * 1024 + 1024 = 65664
        # But mlp is Sequential(Linear, SiLU, Linear) — no params in SiLU
        expected = (
            max_loops * hidden  # embedding
            + hidden * hidden
            + hidden  # first linear + bias
            + hidden * (dim * 2)
            + (dim * 2)  # final linear + bias
        )
        actual = sum(p.numel() for p in film.parameters())
        assert (
            actual == expected
        ), f"Parameter count mismatch: expected {expected}, got {actual}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
