"""
Tests for the bias-free SwiGLU block.

Covers: the MAD default inner width (352 at ``d_model=128``,
``multiple_of=16``) and that the forward equals the literal
``w3(silu(w1 x) * w2 x)`` computation.

Run with: pytest src/research_lib/layers/tests/test_swiglu.py -v
"""

import torch

from ..swiglu import SwiGLU


class TestSwiGLU:
    def test_default_inner_width(self):
        assert SwiGLU(128).hidden_dim == 352
        assert SwiGLU(128, multiple_of=8).hidden_dim % 8 == 0
        # int(8*7/3)=18; 18 % 3 == 0 -> 18 (the docstring's worked case).
        assert SwiGLU(7, multiple_of=3).hidden_dim == 18

    def test_equals_literal_formula(self):
        torch.manual_seed(0)
        block = SwiGLU(128)
        x = torch.randn(2, 32, 128)
        out = block(x)
        literal = block.w3(torch.nn.functional.silu(block.w1(x)) * block.w2(x))
        assert torch.allclose(out, literal, atol=1e-12)

    def test_no_biases_and_shapes(self):
        block = SwiGLU(64)
        for w in (block.w1, block.w2, block.w3):
            assert w.bias is None
        out = block(torch.randn(2, 16, 64))
        assert out.shape == (2, 16, 64)
