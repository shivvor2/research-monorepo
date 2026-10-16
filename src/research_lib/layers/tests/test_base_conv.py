"""
Tests for Zoology's BaseConv short convolution mixer.

Target: src/research_lib/layers/base_conv.py
Run with: pytest src/research_lib/layers/tests/test_base_conv.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from research_lib.layers.base_conv import BaseConv


class TestBaseConv:
    """Tests for BaseConv."""

    @pytest.fixture
    def layer(self) -> BaseConv:
        return BaseConv(d_model=8, kernel_size=3)

    # ------------------------------------------------------------------ #
    # Structure
    # ------------------------------------------------------------------ #

    def test_output_shape(self, layer):
        """Output has the input shape."""
        x = torch.randn(2, 10, 8, dtype=torch.float64)
        layer = layer.double()
        y = layer(x)
        assert y.shape == (2, 10, 8)

    def test_conv_has_no_activation_and_no_bias(self, layer):
        """The depthwise conv is linear (the gate branch is the nonlinearity)."""
        assert layer.conv.activation is None
        assert layer.conv.conv.bias is None
        assert layer.gate.bias is not None  # the "+c" of the formula

    # ------------------------------------------------------------------ #
    # Literal computation
    # ------------------------------------------------------------------ #

    def test_matches_literal_computation(self):
        """y == F.conv1d(pad(x)) * (W x + c) + x, computed literally."""
        torch.manual_seed(0)
        layer = BaseConv(d_model=6, kernel_size=3).double()
        x = torch.randn(3, 12, 6, dtype=torch.float64)

        # Literal: left-pad with zeros, depthwise F.conv1d with the same
        # filter, groups = channels.
        xt = x.transpose(1, 2)  # (B, C, T)
        pad = xt.new_zeros(3, 6, 2)
        weight = layer.conv.conv.weight  # (C, 1, K)
        conv_lit = F.conv1d(torch.cat([pad, xt], dim=-1), weight, groups=6)
        gate_lit = torch.nn.functional.linear(x, layer.gate.weight, layer.gate.bias)
        expected = conv_lit.transpose(1, 2) * gate_lit + x

        y = layer(x)
        torch.testing.assert_close(y, expected, rtol=1e-12, atol=1e-12)

    # ------------------------------------------------------------------ #
    # Causality
    # ------------------------------------------------------------------ #

    def test_causal(self, layer):
        """Perturbing position t' leaves every output before t' unchanged."""
        torch.manual_seed(1)
        layer = layer.double()
        B, T, d = 2, 16, 8
        x = torch.randn(B, T, d, dtype=torch.float64)
        y1 = layer(x)
        t_prime = 9
        x2 = x.clone()
        x2[:, t_prime:, :] = torch.randn(B, T - t_prime, d, dtype=torch.float64)
        y2 = layer(x2)
        torch.testing.assert_close(y1[:, :t_prime], y2[:, :t_prime])
        assert not torch.allclose(y1[:, t_prime:], y2[:, t_prime:])

    # ------------------------------------------------------------------ #
    # Device / dtype factories
    # ------------------------------------------------------------------ #

    def test_dtype_honoured(self):
        """dtype= is passed to the conv filter and the gate."""
        layer = BaseConv(d_model=4, dtype=torch.float64)
        assert layer.conv.conv.weight.dtype == torch.float64
        assert layer.gate.weight.dtype == torch.float64

    def test_kernel_width_is_respected(self):
        """The output at t depends on exactly K inputs (t-K+1 .. t)."""
        torch.manual_seed(2)
        layer = BaseConv(d_model=4, kernel_size=3).double()
        x = torch.randn(1, 10, 4, dtype=torch.float64)
        y1 = layer(x)
        # Changing t=3 may affect outputs at t=3,4,5 but not t>=6.
        x2 = x.clone()
        x2[:, 3, :] += 1.0
        y2 = layer(x2)
        assert not torch.allclose(y1[:, 3], y2[:, 3])
        assert not torch.allclose(y1[:, 5], y2[:, 5])
        torch.testing.assert_close(y1[:, 6:], y2[:, 6:])
