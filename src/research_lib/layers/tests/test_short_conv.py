"""Tests for :class:`CausalShortConv1d` (spec test G09).

Run with::

    pytest src/research_lib/layers/tests/test_short_conv.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from ..short_conv import CausalShortConv1d


def _make_conv(
    channels: int, kernel_size: int, bias: bool = False, activation: str | None = "silu"
) -> CausalShortConv1d:
    conv = CausalShortConv1d(
        channels, kernel_size=kernel_size, bias=bias, activation=activation
    )
    with torch.no_grad():
        conv.conv.weight.copy_(torch.randn_like(conv.conv.weight) * 0.5)
        if bias:
            conv.conv.bias.copy_(torch.randn_like(conv.conv.bias) * 0.1)
    return conv


class TestG09ShortConv:
    def test_G09_equals_fconv1d_left_zero_padding(self):
        channels, K, T, B = 6, 4, 11, 2
        conv = _make_conv(channels, K)
        x = torch.randn(B, T, channels)
        y, _ = conv(x)
        # manual: left-pad with K-1 zeros, depthwise conv, SiLU
        xt = x.transpose(1, 2)  # (B, C, T)
        pad = torch.zeros(B, channels, K - 1)
        expected = F.conv1d(
            torch.cat([pad, xt], dim=-1),
            conv.conv.weight,
            conv.conv.bias,
            groups=channels,
        ).transpose(1, 2)
        expected = F.silu(expected)
        torch.testing.assert_close(y, expected, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize(
        "segments", [[10], [4, 6], [1, 1, 8], [2, 9], [3, 1, 3, 3]]
    )
    def test_G09_segmented_equals_one_pass(self, segments):
        """Cache continuation equals a single pass, incl. segments shorter
        than K-1."""
        channels, K = 5, 4
        conv = _make_conv(channels, K)
        T = sum(segments)
        x = torch.randn(2, T, channels)
        y_full, _ = conv(x)
        cache = None
        outputs = []
        start = 0
        for seg in segments:
            y_seg, cache = conv(
                x[:, start : start + seg], cache=cache, output_final_state=True
            )
            outputs.append(y_seg)
            start += seg
        y_seg_all = torch.cat(outputs, dim=1)
        torch.testing.assert_close(y_seg_all, y_full, atol=1e-6, rtol=1e-6)

    def test_G09_causality(self):
        channels, K, T = 5, 4, 12
        conv = _make_conv(channels, K)
        x = torch.randn(2, T, channels)
        y, _ = conv(x)
        t_prime = 7
        x2 = x.clone()
        x2[:, t_prime] += 10.0
        y2, _ = conv(x2)
        torch.testing.assert_close(
            y[:, :t_prime], y2[:, :t_prime], atol=1e-7, rtol=1e-7
        )
        assert not torch.allclose(y[:, t_prime:], y2[:, t_prime:])

    def test_G09_none_cache_is_zeros(self):
        channels, K, T = 4, 3, 8
        conv = _make_conv(channels, K)
        x = torch.randn(2, T, channels)
        y_none, _ = conv(x, cache=None)
        y_zero, _ = conv(x, cache=torch.zeros(2, channels, K - 1))
        torch.testing.assert_close(y_none, y_zero, atol=1e-7, rtol=1e-7)

    def test_G09_identity_activation(self):
        channels, K, T = 4, 3, 8
        conv = _make_conv(channels, K, activation=None)
        x = torch.randn(2, T, channels)
        y, _ = conv(x)
        xt = x.transpose(1, 2)
        pad = torch.zeros(2, channels, K - 1)
        expected = F.conv1d(
            torch.cat([pad, xt], dim=-1),
            conv.conv.weight,
            None,
            groups=channels,
        ).transpose(1, 2)
        torch.testing.assert_close(y, expected, atol=1e-6, rtol=1e-6)
