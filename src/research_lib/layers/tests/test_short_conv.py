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


class TestKernelSizeOneCache:
    """Sourcery review: kernel_size == 1 must return an empty (B, C, 0)
    cache (-(K-1) is -0, so a naive slice would return the whole padded
    tensor)."""

    def test_kernel_size_1_empty_cache_shape(self):
        conv = _make_conv(channels=5, kernel_size=1)
        x = torch.randn(2, 8, 5)
        y, cache = conv(x, output_final_state=True)
        assert cache is not None
        assert cache.shape == (2, 5, 0)
        # the whole-sequence output must equal the pointwise (K=1) conv
        xt = x.transpose(1, 2)
        expected = F.silu(F.conv1d(xt, conv.conv.weight, None, groups=5))
        torch.testing.assert_close(y, expected.transpose(1, 2), atol=1e-6, rtol=1e-6)

    def test_kernel_size_1_segmented_equals_one_pass(self):
        conv = _make_conv(channels=5, kernel_size=1)
        T = 10
        x = torch.randn(2, T, 5)
        y_full, _ = conv(x)
        cache = None
        outputs = []
        start = 0
        for seg in [3, 1, 6]:
            y_seg, cache = conv(
                x[:, start : start + seg], cache=cache, output_final_state=True
            )
            outputs.append(y_seg)
            start += seg
        torch.testing.assert_close(
            torch.cat(outputs, dim=1), y_full, atol=1e-6, rtol=1e-6
        )

    def test_cache_is_owned_copy(self):
        """The returned cache must not share storage with the caller's input
        (it also must not pin the whole prefill, hence the clone)."""
        channels, K = 5, 4
        conv = _make_conv(channels, K)
        x = torch.randn(2, 16, channels)
        _, cache = conv(x, output_final_state=True)
        assert cache is not None
        assert cache.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        # and it is a genuine copy: mutating the input afterwards leaves the
        # cache unchanged
        x_copy_ptr = cache.untyped_storage().data_ptr()
        x[:, :] = 0.0
        assert cache.untyped_storage().data_ptr() == x_copy_ptr
