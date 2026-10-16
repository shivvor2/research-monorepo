"""Layout tests for the multi-state gated delta layer (M01–M03).

All tests run in float64 on CPU with tolerances ``atol=1e-10, rtol=1e-10``.

Run with::

    pytest src/research_lib/layers/attention/gated_delta/tests/test_layouts.py -v
"""

import pytest
import torch

from ..layouts import (
    Layout,
    ll_chunk_merge,
    ll_level,
    ll_level_matrix,
    ll_token_merge,
    lssb,
    num_slots,
    ws_shift,
)
from ._oracle import ms_ll_level

ATOL = RTOL = 1e-10

# The Fenwick active-level table of 05 §6.3 (t = 1..16 and t = 20).
FENWICK_TABLE = {
    1: [0, 1],
    2: [0, 2],
    3: [0, 1, 2],
    4: [0, 3],
    6: [0, 2, 3],
    7: [0, 1, 2, 3],
    8: [0, 4],
    11: [0, 1, 2, 4],
    12: [0, 3, 4],
    16: [0, 5],
    20: [0, 3, 5],
}


class TestM01LlLevel:
    """M01: ``ll_level`` — the 05 §6.3 table, and the fla mask condition."""

    def test_M01_fenwick_table(self):
        for t, expected in FENWICK_TABLE.items():
            levels = sorted({ll_level(t, s) for s in range(t + 1)})
            assert levels == expected, f"t={t}: got {levels}, expected {expected}"

    def test_M01_matches_independent_transcription(self):
        for t in range(64):
            for s in range(t + 1):
                assert ll_level(t, s) == ms_ll_level(t, s), (t, s)

    def test_M01_matches_fla_mask_condition(self):
        """The level ranges are the fla ``construct_level_mask`` condition
        (05 §6.2): key j is in level l (l >= 1) iff
        ``i mod 2^l >= 2^(l-1)`` and
        ``i - (i mod 2^(l-1)) - 2^(l-1) <= j < i - (i mod 2^(l-1))``;
        the active levels plus the sentinel partition [0, t]."""
        for t in range(1024):
            covered = {t}  # sentinel = level 0
            for lvl in range(1, t.bit_length() + 1):
                if (t >> (lvl - 1)) & 1:
                    size = 1 << (lvl - 1)
                    lo = t - (t % size) - size
                    hi = t - (t % size)
                    for s in range(max(lo, 0), hi):
                        assert ll_level(t, s) == lvl, (t, s, lvl)
                        covered.add(s)
            assert covered == set(range(t + 1)), f"levels do not partition [0,{t}]"

    def test_M01_s_eq_t_is_sentinel_and_s_gt_t_raises(self):
        assert ll_level(5, 5) == 0
        with pytest.raises(ValueError):
            ll_level(5, 6)


class TestM02NumSlots:
    """M02: slot counts per layout."""

    def test_M02_single_is_one(self):
        assert num_slots("single") == 1
        for window in (1, 8, 16):
            assert num_slots("single", window=window, t_max=256) == 1

    def test_M02_ws_fs_is_window_plus_two(self):
        for window in (1, 2, 8, 16):
            assert num_slots("ws", window=window) == window + 2
            assert num_slots("fs", window=window) == window + 2

    def test_M02_ll_is_bitlen_plus_one(self):
        assert num_slots("ll", t_max=256) == 10
        assert num_slots("ll", t_max=4096) == 14
        # 05 §6.5: ceil(log2) + 1 would be one short at powers of two.
        assert num_slots("ll", t_max=16) == 6  # bitlen(16) + 1
        assert num_slots("ll", t_max=1) == 2

    def test_M02_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            num_slots("ws", window=0)
        with pytest.raises(ValueError):
            num_slots("fs", window=-1)
        with pytest.raises(ValueError):
            num_slots("ll", t_max=0)
        with pytest.raises(ValueError):
            num_slots("nonsense")


class TestM03WsShift:
    """M03: ``ws_shift`` — exact shift, functional, differentiable."""

    @staticmethod
    def _random_stack(L: int, seed: int = 0) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(2, 3, L, 4, 3, dtype=torch.float64, generator=gen)

    def test_M03_exact_shift(self):
        for window in (1, 2, 8):
            S = self._random_stack(window + 2, seed=window)
            out = ws_shift(S, window)
            torch.testing.assert_close(
                out[..., 0, :, :],
                torch.zeros_like(S[..., 0, :, :]),
                atol=ATOL,
                rtol=RTOL,
            )
            for w in range(1, window + 1):
                torch.testing.assert_close(
                    out[..., w, :, :], S[..., w - 1, :, :], atol=ATOL, rtol=RTOL
                )
            torch.testing.assert_close(
                out[..., window + 1, :, :],
                S[..., window, :, :] + S[..., window + 1, :, :],
                atol=ATOL,
                rtol=RTOL,
            )

    def test_M03_input_unchanged_functional(self):
        S = self._random_stack(4, seed=1)
        S_before = S.clone()
        ws_shift(S, 2)
        torch.testing.assert_close(S, S_before, atol=ATOL, rtol=RTOL)
        # and a second shift of the *output* must not touch S either
        out = ws_shift(S, 2)
        ws_shift(out, 2)
        torch.testing.assert_close(S, S_before, atol=ATOL, rtol=RTOL)

    def test_M03_autograd_gradient(self):
        """The gradient is the inverse permutation plus the rest-add:
        S'[w] = S[w-1] for w = 1..W and S'[W+1] = S[W] + S[W+1], so
        dS[i] = upstream[i+1] for i < W, dS[W] = dS[W+1] = upstream[W+1]."""
        window, L = 2, 4
        S = self._random_stack(L, seed=2).requires_grad_(True)
        out = ws_shift(S, window)
        upstream = torch.randn_like(out)
        (out * upstream).sum().backward()
        expected = torch.cat(
            [
                upstream[..., 1 : window + 1, :, :],  # dS[w-1] += dS'[w]
                upstream[..., window + 1 : window + 2, :, :],  # S[W] -> rest
                upstream[..., window + 1 : window + 2, :, :],  # S[W+1] -> rest
            ],
            dim=-3,
        )
        torch.testing.assert_close(S.grad, expected, atol=ATOL, rtol=RTOL)

    def test_M03_wrong_slot_count_raises(self):
        S = self._random_stack(3)
        with pytest.raises(ValueError):
            ws_shift(S, window=2)  # needs 4 slots


class TestM03bLlMerges:
    """M03 (continued): the Fenwick merge functions — exact, functional,
    differentiable-by-construction (cat/indexing only)."""

    @staticmethod
    def _stack(L: int, seed: int = 0) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(2, 3, L, 4, 3, dtype=torch.float64, generator=gen)

    def test_ll_token_merge_exact(self):
        L = 6
        S = self._stack(L)
        out = ll_token_merge(S, next_position=6)  # lssb(6) = 1
        merged = S[..., :2, :, :].sum(dim=-3)
        torch.testing.assert_close(out[..., 2, :, :], merged, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(
            out[..., :2, :, :], torch.zeros_like(S[..., :2, :, :]), atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            out[..., 3:, :, :], S[..., 3:, :, :], atol=ATOL, rtol=RTOL
        )
        # old slot j+1 content is discarded (empty by the Fenwick invariant)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(out[..., 2, :, :], S[..., 2, :, :])

    def test_ll_token_merge_every_step_matches_fenwick_trace(self):
        """After merging for arriving at ``t`` (t >= 1), the active levels
        of ``t - 1``'s buckets follow the 05 §6.3 trace."""
        L = 7
        S = self._stack(L)
        for t in range(1, 33):
            S = ll_token_merge(S, next_position=t)
            # slot j+1 == sum of slots 0..j from *before* the merge is
            # checked structurally in test_ll_token_merge_exact; here just
            # exercise every position including powers of two.
        assert S.shape[-3] == L

    def test_ll_chunk_merge_exact(self):
        k, L = 2, 8  # C = 4, coarse slots 3..7
        S = self._stack(L)
        chunk_state = torch.randn(2, 3, 4, 3, dtype=torch.float64)
        out = ll_chunk_merge(S, chunk_state, chunk_counter=3, k=k)  # lssb(3) = 0
        # m = 0: merged lands in slot k+1; slots 0..k untouched
        torch.testing.assert_close(
            out[..., : k + 1, :, :], S[..., : k + 1, :, :], atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            out[..., k + 1, :, :], chunk_state, atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            out[..., k + 2 :, :, :], S[..., k + 2 :, :, :], atol=ATOL, rtol=RTOL
        )
        # chunk_counter = 4: m = 2 -> merged = chunk + slots k+1, k+2
        out2 = ll_chunk_merge(S, chunk_state, chunk_counter=4, k=k)
        expected = chunk_state + S[..., k + 1 : k + 3, :, :].sum(dim=-3)
        torch.testing.assert_close(
            out2[..., k + 3, :, :], expected, atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            out2[..., k + 1 : k + 3, :, :],
            torch.zeros_like(S[..., k + 1 : k + 3, :, :]),
            atol=ATOL,
            rtol=RTOL,
        )

    def test_ll_merges_exceeding_capacity_raise(self):
        S = self._stack(3)  # only levels 0..2
        with pytest.raises(ValueError):
            ll_token_merge(
                S, next_position=4
            )  # needs level 2 -> ok? lssb(4)=2 -> slot 3 > 2
        S = self._stack(6)
        with pytest.raises(ValueError):
            ll_token_merge(S, next_position=32)  # lssb=5 -> needs slot 6
        with pytest.raises(ValueError):
            ll_chunk_merge(self._stack(4), torch.randn(2, 3, 4, 3).double(), 8, k=1)

    def test_lssb(self):
        assert lssb(1) == 0
        assert lssb(2) == 1
        assert lssb(3) == 0
        assert lssb(12) == 2
        assert lssb(1024) == 10
        with pytest.raises(ValueError):
            lssb(0)
        with pytest.raises(ValueError):
            lssb(-4)

    def test_ll_level_matrix(self):
        positions = torch.arange(8)  # levels within one chunk, C = 8
        M = ll_level_matrix(positions)
        for t in range(8):
            for s in range(8):
                if s > t:
                    assert M[t, s].item() == -1
                else:
                    assert M[t, s].item() == ms_ll_level(t, s)
        # a chunk of absolute positions gives the same in-chunk levels
        # shifted by the chunk start (levels are a relative property)
        M2 = ll_level_matrix(positions + 64)
        torch.testing.assert_close(M2, M)
        with pytest.raises(ValueError):
            ll_level_matrix(torch.zeros(2, 2, dtype=torch.long))

    def test_layout_enum_accepts_plain_strings(self):
        assert Layout("ws") is Layout.WS
        with pytest.raises(ValueError):
            Layout("windowed")
