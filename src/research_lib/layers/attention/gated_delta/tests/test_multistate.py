"""Tests for the multi-state gated delta layer (M04–M17, M19).

All tests run in float64 on CPU with tolerances ``atol=1e-10, rtol=1e-10``
(spec §5: fp64 self-consistency; scale-aware).  Sizes per the spec:
``B=2, H=2, d_k=4, d_v=3``; ``T=32``, ``C=B=8``, ``W=2`` for the oracle
comparisons (M04).

The brute-force oracle (`multistate_oracle` in ``_oracle.py``) is the
primary reference: for every ``t`` and ``s <= t`` it places
``P_(s,t] k_s z_s^T`` into the bucket ``s`` occupies as seen from ``t``,
with explicit matrix products.  The oracle's bucket states are the
*post-write* states at ``t``; the layer's canonical cache after ``p``
tokens equals the schedule applied to the oracle's ``t = p - 1`` buckets
(the "pre-arrival" convention, spec §2.2 — tested by M19).

Run with::

    pytest src/research_lib/layers/attention/gated_delta/tests/test_multistate.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from ..functional import apply_gate_mode, chunk_scan, recurrent_scan, transition
from ..layouts import ll_token_merge, ws_shift
from ..multistate import (
    MultistateGatedDelta,
    multistate_chunk_scan,
    multistate_recurrent_scan,
)
from ..reference import GatedDeltaReference
from ._oracle import multistate_oracle

B, H, DK, DV = 2, 2, 4, 3
ATOL = RTOL = 1e-10
GATE_MODES = ["full", "b_only", "w_only", "kda", "gdn"]
LAYOUTS = ["ws", "fs", "ll"]
T_COMMON, C_COMMON, W_COMMON = 32, 8, 2  # spec: T=32, C=B=8, W=2
L_WS = W_COMMON + 2  # 4 slots
T_MAX_LL = 64  # -> 8 slots, enough for positions up to 64
L_LL = 7 + 1


def make_inputs(T: int, seed: int = 0, decay: float = 0.5) -> dict:
    """Random fp64 inputs with unit-norm keys and mild decay."""
    gen = torch.Generator().manual_seed(seed)
    f64 = {"dtype": torch.float64, "generator": gen}
    q = torch.randn(B, T, H, DK, **f64)
    k = F.normalize(torch.randn(B, T, H, DK, **f64), dim=-1)
    v = torch.randn(B, T, H, DV, **f64)
    g = -decay * torch.rand(B, T, H, DK, **f64)
    b = torch.rand(B, T, H, DK, **f64)
    w = torch.rand(B, T, H, DV, **f64)
    return {"q": q, "k": k, "v": v, "g": g, "b": b, "w": w}


def make_lam(T: int, L: int, seed: int = 1) -> torch.Tensor:
    """Random positive λ (the reduction uses exact ones, not these)."""
    gen = torch.Generator().manual_seed(seed)
    return 0.2 + 1.8 * torch.rand(B, T, H, L, dtype=torch.float64, generator=gen)


def layout_params(layout: str, block_size: int = C_COMMON) -> dict:
    """Standard per-layout scan arguments for the given block size."""
    if layout in ("ws", "fs"):
        return {"layout": layout, "window": W_COMMON, "block_size": block_size}
    return {"layout": layout, "window": 0, "block_size": 0}


def lam_levels(layout: str) -> int:
    return L_WS if layout in ("ws", "fs") else L_LL


def sched(S: torch.Tensor, p_next: int, layout: str) -> torch.Tensor:
    """Test-side composition of the schedule for arriving at ``p_next``.

    The oracle's post-write buckets at ``t = p_next - 1`` become the
    canonical cache at ``p_next`` by applying the layout's schedule.
    """
    if layout in ("ws", "fs"):
        if p_next % C_COMMON == 0:
            return ws_shift(S, W_COMMON)
        return S
    return ll_token_merge(S, p_next)


def per_bh(t: torch.Tensor, bi: int, hi: int) -> torch.Tensor:
    """Slice a (B, T, H, ...) tensor to a single (batch, head): (T, ...)."""
    return t[bi, :, hi, ...]


class TestM04RecurrentMatchesOracle:
    """M04: recurrent == brute-force oracle; outputs and every bucket state."""

    @pytest.mark.slow
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_M04_outputs_and_bucket_states(self, layout):
        T = T_COMMON
        inp = make_inputs(T)
        lam = make_lam(T, lam_levels(layout))
        # Per-token recurrent calls capture the canonical state at each t.
        S = None
        for t in range(T):
            o_t, S = multistate_recurrent_scan(
                inp["q"][:, t : t + 1],
                inp["k"][:, t : t + 1],
                inp["v"][:, t : t + 1],
                inp["g"][:, t : t + 1],
                inp["b"][:, t : t + 1],
                inp["w"][:, t : t + 1],
                lam[:, t : t + 1],
                position=t,
                initial_state=S,
                **layout_params(layout),
            )
            for bi in range(B):
                for hi in range(H):
                    o_oracle, Sb = multistate_oracle(
                        per_bh(inp["q"], bi, hi)[: t + 1],
                        per_bh(inp["k"], bi, hi)[: t + 1],
                        per_bh(inp["v"], bi, hi)[: t + 1],
                        per_bh(inp["g"], bi, hi)[: t + 1],
                        per_bh(inp["b"], bi, hi)[: t + 1],
                        per_bh(inp["w"], bi, hi)[: t + 1],
                        per_bh(lam, bi, hi)[: t + 1],
                        layout,
                        block_size=C_COMMON,
                        window=W_COMMON,
                    )
                    # output at t
                    torch.testing.assert_close(
                        o_t[bi, 0, hi],
                        o_oracle[t],
                        atol=ATOL,
                        rtol=RTOL,
                        msg=f"{layout} output at t={t}",
                    )
                    # canonical state at t+1 == sched(oracle post-write at t)
                    expected = sched(Sb[t], t + 1, layout)
                    torch.testing.assert_close(
                        S[bi, hi],
                        expected,
                        atol=ATOL,
                        rtol=RTOL,
                        msg=f"{layout} canonical bucket state at p={t + 1}",
                    )


class TestM05ChunkMatchesRecurrent:
    """M05: chunk == recurrent, every layout × every gate mode; ragged tail."""

    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize("gate_mode", GATE_MODES)
    @pytest.mark.parametrize("T", [32, 37])
    def test_M05_chunk_equals_recurrent(self, layout, gate_mode, T):
        inp = make_inputs(T, seed=hash((layout, gate_mode)) % 1000)
        lam = make_lam(T, lam_levels(layout), seed=2)
        b, w, g = apply_gate_mode(inp["b"], inp["w"], inp["g"], gate_mode)
        args = dict(
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            g=g,
            b=b,
            w=w,
            lam=lam,
            **layout_params(layout),
        )
        o_r, S_r = multistate_recurrent_scan(**args)
        o_c, S_c = multistate_chunk_scan(**args, chunk_size=C_COMMON)
        torch.testing.assert_close(o_c, o_r, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S_c, S_r, atol=ATOL, rtol=RTOL)


class TestM06LambdaOneReductions:
    """M06: λ≡1 reductions and the single-layout reference equivalence."""

    @pytest.mark.parametrize("layout", ["ws", "ll"])
    @pytest.mark.parametrize("path", ["recurrent", "chunk"])
    def test_M06_lambda_one_equals_single_functional(self, layout, path):
        T = T_COMMON
        inp = make_inputs(T)
        lam = torch.ones(B, T, H, lam_levels(layout), dtype=torch.float64)
        args = dict(
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            g=inp["g"],
            b=inp["b"],
            w=inp["w"],
            lam=lam,
            **layout_params(layout),
        )
        if path == "recurrent":
            o_ms, _ = multistate_recurrent_scan(**args)
            o_ref, _ = recurrent_scan(
                inp["q"], inp["k"], inp["v"], inp["g"], inp["b"], inp["w"]
            )
        else:
            o_ms, _ = multistate_chunk_scan(**args, chunk_size=C_COMMON)
            o_ref, _ = chunk_scan(
                inp["q"],
                inp["k"],
                inp["v"],
                inp["g"],
                inp["b"],
                inp["w"],
                chunk_size=C_COMMON,
            )
        torch.testing.assert_close(o_ms, o_ref, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("path", ["recurrent", "chunk"])
    def test_M06_single_layout_is_the_reference_layer(self, path):
        """layout='single' loads a GatedDeltaReference state dict with
        strict=True and produces identical outputs."""
        d = 32
        torch.manual_seed(0)
        ref = GatedDeltaReference(
            hidden_size=d, num_heads=H, head_dim=DK, dtype=torch.float64
        )
        ms = MultistateGatedDelta(
            hidden_size=d,
            num_heads=H,
            head_dim=DK,
            layout="single",
            dtype=torch.float64,
        )
        ms.load_state_dict(ref.state_dict(), strict=True)
        x = torch.randn(B, 8, d, dtype=torch.float64)
        y_ref, _ = ref(x, mode=path)
        y_ms, _ = ms(x, mode=path)
        torch.testing.assert_close(y_ms, y_ref, atol=ATOL, rtol=RTOL)


class TestM07Invariant:
    """M07: the bucket states sum to the single GDN-2 state."""

    @staticmethod
    def _single_state(inp: dict, bi: int, hi: int, p: int) -> torch.Tensor:
        """Post-write single state after the first ``p`` tokens, via the
        transition primitive (independent of the chunk machinery)."""
        S = torch.zeros(DK, DV, dtype=torch.float64)
        for s in range(p):
            k_s = inp["k"][bi, s, hi]
            S = transition(S, k_s, inp["b"][bi, s, hi], inp["g"][bi, s, hi])
            S = S + torch.outer(k_s, inp["w"][bi, s, hi] * inp["v"][bi, s, hi])
        return S

    @pytest.mark.parametrize("layout", ["ws", "ll"])
    def test_M07_recurrent_every_token(self, layout):
        T = T_COMMON
        inp = make_inputs(T)
        L = lam_levels(layout)
        S = None
        for t in range(T):
            _, S = multistate_recurrent_scan(
                inp["q"][:, t : t + 1],
                inp["k"][:, t : t + 1],
                inp["v"][:, t : t + 1],
                inp["g"][:, t : t + 1],
                inp["b"][:, t : t + 1],
                inp["w"][:, t : t + 1],
                torch.ones(B, 1, H, L, dtype=torch.float64),
                position=t,
                initial_state=S,
                **layout_params(layout),
            )
            for bi in range(B):
                for hi in range(H):
                    torch.testing.assert_close(
                        S[bi, hi].sum(dim=0),
                        self._single_state(inp, bi, hi, t + 1),
                        atol=ATOL,
                        rtol=RTOL,
                        msg=f"{layout}: invariant at p={t + 1}",
                    )

    @pytest.mark.parametrize("layout", ["ws", "ll"])
    def test_M07_chunk_boundaries(self, layout):
        T = T_COMMON
        inp = make_inputs(T)
        lam = torch.ones(B, T, H, lam_levels(layout), dtype=torch.float64)
        for n_chunks in range(1, T // C_COMMON + 1):
            p = n_chunks * C_COMMON
            _, S_c = multistate_chunk_scan(
                inp["q"][:, :p],
                inp["k"][:, :p],
                inp["v"][:, :p],
                inp["g"][:, :p],
                inp["b"][:, :p],
                inp["w"][:, :p],
                lam[:, :p],
                chunk_size=C_COMMON,
                **layout_params(layout),
            )
            for bi in range(B):
                for hi in range(H):
                    torch.testing.assert_close(
                        S_c[bi, hi].sum(dim=0),
                        self._single_state(inp, bi, hi, p),
                        atol=ATOL,
                        rtol=RTOL,
                        msg=f"{layout}: chunk-boundary invariant at p={p}",
                    )


class TestM08FsIsARealControl:
    """M08: fs with λ≡1 differs from single when an erase crosses a block."""

    def test_M08_fs_lambda_one_differs_from_single(self):
        T = 24  # crosses block boundaries with B = 8
        inp = make_inputs(T, seed=3)
        lam = torch.ones(B, T, H, L_WS, dtype=torch.float64)
        o_fs, _ = multistate_recurrent_scan(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            inp["b"],
            inp["w"],
            lam,
            **layout_params("fs"),
        )
        o_single, _ = recurrent_scan(
            inp["q"], inp["k"], inp["v"], inp["g"], inp["b"], inp["w"]
        )
        assert not torch.allclose(o_fs, o_single, atol=1e-6, rtol=1e-6), (
            "fs with λ≡1 must differ from single-state GDN-2 when erases "
            "cross block boundaries; otherwise the control is not a control"
        )


class TestM09ToyOverwrite:
    """M09: the exact reads of 05 §5 (unit key, b = w = 1, no decay)."""

    @staticmethod
    def _run(layout: str):
        T, Blk, W = 5, 2, 1
        k = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        v1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        v2 = torch.tensor([-4.0, 0.5, 1.0], dtype=torch.float64)
        q = k.repeat(T, 1)
        kk = k.repeat(T, 1)
        vv = torch.randn(T, DV, dtype=torch.float64)
        gg = torch.zeros(T, DK, dtype=torch.float64)
        bb = torch.zeros(T, DK, dtype=torch.float64)
        ww = torch.zeros(T, DV, dtype=torch.float64)
        # writes (and erases) only at t=0 (v1) and t=4 (v2); other tokens are
        # orthogonal and non-erasing.
        gen = torch.Generator().manual_seed(4)
        for t in range(T):
            if t in (0, 4):
                vv[t] = v1 if t == 0 else v2
                bb[t] = 1.0
                ww[t] = 1.0
            else:
                x = torch.randn(DK, dtype=torch.float64, generator=gen)
                x[0] = 0.0
                kk[t] = F.normalize(x, dim=0)
        lam = torch.tensor([0.7, 1.3, 0.4], dtype=torch.float64).repeat(T, 1)
        _, S = multistate_recurrent_scan(
            q.view(1, T, 1, DK),
            kk.view(1, T, 1, DK),
            vv.view(1, T, 1, DV),
            gg.view(1, T, 1, DK),
            bb.view(1, T, 1, DK),
            ww.view(1, T, 1, DV),
            lam.view(1, T, 1, 3),
            layout=layout,
            window=W,
            block_size=Blk,
        )
        # canonical(5) is post(4) (5 is mid-block): read with q = k.
        read = torch.zeros(DV, dtype=torch.float64)
        for lvl in range(3):
            read += lam[4, lvl] * (S[0, 0, lvl].T @ k)
        return read, lam[4], v1, v2

    def test_M09_ws_reads_only_the_new_value(self):
        read, lam, v1, v2 = self._run("ws")
        torch.testing.assert_close(read, lam[0] * v2, atol=ATOL, rtol=RTOL)

    def test_M09_fs_keeps_the_old_value(self):
        read, lam, v1, v2 = self._run("fs")
        torch.testing.assert_close(
            read, lam[0] * v2 + lam[2] * v1, atol=ATOL, rtol=RTOL
        )


class TestM10Gradients:
    """M10: autograd through the chunk vs the recurrent path, each layout."""

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_M10_chunk_and_recurrent_gradients(self, layout):
        T = T_COMMON
        inp = make_inputs(T, seed=5)
        lam0 = make_lam(T, lam_levels(layout), seed=6)
        S0 = torch.randn(B, H, lam_levels(layout), DK, DV, dtype=torch.float64)
        if layout == "ll":
            # Canonical cache form at a chunk boundary: the intra-chunk
            # levels (slots 0..k) are zero; only the coarse levels carry a
            # state. (The chunk path asserts this; a random S0 would be
            # silently misinterpreted as zero in the fine slots.)
            k = C_COMMON.bit_length() - 1
            S0[..., : k + 1, :, :] = 0
        names = ["q", "k", "v", "g", "b", "w", "lam", "S0"]

        def run(path_fn):
            leaves = {
                "q": inp["q"].clone().requires_grad_(True),
                "k": inp["k"].clone().requires_grad_(True),
                "v": inp["v"].clone().requires_grad_(True),
                "g": inp["g"].clone().requires_grad_(True),
                "b": inp["b"].clone().requires_grad_(True),
                "w": inp["w"].clone().requires_grad_(True),
                "lam": lam0.clone().requires_grad_(True),
                "S0": S0.clone().requires_grad_(True),
            }
            o, S = path_fn(leaves)
            loss = (o * dO).sum() + (S * dS).sum()
            loss.backward()
            return {
                n: (p.grad if p.grad is not None else torch.zeros_like(p))
                for n, p in leaves.items()
            }

        # One shared upstream, injected into both losses.
        gen = torch.Generator().manual_seed(7)
        dO = torch.randn(B, T, H, DV, dtype=torch.float64, generator=gen)
        dS = torch.randn(
            B, H, lam_levels(layout), DK, DV, dtype=torch.float64, generator=gen
        )

        def recurrent_path(leaves):
            return multistate_recurrent_scan(
                leaves["q"],
                leaves["k"],
                leaves["v"],
                leaves["g"],
                leaves["b"],
                leaves["w"],
                leaves["lam"],
                initial_state=leaves["S0"],
                **layout_params(layout),
            )

        def chunk_path(leaves):
            return multistate_chunk_scan(
                leaves["q"],
                leaves["k"],
                leaves["v"],
                leaves["g"],
                leaves["b"],
                leaves["w"],
                leaves["lam"],
                initial_state=leaves["S0"],
                chunk_size=C_COMMON,
                **layout_params(layout),
            )

        grads_r = run(recurrent_path)
        grads_c = run(chunk_path)
        for name in names:
            g_r, g_c = grads_r[name], grads_c[name]
            if name == "S0" and layout == "ll":
                # The canonical cache form constrains the LL fine slots
                # (0..k) to zero; the two paths are gradients of the same
                # function on that submanifold but differ in how they extend
                # perturbations of the constrained zeros (the recurrent
                # carries mix fine slots upward, the chunk path recomputes
                # them from Z).  Compare only the unconstrained coarse slots.
                g_r = g_r[:, :, k + 1 :]
                g_c = g_c[:, :, k + 1 :]
            torch.testing.assert_close(
                g_c,
                g_r,
                atol=ATOL,
                rtol=RTOL,
                msg=f"{layout}: gradient mismatch for {name}",
            )


class TestM11LambdaOneAtInit:
    """M11: λ≡1 at init; identical to the reference; λ gradients nonzero."""

    @staticmethod
    def _build_pair(layout="ws"):
        d = 32
        torch.manual_seed(0)
        ref = GatedDeltaReference(
            hidden_size=d, num_heads=H, head_dim=DK, dtype=torch.float64
        )
        ms = MultistateGatedDelta(
            hidden_size=d,
            num_heads=H,
            head_dim=DK,
            layout=layout,
            lambda_mode="mlp",
            window=W_COMMON,
            block_size=C_COMMON,
            chunk_size=C_COMMON,
            t_max=T_MAX_LL,
            dtype=torch.float64,
        )
        ms.load_state_dict(ref.state_dict(), strict=False)
        return ref, ms

    def test_M11_output_equals_reference_at_init(self):
        ref, ms = self._build_pair("ws")
        torch.manual_seed(1)
        x = torch.randn(B, 16, 32, dtype=torch.float64)
        y_ref, _ = ref(x)
        y_ms, _ = ms(x)
        torch.testing.assert_close(y_ms, y_ref, atol=ATOL, rtol=RTOL)

    def test_M11_shared_gradients_equal_and_lambda_moves(self):
        ref, ms = self._build_pair("ws")
        torch.manual_seed(2)
        x = torch.randn(B, 16, 32, dtype=torch.float64)
        y_ref, _ = ref(x)
        y_ms, _ = ms(x)
        upstream = torch.randn_like(y_ref)  # same upstream for both losses
        (y_ref * upstream).sum().backward()
        (y_ms * upstream).sum().backward()
        ref_params = dict(ref.named_parameters())
        for name, p in ms.named_parameters():
            if name in ref_params:
                torch.testing.assert_close(
                    p.grad,
                    ref_params[name].grad,
                    atol=ATOL,
                    rtol=RTOL,
                    msg=f"shared parameter {name} must have equal gradients",
                )
        lam_names = [
            n for n, _ in ms.named_parameters() if n.startswith("lambda_mixer")
        ]
        assert lam_names, "the multi-state layout must have λ parameters"
        moved = [
            n
            for n, p in ms.named_parameters()
            if n in lam_names and p.grad is not None and p.grad.abs().max() > 0
        ]
        assert moved, "λ parameters must receive nonzero gradients"


class TestM12Causality:
    """M12: perturbing x[:, t'] leaves y[:, :t'] unchanged, all layouts."""

    @pytest.mark.parametrize("layout", LAYOUTS + ["single"])
    @pytest.mark.parametrize("path", ["chunk", "recurrent"])
    def test_M12_causality(self, layout, path):
        d = 32
        torch.manual_seed(0)
        layer = MultistateGatedDelta(
            hidden_size=d,
            num_heads=H,
            head_dim=DK,
            layout=layout,
            window=W_COMMON,
            block_size=C_COMMON,
            chunk_size=C_COMMON,
            t_max=T_MAX_LL,
            dtype=torch.float64,
        )
        torch.manual_seed(3)
        T = 21
        x = torch.randn(B, T, d, dtype=torch.float64)
        y1, _ = layer(x, mode=path)
        t_perturb = 13
        x2 = x.clone()
        x2[:, t_perturb:] = x2[:, t_perturb:] + torch.randn_like(x2[:, t_perturb:])
        y2, _ = layer(x2, mode=path)
        torch.testing.assert_close(
            y1[:, :t_perturb], y2[:, :t_perturb], atol=ATOL, rtol=RTOL
        )


class TestM13DecodeHandoff:
    """M13: chunk prefill then recurrent decode equals one recurrent pass."""

    @pytest.mark.parametrize("layout", ["ws", "ll"])
    def test_M13_prefill_then_decode(self, layout):
        d = 32
        torch.manual_seed(0)
        layer = MultistateGatedDelta(
            hidden_size=d,
            num_heads=H,
            head_dim=DK,
            layout=layout,
            window=W_COMMON,
            block_size=C_COMMON,
            chunk_size=C_COMMON,  # ws crosses block boundaries; ll crosses
            t_max=T_MAX_LL,  # Fenwick carries (incl. position 32)
            dtype=torch.float64,
        )
        torch.manual_seed(4)
        T_prefill, T_decode = 16, 24
        x = torch.randn(B, T_prefill + T_decode, d, dtype=torch.float64)
        # one recurrent pass over everything
        y_ref, _ = layer(x, mode="recurrent")
        # chunk prefill then one-token recurrent decode
        y, cache = layer(x[:, :T_prefill], use_cache=True, mode="chunk")
        outs = [y]
        for t in range(T_prefill, T_prefill + T_decode):
            y_t, cache = layer(
                x[:, t : t + 1], cache=cache, use_cache=True, mode="recurrent"
            )
            outs.append(y_t)
        y_dec = torch.cat(outs, dim=1)
        torch.testing.assert_close(y_dec, y_ref, atol=ATOL, rtol=RTOL)
        assert cache.position == T_prefill + T_decode

    @pytest.mark.parametrize("layout", ["ws", "ll"])
    def test_M13b_prefill_then_chunk_decode(self, layout):
        """M13 variant: decode steps in mode="chunk" too.  A one-token
        chunked call has n_head capped at T=1, so it is handled by the
        head/tail paths — but through the chunk-mode entry, which is what
        a chunk-mode-only decode loop uses."""
        d = 32
        torch.manual_seed(0)
        layer = MultistateGatedDelta(
            hidden_size=d,
            num_heads=H,
            head_dim=DK,
            layout=layout,
            window=W_COMMON,
            block_size=C_COMMON,
            chunk_size=C_COMMON,
            t_max=T_MAX_LL,
            dtype=torch.float64,
        )
        torch.manual_seed(4)
        T_prefill, T_decode = 16, 24
        x = torch.randn(B, T_prefill + T_decode, d, dtype=torch.float64)
        # one recurrent pass over everything
        y_ref, _ = layer(x, mode="recurrent")
        # chunk prefill, then one-token CHUNK-mode decode steps
        y, cache = layer(x[:, :T_prefill], use_cache=True, mode="chunk")
        outs = [y]
        for t in range(T_prefill, T_prefill + T_decode):
            y_t, cache = layer(
                x[:, t : t + 1], cache=cache, use_cache=True, mode="chunk"
            )
            outs.append(y_t)
        y_dec = torch.cat(outs, dim=1)
        torch.testing.assert_close(y_dec, y_ref, atol=ATOL, rtol=RTOL)
        assert cache.position == T_prefill + T_decode


class TestM14bShortChunkCalls:
    """M14b: chunk-mode calls shorter than the distance to the next
    boundary.  The head must be capped at T (a call from position 5 with
    C=16 and T=1 has n_head=1, not 11); uncapped, n_full would go negative
    (and -1 is truthy, so the full-chunk reshape would raise) and t_tail
    would be -1, reprocessing the last token."""

    SHORT_POSITIONS = [17, 5, 31]
    SHORT_T = [1, 3, 15, 16, 17]
    C_SHORT = 16

    @pytest.mark.parametrize("layout", LAYOUTS)
    @pytest.mark.parametrize("position", SHORT_POSITIONS)
    @pytest.mark.parametrize("T", SHORT_T)
    def test_M14b_short_calls_match_recurrent(self, layout, position, T):
        # canonical cache at `position` (built by the recurrent path)
        p_inp = make_inputs(position, seed=21)
        p_lam = make_lam(position, lam_levels(layout), seed=22)
        _, S_p = multistate_recurrent_scan(
            p_inp["q"],
            p_inp["k"],
            p_inp["v"],
            p_inp["g"],
            p_inp["b"],
            p_inp["w"],
            p_lam,
            position=0,
            **layout_params(layout, block_size=self.C_SHORT),
        )
        inp = make_inputs(T, seed=23)
        lam = make_lam(T, lam_levels(layout), seed=24)
        o_c, S_c = multistate_chunk_scan(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            inp["b"],
            inp["w"],
            lam,
            initial_state=S_p,
            position=position,
            chunk_size=self.C_SHORT,
            **layout_params(layout, block_size=self.C_SHORT),
        )
        o_r, S_r = multistate_recurrent_scan(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            inp["b"],
            inp["w"],
            lam,
            initial_state=S_p,
            position=position,
            **layout_params(layout, block_size=self.C_SHORT),
        )
        torch.testing.assert_close(o_c, o_r, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S_c, S_r, atol=ATOL, rtol=RTOL)


class TestM14MisalignedChunkStart:
    """M14: chunk mode from a misaligned cache equals the recurrent path."""

    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_M14_misaligned_start(self, layout):
        T, p_mis = 32, 5
        inp = make_inputs(T, seed=8)
        lam = make_lam(T, lam_levels(layout), seed=9)
        # recurrent pass to position 5, then chunk from the misaligned cache
        _, S5 = multistate_recurrent_scan(
            inp["q"][:, :p_mis],
            inp["k"][:, :p_mis],
            inp["v"][:, :p_mis],
            inp["g"][:, :p_mis],
            inp["b"][:, :p_mis],
            inp["w"][:, :p_mis],
            lam[:, :p_mis],
            position=0,
            **layout_params(layout),
        )
        o_c, S_c = multistate_chunk_scan(
            inp["q"][:, p_mis:],
            inp["k"][:, p_mis:],
            inp["v"][:, p_mis:],
            inp["g"][:, p_mis:],
            inp["b"][:, p_mis:],
            inp["w"][:, p_mis:],
            lam[:, p_mis:],
            initial_state=S5,
            position=p_mis,
            chunk_size=C_COMMON,
            **layout_params(layout),
        )
        o_r, S_r = multistate_recurrent_scan(
            inp["q"][:, p_mis:],
            inp["k"][:, p_mis:],
            inp["v"][:, p_mis:],
            inp["g"][:, p_mis:],
            inp["b"][:, p_mis:],
            inp["w"][:, p_mis:],
            lam[:, p_mis:],
            initial_state=S5,
            position=p_mis,
            **layout_params(layout),
        )
        torch.testing.assert_close(o_c, o_r, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S_c, S_r, atol=ATOL, rtol=RTOL)


class TestChunkScanDtypeContract:
    """The chunk scan returns the output in the input dtype (the head/tail
    recurrent parts already do; the full-chunk parts are cast after the
    concatenation), and the state in the compute dtype."""

    def test_bf16_ragged_head_and_tail_returns_bf16(self):
        T, pos, C = 20, 3, 8  # head 5, one full chunk, ragged tail 7
        gen = torch.Generator().manual_seed(42)
        f = {"dtype": torch.bfloat16, "generator": gen}
        q = torch.randn(B, T, H, DK, **f)
        k = F.normalize(torch.randn(B, T, H, DK, **f), dim=-1)
        v = torch.randn(B, T, H, DV, **f)
        g = -torch.rand(B, T, H, DK, **f)
        b = torch.rand(B, T, H, DK, **f)
        w = torch.rand(B, T, H, DV, **f)
        lam = 0.2 + 1.8 * torch.rand(B, T, H, L_WS, **f)
        o_c, S_c = multistate_chunk_scan(
            q,
            k,
            v,
            g,
            b,
            w,
            lam,
            layout="ws",
            window=W_COMMON,
            block_size=C,
            position=pos,
            chunk_size=C,
        )
        o_r, S_r = multistate_recurrent_scan(
            q,
            k,
            v,
            g,
            b,
            w,
            lam,
            layout="ws",
            window=W_COMMON,
            block_size=C,
            position=pos,
        )
        assert o_c.dtype == torch.bfloat16  # the input dtype, not fp32
        assert S_c.dtype == torch.float32  # the state stays in the compute dtype
        # both paths compute in fp32 and round to bf16; agree to bf16 ulps
        torch.testing.assert_close(o_c, o_r, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(S_c, S_r, atol=1e-4, rtol=1e-4)


class TestM15ParameterAccounting:
    """M15: gate modes identical; layouts differ only by the λ module."""

    def test_M15_gate_modes_have_identical_counts(self):
        d = 32
        counts = {}
        for gm in GATE_MODES:
            layer = MultistateGatedDelta(
                hidden_size=d,
                num_heads=H,
                head_dim=DK,
                gate_mode=gm,
                layout="ws",
                window=W_COMMON,
                block_size=C_COMMON,
                chunk_size=C_COMMON,
                t_max=T_MAX_LL,
                dtype=torch.float64,
            )
            counts[gm] = (
                sum(p.numel() for p in layer.parameters()),
                set(layer.state_dict()),
            )
        base = counts["full"]
        for gm, (num, keys) in counts.items():
            assert num == base[0], gm
            assert keys == base[1], gm

    def test_M15_layouts_differ_only_by_lambda(self):
        d = 32

        def build(layout):
            return MultistateGatedDelta(
                hidden_size=d,
                num_heads=H,
                head_dim=DK,
                layout=layout,
                lambda_mode="mlp",
                window=W_COMMON,
                block_size=C_COMMON,
                chunk_size=C_COMMON,
                t_max=T_MAX_LL,
                dtype=torch.float64,
            )

        single = build("single")
        ws = build("ws")
        ll = build("ll")
        n_single = sum(p.numel() for p in single.parameters())
        n_ws = sum(p.numel() for p in ws.parameters())
        n_ll = sum(p.numel() for p in ll.parameters())
        lam_ws = sum(
            p.numel() for n, p in ws.named_parameters() if n.startswith("lambda_mixer")
        )
        lam_ll = sum(
            p.numel() for n, p in ll.named_parameters() if n.startswith("lambda_mixer")
        )
        # single has no λ parameters at all
        assert not any(
            n.startswith("lambda_mixer") for n, _ in single.named_parameters()
        )
        assert n_ws == n_single + lam_ws
        assert n_ll == n_single + lam_ll
        assert lam_ws != lam_ll  # different slot counts (W+2 vs bitlen+1)

    def test_M15_projections_names_match_reference(self):
        d = 32
        ref = GatedDeltaReference(hidden_size=d, num_heads=H, head_dim=DK)
        ms = MultistateGatedDelta(
            hidden_size=d,
            num_heads=H,
            head_dim=DK,
            layout="single",
        )
        ref_keys = {n.split(".", 1)[1] for n, _ in ref.named_parameters()}
        ms_keys = {n.split(".", 1)[1] for n, _ in ms.named_parameters()}
        assert ms_keys == ref_keys


class TestM16StrongDecay:
    """M16: strong decay (g ≈ -2 per step): forward and backward finite."""

    @pytest.mark.parametrize("layout", ["ws", "ll"])
    def test_M16_strong_decay_finite(self, layout):
        T, C = 32, 16
        inp = make_inputs(T, seed=10)
        inp["g"] = -2.0 - torch.rand(B, T, H, DK, dtype=torch.float64)
        lam = make_lam(T, lam_levels(layout), seed=11)
        if layout == "ws":
            lp = {"window": 2, "block_size": C}
        else:
            lp = {"window": 0, "block_size": 0}
        leaves = [
            inp["q"].requires_grad_(True),
            inp["k"].requires_grad_(True),
            inp["v"].requires_grad_(True),
            inp["g"].requires_grad_(True),
            inp["b"].requires_grad_(True),
            inp["w"].requires_grad_(True),
            lam.requires_grad_(True),
        ]
        o, S = multistate_chunk_scan(
            leaves[0],
            leaves[1],
            leaves[2],
            leaves[3],
            leaves[4],
            leaves[5],
            leaves[6],
            layout=layout,
            chunk_size=C,
            **lp,
        )
        loss = (o * torch.randn_like(o)).sum() + (S * torch.randn_like(S)).sum()
        loss.backward()
        for i, t in enumerate([o, S, *leaves]):
            assert torch.isfinite(t).all(), f"{layout}: non-finite tensor {i}"
        for i, leaf in enumerate(leaves):
            assert torch.isfinite(leaf.grad).all(), f"{layout}: grad {i} non-finite"


class TestM17Validation:
    """M17: constructor and argument validation."""

    def _kwargs(self, **overrides):
        base = {
            "hidden_size": 32,
            "num_heads": H,
            "head_dim": DK,
            "layout": "ws",
            "window": W_COMMON,
            "block_size": C_COMMON,
            "chunk_size": C_COMMON,
            "t_max": T_MAX_LL,
        }
        base.update(overrides)
        return base

    def test_M17_ws_block_size_mismatch(self):
        with pytest.raises(ValueError, match="block_size == chunk_size"):
            MultistateGatedDelta(**self._kwargs(block_size=4, chunk_size=8))

    def test_M17_ll_non_power_of_two_chunk(self):
        with pytest.raises(ValueError, match="power of two"):
            MultistateGatedDelta(**self._kwargs(layout="ll", chunk_size=12))

    def test_M17_ll_t_max_too_small(self):
        with pytest.raises(ValueError, match="t_max"):
            MultistateGatedDelta(**self._kwargs(layout="ll", chunk_size=16, t_max=8))

    def test_M17_unknown_strings(self):
        with pytest.raises(ValueError, match="layout must be one of"):
            MultistateGatedDelta(**self._kwargs(layout="windowed"))
        with pytest.raises(ValueError, match="gate_mode must be one of"):
            MultistateGatedDelta(**self._kwargs(gate_mode="soft"))
        with pytest.raises(ValueError, match="lambda_mode must be one of"):
            MultistateGatedDelta(**self._kwargs(lambda_mode="quadratic"))
        with pytest.raises(ValueError, match="mode must be"):
            MultistateGatedDelta(**self._kwargs(mode="fast"))

    def test_M17_ws_window_zero(self):
        with pytest.raises(ValueError, match="window"):
            MultistateGatedDelta(**self._kwargs(window=0, block_size=8, chunk_size=8))

    def test_M17_forward_mode_override_validation(self):
        layer = MultistateGatedDelta(**self._kwargs())
        x = torch.randn(1, 4, 32)
        with pytest.raises(ValueError, match="mode must be"):
            layer(x, mode="fast")

    def test_M17_scan_input_validation(self):
        inp = make_inputs(4, seed=12)
        lam = torch.ones(B, 4, H, L_WS, dtype=torch.float64)
        bad_state = torch.randn(B, H, 99, DK, DV, dtype=torch.float64)
        with pytest.raises(ValueError, match="initial_state"):
            multistate_recurrent_scan(
                inp["q"],
                inp["k"],
                inp["v"],
                inp["g"],
                inp["b"],
                inp["w"],
                lam,
                initial_state=bad_state,
                **layout_params("ws"),
            )
        with pytest.raises(ValueError, match="block_size == chunk_size"):
            multistate_chunk_scan(
                inp["q"],
                inp["k"],
                inp["v"],
                inp["g"],
                inp["b"],
                inp["w"],
                lam,
                layout="ws",
                window=W_COMMON,
                block_size=4,
                chunk_size=8,
            )
        with pytest.raises(ValueError, match="power of two"):
            multistate_chunk_scan(
                inp["q"],
                inp["k"],
                inp["v"],
                inp["g"],
                inp["b"],
                inp["w"],
                torch.ones(B, 4, H, L_LL, dtype=torch.float64),
                layout="ll",
                window=0,
                block_size=0,
                chunk_size=12,
            )
        with pytest.raises(ValueError, match="single"):
            multistate_recurrent_scan(
                inp["q"],
                inp["k"],
                inp["v"],
                inp["g"],
                inp["b"],
                inp["w"],
                lam,
                layout="single",
                window=0,
                block_size=0,
            )


class TestM19CanonicalCache:
    """M19: cache.state == the oracle's pre-arrival buckets at p."""

    @pytest.mark.slow
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_M19_canonical_form_on_and_off_boundaries(self, layout):
        d = 32
        torch.manual_seed(0)
        layer = MultistateGatedDelta(
            hidden_size=d,
            num_heads=H,
            head_dim=DK,
            layout=layout,
            window=W_COMMON,
            block_size=C_COMMON,
            chunk_size=C_COMMON,
            t_max=T_MAX_LL,
            dtype=torch.float64,
        )
        torch.manual_seed(5)
        T = 21
        x = torch.randn(B, T, d, dtype=torch.float64)
        # projected inputs (what the oracle needs), gate-mode 'full'
        q, k, v, b, w, g, _ = layer.projections.project(x)
        for p in (5, 8, 13, 16, 20):  # off/on block and chunk boundaries
            _, cache_r = layer(x[:, :p], use_cache=True, mode="recurrent")
            _, cache_c = layer(x[:, :p], use_cache=True, mode="chunk")
            # both paths end in the same canonical cache
            torch.testing.assert_close(
                cache_r.state,
                cache_c.state,
                atol=ATOL,
                rtol=RTOL,
                msg=f"{layout}: chunk and recurrent caches differ at p={p}",
            )
            assert cache_r.position == p
            for bi in range(B):
                for hi in range(H):
                    o_oracle, Sb = multistate_oracle(
                        per_bh(q, bi, hi)[:p],
                        per_bh(k, bi, hi)[:p],
                        per_bh(v, bi, hi)[:p],
                        per_bh(g, bi, hi)[:p],
                        per_bh(b, bi, hi)[:p],
                        per_bh(w, bi, hi)[:p],
                        torch.ones(p, lam_levels(layout), dtype=torch.float64),
                        layout,
                        block_size=C_COMMON,
                        window=W_COMMON,
                    )
                    expected = sched(Sb[p - 1], p, layout)
                    torch.testing.assert_close(
                        cache_r.state[bi, hi],
                        expected,
                        atol=ATOL,
                        rtol=RTOL,
                        msg=f"{layout}: canonical cache at p={p}",
                    )
