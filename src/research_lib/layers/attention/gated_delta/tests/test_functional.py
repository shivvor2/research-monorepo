"""Functional-core tests for the gated delta-rule family (G01–G08, G14).

All tests run in float64 on CPU with tolerances ``atol=1e-10, rtol=1e-10``
(fp64 self-consistency; fla cross-checks live in ``test_fla_crosscheck.py``
with fp32 tolerances).  Small sizes per the spec: ``B=2, H=2, d_k=4,
d_v=3``.

Run with::

    pytest src/research_lib/layers/attention/gated_delta/tests/test_functional.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from ..functional import (
    GateMode,
    apply_gate_mode,
    chunk_scan,
    recurrent_scan,
    transition,
)
from ._oracle import brute_force_oracle, literal_gdn, literal_gdn2, literal_kda

B, H, DK, DV = 2, 2, 4, 3
GATE_MODES = ["full", "b_only", "w_only", "kda", "gdn"]
ATOL = RTOL = 1e-10


def _grad_msg(name: str):
    """Message factory that binds ``name`` eagerly (no loop late-binding)."""

    def msg(m: str) -> str:
        return f"gradient mismatch for {name}: {m}"

    return msg


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
    S0 = torch.randn(B, H, DK, DV, **f64)
    return {"q": q, "k": k, "v": v, "g": g, "b": b, "w": w, "S0": S0}


# =============================================================================
# G01: transition vs literal matrix form
# =============================================================================


class TestG01Transition:
    """`transition` must equal the explicit matrix product
    (I - k (b*k)^T) Diag(exp(g)) S."""

    def test_G01_transition_vs_literal_matrix_form(self):
        for seed in range(3):
            inp = make_inputs(T=5, seed=seed)
            eye = torch.eye(DK, dtype=torch.float64)
            for t in range(5):
                for bi in range(B):
                    for hi in range(H):
                        k = inp["k"][bi, t, hi]
                        b = inp["b"][bi, t, hi]
                        g = inp["g"][bi, t, hi]
                        S = inp["S0"][bi, hi]
                        expected = (
                            (eye - torch.outer(k, b * k)) @ torch.diag(g.exp()) @ S
                        )
                        actual = transition(S, k, b, g)
                        torch.testing.assert_close(
                            actual, expected, atol=ATOL, rtol=RTOL
                        )


# =============================================================================
# G02: apply_gate_mode
# =============================================================================


class TestG02ApplyGateMode:
    def test_G02_shapes_preserved(self):
        inp = make_inputs(T=6)
        for mode in GATE_MODES:
            b, w, g = apply_gate_mode(inp["b"], inp["w"], inp["g"], mode)
            assert b.shape == inp["b"].shape
            assert w.shape == inp["w"].shape
            assert g.shape == inp["g"].shape

    def test_G02_full_is_identity(self):
        inp = make_inputs(T=6)
        b, w, g = apply_gate_mode(inp["b"], inp["w"], inp["g"], "full")
        assert b is inp["b"]
        assert w is inp["w"]
        assert g is inp["g"]

    def test_G02_b_only_scalarizes_w(self):
        inp = make_inputs(T=6)
        b, w, g = apply_gate_mode(inp["b"], inp["w"], inp["g"], "b_only")
        torch.testing.assert_close(b, inp["b"], atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(g, inp["g"], atol=ATOL, rtol=RTOL)
        expected = inp["w"].mean(dim=-1, keepdim=True).expand_as(inp["w"])
        torch.testing.assert_close(w, expected, atol=ATOL, rtol=RTOL)
        # all channels of the scalarized gate are equal
        assert torch.all(w[..., 0] == w[..., -1])

    def test_G02_w_only_scalarizes_b(self):
        inp = make_inputs(T=6)
        b, w, g = apply_gate_mode(inp["b"], inp["w"], inp["g"], "w_only")
        torch.testing.assert_close(w, inp["w"], atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(g, inp["g"], atol=ATOL, rtol=RTOL)
        expected = inp["b"].mean(dim=-1, keepdim=True).expand_as(inp["b"])
        torch.testing.assert_close(b, expected, atol=ATOL, rtol=RTOL)

    def test_G02_kda_ties_b_and_w_to_same_scalar(self):
        inp = make_inputs(T=6)
        b, w, g = apply_gate_mode(inp["b"], inp["w"], inp["g"], "kda")
        beta = inp["b"].mean(dim=-1)
        torch.testing.assert_close(
            b, beta[..., None].expand(B, 6, H, DK), atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            w, beta[..., None].expand(B, 6, H, DV), atol=ATOL, rtol=RTOL
        )
        # the tie is exact: same scalar for both gates
        torch.testing.assert_close(b[..., 0], w[..., 0], atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(g, inp["g"], atol=ATOL, rtol=RTOL)

    def test_G02_gdn_also_averages_g(self):
        inp = make_inputs(T=6)
        b, w, g = apply_gate_mode(inp["b"], inp["w"], inp["g"], "gdn")
        beta = inp["b"].mean(dim=-1)
        torch.testing.assert_close(
            b, beta[..., None].expand(B, 6, H, DK), atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(
            w, beta[..., None].expand(B, 6, H, DV), atol=ATOL, rtol=RTOL
        )
        g_scalar = inp["g"].mean(dim=-1)
        torch.testing.assert_close(
            g, g_scalar[..., None].expand(B, 6, H, DK), atol=ATOL, rtol=RTOL
        )

    def test_G02_accepts_plain_strings_and_enum(self):
        inp = make_inputs(T=4)
        b1, w1, g1 = apply_gate_mode(inp["b"], inp["w"], inp["g"], "kda")
        b2, w2, g2 = apply_gate_mode(inp["b"], inp["w"], inp["g"], GateMode.KDA)
        assert b1 is b2 or torch.equal(b1, b2)
        assert torch.equal(w1, w2)
        assert torch.equal(g1, g2)


# =============================================================================
# G03: chunk_scan == recurrent_scan
# =============================================================================


class TestG03ChunkMatchesRecurrent:
    @pytest.mark.parametrize("gate_mode", GATE_MODES)
    @pytest.mark.parametrize("T", [32, 37])
    @pytest.mark.parametrize("C", [4, 16])
    def test_G03_chunk_matches_recurrent(self, gate_mode: str, T: int, C: int):
        """T=37 exercises the short tail chunk; C in {4, 16}."""
        inp = make_inputs(T=T)
        b, w, g = apply_gate_mode(inp["b"], inp["w"], inp["g"], gate_mode)
        o_chunk, S_chunk = chunk_scan(
            inp["q"],
            inp["k"],
            inp["v"],
            g,
            b,
            w,
            initial_state=inp["S0"],
            chunk_size=C,
        )
        o_rec, S_rec = recurrent_scan(
            inp["q"], inp["k"], inp["v"], g, b, w, initial_state=inp["S0"]
        )
        torch.testing.assert_close(o_chunk, o_rec, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S_chunk, S_rec, atol=ATOL, rtol=RTOL)

    def test_G03_zero_initial_state_default(self):
        inp = make_inputs(T=32)
        o1, S1 = chunk_scan(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            inp["b"],
            inp["w"],
            initial_state=None,
            chunk_size=16,
        )
        zeros = torch.zeros(B, H, DK, DV, dtype=torch.float64)
        o2, S2 = chunk_scan(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            inp["b"],
            inp["w"],
            initial_state=zeros,
            chunk_size=16,
        )
        torch.testing.assert_close(o1, o2, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S1, S2, atol=ATOL, rtol=RTOL)


# =============================================================================
# G04: recurrent_scan == literal loops
# =============================================================================


class TestG04LiteralLoops:
    def test_G04_full_vs_literal_gdn2(self):
        inp = make_inputs(T=12)
        o_rec, S_rec = recurrent_scan(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            inp["b"],
            inp["w"],
            initial_state=inp["S0"],
        )
        for bi in range(B):
            for hi in range(H):
                o_lit, S_lit = literal_gdn2(
                    inp["q"][bi, :, hi],
                    inp["k"][bi, :, hi],
                    inp["v"][bi, :, hi],
                    inp["g"][bi, :, hi],
                    inp["b"][bi, :, hi],
                    inp["w"][bi, :, hi],
                    inp["S0"][bi, hi],
                )
                torch.testing.assert_close(
                    o_rec[bi, :, hi], o_lit, atol=ATOL, rtol=RTOL
                )
                torch.testing.assert_close(S_rec[bi, hi], S_lit, atol=ATOL, rtol=RTOL)

    def test_G04_kda_vs_literal_kda(self):
        inp = make_inputs(T=12)
        b, w, _ = apply_gate_mode(inp["b"], inp["w"], inp["g"], "kda")
        beta = inp["b"].mean(dim=-1)  # (B, T, H)
        o_rec, S_rec = recurrent_scan(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            b,
            w,
            initial_state=inp["S0"],
        )
        for bi in range(B):
            for hi in range(H):
                o_lit, S_lit = literal_kda(
                    inp["q"][bi, :, hi],
                    inp["k"][bi, :, hi],
                    inp["v"][bi, :, hi],
                    inp["g"][bi, :, hi],
                    beta[bi, :, hi],
                    inp["S0"][bi, hi],
                )
                torch.testing.assert_close(
                    o_rec[bi, :, hi], o_lit, atol=ATOL, rtol=RTOL
                )
                torch.testing.assert_close(S_rec[bi, hi], S_lit, atol=ATOL, rtol=RTOL)

    def test_G04_gdn_vs_literal_gdn(self):
        inp = make_inputs(T=12)
        b, w, g = apply_gate_mode(inp["b"], inp["w"], inp["g"], "gdn")
        beta = inp["b"].mean(dim=-1)
        g_scalar = inp["g"].mean(dim=-1)
        o_rec, S_rec = recurrent_scan(
            inp["q"], inp["k"], inp["v"], g, b, w, initial_state=inp["S0"]
        )
        for bi in range(B):
            for hi in range(H):
                o_lit, S_lit = literal_gdn(
                    inp["q"][bi, :, hi],
                    inp["k"][bi, :, hi],
                    inp["v"][bi, :, hi],
                    g_scalar[bi, :, hi],
                    beta[bi, :, hi],
                    inp["S0"][bi, hi],
                )
                torch.testing.assert_close(
                    o_rec[bi, :, hi], o_lit, atol=ATOL, rtol=RTOL
                )
                torch.testing.assert_close(S_rec[bi, hi], S_lit, atol=ATOL, rtol=RTOL)


# =============================================================================
# G05: recurrent_scan == brute-force oracle
# =============================================================================


class TestG05BruteForce:
    def test_G05_recurrent_scan_vs_brute_force_oracle(self):
        inp = make_inputs(T=12, seed=3)
        o_rec, S_rec = recurrent_scan(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            inp["b"],
            inp["w"],
            initial_state=None,
        )
        for bi in range(B):
            for hi in range(H):
                o_or, S_or = brute_force_oracle(
                    inp["q"][bi, :, hi],
                    inp["k"][bi, :, hi],
                    inp["v"][bi, :, hi],
                    inp["g"][bi, :, hi],
                    inp["b"][bi, :, hi],
                    inp["w"][bi, :, hi],
                )
                torch.testing.assert_close(o_rec[bi, :, hi], o_or, atol=ATOL, rtol=RTOL)
                torch.testing.assert_close(S_rec[bi, hi], S_or, atol=ATOL, rtol=RTOL)


# =============================================================================
# G06: gradients
# =============================================================================


class TestG06Gradients:
    def _run(self, fn, inp):
        leaves = {
            name: inp[name].detach().clone().requires_grad_(True)
            for name in ("q", "k", "v", "g", "b", "w")
        }
        S0 = inp["S0"].detach().clone().requires_grad_(True)
        o, S = fn(
            leaves["q"],
            leaves["k"],
            leaves["v"],
            leaves["g"],
            leaves["b"],
            leaves["w"],
            initial_state=S0,
        )
        gen = torch.Generator().manual_seed(7)
        dO = torch.empty_like(o).normal_(generator=gen)
        dS = torch.empty_like(S).normal_(generator=gen)
        loss = (o * dO).sum() + (S * dS).sum()
        loss.backward()
        grads = {name: t.grad.clone() for name, t in leaves.items()}
        grads["S0"] = S0.grad.clone()
        return grads

    @pytest.mark.slow
    def test_G06_chunk_gradients_match_recurrent(self):
        """Autograd through chunk_scan vs recurrent_scan, incl. the initial state."""
        inp = make_inputs(T=24, seed=1)
        grads_rec = self._run(lambda *a, **kw: recurrent_scan(*a, **kw), inp)
        grads_chunk = self._run(
            lambda *a, **kw: chunk_scan(*a, chunk_size=4, **kw), inp
        )
        for name, gr in grads_rec.items():
            torch.testing.assert_close(
                grads_chunk[name],
                gr,
                atol=1e-8,
                rtol=1e-8,
                msg=_grad_msg(name),
            )


# =============================================================================
# G07: log space under strong decay
# =============================================================================


class TestG07StrongDecay:
    @pytest.mark.slow
    def test_G07_log_space_strong_decay(self):
        """g ~ -2 per step, C=64: finite forward/backward, equals recurrent."""
        inp = make_inputs(T=64, seed=2, decay=4.0)
        inp["g"] = -2.0 - 0.1 * torch.rand_like(inp["g"])

        leaves = {
            n: inp[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "b", "w")
        }
        o, S = chunk_scan(
            leaves["q"],
            leaves["k"],
            leaves["v"],
            leaves["g"],
            leaves["b"],
            leaves["w"],
            initial_state=None,
            chunk_size=64,
        )
        assert torch.isfinite(o).all() and torch.isfinite(S).all()
        (o.sum() + S.sum()).backward()
        for name, t in leaves.items():
            assert torch.isfinite(t.grad).all(), f"{name} grad not finite"

        o_rec, S_rec = recurrent_scan(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            inp["b"],
            inp["w"],
            initial_state=None,
        )
        torch.testing.assert_close(o, o_rec, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S, S_rec, atol=ATOL, rtol=RTOL)


# =============================================================================
# G08: continuation
# =============================================================================


class TestG08Continuation:
    @pytest.mark.parametrize("T1, T2", [(20, 13), (16, 16), (18, 2)])
    def test_G08_chunk_continuation(self, T1: int, T2: int):
        """chunk_scan on T1 then on T2 from the returned state == one pass."""
        inp1 = make_inputs(T=T1, seed=4)
        inp2 = make_inputs(T=T2, seed=5)
        full = {
            name: torch.cat([inp1[name], inp2[name]], dim=1)
            for name in ("q", "k", "v", "g", "b", "w")
        }
        C = 16
        o1, S_mid = chunk_scan(
            inp1["q"],
            inp1["k"],
            inp1["v"],
            inp1["g"],
            inp1["b"],
            inp1["w"],
            initial_state=inp1["S0"],
            chunk_size=C,
        )
        o2, S_end = chunk_scan(
            inp2["q"],
            inp2["k"],
            inp2["v"],
            inp2["g"],
            inp2["b"],
            inp2["w"],
            initial_state=S_mid,
            chunk_size=C,
        )
        o_full, S_full = chunk_scan(
            full["q"],
            full["k"],
            full["v"],
            full["g"],
            full["b"],
            full["w"],
            initial_state=inp1["S0"],
            chunk_size=C,
        )
        torch.testing.assert_close(
            torch.cat([o1, o2], dim=1), o_full, atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(S_end, S_full, atol=ATOL, rtol=RTOL)


# =============================================================================
# G14: gate reduction at the functional level
# =============================================================================


class TestG14GateReduction:
    def test_G14_full_with_scalar_gates_equals_kda(self):
        """full with b = w = beta * 1 equals kda with the same beta."""
        inp = make_inputs(T=32, seed=6)
        beta = inp["b"].mean(dim=-1, keepdim=True)  # (B, T, H, 1)
        b_const = beta.expand(B, 32, H, DK)
        w_const = beta.expand(B, 32, H, DV)
        o_full, S_full = chunk_scan(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            b_const,
            w_const,
            initial_state=inp["S0"],
            chunk_size=16,
        )
        o_kda, S_kda = chunk_scan(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            b_const,
            w_const,
            initial_state=inp["S0"],
            chunk_size=16,
        )
        # kda-mode scalarization of b_const, w_const is a no-op (already
        # constant), so this checks the reduction via the mode path:
        b_kda, w_kda, _ = apply_gate_mode(inp["b"], inp["w"], inp["g"], "kda")
        o_mode, S_mode = chunk_scan(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            b_kda,
            w_kda,
            initial_state=inp["S0"],
            chunk_size=16,
        )
        torch.testing.assert_close(o_full, o_kda, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S_full, S_kda, atol=ATOL, rtol=RTOL)
        # and the kda mode applied to the raw gates equals the explicitly
        # const-constructed full run
        torch.testing.assert_close(o_mode, o_full, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S_mode, S_full, atol=ATOL, rtol=RTOL)
