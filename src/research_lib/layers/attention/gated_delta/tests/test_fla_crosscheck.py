"""fla cross-checks (spec test G16).

Compares :func:`recurrent_scan` in the reduction modes against fla's naive
reference implementations.  fla's naive references cast to fp32, so the
comparison uses fp32 inputs and tolerances (``atol=rtol=1e-4``) per the
precision policy.

The checks are **reduction anchors**: our ``kda`` gate mode must equal
fla's KDA recurrence and our ``gdn`` mode fla's Gated DeltaNet recurrence.
The ``full`` (GDN-2) check targets ``fla.ops.gdn2.naive_recurrent_gdn2``,
which is not present in every fla release; it is skipped when unavailable.

Run with::

    pytest src/research_lib/layers/attention/gated_delta/tests/test_fla_crosscheck.py -v
"""

import importlib

import pytest
import torch

from ..functional import apply_gate_mode, recurrent_scan

ATOL = RTOL = 1e-4  # fla's naive references cast to fp32

B, H, DK, DV = 2, 2, 8, 6


def _fla_naive(module_names: tuple[str, ...], func_name: str):
    """Import a fla naive reference or return None (release difference).

    Tries each candidate module path in order (fla has moved the naive
    references into ``*.naive`` submodules across releases).
    """
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        func = getattr(module, func_name, None)
        if func is not None:
            return func
    return None


def _make_inputs(T: int, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    f32 = {"dtype": torch.float32, "generator": gen}
    return {
        "q": torch.randn(B, T, H, DK, **f32),
        "k": torch.randn(B, T, H, DK, **f32),
        "v": torch.randn(B, T, H, DV, **f32),
        "g": -0.3 * torch.rand(B, T, H, DK, **f32),
        "b": torch.rand(B, T, H, DK, **f32),
        "w": torch.rand(B, T, H, DV, **f32),
    }


class TestG16FlaCrosscheck:
    def test_G16_full_vs_fla_gdn2(self):
        """full mode vs fla.ops.gdn2.naive_recurrent_gdn2 (skipped when the
        installed fla release has no gdn2 module)."""
        pytest.importorskip("fla")
        naive = _fla_naive(
            ("fla.ops.gdn2.naive", "fla.ops.gdn2"), "naive_recurrent_gdn2"
        )
        if naive is None:
            pytest.skip(
                "fla.ops.gdn2.naive_recurrent_gdn2 not available in the "
                "installed fla release"
            )
        T = 16
        inp = _make_inputs(T, seed=0)
        o_ours, S_ours = recurrent_scan(
            inp["q"], inp["k"], inp["v"], inp["g"], inp["b"], inp["w"]
        )
        o_fla, S_fla = naive(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            inp["b"],
            inp["w"],
            scale=1.0,
            output_final_state=True,
        )
        torch.testing.assert_close(o_ours, o_fla, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S_ours, S_fla, atol=ATOL, rtol=RTOL)

    def test_G16_kda_vs_fla_kda(self):
        """kda mode vs fla.ops.kda.naive.naive_recurrent_kda with
        beta = mean(b) of shape [B, T, H]."""
        pytest.importorskip("fla")
        naive = _fla_naive(("fla.ops.kda.naive", "fla.ops.kda"), "naive_recurrent_kda")
        if naive is None:
            pytest.skip("fla.ops.kda.naive not available")
        T = 16
        inp = _make_inputs(T, seed=1)
        b, w, g = apply_gate_mode(inp["b"], inp["w"], inp["g"], "kda")
        o_ours, S_ours = recurrent_scan(inp["q"], inp["k"], inp["v"], g, b, w)
        beta = inp["b"].mean(dim=-1)  # [B, T, H]
        o_fla, S_fla = naive(
            inp["q"],
            inp["k"],
            inp["v"],
            inp["g"],
            beta,
            scale=1.0,
            output_final_state=True,
        )
        torch.testing.assert_close(o_ours, o_fla, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S_ours, S_fla, atol=ATOL, rtol=RTOL)

    def test_G16_gdn_vs_fla_gated_delta_rule(self):
        """gdn mode vs fla.ops.gated_delta_rule.naive.
        naive_recurrent_gated_delta_rule (note: beta before g, scalar g)."""
        pytest.importorskip("fla")
        naive = _fla_naive(
            ("fla.ops.gated_delta_rule.naive", "fla.ops.gated_delta_rule"),
            "naive_recurrent_gated_delta_rule",
        )
        if naive is None:
            pytest.skip("fla.ops.gated_delta_rule.naive not available")
        T = 16
        inp = _make_inputs(T, seed=2)
        b, w, g = apply_gate_mode(inp["b"], inp["w"], inp["g"], "gdn")
        o_ours, S_ours = recurrent_scan(inp["q"], inp["k"], inp["v"], g, b, w)
        beta = inp["b"].mean(dim=-1)  # [B, T, H]
        g_scalar = inp["g"].mean(dim=-1)  # [B, T, H]
        o_fla, S_fla = naive(
            inp["q"],
            inp["k"],
            inp["v"],
            beta,
            g_scalar,
            scale=1.0,
            output_final_state=True,
        )
        # fla's naive gdr returns o in (B, T, H, DV); S in (B, H, DK, DV)
        torch.testing.assert_close(o_ours, o_fla, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(S_ours, S_fla, atol=ATOL, rtol=RTOL)
