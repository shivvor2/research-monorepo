"""Layer-level tests for :class:`GatedDeltaReference` (G10–G13, G15).

The module is built in float64 so the chunk and recurrent paths must agree
to fp64 self-consistency tolerance (``atol=rtol=1e-10``).

Run with::

    pytest src/research_lib/layers/attention/gated_delta/tests/test_reference.py -v
"""

import pytest
import torch

from ..projections import GatedDeltaProjections
from ..reference import GatedDeltaReference

GATE_MODES = ["full", "b_only", "w_only", "kda", "gdn"]
ATOL = RTOL = 1e-10

HIDDEN, HEADS, HEAD_DIM, EXPAND_V = 16, 2, 8, 1.5


def _make_layer(gate_mode: str, **kwargs) -> GatedDeltaReference:
    defaults = {
        "hidden_size": HIDDEN,
        "num_heads": HEADS,
        "head_dim": HEAD_DIM,
        "expand_v": EXPAND_V,
        "gate_mode": gate_mode,
        "chunk_size": 4,
        "dtype": torch.float64,
    }
    defaults.update(kwargs)
    return GatedDeltaReference(**defaults)


# =============================================================================
# G10: layer chunk == recurrent
# =============================================================================


class TestG10LayerPaths:
    @pytest.mark.parametrize("gate_mode", GATE_MODES)
    def test_G10_chunk_matches_recurrent(self, gate_mode: str):
        torch.manual_seed(0)
        layer = _make_layer(gate_mode)
        x = torch.randn(2, 20, HIDDEN, dtype=torch.float64)
        y_chunk, _ = layer(x, mode="chunk")
        y_rec, _ = layer(x, mode="recurrent")
        torch.testing.assert_close(y_chunk, y_rec, atol=ATOL, rtol=RTOL)

    def test_G10_no_short_conv_chunk_matches_recurrent(self):
        torch.manual_seed(1)
        layer = _make_layer("full", use_short_conv=False)
        x = torch.randn(2, 20, HIDDEN, dtype=torch.float64)
        y_chunk, _ = layer(x, mode="chunk")
        y_rec, _ = layer(x, mode="recurrent")
        torch.testing.assert_close(y_chunk, y_rec, atol=ATOL, rtol=RTOL)


# =============================================================================
# G11: decode handoff
# =============================================================================


class TestG11DecodeHandoff:
    @pytest.mark.slow
    def test_G11_prefill_then_decode_equals_one_recurrent_pass(self):
        """Chunk prefill (T0) with use_cache, then one token at a time in
        recurrent mode, equals one recurrent pass over all tokens — including
        the conv cache."""
        torch.manual_seed(2)
        layer = _make_layer("full")
        T0, n_decode = 17, 6
        x = torch.randn(2, T0 + n_decode, HIDDEN, dtype=torch.float64)

        y_ref, cache_ref = layer(x, mode="recurrent", use_cache=True)
        expected_state = cache_ref.state

        # prefill in chunk mode
        y_prefill, cache = layer(x[:, :T0], mode="chunk", use_cache=True)
        assert cache is not None
        assert cache.position == T0
        torch.testing.assert_close(y_prefill, y_ref[:, :T0], atol=ATOL, rtol=RTOL)
        # decode one token at a time in recurrent mode
        decode_outputs = []
        for t in range(T0, T0 + n_decode):
            y_t, cache = layer(
                x[:, t : t + 1],
                cache=cache,
                use_cache=True,
                mode="recurrent",
            )
            decode_outputs.append(y_t)
            assert cache.position == t + 1
        y_decode = torch.cat(decode_outputs, dim=1)
        torch.testing.assert_close(y_decode, y_ref[:, T0:], atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(cache.state, expected_state, atol=ATOL, rtol=RTOL)

    def test_G11_cache_shape_and_dtype(self):
        torch.manual_seed(3)
        layer = _make_layer("full")
        x = torch.randn(2, 9, HIDDEN, dtype=torch.float64)
        _, cache = layer(x, use_cache=True)
        assert cache.state.shape == (2, HEADS, HEAD_DIM, int(HEAD_DIM * EXPAND_V))
        assert cache.state.dtype == torch.float64  # fp64 module -> fp64 state
        assert cache.conv_state is not None
        for c in cache.conv_state:
            # (B, H*d, K-1) pre-conv inputs
            assert (
                c.shape[0] == 2
                and c.shape[-1] == layer.projections.conv_q.kernel_size - 1
            )


# =============================================================================
# G12: causality
# =============================================================================


class TestG12Causality:
    @pytest.mark.parametrize("mode", ["chunk", "recurrent"])
    def test_G12_perturbation_is_causal(self, mode: str):
        torch.manual_seed(4)
        layer = _make_layer("full")
        T, t_prime = 20, 12
        x = torch.randn(2, T, HIDDEN, dtype=torch.float64)
        x2 = x.clone()
        x2[:, t_prime] = x2[:, t_prime] + 3.0
        y, _ = layer(x, mode=mode)
        y2, _ = layer(x2, mode=mode)
        torch.testing.assert_close(
            y[:, :t_prime], y2[:, :t_prime], atol=ATOL, rtol=RTOL
        )
        assert not torch.allclose(y[:, t_prime:], y2[:, t_prime:])


# =============================================================================
# G13: parameter identity across gate modes
# =============================================================================


class TestG13ParameterIdentity:
    def test_G13_same_param_count_and_keys(self):
        ref = _make_layer("full")
        ref_count = sum(p.numel() for p in ref.parameters())
        ref_keys = set(ref.state_dict().keys())
        for gate_mode in GATE_MODES:
            layer = _make_layer(gate_mode)
            assert (
                sum(p.numel() for p in layer.parameters()) == ref_count
            ), f"parameter count differs for gate_mode={gate_mode}"
            assert (
                set(layer.state_dict().keys()) == ref_keys
            ), f"state_dict keys differ for gate_mode={gate_mode}"


# =============================================================================
# G15: output gate option
# =============================================================================


class TestG15OutputGate:
    def test_G15_default_is_silu(self):
        layer = _make_layer("full")
        assert layer.projections.output_gate_activation == "silu"

    @pytest.mark.parametrize("activation", ["silu", "sigmoid"])
    def test_G15_both_activations_run(self, activation: str):
        torch.manual_seed(5)
        layer = _make_layer("full", output_gate_activation=activation)
        x = torch.randn(2, 10, HIDDEN, dtype=torch.float64)
        y, _ = layer(x)
        assert y.shape == (2, 10, HIDDEN)
        assert torch.isfinite(y).all()


# =============================================================================
# Sourcery review: device/dtype factory kwargs must reach every parameter
# =============================================================================


class TestDeviceDtypeFactoryKwargs:
    def test_all_parameters_on_meta_device(self):
        """device="meta" construction puts every parameter on that device
        (no real allocation; works without CUDA)."""
        proj = GatedDeltaProjections(
            hidden_size=HIDDEN,
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            expand_v=EXPAND_V,
            device="meta",
        )
        for name, p in proj.named_parameters():
            assert p.device.type == "meta", f"{name} is on {p.device}"

    def test_layer_all_parameters_on_meta_device(self):
        layer = _make_layer("full", device="meta")
        for name, p in layer.named_parameters():
            assert p.device.type == "meta", f"{name} is on {p.device}"

    def test_dtype_factory_kwarg_reaches_all_parameters(self):
        layer = _make_layer("full", dtype=torch.float32)
        for name, p in layer.named_parameters():
            assert p.dtype == torch.float32, f"{name} has dtype {p.dtype}"
