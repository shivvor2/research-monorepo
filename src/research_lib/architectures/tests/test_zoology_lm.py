"""
Tests for the Zoology-style two-layer language model.

Target: src/research_lib/architectures/zoology_lm.py
Run with: pytest src/research_lib/architectures/tests/test_zoology_lm.py -v
"""

import pytest
import torch

from research_lib.architectures.config import ZoologyLMConfig
from research_lib.architectures.zoology_lm import ZoologyLM
from research_lib.layers.attention import MultistateGatedDelta
from research_lib.layers.base_conv import BaseConv

LAYOUTS = ("single", "ws", "fs", "ll")
GATE_MODES = ("full", "b_only", "w_only", "kda", "gdn")


def tiny_config(**overrides) -> ZoologyLMConfig:
    defaults = dict(
        vocab_size=64,
        d_model=16,
        n_layer=2,
        num_heads=1,
        t_max=32,
        window=4,
    )
    defaults.update(overrides)
    return ZoologyLMConfig(**defaults)


class TestZoologyLM:
    """Tests for ZoologyLM."""

    # ------------------------------------------------------------------ #
    # Config validation
    # ------------------------------------------------------------------ #

    def test_config_rejects_invalid_values(self):
        with pytest.raises(ValueError):
            ZoologyLMConfig(layer_pattern="alternating")
        with pytest.raises(ValueError):
            ZoologyLMConfig(state_mixer="relu")
        with pytest.raises(ValueError):
            ZoologyLMConfig(d_model=33, num_heads=2)
        with pytest.raises(ValueError):
            ZoologyLMConfig(t_max=0)

    # ------------------------------------------------------------------ #
    # Structure and shapes
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("pattern", ["zoology", "mixer_x2"])
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_forward_shapes(self, pattern, layout):
        """Forward shapes hold for both patterns and all four layouts."""
        torch.manual_seed(0)
        model = ZoologyLM(tiny_config(layer_pattern=pattern, layout=layout))
        logits = model(torch.zeros(2, 12, dtype=torch.long))
        assert logits.shape == (2, 12, 64)

    def test_zoology_pattern_mixers(self):
        """Layer 0 is BaseConv, layer 1 the gated-delta mixer."""
        model = ZoologyLM(tiny_config())
        assert isinstance(model.blocks[0].mixer, BaseConv)
        assert isinstance(model.blocks[1].mixer, MultistateGatedDelta)

    def test_mixer_x2_pattern_mixers(self):
        """Every block uses the gated-delta mixer."""
        model = ZoologyLM(tiny_config(layer_pattern="mixer_x2"))
        assert all(isinstance(b.mixer, MultistateGatedDelta) for b in model.blocks)

    def test_no_positional_embedding(self):
        """Token embedding only; nothing depends on position inputs."""
        model = ZoologyLM(tiny_config())
        assert not any("pos" in name for name, _ in model.named_parameters())

    def test_state_mixer_mlp_adds_parameters(self):
        base = ZoologyLM(tiny_config())
        mlp = ZoologyLM(tiny_config(state_mixer="mlp", ff_mult=2))
        n_base = sum(p.numel() for p in base.parameters())
        n_mlp = sum(p.numel() for p in mlp.parameters())
        assert n_mlp > n_base
        assert hasattr(mlp.blocks[0], "mlp")

    # ------------------------------------------------------------------ #
    # Causality (end to end)
    # ------------------------------------------------------------------ #

    def test_causal(self):
        """Changing tokens at t' leaves logits before t' unchanged."""
        torch.manual_seed(1)
        model = ZoologyLM(tiny_config()).double()
        ids = torch.randint(0, 64, (2, 16))
        logits1 = model(ids)
        ids2 = ids.clone()
        ids2[:, 9:] = torch.randint(0, 64, (2, 7))
        logits2 = model(ids2)
        torch.testing.assert_close(logits1[:, :9], logits2[:, :9])
        assert not torch.allclose(logits1[:, 9:], logits2[:, 9:])

    # ------------------------------------------------------------------ #
    # Parameter accounting
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("gate_mode", GATE_MODES)
    def test_param_counts_identical_across_gate_modes(self, gate_mode):
        """All five gate modes have identical parameter counts (D10)."""
        model = ZoologyLM(tiny_config(layout="ws", gate_mode=gate_mode))
        counts = {name: tuple(p.shape) for name, p in model.named_parameters()}
        reference = ZoologyLM(tiny_config(layout="ws", gate_mode="full"))
        ref_counts = {name: tuple(p.shape) for name, p in reference.named_parameters()}
        assert counts.keys() == ref_counts.keys()
        for key in ref_counts:
            assert counts[key] == ref_counts[key], key

    @pytest.mark.parametrize("layout", ["ws", "fs", "ll"])
    def test_layouts_differ_only_by_lambda_parameters(self, layout):
        """Every layout has exactly the single-state parameters plus the
        lambda_mixer ones, with identical shapes for the shared part."""
        single = ZoologyLM(tiny_config(layout="single"))
        multi = ZoologyLM(tiny_config(layout=layout))
        single_names = {name for name, _ in single.named_parameters()}
        multi_names = {name for name, _ in multi.named_parameters()}
        lambda_names = {n for n in multi_names if "lambda_mixer" in n}
        assert multi_names - single_names == lambda_names
        assert single_names - multi_names == set()
        for name, p in single.named_parameters():
            assert multi.state_dict()[name].shape == p.shape, name

    def test_single_has_no_lambda_parameters(self):
        model = ZoologyLM(tiny_config(layout="single"))
        assert not any("lambda" in name for name, _ in model.named_parameters())
        assert model.gated_delta_layers()[0].lambda_mixer is None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def test_gated_delta_layers(self):
        model = ZoologyLM(tiny_config())
        layers = model.gated_delta_layers()
        assert len(layers) == 1
        assert layers[0] is model.blocks[1].mixer

    @pytest.mark.parametrize("layout", ["ws", "ll"])
    def test_set_collect_stats_fills_latest_stats(self, layout):
        """set_collect_stats(True) fills latest_stats on every gated-delta
        layer after a forward pass."""
        model = ZoologyLM(
            tiny_config(layout=layout, layer_pattern="mixer_x2", collect_stats=False)
        )
        model.set_collect_stats(True)
        assert all(layer.collect_stats for layer in model.gated_delta_layers())
        model(torch.zeros(1, 8, dtype=torch.long))
        for layer in model.gated_delta_layers():
            stats = layer.latest_stats
            assert stats is not None
            assert "erase_strength" in stats
            assert "lambda_mean" in stats and "lambda_std" in stats
            L = layer.num_slots
            assert stats["lambda_mean"].shape == (layer.projections.num_heads, L)
        model.set_collect_stats(False)
        assert not any(layer.collect_stats for layer in model.gated_delta_layers())

    def test_single_layer_latest_stats_has_no_lambda(self):
        """layout='single' collects only the erase strength (no λ)."""
        model = ZoologyLM(tiny_config(layout="single", collect_stats=True))
        model(torch.zeros(1, 8, dtype=torch.long))
        stats = model.gated_delta_layers()[0].latest_stats
        assert set(stats.keys()) == {"erase_strength"}

    def test_state_bytes_per_mixer(self):
        """num_heads * num_slots * d_k * d_v * 4 bytes of recurrent state."""
        model = ZoologyLM(tiny_config(layout="ws", window=4))
        layer = model.gated_delta_layers()[0]
        expected = 1 * layer.num_slots * 16 * 16 * 4
        assert model.state_bytes_per_mixer() == expected
        assert layer.num_slots == 4 + 2  # window + 2

    def test_state_bytes_single_layout(self):
        """The single-state layer is the L = 1 case."""
        model = ZoologyLM(tiny_config(layout="single"))
        assert model.state_bytes_per_mixer() == 1 * 1 * 16 * 16 * 4
