"""
Tests for the Zoology-style language-model harness.

Target: src/research_lib/architectures/zoology_lm.py
Run with: pytest src/research_lib/architectures/tests/test_zoology_lm.py -v

The harness is mixer-agnostic, so these tests use a cheap mixer
(``BaseConv`` in every block) plus one ``MultistateGatedDelta`` case; the
gated-delta-specific model tests (parameter counts across gate modes,
lambda-only parameter additions, state sizes) live with the experiment's
builder in
``experiments/03_multistate_gdn2/01_synthetic_pilot/test_01_train.py``.
"""

from typing import Any

import pytest
import torch
import torch.nn as nn

from research_lib.architectures.config import StateMixer, ZoologyLMConfig
from research_lib.architectures.zoology_lm import (
    LayerPattern,
    ZoologyLM,
    zoology_mixers,
)
from research_lib.layers.attention import MultistateGatedDelta
from research_lib.layers.base_conv import BaseConv


def tiny_config(**overrides) -> ZoologyLMConfig:
    # Any-valued: the literal dict infers dict[str, int], which would fail
    # the str-typed fields at **defaults.
    defaults: dict[str, Any] = {"vocab_size": 64, "d_model": 16}
    defaults.update(overrides)
    return ZoologyLMConfig(**defaults)


def base_conv_mixers(n_layer: int = 2, d_model: int = 16) -> list[nn.Module]:
    """A cheap mixer: BaseConv in every block."""
    return zoology_mixers(
        n_layer, d_model, make_mixer=lambda: BaseConv(d_model=d_model)
    )


class TupleMixer(nn.Module):
    """A mixer that returns ``(y, None)``: the tuple-output contract."""

    def __init__(self, d_model: int = 16):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear(x), None


class TestZoologyMixers:
    def test_zoology_pattern_layout(self):
        mixers = zoology_mixers(
            3, 16, make_mixer=lambda: BaseConv(d_model=16), pattern="zoology"
        )
        assert [type(m) for m in mixers] == [BaseConv, BaseConv, BaseConv]
        # The real check: with a distinct make_mixer the odd blocks differ.
        mixers = zoology_mixers(
            4,
            16,
            make_mixer=lambda: TupleMixer(16),
            pattern=LayerPattern.ZOOLOGY,
        )
        assert isinstance(mixers[0], BaseConv)
        assert isinstance(mixers[1], TupleMixer)
        assert isinstance(mixers[2], BaseConv)
        assert isinstance(mixers[3], TupleMixer)

    def test_mixer_x2_pattern(self):
        mixers = zoology_mixers(
            3, 16, make_mixer=lambda: TupleMixer(16), pattern="mixer_x2"
        )
        assert all(isinstance(m, TupleMixer) for m in mixers)

    def test_invalid_pattern_and_n_layer_rejected(self):
        with pytest.raises(ValueError):
            zoology_mixers(2, 16, make_mixer=lambda: TupleMixer(16), pattern="stacked")
        with pytest.raises(ValueError):
            zoology_mixers(0, 16, make_mixer=lambda: TupleMixer(16))


class TestZoologyLM:
    """The harness, with cheap mixers."""

    @pytest.mark.parametrize("pattern", ["zoology", "mixer_x2"])
    def test_forward_shapes(self, pattern):
        torch.manual_seed(0)
        mixers = zoology_mixers(
            2, 16, make_mixer=lambda: TupleMixer(16), pattern=pattern
        )
        model = ZoologyLM(tiny_config(), mixers)
        logits = model(torch.zeros(2, 12, dtype=torch.long))
        assert logits.shape == (2, 12, 64)

    def test_mixer_contract_single_output(self):
        # A mixer returning a plain (B, T, d) tensor also works.
        mixers = base_conv_mixers()
        model = ZoologyLM(tiny_config(), mixers)
        assert model(torch.zeros(1, 8, dtype=torch.long)).shape == (1, 8, 64)

    def test_causal(self):
        """Changing tokens at t' leaves logits before t' unchanged."""
        torch.manual_seed(1)
        mixers = base_conv_mixers()
        model = ZoologyLM(tiny_config(), mixers).double()
        ids = torch.randint(0, 64, (2, 16))
        logits1 = model(ids)
        ids2 = ids.clone()
        ids2[:, 9:] = torch.randint(0, 64, (2, 7))
        logits2 = model(ids2)
        torch.testing.assert_close(logits1[:, :9], logits2[:, :9])
        assert not torch.allclose(logits1[:, 9:], logits2[:, 9:])

    def test_n_layer_follows_seq_mixers(self):
        for n in (1, 2, 4):
            model = ZoologyLM(tiny_config(), base_conv_mixers(n_layer=n))
            assert len(model.blocks) == n

    def test_empty_seq_mixers_rejected(self):
        with pytest.raises(ValueError, match="seq_mixers"):
            ZoologyLM(tiny_config(), [])

    def test_parameter_names_are_block_mixer(self):
        model = ZoologyLM(tiny_config(), base_conv_mixers())
        names = [name for name, _ in model.named_parameters()]
        assert any(name.startswith("blocks.0.mixer.") for name in names)
        assert any(name.startswith("blocks.1.mixer.") for name in names)
        assert "embedding.weight" in names
        assert "lm_head.weight" in names

    def test_state_mixer_mlp_adds_parameters(self):
        base = ZoologyLM(tiny_config(), base_conv_mixers())
        mlp = ZoologyLM(tiny_config(state_mixer="mlp", ff_mult=2), base_conv_mixers())
        assert sum(p.numel() for p in mlp.parameters()) > sum(
            p.numel() for p in base.parameters()
        )
        assert hasattr(mlp.blocks[0], "mlp")

    def test_multistate_gated_delta_case(self):
        """One gated-delta mixer works through the same harness (ws)."""
        torch.manual_seed(0)
        mixers = [
            BaseConv(d_model=32),
            MultistateGatedDelta(
                hidden_size=32, num_heads=2, head_dim=16, layout="ws", t_max=32
            ),
        ]
        model = ZoologyLM(ZoologyLMConfig(vocab_size=64, d_model=32), mixers)
        y, _ = model.blocks[1].mixer(torch.randn(1, 4, 32), use_cache=True)
        assert y.shape == (1, 4, 32)
        logits = model(torch.zeros(2, 12, dtype=torch.long))
        assert logits.shape == (2, 12, 64)


class TestAnswerPositionHead:
    """forward == head(hidden_states), exactly."""

    def test_forward_equals_head_of_hidden_states(self):
        torch.manual_seed(0)
        mixers = zoology_mixers(2, 16, make_mixer=lambda: TupleMixer(16))
        model = ZoologyLM(tiny_config(), mixers)
        ids = torch.randint(0, 64, (2, 12))
        torch.testing.assert_close(
            model(ids), model.head(model.hidden_states(ids)), rtol=0, atol=0
        )

    def test_hidden_states_shape_and_head_contract(self):
        """hidden_states is (B, T, d) after the final norm; head maps (..., d)
        to (..., V) for any leading shape."""
        torch.manual_seed(1)
        mixers = zoology_mixers(2, 16, make_mixer=lambda: TupleMixer(16))
        model = ZoologyLM(tiny_config(), mixers)
        ids = torch.randint(0, 64, (3, 7))
        hidden = model.hidden_states(ids)
        assert hidden.shape == (3, 7, 16)
        assert model.head(hidden).shape == (3, 7, 64)
        # A single position works too (the answer rows are gathered first).
        assert model.head(hidden[1, 3]).shape == (64,)


class TestZoologyLMConfig:
    def test_state_mixer_coercion(self):
        config = ZoologyLMConfig(state_mixer="mlp")
        assert config.state_mixer is StateMixer.MLP
        assert ZoologyLMConfig().state_mixer is StateMixer.NONE
        # The enum member works too.
        assert ZoologyLMConfig(state_mixer=StateMixer.MLP).state_mixer is StateMixer.MLP

    def test_invalid_state_mixer_rejected(self):
        with pytest.raises(ValueError):
            ZoologyLMConfig(state_mixer="relu")

    def test_invalid_dimensions_rejected(self):
        with pytest.raises(ValueError):
            ZoologyLMConfig(d_model=0)
        with pytest.raises(ValueError):
            ZoologyLMConfig(ff_mult=0)
