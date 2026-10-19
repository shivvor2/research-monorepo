"""Tests for the LambdaMixer (M18) and the λ≡1 initialization (part of M11).

All tests run in float64 on CPU (fp32 variants where the spec says fp32)
with tolerances ``atol=1e-10, rtol=1e-10`` (fp64) / ``atol=1e-6`` (fp32
around the λ≡1 init).

Run with::

    pytest src/research_lib/layers/attention/gated_delta/tests/test_mixing.py -v
"""

import pytest
import torch

from ..mixing import LambdaMixer

ATOL = RTOL = 1e-10
LAMBDA_MODES = ["static", "lin", "mlp"]


def _x(B: int, T: int, d: int, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, d, dtype=dtype, generator=gen)


class TestM18LambdaMixer:
    """M18: shapes, nonnegativity, fp32 output, static independence of x."""

    @pytest.mark.parametrize("mode", LAMBDA_MODES)
    def test_M18_shape_and_nonnegative(self, mode):
        d, H, L = 32, 2, 10
        mixer = LambdaMixer(hidden_size=d, num_heads=H, num_levels=L, mode=mode)
        x = _x(2, 16, d, torch.float32)
        lam = mixer(x)
        assert lam.shape == (2, 16, H, L)
        assert lam.dtype == torch.float32  # fp32 output for fp32 input
        assert torch.all(lam >= 0)

    @pytest.mark.parametrize("mode", LAMBDA_MODES)
    def test_M18_fp64_module_stays_fp64(self, mode):
        d, H, L = 32, 2, 10
        mixer = LambdaMixer(
            hidden_size=d, num_heads=H, num_levels=L, mode=mode, dtype=torch.float64
        )
        lam = mixer(_x(2, 16, d, torch.float64))
        assert lam.dtype == torch.float64

    def test_M18_static_is_independent_of_x(self):
        d, H, L = 32, 2, 10
        mixer = LambdaMixer(hidden_size=d, num_heads=H, num_levels=L, mode="static")
        x1 = _x(2, 16, d, torch.float32, seed=0)
        x2 = _x(2, 16, d, torch.float32, seed=1)
        torch.testing.assert_close(mixer(x1), mixer(x2), atol=ATOL, rtol=RTOL)

    def test_M18_lin_and_mlp_depend_on_x(self):
        """λ depends on x once the projection weights are nonzero. (At init
        they are zero by design, so λ≡1 for every x — that is D9.)"""
        d, H, L = 32, 2, 10
        x1 = _x(2, 16, d, torch.float32, seed=0)
        x2 = _x(2, 16, d, torch.float32, seed=1)
        lin = LambdaMixer(hidden_size=d, num_heads=H, num_levels=L, mode="lin")
        with torch.no_grad():
            lin.lambda_proj.weight.normal_(0, 0.1)
        assert not torch.allclose(lin(x1), lin(x2), atol=1e-8)
        mlp = LambdaMixer(hidden_size=d, num_heads=H, num_levels=L, mode="mlp")
        with torch.no_grad():
            mlp.mlp_out.weight.normal_(0, 0.1)
        assert not torch.allclose(mlp(x1), mlp(x2), atol=1e-8)

    def test_M18_lambda_parameters_marked_no_weight_decay(self):
        d, H, L = 32, 2, 10
        for mode in LAMBDA_MODES:
            mixer = LambdaMixer(hidden_size=d, num_heads=H, num_levels=L, mode=mode)
            lam_params = [n for n, _ in mixer.named_parameters()]
            assert lam_params, f"{mode}: expected λ parameters"
            for name, p in mixer.named_parameters():
                assert getattr(p, "_no_weight_decay", False) is True, (mode, name)

    def test_M18_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            LambdaMixer(hidden_size=8, num_heads=1, num_levels=2, mode="quadratic")

    def test_M18_unknown_nonlinearity_raises(self):
        with pytest.raises(ValueError, match="nonlinearity"):
            LambdaMixer(hidden_size=8, num_heads=1, num_levels=2, nonlinearity="relu")

    def test_M18_hidden_size_mismatch_raises(self):
        mixer = LambdaMixer(hidden_size=8, num_heads=1, num_levels=2)
        with pytest.raises(ValueError, match="hidden size"):
            mixer(torch.randn(1, 4, 9))

    def test_M18_mlp_hidden_override(self):
        mixer = LambdaMixer(
            hidden_size=8, num_heads=1, num_levels=2, mode="mlp", mlp_hidden=5
        )
        assert mixer.mlp_in.weight.shape == (5, 8)
        lam = mixer(_x(1, 4, 8, torch.float32))
        assert lam.shape == (1, 4, 1, 2)
        with pytest.raises(ValueError, match="mlp_hidden"):
            LambdaMixer(hidden_size=8, num_heads=1, num_levels=2, mlp_hidden=0)


class TestLambdaInit:
    """λ≡1 exactly at initialization, every mode."""

    @pytest.mark.parametrize("mode", LAMBDA_MODES)
    def test_lambda_is_one_at_init_fp64(self, mode):
        d, H, L = 16, 2, 6
        mixer = LambdaMixer(
            hidden_size=d, num_heads=H, num_levels=L, mode=mode, dtype=torch.float64
        )
        lam = mixer(_x(2, 8, d, torch.float64))
        torch.testing.assert_close(
            lam,
            torch.ones_like(lam),
            atol=ATOL,
            rtol=RTOL,
            msg=f"{mode}: λ must be exactly 1 at init in fp64",
        )

    @pytest.mark.parametrize("mode", LAMBDA_MODES)
    def test_lambda_is_one_at_init_fp32(self, mode):
        # fp32: softplus(log(e-1)) is 1 up to rounding; the init is the
        # inverse softplus at 1, so this holds to fp32 precision.
        d, H, L = 16, 2, 6
        mixer = LambdaMixer(hidden_size=d, num_heads=H, num_levels=L, mode=mode)
        lam = mixer(_x(2, 8, d, torch.float32))
        assert torch.max((lam - 1).abs()).item() < 1e-6

    @pytest.mark.parametrize("mode", LAMBDA_MODES)
    def test_lambda_moves_with_gradient(self, mode):
        """λ is trainable: a step of SGD moves it off 1."""
        d, H, L = 16, 2, 6
        mixer = LambdaMixer(
            hidden_size=d, num_heads=H, num_levels=L, mode=mode, dtype=torch.float64
        )
        x = _x(2, 8, d, torch.float64).requires_grad_(True)
        lam = mixer(x)
        loss = (lam * torch.randn_like(lam)).sum()
        loss.backward()
        moved = False
        for p in mixer.parameters():
            if p.grad is not None and p.grad.abs().max() > 0:
                moved = True
        assert moved, f"{mode}: λ parameters must receive gradients"
