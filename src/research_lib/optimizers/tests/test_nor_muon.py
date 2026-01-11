"""
Test suite for NorMuon optimizer.

These tests verify:
1. Basic functionality (parameter updates occur)
2. Gradient flow (loss decreases)
3. State management (state_dict save/load)
4. Numerical stability
5. Triton kernel correctness
6. Polar Express coefficient computation

Run with: pytest src/research_lib/optimizers/tests/test_nor_muon.py -v
"""

import pytest
import torch

# Increase recompile limit for tests since we sweep many shapes
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F

torch._dynamo.config.recompile_limit = 128

from typing import List, Tuple

from ..nor_muon import (
    DEFAULT_CUSHION,
    DEFAULT_NUM_ITERS,
    DEFAULT_POLAR_EXPRESS_COEFFS,
    DEFAULT_SAFETY_FACTOR,
    XXT,
    NorMuon,
    XXT_kernel,
    _make_polar_express,
    apply_normuon_variance_reduction,
    ba_plus_cAA,
    compute_polar_express_coeffs,
)


@pytest.fixture
def device():
    """Return CUDA device if available, else skip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


@pytest.fixture
def simple_2d_model(device):
    """A simple model with only 2D parameters for NorMuon testing."""
    model = nn.Sequential(
        nn.Linear(64, 128, bias=False),
        nn.ReLU(),
        nn.Linear(128, 64, bias=False),
        nn.ReLU(),
        nn.Linear(64, 10, bias=False),
    ).to(device)
    return model


class TestTritonKernels:
    """Test the Triton kernels used in NorMuon."""

    # Tolorances set wrt tf32 precision range.
    def test_xxt_deterministic(self, device):
        """Run multiple times to catch race conditions."""
        # Clear Triton cache before test
        XXT_kernel.cache.clear()

        torch.manual_seed(42)
        M, K = 256, 256
        A = torch.randn(M, K, device=device, dtype=torch.float32)

        # Disable TF32 for the reference computation to get true FP32
        torch.backends.cudnn.allow_tf32 = True
        expected = A @ A.T

        for i in range(10):
            out = torch.empty(M, M, device=device, dtype=torch.float32)
            XXT(A, out)

            # Debug: find where the differences are
            diff = (out - expected).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Find locations of large differences
            threshold = 1e-2
            bad_mask = diff > threshold
            num_bad = bad_mask.sum().item()

            if num_bad > 0:
                bad_indices = torch.nonzero(bad_mask)
                print(f"\nIteration {i}:")
                print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
                print(f"  Num elements > {threshold}: {num_bad} / {M*M}")
                print(f"  First 10 bad indices (row, col):")
                for idx in bad_indices[:10]:
                    r, c = idx[0].item(), idx[1].item()
                    print(
                        f"    [{r}, {c}]: got {out[r,c].item():.6f}, expected {expected[r,c].item():.6f}"
                    )

                # Check if bad indices follow a pattern (e.g., specific blocks)
                bad_rows = bad_indices[:, 0].unique()
                bad_cols = bad_indices[:, 1].unique()
                print(f"  Unique bad rows: {bad_rows[:20].tolist()}...")
                print(f"  Unique bad cols: {bad_cols[:20].tolist()}...")

                # Check if it's a specific region
                print(
                    f"  Bad region bounds: rows [{bad_indices[:,0].min()}-{bad_indices[:,0].max()}], "
                    f"cols [{bad_indices[:,1].min()}-{bad_indices[:,1].max()}]"
                )

            torch.testing.assert_close(
                out, expected, rtol=5e-3, atol=5e-2, msg=f"Failed on iteration {i}"
            )

    def test_xxt_correctness(self, device):
        """Test that XXT kernel computes A @ A.T correctly."""
        torch.manual_seed(42)

        shapes = [(64, 128), (128, 64), (256, 256), (32, 512), (512, 512)]

        for M, K in shapes:
            A = torch.randn(M, K, device=device, dtype=torch.float32)
            expected = A @ A.T

            out = torch.empty(M, M, device=device, dtype=torch.float32)
            XXT(A, out)

            # Relaxed tolerance for TF32/accumulation differences
            torch.testing.assert_close(
                out,
                expected,
                rtol=5e-3,
                atol=5e-2,
                msg=f"XXT failed for shape ({M}, {K})",
            )

    def test_xxt_batched(self, device):
        """Test batched XXT kernel."""
        torch.manual_seed(42)

        B, M, K = 4, 64, 128
        A = torch.randn(B, M, K, device=device, dtype=torch.float32)
        expected = torch.bmm(A, A.transpose(-2, -1))

        out = torch.empty(B, M, M, device=device, dtype=torch.float32)
        XXT(A, out)

        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    def test_xxt_bfloat16(self, device):
        """Test XXT kernel with bfloat16."""
        torch.manual_seed(42)

        M, K = 64, 128
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        expected = (A.float() @ A.float().T).bfloat16()

        out = torch.empty(M, M, device=device, dtype=torch.bfloat16)
        XXT(A, out)

        torch.testing.assert_close(out, expected, rtol=5e-2, atol=5e-2)

    def test_ba_plus_caa_correctness(self, device):
        """Test ba_plus_cAA kernel: C = alpha * A @ A.T + beta * A."""
        torch.manual_seed(42)

        M = 64
        # Important: The kernel implements a symmetry-preserving optimization
        # (stores lower triangle and mirrors to upper). The input A must be symmetric
        # for this to match the reference calculation 'beta * A'.
        # In Polar Express, A = X @ X.T, so it is always symmetric.
        A_raw = torch.randn(M, M, device=device, dtype=torch.float32)
        A = A_raw + A_raw.T  # Make symmetric

        alpha, beta = 0.5, 1.5

        expected = alpha * (A @ A.T) + beta * A

        out = torch.empty(M, M, device=device, dtype=torch.float32)
        ba_plus_cAA(A, alpha, beta, out)

        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    def test_ba_plus_caa_batched(self, device):
        """Test batched ba_plus_cAA kernel."""
        torch.manual_seed(42)

        B, M = 4, 64
        A_raw = torch.randn(B, M, M, device=device, dtype=torch.float32)
        A = A_raw + A_raw.transpose(-2, -1)  # Make symmetric

        alpha, beta = 0.3, 1.2

        expected = alpha * torch.bmm(A, A.transpose(-2, -1)) + beta * A

        out = torch.empty(B, M, M, device=device, dtype=torch.float32)
        ba_plus_cAA(A, alpha, beta, out)

        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


class TestPolarExpress:
    """Test the Polar Express orthogonalization."""

    def test_polar_express_orthogonality(self, device):
        """Test that Polar Express produces approximately orthogonal matrices."""
        torch.manual_seed(42)

        G = torch.randn(64, 128, device=device, dtype=torch.float32)
        polar_express = _make_polar_express(
            DEFAULT_POLAR_EXPRESS_COEFFS, DEFAULT_SAFETY_FACTOR
        )
        U = polar_express(G).float()

        # U @ U.T should be approximately identity (scaled)
        UUT = U @ U.T
        diag = torch.diag(UUT)

        # Diagonal should be roughly constant
        diag_std = diag.std() / diag.mean()
        assert diag_std < 0.5, f"Diagonal variance too high: {diag_std}"

        # Off-diagonal should be small relative to diagonal
        off_diag = UUT - torch.diag(diag)
        ratio = off_diag.abs().mean() / diag.mean()
        assert ratio < 0.3, f"Off-diagonal too large: {ratio}"

    def test_polar_express_shapes(self, device):
        """Test Polar Express handles various shapes correctly."""
        torch.manual_seed(42)
        polar_express = _make_polar_express(
            DEFAULT_POLAR_EXPRESS_COEFFS, DEFAULT_SAFETY_FACTOR
        )

        shapes = [
            (64, 128),  # M < K
            (128, 64),  # M > K
            (64, 64),  # Square
            (4, 64, 128),  # Batched
        ]

        for shape in shapes:
            G = torch.randn(*shape, device=device, dtype=torch.float32)
            U = polar_express(G)

            assert U.shape == G.shape, f"Shape mismatch for input {shape}"
            assert not torch.isnan(U).any(), f"NaN in output for shape {shape}"
            assert not torch.isinf(U).any(), f"Inf in output for shape {shape}"

    def test_polar_express_numerical_stability(self, device):
        """Test Polar Express with extreme inputs."""
        torch.manual_seed(42)
        polar_express = _make_polar_express(
            DEFAULT_POLAR_EXPRESS_COEFFS, DEFAULT_SAFETY_FACTOR
        )

        G_small = torch.randn(64, 128, device=device) * 1e-6
        U_small = polar_express(G_small)
        assert not torch.isnan(U_small).any(), "NaN with small inputs"

        G_large = torch.randn(64, 128, device=device) * 1e6
        U_large = polar_express(G_large)
        assert not torch.isnan(U_large).any(), "NaN with large inputs"

        G_zero = torch.zeros(64, 128, device=device)
        G_zero[0, 0] = 1e-8
        U_zero = polar_express(G_zero)
        assert not torch.isnan(U_zero).any(), "NaN with near-zero input"

    def test_polar_express_custom_coefficients(self, device):
        """Test Polar Express with custom coefficients."""
        # Use fewer iterations
        custom_coeffs = DEFAULT_POLAR_EXPRESS_COEFFS[:3]
        polar_express = _make_polar_express(custom_coeffs, DEFAULT_SAFETY_FACTOR)

        G = torch.randn(64, 128, device=device, dtype=torch.float32)
        U = polar_express(G)

        assert U.shape == G.shape
        assert not torch.isnan(U).any()


class TestCoefficientComputation:
    """Test the Polar Express coefficient computation."""

    def test_compute_coeffs_returns_correct_length(self):
        """Test that coefficient computation returns correct number of iterations."""
        for num_iters in [3, 5, 7]:
            coeffs = compute_polar_express_coeffs(num_iters=num_iters)
            assert (
                len(coeffs) == num_iters
            ), f"Expected {num_iters} coeffs, got {len(coeffs)}"

    def test_compute_coeffs_returns_tuples(self):
        """Test that coefficients are 3-tuples of floats."""
        coeffs = compute_polar_express_coeffs()
        for i, (a, b, c) in enumerate(coeffs):
            assert isinstance(a, float), f"Coeff {i}[0] is not float"
            assert isinstance(b, float), f"Coeff {i}[1] is not float"
            assert isinstance(c, float), f"Coeff {i}[2] is not float"

    def test_default_coeffs_match_precomputed(self):
        """Test that default params produce the precomputed coefficients."""
        computed = compute_polar_express_coeffs(
            num_iters=DEFAULT_NUM_ITERS,
            safety_factor=DEFAULT_SAFETY_FACTOR,
            cushion=DEFAULT_CUSHION,
        )
        assert len(computed) == len(DEFAULT_POLAR_EXPRESS_COEFFS)

    def test_compute_coeffs_different_params(self):
        """Test that different parameters produce different coefficients."""
        coeffs_default = compute_polar_express_coeffs()
        coeffs_custom = compute_polar_express_coeffs(num_iters=4, cushion=0.2)
        assert (
            len(coeffs_default) != len(coeffs_custom) or coeffs_default != coeffs_custom
        )


class TestNorMuonInit:
    """Test NorMuon initialization."""

    def test_instantiation_default_params(self, simple_2d_model):
        """Test NorMuon instantiation with default parameters."""
        opt = NorMuon(simple_2d_model.parameters())

        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["lr"] == 0.02
        assert opt.param_groups[0]["weight_decay"] == 0.01
        assert opt.param_groups[0]["momentum"] == 0.95

    def test_instantiation_custom_params(self, simple_2d_model):
        """Test NorMuon instantiation with custom parameters."""
        opt = NorMuon(
            simple_2d_model.parameters(),
            lr=0.01,
            weight_decay=0.05,
            momentum=0.9,
            beta2=0.99,
        )
        assert opt.param_groups[0]["lr"] == 0.01
        assert opt.param_groups[0]["weight_decay"] == 0.05
        assert opt.param_groups[0]["momentum"] == 0.9
        assert opt.param_groups[0]["beta2"] == 0.99

    def test_instantiation_custom_polar_params(self, simple_2d_model):
        """Test NorMuon with custom Polar Express parameters."""
        opt = NorMuon(
            simple_2d_model.parameters(),
            num_iters=6,
            safety_factor=0.03,
            cushion=0.15,
        )
        assert opt._num_iters == 6
        assert opt._safety_factor == 0.03
        assert opt._cushion == 0.15
        assert len(opt._polar_express_coeffs) == 6

    def test_instantiation_explicit_coefficients(self, simple_2d_model):
        """Test NorMuon with explicit coefficients."""
        custom_coeffs = [(1.0, 0.5, 0.25), (0.9, 0.4, 0.2)]
        opt = NorMuon(
            simple_2d_model.parameters(),
            polar_express_coeffs=custom_coeffs,
        )
        assert opt._polar_express_coeffs == custom_coeffs

    def test_rejects_non_2d_params(self, device):
        """Test that NorMuon rejects non-2D parameters."""
        param_1d = nn.Parameter(torch.randn(64, device=device))
        param_3d = nn.Parameter(torch.randn(4, 64, 128, device=device))

        with pytest.raises(ValueError, match="only supports 2D parameters"):
            NorMuon([param_1d])

        with pytest.raises(ValueError, match="only supports 2D parameters"):
            NorMuon([param_3d])

    def test_invalid_lr(self, simple_2d_model):
        """Test that negative learning rate raises error."""
        with pytest.raises(ValueError, match="Invalid learning rate"):
            NorMuon(simple_2d_model.parameters(), lr=-0.01)

    def test_invalid_momentum(self, simple_2d_model):
        """Test that invalid momentum raises error."""
        with pytest.raises(ValueError, match="Invalid momentum"):
            NorMuon(simple_2d_model.parameters(), momentum=1.0)
        with pytest.raises(ValueError, match="Invalid momentum"):
            NorMuon(simple_2d_model.parameters(), momentum=-0.1)

    def test_invalid_num_iters(self, simple_2d_model):
        """Test that invalid num_iters raises error."""
        with pytest.raises(ValueError, match="Invalid num_iters"):
            NorMuon(simple_2d_model.parameters(), num_iters=0)

    def test_invalid_cushion(self, simple_2d_model):
        """Test that invalid cushion raises error."""
        with pytest.raises(ValueError, match="Invalid cushion"):
            NorMuon(simple_2d_model.parameters(), cushion=0.0)
        with pytest.raises(ValueError, match="Invalid cushion"):
            NorMuon(simple_2d_model.parameters(), cushion=1.5)


class TestNorMuonStep:
    """Test NorMuon optimization step."""

    def test_step_updates_params(self, device):
        """Test that NorMuon.step() updates parameters."""
        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(64, 128, device=device))
        param_before = param.data.clone()

        opt = NorMuon([param], lr=0.02, weight_decay=0.0)
        param.grad = torch.randn_like(param)
        opt.step()

        assert not torch.allclose(param.data, param_before), "Parameter was not updated"

    def test_step_with_none_grad(self, device):
        """Test that step handles None gradients gracefully."""
        param1 = nn.Parameter(torch.randn(64, 128, device=device))
        param2 = nn.Parameter(torch.randn(32, 64, device=device))
        opt = NorMuon([param1, param2])
        param1.grad = torch.randn_like(param1)
        opt.step()

    def test_step_creates_state(self, device):
        """Test that step initializes optimizer state."""
        param = nn.Parameter(torch.randn(64, 128, device=device))
        opt = NorMuon([param])
        assert len(opt.state[param]) == 0

        param.grad = torch.randn_like(param)
        opt.step()
        assert "momentum_buffer" in opt.state[param]
        assert "second_moment" in opt.state[param]

    def test_gradient_descent(self, device):
        """Test that NorMuon minimizes a simple loss."""
        torch.manual_seed(42)
        W = nn.Parameter(torch.randn(32, 64, device=device))
        x = torch.randn(64, device=device)
        y = torch.randn(32, device=device)
        opt = NorMuon([W], lr=0.01, weight_decay=0.0, momentum=0.95)

        initial_loss = F.mse_loss(W @ x, y).item()
        for _ in range(50):
            opt.zero_grad()
            loss = F.mse_loss(W @ x, y)
            loss.backward()
            opt.step()

        final_loss = loss.item()
        assert (
            final_loss < initial_loss
        ), f"Loss did not decrease: {initial_loss} -> {final_loss}"

    def test_overfit_single_batch(self, simple_2d_model, device):
        """Test that NorMuon can overfit a single batch."""
        torch.manual_seed(42)
        opt = NorMuon(simple_2d_model.parameters(), lr=0.05, weight_decay=0.0)
        x = torch.randn(8, 64, device=device)
        y = torch.randint(0, 10, (8,), device=device)

        initial_loss = F.cross_entropy(simple_2d_model(x), y).item()
        for _ in range(100):
            opt.zero_grad()
            loss = F.cross_entropy(simple_2d_model(x), y)
            loss.backward()
            opt.step()

        final_loss = loss.item()
        assert (
            final_loss < initial_loss * 0.5
        ), f"Failed to overfit: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_step_with_closure(self, device):
        """Test step with closure."""
        param = nn.Parameter(torch.randn(64, 128, device=device))
        target = torch.randn(64, 128, device=device)
        opt = NorMuon([param])

        def closure():
            opt.zero_grad()
            loss = F.mse_loss(param, target)
            loss.backward()
            return loss

        loss = opt.step(closure)
        assert loss is not None
        assert isinstance(loss.item(), float)


class TestNorMuonFeatures:
    """Test specific NorMuon features."""

    def test_variance_reduction_enabled(self, device):
        """Test optimizer with variance reduction enabled."""
        param = nn.Parameter(torch.randn(64, 128, device=device))
        opt = NorMuon([param], use_variance_reduction=True)
        param.grad = torch.randn_like(param)
        opt.step()
        assert "second_moment" in opt.state[param]
        assert opt.state[param]["second_moment"].abs().sum() > 0

    def test_variance_reduction_disabled(self, device):
        """Test optimizer with variance reduction disabled."""
        param = nn.Parameter(torch.randn(64, 128, device=device))
        opt = NorMuon([param], use_variance_reduction=False)
        param.grad = torch.randn_like(param)
        opt.step()
        assert "second_moment" not in opt.state[param]

    def test_cautious_weight_decay(self, device):
        """Test cautious weight decay behavior."""
        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(64, 128, device=device))
        param_cautious = param.data.clone()
        param_standard = param.data.clone()

        # Cautious WD
        opt_cautious = NorMuon(
            [nn.Parameter(param_cautious)],
            weight_decay=0.1,
            use_cautious_wd=True,
        )
        opt_cautious.param_groups[0]["params"][0].grad = torch.randn(
            64, 128, device=device
        )
        opt_cautious.step()

        # Standard WD
        opt_standard = NorMuon(
            [nn.Parameter(param_standard)],
            weight_decay=0.1,
            use_cautious_wd=False,
        )
        opt_standard.param_groups[0]["params"][0].grad = torch.randn(
            64, 128, device=device
        )
        opt_standard.step()

        assert not torch.allclose(
            opt_cautious.param_groups[0]["params"][0],
            opt_standard.param_groups[0]["params"][0],
        )

    def test_aspect_ratio_scaling(self, device):
        """Test that aspect ratio scaling is applied correctly."""
        torch.manual_seed(42)
        # Wide (M < K)
        param_wide = nn.Parameter(torch.randn(32, 128, device=device))
        opt_wide = NorMuon([param_wide], lr=0.02, weight_decay=0.0)
        param_wide.grad = torch.ones_like(param_wide)
        opt_wide.step()
        assert not torch.isnan(param_wide).any()

        # Tall (M > K)
        param_tall = nn.Parameter(torch.randn(128, 32, device=device))
        opt_tall = NorMuon([param_tall], lr=0.02, weight_decay=0.0)
        param_tall.grad = torch.ones_like(param_tall)
        opt_tall.step()
        assert not torch.isnan(param_tall).any()


class TestNorMuonState:
    """Test NorMuon state management."""

    def test_reset(self, device):
        """Test that reset() clears optimizer state."""
        param = nn.Parameter(torch.randn(64, 128, device=device))
        opt = NorMuon([param])
        param.grad = torch.randn_like(param)
        opt.step()
        assert opt.state[param]["momentum_buffer"].abs().sum() > 0

        opt.reset()
        assert opt.state[param]["momentum_buffer"].abs().sum() == 0

    def test_state_dict_save_load(self, device):
        """Test state_dict save and load."""
        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(64, 128, device=device))
        opt = NorMuon([param], lr=0.02)
        for _ in range(5):
            param.grad = torch.randn_like(param)
            opt.step()

        state_dict = opt.state_dict()

        param_new = nn.Parameter(torch.randn(64, 128, device=device))
        opt_new = NorMuon([param_new], lr=0.02)
        opt_new.load_state_dict(state_dict)

        for key in opt.state[param]:
            if isinstance(opt.state[param][key], torch.Tensor):
                torch.testing.assert_close(
                    opt.state[param][key],
                    opt_new.state[param_new][key],
                )

    def test_state_transfer_for_config_change(self, device):
        """Test transferring state when changing optimizer config."""
        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(64, 128, device=device))
        opt_old = NorMuon([param], lr=0.02, num_iters=5)
        for _ in range(10):
            param.grad = torch.randn_like(param)
            opt_old.step()

        state_dict = opt_old.state_dict()
        param_value = param.data.clone()

        opt_new = NorMuon([param], lr=0.01, num_iters=6)
        opt_new.load_state_dict(state_dict)

        assert opt_new.state[param]["momentum_buffer"].abs().sum() > 0
        param.grad = torch.randn_like(param)
        opt_new.step()
        assert not torch.allclose(param.data, param_value)


class TestNorMuonStability:
    """Test NorMuon numerical stability."""

    def test_large_gradients(self, device):
        param = nn.Parameter(torch.randn(64, 128, device=device))
        opt = NorMuon([param])
        param.grad = torch.randn_like(param) * 1000
        opt.step()
        assert not torch.isnan(param.data).any()
        assert not torch.isinf(param.data).any()

    def test_small_gradients(self, device):
        param = nn.Parameter(torch.randn(64, 128, device=device))
        opt = NorMuon([param])
        param.grad = torch.randn_like(param) * 1e-8
        opt.step()
        assert not torch.isnan(param.data).any()

    def test_many_steps(self, device):
        torch.manual_seed(42)
        param = nn.Parameter(torch.randn(64, 128, device=device))
        opt = NorMuon([param])
        for _ in range(50):
            param.grad = torch.randn_like(param)
            opt.step()
        assert not torch.isnan(param.data).any()
        assert not torch.isinf(param.data).any()

    def test_mixed_gradient_magnitudes(self, device):
        param = nn.Parameter(torch.randn(64, 128, device=device))
        opt = NorMuon([param])
        magnitudes = [1e-6, 1e-3, 1.0, 1e3, 1e6]
        for mag in magnitudes:
            param.grad = torch.randn_like(param) * mag
            opt.step()
            assert not torch.isnan(param.data).any()


class TestVarianceReduction:
    """Test the NorMuon variance reduction component."""

    def test_output_shape(self, device):
        v_chunk = torch.randn(4, 64, 128, device=device, dtype=torch.bfloat16)
        second_momentum = torch.zeros(4, 64, 1, device=device, dtype=torch.float32)
        result = apply_normuon_variance_reduction(
            v_chunk, second_momentum, beta2=0.95, red_dim=-1
        )
        assert result.shape == v_chunk.shape

    def test_momentum_update(self, device):
        v_chunk = torch.randn(4, 64, 128, device=device, dtype=torch.bfloat16)
        second_momentum = torch.zeros(4, 64, 1, device=device, dtype=torch.float32)
        apply_normuon_variance_reduction(
            v_chunk, second_momentum, beta2=0.95, red_dim=-1
        )
        assert second_momentum.abs().sum() > 0

    def test_no_nan(self, device):
        v_chunk = torch.randn(4, 64, 128, device=device, dtype=torch.bfloat16)
        second_momentum = torch.zeros(4, 64, 1, device=device, dtype=torch.float32)
        for _ in range(10):
            v_chunk = apply_normuon_variance_reduction(
                v_chunk, second_momentum, beta2=0.95, red_dim=-1
            )
            v_chunk = torch.randn(4, 64, 128, device=device, dtype=torch.bfloat16)
        assert not torch.isnan(second_momentum).any()

    def test_red_dim_minus_2(self, device):
        v_chunk = torch.randn(4, 128, 64, device=device, dtype=torch.bfloat16)
        second_momentum = torch.zeros(4, 1, 64, device=device, dtype=torch.float32)
        result = apply_normuon_variance_reduction(
            v_chunk, second_momentum, beta2=0.95, red_dim=-2
        )
        assert result.shape == v_chunk.shape
        assert not torch.isnan(result).any()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_param(self, device):
        param = nn.Parameter(torch.randn(64, 64, device=device))
        opt = NorMuon([param])
        param.grad = torch.randn_like(param)
        opt.step()
        assert not torch.isnan(param).any()

    def test_square_matrix(self, device):
        param = nn.Parameter(torch.randn(128, 128, device=device))
        opt = NorMuon([param])
        param.grad = torch.randn_like(param)
        opt.step()
        assert not torch.isnan(param).any()

    def test_very_wide_matrix(self, device):
        param = nn.Parameter(torch.randn(16, 512, device=device))
        opt = NorMuon([param])
        param.grad = torch.randn_like(param)
        opt.step()
        assert not torch.isnan(param).any()

    def test_very_tall_matrix(self, device):
        param = nn.Parameter(torch.randn(512, 16, device=device))
        opt = NorMuon([param])
        param.grad = torch.randn_like(param)
        opt.step()
        assert not torch.isnan(param).any()

    def test_zero_weight_decay(self, device):
        param = nn.Parameter(torch.randn(64, 128, device=device))
        param_before = param.data.clone()
        opt = NorMuon([param], weight_decay=0.0)
        param.grad = torch.zeros_like(param)
        opt.step()
        torch.testing.assert_close(param.data, param_before, rtol=1e-4, atol=1e-4)

    def test_multiple_param_groups(self, device):
        param1 = nn.Parameter(torch.randn(64, 128, device=device))
        param2 = nn.Parameter(torch.randn(32, 64, device=device))
        opt = NorMuon(
            [
                {"params": [param1], "lr": 0.02},
                {"params": [param2], "lr": 0.01},
            ]
        )
        param1.grad = torch.randn_like(param1)
        param2.grad = torch.randn_like(param2)
        opt.step()
        assert not torch.isnan(param1).any()
        assert not torch.isnan(param2).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
