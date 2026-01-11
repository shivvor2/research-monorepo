"""
Simplified NorMuon optimizer without optimizations for distributed training

Reference implementation (hardcoded for use in 8xH100 cluster):
https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py#L463

This is a non-distributed version of NorMuon that can be used for local testing
and single-GPU training.

Includes the following algorithmic improvements:
1. Polar Express orthogonalization (instead of Newton-Schulz)
2. NorMuon variance reduction (Adafactor-style)
3. Cautious weight decay

Usage:
    optimizer = NorMuon(
        params=model.parameters(),
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        beta2=0.95,
    )

To change polar express parameters mid-training (e.g., for annealing):
    new_optimizer = NorMuon(params, num_iters=6, ...)
    new_optimizer.load_state_dict(old_optimizer.state_dict())
"""

from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch import Tensor
from torch.optim.optimizer import Optimizer


@torch.compile(dynamic=False, fullgraph=True)
def apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2, red_dim):
    """NorMuon variance reduction. Algebraically fuses the normalization steps to minimize memory ops."""
    v_mean = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v_chunk.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
    v_norm = v_norm_sq.sqrt_()
    second_momentum_buffer.lerp_(
        v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2
    )
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
    return v_chunk.mul_(final_scale.type_as(v_chunk))


def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]


@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def XXT_kernel(
    A_ptr,
    C_ptr,
    M,
    K,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    ALLOW_TF32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed (MATCH REFERENCE EXACTLY)
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def XXT(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    # Follow torch's global flag for tf32 availiability
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size
        * triton.cdiv(M, meta["BLOCK_SIZE_M"])
        * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    XXT_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        ALLOW_TF32=allow_tf32,
    )
    return out


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ba_plus_cAA_kernel(
    A_ptr,
    C_ptr,
    M,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    alpha,
    beta,
    ALLOW_TF32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from XXT_kernel, but also loads and adds a block of A
    # Performance is slightly slower than XXT_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    allow_tf32 = torch.backends.cuda.matmul.allow_tf32

    grid = lambda meta: (
        batch_size
        * triton.cdiv(M, meta["BLOCK_SIZE_M"])
        * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ba_plus_cAA_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
        ALLOW_TF32=allow_tf32,
    )
    return out


# =============================================================================
# Polar Express Coefficients
# =============================================================================

# Default coefficients computed for num_iters=5, safety_factor=0.02, cushion=0.1
# These provide a good balance of speed and accuracy for typical training scenarios.
# See: "The Polar Express" (Amsel et al., 2025) https://arxiv.org/abs/2505.16932
DEFAULT_POLAR_EXPRESS_COEFFS: List[Tuple[float, float, float]] = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

DEFAULT_NUM_ITERS = 5
DEFAULT_SAFETY_FACTOR = 0.02
DEFAULT_CUSHION = 0.1


def compute_polar_express_coeffs(
    num_iters: int = DEFAULT_NUM_ITERS,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    cushion: float = DEFAULT_CUSHION,
    lower_bound: float = 1e-3,
) -> List[Tuple[float, float, float]]:
    r"""Compute Polar Express coefficients using the Remez algorithm.

    This implements the offline stage of the Polar Express algorithm, which
    precomputes optimal polynomial coefficients for matrix sign approximation.

    See: "The Polar Express: Optimal Matrix Sign Methods and Their Application
    to the Muon Algorithm" (Amsel et al., 2025) https://arxiv.org/abs/2505.16932

    Args:
        num_iters: Number of Polar Express iterations. Higher values give more
            accurate orthogonalization but are slower. Default: 5.
            - Use 4 if optimizer compute is a bottleneck
            - Use 6-7 for optimization instability or ill-conditioned gradients
        safety_factor: Factor to prevent singular values from exceeding 1 due to
            numerical errors. Increase if seeing NaN/Inf. Default: 0.02.
        cushion: Minimum ratio of lower to upper bound for polynomial selection.
            Prevents instability from extremely small singular values. Default: 0.1.
        lower_bound: Initial lower bound for singular values. Default: 1e-3.

    Returns:
        List of (a, b, c) coefficient tuples for each iteration, where the
        polynomial is p(x) = a*x + b*x*(x@x.T) + c*x*(x@x.T)@(x@x.T).
    """
    import numpy as np
    from numpy.polynomial import chebyshev as cheb

    def remez_sign_approx(lower: float, upper: float, degree: int) -> np.ndarray:
        """Compute optimal polynomial approximation to sign(x) on [lower, upper]."""
        # Transform to [-1, 1] for Chebyshev basis
        scale = 2.0 / (upper - lower)
        shift = -(upper + lower) / (upper - lower)

        def sign_transformed(x):
            # sign((x - shift) / scale) = sign(x) for x in [-1, 1] mapped to [lower, upper]
            return np.sign((x - shift) / scale)

        # Use Chebyshev nodes for initial approximation
        n_points = degree + 2
        cheb_nodes = np.cos(np.pi * np.arange(n_points) / (n_points - 1))

        # Remez exchange algorithm
        nodes = cheb_nodes.copy()
        for _ in range(50):  # Max iterations
            # Solve for polynomial coefficients
            V = np.vander(nodes[:-1], degree + 1, increasing=True)
            signs = np.array([(-1) ** i for i in range(degree + 1)])
            V = np.column_stack([V, signs])

            target = sign_transformed(nodes[:-1])
            target = np.append(target, sign_transformed(nodes[-1]))

            try:
                coeffs_and_err = np.linalg.solve(
                    np.vstack(
                        [
                            V,
                            np.append(
                                np.vander([nodes[-1]], degree + 1, increasing=True)[0],
                                (-1) ** (degree + 1),
                            ),
                        ]
                    ),
                    target,
                )
            except np.linalg.LinAlgError:
                break

            coeffs = coeffs_and_err[:-1]

            # Find new extrema
            test_points = np.linspace(-1, 1, 1000)
            poly_vals = np.polyval(coeffs[::-1], test_points)
            errors = sign_transformed(test_points) - poly_vals

            # Find local extrema of error
            extrema_idx = []
            for i in range(1, len(errors) - 1):
                if (errors[i] > errors[i - 1] and errors[i] > errors[i + 1]) or (
                    errors[i] < errors[i - 1] and errors[i] < errors[i + 1]
                ):
                    extrema_idx.append(i)

            if len(extrema_idx) >= degree + 2:
                nodes = test_points[extrema_idx[: degree + 2]]
            else:
                break

        return coeffs

    def poly_to_iteration_coeffs(
        coeffs: np.ndarray, lower: float, upper: float
    ) -> Tuple[float, float, float]:
        """Convert polynomial coefficients to (a, b, c) iteration form."""
        # The iteration computes: X_new = a*X + b*X@(X@X.T) + c*X@(X@X.T)@(X@X.T)
        # which corresponds to p(x) = a + b*x^2 + c*x^4 applied to singular values
        # We need to extract these from the Remez polynomial

        # For a degree-5 odd polynomial p(x) = c1*x + c3*x^3 + c5*x^5
        # The iteration form uses X, X@X.T@X, X@(X.T@X)^2
        # which applies q(s) = a + b*s + c*s^2 to singular values squared

        if len(coeffs) >= 6:
            # Odd polynomial: p(x) = c1*x + c3*x^3 + c5*x^5
            c1 = coeffs[1] if len(coeffs) > 1 else 0
            c3 = coeffs[3] if len(coeffs) > 3 else 0
            c5 = coeffs[5] if len(coeffs) > 5 else 0
            return (float(c1), float(c3), float(c5))
        else:
            # Fallback for lower degree
            return (1.0, 0.0, 0.0)

    # Compute coefficients for each iteration
    coeffs_list = []
    current_lower = lower_bound
    current_upper = 1.0

    for i in range(num_iters):
        # Apply cushion: don't let lower bound get too small relative to upper
        effective_lower = max(current_lower, current_upper * cushion)

        # Compute optimal polynomial for this iteration
        # Degree 5 gives the (a, b, c) three-term form
        poly_coeffs = remez_sign_approx(effective_lower, current_upper, degree=5)

        # Convert to iteration coefficients
        a, b, c = poly_to_iteration_coeffs(poly_coeffs, effective_lower, current_upper)
        coeffs_list.append((a, b, c))

        # Update bounds for next iteration
        # After applying p(x), singular values in [l, u] map to [p(l), p(u)]
        # For sign approximation, this contracts toward 1
        current_lower = effective_lower**2 * abs(a + b + c)  # Rough approximation
        current_upper = 1.0 / (1.0 + safety_factor)

    return coeffs_list


def _make_polar_express(
    coeffs: List[Tuple[float, float, float]],
    safety_factor: float,
):
    """Factory function to create a compiled polar_express function with fixed coefficients.

    The coefficients must be fixed at function definition time for torch.compile
    to work efficiently with dynamic=False.

    Args:
        coeffs: List of (a, b, c) coefficient tuples for each iteration.
        safety_factor: Safety factor for spectral norm normalization.

    Returns:
        Compiled polar_express function.
    """
    # Capture coefficients as closure variables (compile-time constants)
    _coeffs = tuple(coeffs)
    _safety_mult = 1.0 + safety_factor

    @torch.compile(dynamic=False, fullgraph=True)
    def polar_express(G: Tensor, split_baddbmm: bool = False) -> Tensor:
        """Polar Express Sign Method for matrix orthogonalization.

        See: "The Polar Express" (Amsel et al., 2025) https://arxiv.org/abs/2505.16932

        Args:
            G: Input gradient tensor of shape (..., M, K).
            split_baddbmm: If True, split baddbmm into separate ops to avoid
                cudaMemcpyAsync for large matrices.

        Returns:
            Orthogonalized tensor of the same shape as G.
        """
        X = G.bfloat16()
        if G.size(-2) > G.size(-1):
            X = X.mT

        # Ensure spectral norm is at most 1
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * _safety_mult + 1e-6)

        # Allocate buffers
        X = X.contiguous()
        A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        # Select batched vs unbatched matmul
        if split_baddbmm:
            BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

        # Perform the iterations
        for a, b, c in _coeffs:
            XXT(X, out=A)
            ba_plus_cAA(A, alpha=c, beta=b, out=B)

            if split_baddbmm:
                BX_matmul(B, X, out=C)
                C.add_(X, alpha=a)
            else:
                aX_plus_BX(X, B, X, beta=a, out=C)

            X, C = C, X

        if G.size(-2) > G.size(-1):
            X = X.mT
        return X

    return polar_express


# Default compiled polar_express for use when default coefficients are used
_default_polar_express = _make_polar_express(
    DEFAULT_POLAR_EXPRESS_COEFFS,
    DEFAULT_SAFETY_FACTOR,
)


# =============================================================================
# NorMuon Optimizer
# =============================================================================


class NorMuon(Optimizer):
    r"""Single-device NorMuon optimizer.

    NorMuon combines Muon (Momentum Orthogonalized Update) with several
    improvements for better training stability and efficiency:

    1. **Polar Express orthogonalization**: Faster and more accurate than
       Newton-Schulz iteration for computing the matrix sign function.
    2. **Variance reduction**: Adafactor-style second moment estimation
       to reduce gradient variance.
    3. **Cautious weight decay**: Only applies weight decay where the
       gradient agrees with the parameter sign.

    This optimizer should only be used for 2D weight matrices (attention, MLP).
    Use AdamW for embeddings, biases, and normalization parameters.

    .. note::
        To change Polar Express parameters mid-training (e.g., increasing
        iterations during annealing), create a new optimizer and transfer state::

            new_opt = NorMuon(params, num_iters=6, ...)
            new_opt.load_state_dict(old_opt.state_dict())

    .. note:: **Precision and Compilation**
        This optimizer respects the global ``torch.backends.cuda.matmul.allow_tf32`` flag
        for its internal Triton kernels. On Ampere+ GPUs, this defaults to True.

        Because the optimizer core is compiled with ``torch.compile(dynamic=False)``,
        the value of this flag is captured at the time of the **first step**.
        Changing ``allow_tf32`` after the first step will not affect the compilation
        unless a re-compilation is triggered. It is recommended to set your desired precision
        before the training loop begins.

    Args:
        params: Iterable of parameters to optimize. Must be 2D tensors.
        lr: Learning rate. Default: 0.02.
        weight_decay: Weight decay coefficient. Default: 0.01.
        momentum: Momentum coefficient for Nesterov momentum. Default: 0.95.
        beta2: Second moment coefficient for variance reduction. Default: 0.95.
        use_variance_reduction: Whether to use NorMuon variance reduction.
            Default: True.
        use_cautious_wd: Whether to use cautious weight decay. Default: True.
        num_iters: Number of Polar Express iterations. Higher values give more
            accurate orthogonalization but are slower. Default: 5.

            - Use 4 if optimizer compute is a bottleneck
            - Use 6-7 for optimization instability or ill-conditioned gradients
        safety_factor: Safety factor for spectral norm normalization to prevent
            numerical overflow. Increase if seeing NaN/Inf. Default: 0.02.
        cushion: Minimum ratio of lower to upper singular value bound for
            polynomial selection. Default: 0.1.
        polar_express_coeffs: Pre-computed Polar Express coefficients. If
            provided, ``num_iters``, ``safety_factor``, and ``cushion`` are
            ignored for coefficient computation (though ``safety_factor`` is
            still used in the online normalization step). Default: None.

    References:
        - Muon: https://arxiv.org/abs/2510.05491
        - Polar Express: https://arxiv.org/abs/2505.16932

    Example::

        >>> model = nn.Linear(512, 256, bias=False)
        >>> optimizer = NorMuon(model.parameters(), lr=0.02)
        >>> optimizer.zero_grad()
        >>> loss = model(input).sum()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        beta2: float = 0.95,
        use_variance_reduction: bool = True,
        use_cautious_wd: bool = True,
        *,
        num_iters: int = DEFAULT_NUM_ITERS,
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
        cushion: float = DEFAULT_CUSHION,
        polar_express_coeffs: Optional[List[Tuple[float, float, float]]] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if num_iters < 1:
            raise ValueError(f"Invalid num_iters value: {num_iters}")
        if safety_factor < 0.0:
            raise ValueError(f"Invalid safety_factor value: {safety_factor}")
        if not 0.0 < cushion <= 1.0:
            raise ValueError(f"Invalid cushion value: {cushion}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            beta2=beta2,
            use_variance_reduction=use_variance_reduction,
            use_cautious_wd=use_cautious_wd,
        )
        super().__init__(params, defaults)

        # Validate that all params are 2D
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"NorMuon only supports 2D parameters, got shape {p.shape}. "
                        "Use AdamW for embeddings, biases, and LayerNorm parameters."
                    )

        # Store polar express configuration (immutable after init)
        self._num_iters = num_iters
        self._safety_factor = safety_factor
        self._cushion = cushion

        # Determine coefficients and create compiled polar_express function
        if polar_express_coeffs is not None:
            self._polar_express_coeffs = list(polar_express_coeffs)
            self._polar_express = _make_polar_express(
                self._polar_express_coeffs,
                safety_factor,
            )
        elif (
            num_iters == DEFAULT_NUM_ITERS
            and safety_factor == DEFAULT_SAFETY_FACTOR
            and cushion == DEFAULT_CUSHION
        ):
            # Use precomputed defaults for efficiency
            self._polar_express_coeffs = DEFAULT_POLAR_EXPRESS_COEFFS
            self._polar_express = _default_polar_express
        else:
            # Compute custom coefficients
            self._polar_express_coeffs = compute_polar_express_coeffs(
                num_iters=num_iters,
                safety_factor=safety_factor,
                cushion=cushion,
            )
            self._polar_express = _make_polar_express(
                self._polar_express_coeffs,
                safety_factor,
            )

    def reset(self) -> None:
        r"""Reset optimizer state.

        Clears all momentum buffers and second moment estimates, effectively
        restarting the optimizer while preserving configuration.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "momentum_buffer" in state:
                    state["momentum_buffer"].zero_()
                if "second_moment" in state:
                    state["second_moment"].zero_()

    @torch.no_grad()
    def step(self, closure=None):
        r"""Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                Optional.

        Returns:
            Loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            beta2 = group["beta2"]
            use_vr = group["use_variance_reduction"]
            use_cautious = group["use_cautious_wd"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    if use_vr:
                        red_dim = -1 if p.shape[-2] >= p.shape[-1] else -2
                        if red_dim == -1:
                            state["second_moment"] = torch.zeros(
                                p.shape[-2], 1, device=p.device, dtype=torch.float32
                            )
                        else:
                            state["second_moment"] = torch.zeros(
                                1, p.shape[-1], device=p.device, dtype=torch.float32
                            )
                        state["red_dim"] = red_dim

                # Momentum update
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)

                # Nesterov momentum
                update = grad.lerp_(buf, momentum)

                # Polar Express orthogonalization
                update = self._polar_express(
                    update.unsqueeze(0), split_baddbmm=False
                ).squeeze(0)

                # Scale by aspect ratio
                update = update * max(1, p.shape[-2] / p.shape[-1]) ** 0.5

                # Variance reduction (NorMuon)
                if use_vr:
                    red_dim = state["red_dim"]
                    second_moment = state["second_moment"]
                    update = (
                        apply_normuon_variance_reduction(
                            update.unsqueeze(0).bfloat16(),
                            second_moment.unsqueeze(0),
                            beta2,
                            red_dim,
                        )
                        .squeeze(0)
                        .float()
                    )

                # Weight decay (cautious or standard)
                if wd != 0:
                    if use_cautious:
                        mask = (update * p.float()) >= 0
                        p.add_(p * mask * (-lr * wd))
                    else:
                        p.add_(p, alpha=-lr * wd)

                # Parameter update
                p.add_(update.to(p.dtype), alpha=-lr)

        return loss
