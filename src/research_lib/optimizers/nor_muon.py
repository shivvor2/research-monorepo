"""
NorMuon Implementation copied from https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py#L463
"""

from collections import defaultdict

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch import Tensor, nn

# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99


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
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
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
        accumulator = tl.dot(a, at, accumulator)
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
    )
    return out


# Computed for num_iters=5, safety_factor=2e-2, cushion=2
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(
    dynamic=False, fullgraph=True
)  # Must use dynamic=False or else it's much slower
def polar_express(G: torch.Tensor, split_baddbmm: bool = False):
    """
    Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.
    """
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    # Select batched vs unbatched
    if split_baddbmm:
        BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
    else:
        aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)  # A = X @ X.mT
        ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A

        # Referencing X twice causes pytorch to make a defensive copy,
        # resulting in a cudaMemcpyAsync in baddbmm.
        # For large matrices (i.e., the mlp weights), it's faster to split
        # the operation into two kernels to avoid this.
        if split_baddbmm:
            BX_matmul(B, X, out=C)  # C = B @ X
            C.add_(X, alpha=a)  # C = C + a*X  (in-place, X only read)
        else:
            aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X

        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile(dynamic=False, fullgraph=True)
def cautious_wd_and_update_inplace(p, mantissa, grad, wd_tensor, lr_tensor):
    """
    Cautious weight decay + parameter update. wd_tensor and lr_tensor are 0-D CPU tensors.
    Mantissa is tracked to enable higher precision updates on bfloat16 parameters.
    bfloat16 format: 1 sign bit + 8 exponent bits + 7 mantissa bits = 16 bits total
    float32 format: 1 sign bit + 8 exponent bits + 23 mantissa bits = 32 bits total
    """
    assert p.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    wd_factor = wd_tensor.to(torch.float32)
    lr_factor = lr_tensor.to(torch.float32)
    p_precise_raw = (p.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    p_precise = p_precise_raw.view(torch.float32)
    mask = (grad * p_precise) >= 0
    p_precise.copy_(
        p_precise - (p_precise * mask * wd_factor * lr_factor) - (grad * lr_factor)
    )
    p.copy_((p_precise_raw >> 16).to(torch.uint16))
    mantissa.copy_(p_precise_raw.to(torch.uint16))


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


# -----------------------------------------------------------------------------
# NorMuon optimizer


class NorMuon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).

    Differences from standard Muon:
    - Newton-Shulz is replaced with Polar Express for the orthogonalization step
    - NorMuon adds a low-rank variance estimator similar to Adafactor. https://arxiv.org/pdf/2510.05491
    - small 1D parameters handled here instead of in Adam
    - Cautious weight decay, a gated version of decoupled weight decay
    - Custom distributed sizing:
    The model stores all attn and mlp weights in the same shape, and then updates the view as
    needed on the forward pass. This enables attn and mlp weights to be contained within the same
    dist.reduce_scatter_tensor() call. The model architecture has been customized to enable
    (n_attn_layers+n_mlp_layers*2)%8==0 for batching across 8 GPUs with zero padding on mlp and attn.
    The scheduling is:
        1. reduce scatter attn/mlp round 1 (10 attn params 6 mlp params)
        2. reduce scatter attn/mlp round 2 (16 mlp params)
        3. wait on step 1, then compute update of 1 and schedule all gather
        4. wait on step 2, then compute update of 2 and schedule all gather
            GPUs receive [2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 MLP, 2 MLP, 2 MLP]
            GPUs that receive params of type attn reshape before computing update
        5. wait for each all gather to complete and update params
    """

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        beta2=0.95,
        custom_sizing=True,
    ):
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2
        )
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        # custom sizing requires 8 GPUs
        if custom_sizing and dist.get_world_size() == 8:
            param_groups = self.generate_custom_param_groups(params)
        else:
            param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)

    def reset(self):
        # expose a reset for clearing buffers
        for group in self.param_groups:
            if "momentum_buffer" in group:
                group["momentum_buffer"].zero_()
                group["mantissa"].zero_()
                group["second_momentum_buffer"].zero_()

    def generate_standard_param_groups(self, params):
        """
        Use this method if running on less than 8 GPU or experimenting with additional attn or mlp modules.
        Creates one param group per module.
        """
        groups = defaultdict(list)
        for param in params:
            groups[param.label].append(param)

        param_groups = []
        for module_name, group_params in groups.items():
            chunk_size = (len(group_params) + self.world_size - 1) // self.world_size
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))

        return param_groups

    def generate_custom_param_groups(self, params):
        """
        Implementation requires that a single GPU does not receive both attn
        and mlp params when a param group is split across GPUs.
        """
        params_list = list(params)
        module_group_order = ["attn", "mlp"]
        group_sizes = [16, 16]  # 10 attn + 6 mlp, then 16 mlp
        params_list.sort(key=lambda x: module_group_order.index(x.label))

        idx = 0
        assert len(params_list) == sum(group_sizes)
        param_groups = []
        for size in group_sizes:
            chunk_size = (size + self.world_size - 1) // self.world_size
            group_params = params_list[idx : idx + size]
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))
            idx += size

        return param_groups

    def step(self):
        self.step_p1()
        self.step_p2()
        self.step_p3()

    @torch.no_grad()
    def step_p1(self):
        """
        Part 1: Launch distributed reduce_scatter operations for parameter groups.
        """
        rank = dist.get_rank()
        self.group_infos = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue

            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            stacked_grads = torch.empty(
                (padded_num_params, *params[0].shape),
                dtype=params[0].dtype,
                device=params[0].device,
            )
            for i, p in enumerate(params):
                stacked_grads[i].copy_(p.grad, non_blocking=True)
            if len(params) < padded_num_params:
                stacked_grads[len(params) :].zero_()

            grad_chunk = torch.empty_like(stacked_grads[:chunk_size])

            reduce_future = dist.reduce_scatter_tensor(
                grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
            ).get_future()

            self.group_infos.append(
                dict(grad_chunk=grad_chunk, reduce_future=reduce_future)
            )

    @torch.no_grad()
    def step_p2(self):
        """
        Part 2: Compute gradient updates and launch all_gather operations
        Wait for all gather to complete for all parameter groups except final one
        """
        # Efficient distributed step by @YouJiacheng, @KonstantinWilleke, @alexrgilbert,
        # @adricarda, @tuttyfrutyee, @vdlad, @ryanyang0, @vagrawal, @varunneal, @chrisjmccormick
        rank = dist.get_rank()
        group_infos = self.group_infos

        self.all_gather_infos = []
        # Second pass: wait for gradients, compute updates for the local shard of parameters,
        # and launch all async all_gather operations.
        for group, info in zip(self.param_groups, group_infos):
            info["reduce_future"].wait()

            params = group["params"]
            grad_chunk = info["grad_chunk"].float()
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            start_idx = rank * chunk_size
            module_idx = start_idx if start_idx < len(params) else 0

            num_params = min(
                chunk_size, max(0, len(params) - start_idx)
            )  # num params for this rank

            if "momentum_buffer" not in group:
                group["momentum_buffer"] = torch.zeros_like(
                    grad_chunk[:num_params], dtype=torch.float32
                )

            momentum_buffer = group["momentum_buffer"]
            # Apply momentum update to the persistent momentum buffer in-place
            momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
            updated_grads = grad_chunk[:num_params].lerp_(
                momentum_buffer, group["momentum"]
            )

            grad_shape = updated_grads.shape
            if params[module_idx].label == "attn":
                for p in params[module_idx : module_idx + num_params]:
                    assert p.label == "attn"
                updated_grads = updated_grads.view(
                    4 * grad_shape[0], grad_shape[1] // 4, grad_shape[2]
                )

            ref_param = params[module_idx]
            param_shape = ref_param.shape

            if "second_momentum_buffer" not in group:
                group["second_momentum_buffer"] = (
                    torch.zeros_like(updated_grads[..., :, :1], dtype=torch.float32)
                    if param_shape[-2] >= param_shape[-1]
                    else torch.zeros_like(updated_grads[..., :1, :])
                )
            second_momentum_buffer = group["second_momentum_buffer"]

            if "param_lr_cpu" not in group:
                # Define multipliers for ALL params in this group (global, not per-shard)
                lr_mults = []
                wd_mults = []
                for p in params:
                    # Increase learning rate for modules with larger inputs than outputs.
                    # This shape check also assumes rows=input, columns=output, so take care
                    # when changing memory layouts. @chrisjmccormick
                    shape = p.shape
                    if len(shape) >= 2:
                        shape_mult = max(1.0, shape[-2] / shape[-1]) ** 0.5
                    else:
                        shape_mult = 1.0
                    lr_mults.append(shape_mult * getattr(p, "lr_mul", 1.0))
                    wd_mults.append(getattr(p, "wd_mul", 1.0))
                # Define as cpu tensors to enable Inductor constant folding
                group["param_lr_cpu"] = torch.tensor(
                    lr_mults, dtype=torch.float32, device="cpu"
                )
                group["param_wd_cpu"] = torch.tensor(
                    wd_mults, dtype=torch.float32, device="cpu"
                )

            eff_lr_all = group["param_lr_cpu"] * group["lr"]
            eff_wd_all = group["param_wd_cpu"] * group["weight_decay"] * group["lr"]

            # Slice the portion corresponding to this rank's shard
            eff_lr_cpu = eff_lr_all[module_idx : module_idx + num_params]
            eff_wd_cpu = eff_wd_all[module_idx : module_idx + num_params]

            # Compute zeropower for the entire chunk in a single, batched call.
            if num_params == 0:
                v_chunk = updated_grads
            else:
                v_chunk = polar_express(
                    updated_grads, split_baddbmm=(ref_param.label == "mlp")
                )

            # Note that the head orientation in O is transposed relative to QKV, so red_dim
            # is 'incorrect' for O. However, correcting this showed no improvement. @chrisjmccormick
            red_dim = -1 if param_shape[-2] >= param_shape[-1] else -2

            v_chunk = apply_normuon_variance_reduction(
                v_chunk, second_momentum_buffer, group["beta2"], red_dim
            )

            v_chunk = v_chunk.view(grad_shape)

            # # "Cautious" weight decay (https://arxiv.org/abs/2510.12402)
            updated_params = torch.empty_like(grad_chunk, dtype=torch.bfloat16)
            if num_params > 0:
                # Work on a stacked copy to avoid touching original params
                param_chunk = torch.stack(params[module_idx : module_idx + num_params])

                if "mantissa" not in group:
                    group["mantissa"] = torch.zeros_like(
                        param_chunk, dtype=torch.uint16
                    )
                mantissa = group["mantissa"]

                for local_idx in range(num_params):
                    cautious_wd_and_update_inplace(
                        param_chunk[local_idx].view(torch.uint16),
                        mantissa[local_idx],
                        v_chunk[local_idx],
                        eff_wd_cpu[local_idx],
                        eff_lr_cpu[local_idx],
                    )
            else:
                param_chunk = torch.zeros_like(v_chunk)

            updated_params[:num_params].copy_(param_chunk)
            if num_params < chunk_size:
                updated_params[num_params:].zero_()

            stacked_params = torch.empty(
                (padded_num_params, *param_shape),
                dtype=updated_params.dtype,
                device=updated_params.device,
            )

            gather_future = dist.all_gather_into_tensor(
                stacked_params, updated_params, async_op=True
            ).get_future()

            self.all_gather_infos.append(
                {
                    "gather_future": gather_future,
                    "stacked_params": stacked_params,
                    "orig_params": params,
                }
            )

        # Final pass: wait for all_gather to complete for all except final and copy results back into original parameter tensors.
        for info in self.all_gather_infos[:-1]:
            info["gather_future"].wait()
            stacked_params = info["stacked_params"]
            orig_params = info["orig_params"]

            unstacked_params = torch.unbind(stacked_params)
            for i, p in enumerate(orig_params):
                p.copy_(unstacked_params[i], non_blocking=True)

    @torch.no_grad()
    def step_p3(self):
        """
        Part 3: Wait for final all gather to complete and copy results back into original parameter tensors
        """
        info = self.all_gather_infos[-1]
        info["gather_future"].wait()
        stacked_params = info["stacked_params"]
        orig_params = info["orig_params"]

        unstacked_params = torch.unbind(stacked_params)
        for i, p in enumerate(orig_params):
            p.copy_(unstacked_params[i], non_blocking=True)
