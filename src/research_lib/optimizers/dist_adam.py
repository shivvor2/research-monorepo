"""
Copied from https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py
"""

from collections import defaultdict

import torch
import torch.distributed as dist


class DistAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        label_order: list[str],
        betas: list[list[float]],
        lr: float = 1e-3,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        params = list(params)
        # Group by label, with explicit ordering for execution control.
        params_by_label = defaultdict(list)
        for p in params:
            params_by_label[getattr(p, "label", None)].append(p)
        param_groups = []
        for idx, label in enumerate(label_order):
            if label in params_by_label:
                param_groups.append(
                    dict(params=params_by_label[label], betas=betas[idx])
                )
        # include any unlabeled params at the end (processed last)
        if None in params_by_label:
            param_groups.append(dict(params=params_by_label[None]))
        super().__init__(param_groups, defaults)
        # init state: small params (numel < 1024) use full-sized state, others use sharded
        for p in params:
            chunk = p if p.numel() < 1024 else p[: p.size(0) // self.world_size]
            exp_avg = torch.zeros_like(chunk, dtype=torch.float32, device=p.device)
            self.state[p] = dict(
                step=0, exp_avg=exp_avg, exp_avg_sq=torch.zeros_like(exp_avg)
            )

        # tag the final param for optimizer pipelining, run all gather after muon copy
        param_groups[-1]["params"][-1].is_final_param = True

        # DistributedAdam implementation by @vagrawal, @akash5474
        self.should_sync = False
        self._reduce_scatter_hooks = []
        self._reduce_scatter_futures = {}
        # 0-D CPU tensors to avoid recompilation in _update_step
        self._step_size_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self.register_backward_hooks()

    def register_backward_hooks(self):
        for group in self.param_groups:
            for param in group["params"]:
                self._reduce_scatter_hooks.append(
                    param.register_post_accumulate_grad_hook(self._sync_gradient)
                )

    def load_state_dict(self, state_dict):
        """Override to preserve optimizer state dtypes (avoid BFloat16->Float32 cast that causes recompilation)."""
        # Save original state dtypes before loading
        original_dtypes = {}
        for p, s in self.state.items():
            original_dtypes[p] = {
                k: v.dtype for k, v in s.items() if isinstance(v, torch.Tensor)
            }

        # Call parent load_state_dict (which may cast dtypes to match param dtype)
        super().load_state_dict(state_dict)

        # Restore original dtypes
        for p, s in self.state.items():
            if p in original_dtypes:
                for k, v in s.items():
                    if isinstance(v, torch.Tensor) and k in original_dtypes[p]:
                        if v.dtype != original_dtypes[p][k]:
                            s[k] = v.to(original_dtypes[p][k])

    @torch.no_grad()
    def _sync_gradient(self, param):
        if not self.should_sync:
            return

        grad = param.grad
        if param.numel() < 1024:
            # Small params: use all_reduce (no scatter/gather needed)
            self._reduce_scatter_futures[param] = (
                dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future(),
                grad,
            )
        else:
            rank_size = grad.shape[0] // self.world_size
            if grad is not None:
                grad_slice = torch.empty_like(grad[:rank_size])
                self._reduce_scatter_futures[param] = (
                    dist.reduce_scatter_tensor(
                        grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
                    ).get_future(),
                    grad_slice,
                )

    def copy_lm_to_embed(self):
        # run at 2/3 of training
        lm_head = self.param_groups[0]["params"][0]
        embed = self.param_groups[-2]["params"][0]
        lm_head_state = self.state[lm_head]
        embed_state = self.state[embed]
        embed_state["step"] = lm_head_state["step"]
        embed_state["exp_avg"] = lm_head_state["exp_avg"].clone()
        embed_state["exp_avg_sq"] = lm_head_state["exp_avg_sq"].clone()
        embed.data.copy_(lm_head.data)

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _update_step(
        p_slice, g_slice, exp_avg, exp_avg_sq, beta1, beta2, eps, step_size_t, eff_wd_t
    ):
        """Compiled Adam update step. step_size_t and eff_wd_t are 0-D CPU tensors to avoid recompilation."""
        exp_avg.mul_(beta1).add_(
            g_slice, alpha=1 - beta1
        )  # exp_avg = beta1 * exp_avg + (1 - beta1) * g_slice
        exp_avg_sq.mul_(beta2).addcmul_(
            g_slice, g_slice, value=1 - beta2
        )  # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * g_slice^2
        # compute step
        update = exp_avg.div(exp_avg_sq.sqrt().add_(eps)).mul_(
            step_size_t
        )  # update = (exp_avg / (sqrt(exp_avg_sq) + eps)) * step_size
        # cautious weight decay
        mask = (update * p_slice) > 0
        update.addcmul_(
            p_slice, mask, value=eff_wd_t
        )  # update += eff_wd_t * p_slice * mask
        p_slice.add_(other=update, alpha=-1.0)  # p_slice -= update

    @torch.no_grad()
    def step(self, muon_opt):
        muon_opt.step_p1()
        rank = dist.get_rank()
        all_gather_futures: list[torch.Future] = []

        last_param = None
        last_p_slice = None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for param in group["params"]:
                if param not in self._reduce_scatter_futures:
                    continue

                fut, g_slice = self._reduce_scatter_futures[param]
                fut.wait()

                is_small = param.numel() < 1024
                if is_small:
                    # Small params: g_slice is actually full grad, p_slice is full param
                    p_slice = param
                else:
                    rank_size = param.shape[0] // self.world_size
                    p_slice = param[rank * rank_size : (rank + 1) * rank_size]

                lr = group["lr"] * getattr(param, "lr_mul", 1.0)
                state = self.state[param]
                state["step"] += 1
                t = state["step"]

                # Pre-compute changing values as 0-D CPU tensors to avoid recompilation.
                # `.fill_(value)` is the same as "= value", but doesn't change the tensor object.
                bias1, bias2 = 1 - beta1**t, 1 - beta2**t
                self._step_size_t.fill_(lr * (bias2**0.5 / bias1))
                self._eff_wd_t.fill_(
                    lr * lr * wd * getattr(param, "wd_mul", 1.0)
                )  # `lr` included twice to serve as weight decay schedule.

                DistAdam._update_step(
                    p_slice,
                    g_slice,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    beta1,
                    beta2,
                    eps,
                    self._step_size_t,
                    self._eff_wd_t,
                )

                if not is_small:
                    if getattr(param, "is_final_param", False):
                        last_param = param
                        last_p_slice = p_slice
                    else:
                        all_gather_futures.append(
                            dist.all_gather_into_tensor(
                                param, p_slice, async_op=True
                            ).get_future()
                        )
        self._reduce_scatter_futures.clear()

        muon_opt.step_p2()
        torch.futures.collect_all(all_gather_futures).wait()

        if last_param is not None:
            last_all_gather_future = dist.all_gather_into_tensor(
                last_param, last_p_slice, async_op=True
            ).get_future()
        muon_opt.step_p3()
        torch.futures.collect_all([last_all_gather_future]).wait()
