import math

import torch
from torch.optim import AdamW


class CautiousAdamW(AdamW):
    r"""Implements AdamW algorithm with Cautious Weight Decay.

    It inherits from :class:`torch.optim.AdamW` and modifies the update step
    to gate the weight decay update. Weight decay is only applied when the
    parameter update aligns with the parameter's current value (i.e.,
    pushing the weight towards zero).

    This helps prevent weight decay from fighting against "active" learning
    directions.

    .. math::
       \begin{aligned}
            &\dots \\
            &\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big) \\
            &\textbf{if } (\theta_t - \theta_{t-1}) \cdot \theta_{t-1} > 0: \\
            &\hspace{10mm} \theta_t \leftarrow \theta_t - \gamma \lambda \theta_{t-1}
       \end{aligned}

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm (default: False)
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            max_exp_avg_sqs = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "CautiousAdamW does not support sparse gradients"
                        )
                    grads.append(p.grad)

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = (
                            torch.zeros((1,), dtype=torch.float, device=p.device)
                            if group["capturable"]
                            else torch.tensor(0.0)
                        )
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    state_steps.append(state["step"])
                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

            beta1, beta2 = group["betas"]

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]

                step_t += 1

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step = step_t.item()

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                if group["amsgrad"]:
                    max_exp_avg_sq = max_exp_avg_sqs[i]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                else:
                    denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)

                denom.add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                update = exp_avg.div(denom).mul_(-step_size)

                if group["weight_decay"] > 0:
                    mask = (update * param) < 0

                    # Corrected update using addcmul_
                    # param += value * (mask * param)
                    # param -= lr * wd * (mask * param)
                    param.addcmul_(
                        param, mask, value=-group["lr"] * group["weight_decay"]
                    )

                param.add_(update)

        return loss
