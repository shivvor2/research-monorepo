"""
Preset configurations for common optimizers.

Each preset returns a tuple of (OptimizerConfig, ScheduleConfig) that work
well together. These are starting points—adjust based on your specific use case.

Available Presets:
    - :func:`default_adamw_config`: AdamW with warmup-stable-decay LR
    - :func:`default_muon_config`: Muon with LR and momentum schedules
    - :func:`default_sgd_config`: SGD with warmup and linear decay

Example:
    Using presets::

        from research_lib.training.presets import default_adamw_config
        from research_lib.training.configs import build_optimizer
        from research_lib.training.scheduling import ParamScheduler

        opt_config, schedule_config = default_adamw_config(lr=1e-4)
        optimizer = build_optimizer(opt_config, model.parameters())
        scheduler = ParamScheduler(
            optimizer,
            schedule_config.global_schedules,
            total_steps=100000
        )

See Also:
    - :class:`research_lib.training.configs.OptimizerConfig` for optimizer config
    - :class:`research_lib.training.configs.ScheduleConfig` for schedule config
    - :class:`research_lib.training.scheduling.ParamScheduler` for runtime scheduling
"""

from __future__ import annotations

from typing import Tuple

import torch.optim

from .configs import OptimizerConfig, ScheduleConfig
from .scheduling import WarmupStableDecaySchedule


def default_adamw_config(
    lr: float = 3e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    warmup_frac: float = 0.1,
    stable_frac: float = 0.7,
) -> Tuple[OptimizerConfig, ScheduleConfig]:
    """Default AdamW configuration with warmup-stable-decay LR schedule.

    Args:
        lr: Peak learning rate. Default: 3e-4.
        betas: Adam beta parameters. Default: (0.9, 0.95).
        weight_decay: Weight decay coefficient. Default: 0.1.
        warmup_frac: Fraction of training for LR warmup. Default: 0.1.
        stable_frac: Fraction of training at peak LR. Default: 0.7.

    Returns:
        Tuple of (OptimizerConfig, ScheduleConfig).
    """
    optimizer_config = OptimizerConfig(
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
        },
    )

    # Calculate cooldown fraction based on remainder
    # Total = warmup + stable + cooldown -> cooldown = 1 - warmup - stable
    cooldown_frac = max(0.0, 1.0 - warmup_frac - stable_frac)

    schedule_config = ScheduleConfig(
        global_schedules=[
            WarmupStableDecaySchedule(
                param_name="lr",
                max_value=lr,
                min_value=0.0,
                warmup_frac=warmup_frac,
                cooldown_frac=cooldown_frac,
                decay_type="cosine",
            ),
        ],
    )

    return optimizer_config, schedule_config


def default_muon_config(
    lr: float = 0.02,
    momentum: float = 0.95,
    warmup_frac: float = 0.1,
    stable_frac: float = 0.7,
) -> Tuple[OptimizerConfig, ScheduleConfig]:
    """Default Muon configuration with warmup-stable-decay schedules for LR and momentum.

    Muon is an optimizer designed for training neural networks with orthogonal
    updates. This preset provides sensible defaults based on modded-nanogpt runs.

    Args:
        lr: Peak learning rate. Default: 0.02.
        momentum: Peak momentum. Default: 0.95.
        warmup_frac: Fraction of training for warmup. Default: 0.1.
        stable_frac: Fraction of training at peak values. Default: 0.7.

    Returns:
        Tuple of (OptimizerConfig, ScheduleConfig).

    Raises:
        ImportError: If torch.optim.Muon is not available (requires PyTorch 2.5+).
    """
    if not hasattr(torch.optim, "Muon"):
        raise ImportError(
            "torch.optim.Muon not available. Requires PyTorch 2.5+ or "
            "install standalone package: pip install muon"
        )

    optimizer_config = OptimizerConfig(
        optimizer_class=torch.optim.Muon,
        optimizer_kwargs={
            "lr": lr,
            "momentum": momentum,
        },
    )

    cooldown_frac = max(0.0, 1.0 - warmup_frac - stable_frac)

    schedule_config = ScheduleConfig(
        global_schedules=[
            WarmupStableDecaySchedule(
                param_name="lr",
                max_value=lr,
                min_value=0.0,
                warmup_frac=warmup_frac,
                cooldown_frac=cooldown_frac,
                decay_type="cosine",
            ),
            # Momentum typically mimics LR phases but acts as "damping"
            # Here we follow simple high-momentum strategy
            WarmupStableDecaySchedule(
                param_name="momentum",
                max_value=momentum,
                min_value=0.85,
                warmup_frac=warmup_frac,
                cooldown_frac=cooldown_frac,
                decay_type="cosine",
            ),
        ],
    )

    return optimizer_config, schedule_config


def default_sgd_config(
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    warmup_frac: float = 0.05,
    stable_frac: float = 0.0,
) -> Tuple[OptimizerConfig, ScheduleConfig]:
    """Default SGD configuration with warmup and linear decay.

    Uses a short warmup then linear decay (no stable phase by default),
    which is common for SGD in vision tasks.

    Args:
        lr: Peak learning rate. Default: 0.1.
        momentum: Momentum value. Default: 0.9.
        weight_decay: Weight decay coefficient. Default: 1e-4.
        warmup_frac: Fraction of training for LR warmup. Default: 0.05.
        stable_frac: Fraction of training at peak LR. Default: 0.0.

    Returns:
        Tuple of (OptimizerConfig, ScheduleConfig).
    """
    optimizer_config = OptimizerConfig(
        optimizer_class=torch.optim.SGD,
        optimizer_kwargs={
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        },
    )

    cooldown_frac = max(0.0, 1.0 - warmup_frac - stable_frac)

    schedule_config = ScheduleConfig(
        global_schedules=[
            WarmupStableDecaySchedule(
                param_name="lr",
                max_value=lr,
                min_value=0.0,
                warmup_frac=warmup_frac,
                cooldown_frac=cooldown_frac,
                decay_type="linear",
            ),
        ],
    )

    return optimizer_config, schedule_config
