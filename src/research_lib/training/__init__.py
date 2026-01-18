"""
Training utilities for neural network training with multiple optimizers.

This module provides a flexible training infrastructure supporting:
    - Multiple optimizers with different parameter groups
    - Independent LR and param_group scheduling
    - Gradient accumulation and clipping
    - PyTorch Lightning integration

Typical usage with Muon + AdamW::

    from research_lib.training import (
        DualOptimizerModule,
        TrainingConfig,
        OptimizerConfig,
        SchedulerConfig,
        default_muon_config,
        default_adam_config,
    )

    # Create configs
    training_config = TrainingConfig(total_steps=10000, grad_accum_steps=8)
    muon_config = default_muon_config(lr=0.02)
    adam_config = default_adam_config(lr=0.001)

    # Create module
    module = DualOptimizerModule(
        model=my_model,
        training_config=training_config,
        matrix_optimizer_config=muon_config,
        vector_optimizer_config=adam_config,
        target_modules=["attn", "mlp"],
    )

    # Configure logging and checkpointing at Trainer level
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint

    trainer = L.Trainer(
        max_steps=10000,
        logger=WandbLogger(project="my-project"),
        callbacks=[ModelCheckpoint(dirpath="checkpoints/", monitor="val/loss")],
    )
    trainer.fit(module, train_dataloader)

For custom param_group scheduling::

    from research_lib.training import CustomScheduleConfig, SchedulerConfig

    # Schedule any optimizer param_group value
    beta1_schedule = CustomScheduleConfig(
        param_name="betas",
        param_index=0,
        warmup_steps=100,
        cooldown_steps=50,
        min_value=0.85,
        max_value=0.95,
    )

    scheduler_config = SchedulerConfig(
        warmup_steps=100,
        custom_schedules=[beta1_schedule],
    )

For multi-group parameter partitioning (3+ optimizer groups)::

    from research_lib.training import partition_parameters_multi

    pattern_groups = [
        ["attn"],    # Group 0: Attention weights
        ["mlp"],     # Group 1: MLP weights
        [],          # Group 2: Everything else (catch-all)
    ]
    attn_params, mlp_params, other_params = partition_parameters_multi(
        model, pattern_groups
    )

Note on Logging and Checkpointing:
    Lightning modules in this package use the standard `self.log()` interface
    and do not hardcode any logging or checkpointing backends. Configure your
    preferred systems (WandB, TensorBoard, HuggingFace, etc.) at the Trainer level.
"""

from .configs import (
    CustomScheduleConfig,
    MomentumScheduleConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    default_adam_config,
    default_muon_config,
)
from .modules import DualOptimizerModule
from .param_utils import (
    ParameterCounts,
    count_parameters,
    count_parameters_by_status,
    partition_parameters,
    partition_parameters_multi,
    partition_parameters_multi_with_names,
    partition_parameters_with_names,
    select_parameters,
    select_parameters_with_names,
    summarize_partition,
)
from .scheduling import (
    get_current_lr,
    get_param_group_value,
    update_param_group_schedules,
    update_single_param_schedule,
)

__all__ = [
    # Configs
    "TrainingConfig",
    "SchedulerConfig",
    "OptimizerConfig",
    "MomentumScheduleConfig",
    "CustomScheduleConfig",
    "default_muon_config",
    "default_adam_config",
    # Lightning Modules
    "DualOptimizerModule",
    # Parameter utilities - Primitives
    "select_parameters",
    "select_parameters_with_names",
    # Parameter utilities - Convenience functions
    "partition_parameters",
    "partition_parameters_with_names",
    "partition_parameters_multi",
    "partition_parameters_multi_with_names",
    # Parameter utilities - Counting and summary
    "ParameterCounts",
    "count_parameters",
    "count_parameters_by_status",
    "summarize_partition",
    # Scheduling utilities
    "update_param_group_schedules",
    "update_single_param_schedule",
    "get_current_lr",
    "get_param_group_value",
]
