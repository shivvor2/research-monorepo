"""
Main training script for NanoGPT baseline on FineWeb-Edu.

Usage:
    # Run with default config
    python 02_pretrain.py

    # Override via CLI
    python 02_pretrain.py training.max_steps=1000 data.batch_size=2

    # Use test config for quick sanity check
    python 02_pretrain.py --config-name=test

    # Resume from checkpoint
    python 02_pretrain.py ckpt_path=checkpoints/last.ckpt
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import lightning as L
import torch
from hydra.utils import get_original_cwd, to_absolute_path
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

# Library imports
from research_lib.architectures.config import NanoGPTConfig
from research_lib.architectures.modded_nanogpt_base import ModdedNanoGPT
from research_lib.data import FineWebDataModule
from research_lib.optimizers import CautiousAdamW, NorMuon
from research_lib.training.callbacks import (
    ExperimentStateCallback,
    resolve_checkpoint_path,
)
from research_lib.training.configs import OptimizerConfig, ScheduleConfig
from research_lib.training.modules import DualOptimizerModule
from research_lib.training.scheduling import WarmupStableDecaySchedule
from research_lib.utils.secrets import check_auth

# =============================================================================
# Torch Flags
# =============================================================================

torch.set_float32_matmul_precision("high")

# =============================================================================
# Parameter Targeting
# =============================================================================

# Parameters that should use AdamW (vector optimizer)
# Everything else (attention projections, MLP weights) uses NorMuon
VECTOR_TARGET_MODULES: List[str] = [
    "embedding",  # Token embeddings (lookup table, not compute)
    "output",  # LM head (conceptually tied to embeddings)
    "norm_1",  # Post-attention RMSNorm
    "norm_2",  # Post-FFN RMSNorm
    "norm_f",  # Final RMSNorm
    "q_norm",  # QK-Norm query normalization (1D)
    "k_norm",  # QK-Norm key normalization (1D)
]

# =============================================================================
# Factory Functions
# =============================================================================


def create_model(cfg: DictConfig) -> ModdedNanoGPT:
    """Create model from Hydra config."""
    model_config = NanoGPTConfig(
        vocab_size=cfg.model.vocab_size,
        block_size=cfg.model.block_size,
        n_layer=cfg.model.n_layer,
        n_embd=cfg.model.n_embd,
        n_head=cfg.model.n_head,
        ff_dim=cfg.model.ff_dim,
        bias=cfg.model.bias,
        dropout=cfg.model.dropout,
        padding_idx=cfg.model.padding_idx,
    )
    return ModdedNanoGPT(model_config)


def get_optimizer_class(class_path: str):
    """Get optimizer class from string path."""
    # Simple mapping for our optimizers
    mapping = {
        "research_lib.optimizers.NorMuon": NorMuon,
        "research_lib.optimizers.CautiousAdamW": CautiousAdamW,
        "torch.optim.AdamW": torch.optim.AdamW,
    }
    if class_path not in mapping:
        raise ValueError(f"Unknown optimizer: {class_path}")
    return mapping[class_path]


def create_schedule(
    schedule_cfg: DictConfig, param_name: str
) -> WarmupStableDecaySchedule:
    return WarmupStableDecaySchedule(
        param_name=param_name,
        min_value=schedule_cfg.min_value,
        max_value=schedule_cfg.max_value,
        warmup_frac=schedule_cfg.warmup_frac,
        cooldown_frac=schedule_cfg.cooldown_frac,
        decay_type=schedule_cfg.decay_type,
    )


def create_training_module(
    model: ModdedNanoGPT,
    cfg: DictConfig,
) -> DualOptimizerModule:
    """Create Lightning training module with dual optimizers."""

    matrix_cfg = cfg.optimizer.matrix
    vector_cfg = cfg.optimizer.vector

    # --- Matrix Optimizer Config (NorMuon) ---
    matrix_optimizer_config = OptimizerConfig(
        optimizer_class=get_optimizer_class(matrix_cfg.class_path),
        optimizer_kwargs={
            "lr": 0.0,  # Start at 0, scheduler warms up
            "weight_decay": matrix_cfg.weight_decay,
            "momentum": matrix_cfg.schedule.momentum.min_value,  # Start at min
            "beta2": matrix_cfg.beta2,
        },
    )

    # Build schedules from config
    matrix_schedules = [
        create_schedule(matrix_cfg.schedule.lr, "lr"),
    ]
    # Add momentum schedule if present
    if "momentum" in matrix_cfg.schedule:
        matrix_schedules.append(
            create_schedule(matrix_cfg.schedule.momentum, "momentum")
        )

    matrix_schedule_config = ScheduleConfig(global_schedules=matrix_schedules)

    # --- Vector Optimizer Config (CautiousAdamW) ---
    vector_optimizer_config = OptimizerConfig(
        optimizer_class=get_optimizer_class(vector_cfg.class_path),
        optimizer_kwargs={
            "lr": 0.0,  # Start at 0, scheduler warms up
            "betas": tuple(vector_cfg.betas),
            "weight_decay": vector_cfg.weight_decay,
        },
    )

    vector_schedules = [
        create_schedule(vector_cfg.schedule.lr, "lr"),
    ]

    vector_schedule_config = ScheduleConfig(global_schedules=vector_schedules)

    # --- Create Module ---
    module = DualOptimizerModule(
        model=model,
        matrix_optimizer_config=matrix_optimizer_config,
        vector_optimizer_config=vector_optimizer_config,
        matrix_schedule_config=matrix_schedule_config,
        vector_schedule_config=vector_schedule_config,
        vector_target_modules=VECTOR_TARGET_MODULES,
        grad_accum=cfg.training.grad_accum,
        grad_clip_val=cfg.training.grad_clip_val,
    )

    return module


def create_callbacks(cfg: DictConfig) -> List:
    """Create Lightning callbacks."""
    callbacks = [
        # Track experiment state (only updates on valid save)
        ExperimentStateCallback(
            experiment_name=cfg.experiment.name, base_dir=Path(get_original_cwd())
        ),
        # Standard Checkpointing
        ModelCheckpoint(
            dirpath=cfg.checkpoint.dir,
            filename="step_{train/optimizer_step}",
            save_top_k=cfg.training.save_top_k,
            auto_insert_metric_name=False,
            monitor="val/loss",
            mode="min",
            every_n_train_steps=cfg.training.checkpoint_every_n_steps,
            save_last=True,
            save_on_train_epoch_end=True,
        ),
        # LR logging
        LearningRateMonitor(logging_interval="step"),
        # Progress bar
        RichProgressBar(),
    ]
    return callbacks


def create_loggers(cfg: DictConfig) -> Optional[List]:
    """Create list of loggers based on config."""
    loggers = []

    # CSV logger (for local EDA)
    if cfg.logging.csv.enabled:
        csv_logger = CSVLogger(
            save_dir=cfg.logging.csv.save_dir,
            name="metrics",
        )
        loggers.append(csv_logger)
        print(f"CSV logging enabled: {cfg.logging.csv.save_dir}")

    # WandB logger
    if cfg.logging.wandb.enabled:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb.project,
            name=cfg.logging.wandb.name,
            save_dir="logs/wandb",
            log_model=False,
        )
        loggers.append(wandb_logger)
        print(f"WandB logging enabled: project={cfg.logging.wandb.project}")

    return loggers if loggers else None


# =============================================================================
# Main Entry Point
# =============================================================================


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""

    print("=" * 80)
    check_auth()
    print("Checking auth")
    print("=" * 80)

    # Print resolved config
    print("=" * 80)
    print("Resolved Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Seed for reproducibility
    if cfg.seed is not None:
        L.seed_everything(cfg.seed)
    else:
        print("No seed set - training will be non-deterministic.")

    # 1. Create (and compile) Model
    print("\n[1/4] Creating model...")
    model = create_model(cfg)
    # No-one runs torch code on native windows, RIGHT?
    model = torch.compile(model, mode=cfg.training.compile_mode)

    # 2. Count and report parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model has {num_params:,} parameters ({num_params/1e6:.1f}M)")

    # 2. Resolve Checkpoint
    print("\n[Checkpoint] Resolving auto-resume status...")
    ckpt_path = resolve_checkpoint_path(
        resume_from=cfg.checkpoint.resume_from,
        experiment_name=cfg.experiment.name,
        base_dir=Path(get_original_cwd()),
        checkpoint_subdir=cfg.checkpoint.dir,
    )

    print("\n[2/4] Creating training module...")
    module = create_training_module(model, cfg)

    print("\n[3/4] Creating data module...")
    datamodule = FineWebDataModule(
        num_train_shards=cfg.training.get("num_train_shards", 99),
        seq_len=cfg.data.seq_len,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # Create callbacks and loggers
    print("\n[4/4] Setting up trainer...")
    callbacks = create_callbacks(cfg)
    loggers = create_loggers(cfg)

    # Create trainer
    trainer = L.Trainer(
        max_steps=cfg.training.max_steps,
        limit_val_batches=cfg.training.get("limit_val_batches", 1.0),
        val_check_interval=cfg.training.val_check_interval,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=loggers,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        enable_progress_bar=True,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    # Calculate and print training info
    tokens_per_step = cfg.data.batch_size * cfg.data.seq_len * cfg.training.grad_accum
    total_tokens = cfg.training.max_steps * tokens_per_step
    print(f"\nTraining info:")
    print(f"  Physical batch size: {cfg.data.batch_size}")
    print(f"  Gradient accumulation: {cfg.training.grad_accum}")
    print(
        f"  Effective batch size: {cfg.data.batch_size * cfg.training.grad_accum} sequences"
    )
    print(f"  Tokens per step: {tokens_per_step:,}")
    print(f"  Total steps: {cfg.training.max_steps:,}")
    print(f"  Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")

    # Train!
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
