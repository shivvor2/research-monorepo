"""
Main training script for NanoMythos on FineWeb-Edu.

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

import logging
from pathlib import Path
from typing import List, Optional

import hydra
import lightning as L
import torch
import torch._dynamo
from hydra.utils import get_original_cwd
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

# Library imports
from research_lib.architectures.config import NanoMythosConfig
from research_lib.architectures.nano_mythos import NanoMythosAttnRes
from research_lib.data import FineWebDataModule
from research_lib.optimizers import CautiousAdamW, NorMuon
from research_lib.training.callbacks import (
    ExperimentStateCallback,
    resolve_checkpoint_path,
)
from research_lib.training.configs import (
    GradAccumSchedule,
    OptimizerConfig,
    ScheduleConfig,
)
from research_lib.training.modules import DualOptimizerModule
from research_lib.training.scheduling import WarmupStableDecaySchedule
from research_lib.utils.secrets import check_auth

logger = logging.getLogger(__name__)

# =============================================================================
# Torch Flags
# =============================================================================

torch.set_float32_matmul_precision("high")

# Higher recompile limit since _make_polar_express uses fullgraph = True
# Have to trace the shapes for all modules we are using
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.recompile_limit = 64

# =============================================================================
# Parameter Targeting
# =============================================================================

# Parameters that should use AdamW (vector optimizer).
# Everything else (attention projections, MLP weights) uses NorMuon.
#
# Rationale:
#   - Embeddings, norm scales, and biases are 1D / element-wise → AdamW
#   - KDA state params (A_log, dt_bias) are small vectors → AdamW
#   - KDA short-conv weights are 3D depthwise filters → AdamW
#   - LTI injection (log_A, log_dt, B) are 1D / scalar → AdamW
#   - ACT halting (small classification head) → AdamW
#   - AttnRes pseudo-queries / readout query → AdamW (per-depth query vectors)
#   - All bias terms (any param ending in .bias or _bias) → AdamW
#   - DepthFiLM embedding tables → AdamW
#   - All 2D weight matrices (Linear projections) → NorMuon

VECTOR_TARGET_MODULES: List[str] = [
    # -- Embeddings and heads (conceptually non-matrix) --
    "embedding",  # Token embeddings (lookup table)
    "depth_emb",  # DepthFiLM embedding tables
    "output",  # LM head (conceptually tied to embeddings)
    # -- AttnRes learnable queries --
    "pseudo_queries",  # Per-depth query vectors [n_layers, n_embd]
    "readout_query",  # Final readout query [1, n_embd]
    # -- Normalization layers (all are 1D scale parameters) --
    "norm",  # Plain norm modules (prelude/coda/loop block norms)
    "norm_f",  # Final output RMSNorm
    "attn_norm",  # RecurrentTransformerBlock attention norm
    "ffn_norm",  # RecurrentTransformerBlock FFN norm
    "q_norm",  # QK-norm in RotaryMultiheadAttention
    "k_norm",  # QK-norm in RotaryMultiheadAttention
    "o_norm",  # KDA output RMSNorm
    # -- KDA-specific vector parameters (1D / embedding-like) --
    "A_log",  # Delta attention state decay log-parameter [num_heads]
    "dt_bias",  # Delta time bias [hidden_size]
    # -- Recurrent block vector parameters --
    "injection",  # LTIInjection: log_A, log_dt, B (all 1D / scalar)
    "halt",  # ACT halting linear layer (conceptually a small head)
    # -- KDA short-conv weights (3D, depthwise filters - not true matrices) --
    ".*conv1d.*",  # q_conv1d, k_conv1d, v_conv1d weights
    # -- All bias terms (1D, never matrix) --
    ".*\\.bias$",  # Anything ending in .bias (e.g. halt.bias, out_proj.bias)
    ".*_bias$",  # Anything ending in _bias (e.g. in_proj_bias, q_proj_bias)
]

# =============================================================================
# Factory Functions
# =============================================================================


def create_model(cfg: DictConfig) -> NanoMythosAttnRes:
    """Create NanoMythos model from Hydra config."""
    model_config = NanoMythosConfig(
        vocab_size=cfg.model.vocab_size,
        block_size=cfg.model.block_size,
        n_embd=cfg.model.n_embd,
        n_head=cfg.model.n_head,
        ff_dim=cfg.model.ff_dim,
        bias=cfg.model.bias,
        dropout=cfg.model.dropout,
        padding_idx=cfg.model.padding_idx,
        n_prelude_blocks=cfg.model.n_prelude_blocks,
        n_loop_blocks=cfg.model.n_loop_blocks,
        n_coda_blocks=cfg.model.n_coda_blocks,
        linear_to_full_ratio=cfg.model.linear_to_full_ratio,
        max_loop_iters=cfg.model.max_loop_iters,
        act_threshold=cfg.model.act_threshold,
        use_lti=cfg.model.use_lti,
        loop_dim_fraction=cfg.model.loop_dim_fraction,
        loop_theta=cfg.model.loop_theta,
        film_hidden=cfg.model.film_hidden,
        attnres_block_size=cfg.model.get("attnres_block_size", None),
        attnres_rmsnorm_eps=cfg.model.attnres_rmsnorm_eps,
        kda_head_dim=cfg.model.kda_head_dim,
        kda_expand_v=cfg.model.kda_expand_v,
        kda_use_short_conv=cfg.model.kda_use_short_conv,
        kda_conv_size=cfg.model.kda_conv_size,
        kda_mode=cfg.model.kda_mode,
    )
    return NanoMythosAttnRes(model_config)


def get_optimizer_class(class_path: str):
    """Get optimizer class from string path."""
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
    model: NanoMythosAttnRes,
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
            "momentum": matrix_cfg.schedule.momentum.min_value,
            "beta2": matrix_cfg.beta2,
        },
    )

    # Build schedules from config
    matrix_schedules = [
        create_schedule(matrix_cfg.schedule.lr, "lr"),
    ]
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

    # --- Gradient Accumulation ---
    grad_accum: Optional[int] = None
    grad_accum_schedule: Optional[GradAccumSchedule] = None

    if (
        hasattr(cfg.training, "grad_accum_schedule")
        and cfg.training.grad_accum_schedule is not None
    ):
        # Convert OmegaConf DictConfig to dict with int keys
        sched_dict = {
            int(k): int(v) for k, v in cfg.training.grad_accum_schedule.items()
        }
        grad_accum_schedule = GradAccumSchedule(schedule=sched_dict)
        logger.info(f"Using grad-accum schedule: {sched_dict}")
    elif hasattr(cfg.training, "grad_accum") and cfg.training.grad_accum is not None:
        grad_accum = cfg.training.grad_accum
        logger.info(f"Using constant grad-accum: {grad_accum}")
    else:
        grad_accum = 1
        logger.info("Using default grad-accum: 1")

    # --- Create Module ---
    module = DualOptimizerModule(
        model=model,
        matrix_optimizer_config=matrix_optimizer_config,
        vector_optimizer_config=vector_optimizer_config,
        matrix_schedule_config=matrix_schedule_config,
        vector_schedule_config=vector_schedule_config,
        vector_target_modules=VECTOR_TARGET_MODULES,
        grad_accum=grad_accum,
        grad_accum_schedule=grad_accum_schedule,
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
        logger.info(f"CSV logging enabled: {cfg.logging.csv.save_dir}")

    # WandB logger
    if cfg.logging.wandb.enabled:
        wandb_kwargs = {
            "project": cfg.logging.wandb.project,
            "name": cfg.logging.wandb.name,
            "save_dir": "logs/wandb",
            "log_model": False,
        }
        # Optional: override wandb entity (defaults to API key's default entity)
        if (
            hasattr(cfg.logging.wandb, "entity")
            and cfg.logging.wandb.entity is not None
        ):
            wandb_kwargs["entity"] = cfg.logging.wandb.entity

        wandb_logger = WandbLogger(**wandb_kwargs)
        loggers.append(wandb_logger)
        logger.info(f"WandB logging enabled: project={cfg.logging.wandb.project}")

    return loggers if loggers else None


# =============================================================================
# Main Entry Point
# =============================================================================


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 80)
    check_auth()
    logger.info("Checking auth")
    logger.info("=" * 80)

    # Print resolved config
    logger.info("=" * 80)
    logger.info("Resolved Configuration:")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    # Seed for reproducibility
    if cfg.seed is not None:
        L.seed_everything(cfg.seed)
    else:
        logger.info("No seed set - training will be non-deterministic.")

    # 1. Create (and compile) Model
    logger.info("\n[1/4] Creating model...")
    model = create_model(cfg)
    model = torch.compile(model, mode=cfg.training.compile_mode)

    # 2. Count and report parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model has {num_params:,} parameters ({num_params / 1e6:.1f}M)")

    # 3. Resolve Checkpoint
    logger.info("\n[Checkpoint] Resolving auto-resume status...")
    ckpt_path = resolve_checkpoint_path(
        resume_from=cfg.checkpoint.resume_from,
        experiment_name=cfg.experiment.name,
        base_dir=Path(get_original_cwd()),
        checkpoint_subdir=cfg.checkpoint.dir,
    )

    logger.info("\n[2/4] Creating training module...")
    module = create_training_module(model, cfg)

    logger.info("\n[3/4] Creating data module...")
    datamodule = FineWebDataModule(
        num_train_shards=cfg.training.get("num_train_shards", 99),
        seq_len=cfg.data.seq_len,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # Create callbacks and loggers
    logger.info("\n[4/4] Setting up trainer...")
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
    # Resolve current grad-accum for token counting
    if (
        hasattr(cfg.training, "grad_accum_schedule")
        and cfg.training.grad_accum_schedule is not None
    ):
        sched_dict = {
            int(k): int(v) for k, v in cfg.training.grad_accum_schedule.items()
        }
        current_accum = max(sched_dict.values())  # conservative estimate
        accum_label = f"variable (schedule: {sched_dict})"
    else:
        current_accum = cfg.training.get("grad_accum", 1)
        accum_label = str(current_accum)

    tokens_per_step = cfg.data.batch_size * cfg.data.seq_len * current_accum
    total_tokens = cfg.training.max_steps * tokens_per_step
    logger.info("\nTraining info:")
    logger.info(f"  Physical batch size: {cfg.data.batch_size}")
    logger.info(f"  Gradient accumulation: {accum_label}")
    logger.info(
        f"  Effective batch size: {cfg.data.batch_size * current_accum} sequences (at max accum)"
    )
    logger.info(f"  Tokens per step: {tokens_per_step:,} (at max accum)")
    logger.info(f"  Total steps: {cfg.training.max_steps:,}")
    logger.info(f"  Total tokens: {total_tokens:,} ({total_tokens / 1e9:.2f}B)")

    # Train!
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80 + "\n")

    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
