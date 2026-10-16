"""
One training run of the multi-state GDN-2 synthetic pilot.

Usage:
    # One run with defaults (needs a frozen protocol or an explicit lr):
    python 01_train.py optimizer.lr=0.003

    # Override via CLI:
    python 01_train.py task=overwrite model.layout=ll seed=3

    # One sweep tier (grid from conf/sweep/):
    python 01_train.py +sweep=tier1_core --multirun

    # Tiny CPU smoke config / debug multirun over every layout:
    python 01_train.py --config-name=test
    python 01_train.py --config-name=debug --multirun

One process = one run = one JSON under ``runs/<task>/<run_id>.json``
(the protocol's unit of record; ``06_experiment_protocol.md`` §10).
``experiment.phase >= 2`` refuses to run until
``conf/frozen_protocol.yaml`` has ``frozen: true``, and ``optimizer.lr``
is resolved from the frozen LR table by (task, layout, gate_mode).

Precision: fp32 throughout; ``torch.set_float32_matmul_precision`` is
left at its default (TF32 would lower the reference layers' fp32 matmuls).
"""

import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

# Library imports
from research_lib.architectures.config import ZoologyLMConfig
from research_lib.architectures.zoology_lm import ZoologyLM
from research_lib.data.synthetic import (
    MQARConfig,
    OverwriteConfig,
    SyntheticRecallDataModule,
)
from research_lib.training.callbacks import RunSummaryCallback
from research_lib.training.configs import OptimizerConfig, ScheduleConfig
from research_lib.training.modules import SyntheticRecallModule
from research_lib.training.scheduling import WarmupStableDecaySchedule

logger = logging.getLogger(__name__)

# =============================================================================
# Parameter targeting (weight-decay groups, protocol §2 / D-optimizer note)
# =============================================================================

# Parameters routed to the VECTOR optimizer (AdamW, weight decay 0):
# embeddings, the untied LM head, every norm weight, every bias, the
# depthwise conv filters, the gated-delta decay parameters (A_log,
# dt_bias) and every LambdaMixer parameter (weight decay would pull λ
# away from its λ≡1 anchor). Everything else — the 2-D linear weights —
# goes to the MATRIX optimizer (AdamW, weight decay 0.1).
VECTOR_TARGET_MODULES: list[str] = [
    "embedding",
    "lm_head",
    r".*norm.*",
    "A_log",
    "dt_bias",
    "lambda_mixer",
    "conv",
    r".*\.bias$",
]

TASK_CONFIGS = {"mqar": MQARConfig, "overwrite": OverwriteConfig}

# =============================================================================
# Helpers
# =============================================================================


def load_frozen_protocol() -> DictConfig:
    """Load conf/frozen_protocol.yaml (from this experiment's conf dir)."""
    path = Path(__file__).resolve().parent / "conf" / "frozen_protocol.yaml"
    return OmegaConf.load(path)


def resolve_lr(cfg: DictConfig, frozen: DictConfig) -> float:
    """The peak LR: cfg.optimizer.lr, else the frozen-protocol entry.

    Fails loudly if neither exists.
    """
    if cfg.optimizer.lr is not None:
        return float(cfg.optimizer.lr)
    task = str(cfg.task.name)
    layout = str(cfg.model.layout)
    gate_mode = str(cfg.model.gate_mode)
    try:
        value = frozen.lr_table[task][layout][gate_mode]
    except (KeyError, AttributeError) as e:
        raise RuntimeError(
            f"optimizer.lr is null and no frozen-protocol LR entry exists for "
            f"(task={task!r}, layout={layout!r}, gate_mode={gate_mode!r}); run "
            f"the calibration (conf/sweep/calib_lr.yaml) and fill in "
            f"conf/frozen_protocol.yaml, or pass optimizer.lr=<value>"
        ) from e
    if value is None:
        raise RuntimeError(
            f"the frozen-protocol LR entry for (task={task!r}, layout={layout!r}, "
            f"gate_mode={gate_mode!r}) is empty; fill in conf/frozen_protocol.yaml "
            f"at the end of calibration, or pass optimizer.lr=<value>"
        )
    return float(value)


def build_task_config(cfg: DictConfig):
    """The generator config for the selected task."""
    name = str(cfg.task.name)
    if name == "mqar":
        return MQARConfig(
            vocab_size=cfg.task.vocab_size,
            seq_len=cfg.task.seq_len,
            num_kv_pairs=tuple(cfg.task.num_kv_pairs),
            power_a=cfg.task.power_a,
            random_non_queries=cfg.task.random_non_queries,
        )
    if name == "overwrite":
        return OverwriteConfig(
            vocab_size=cfg.task.vocab_size,
            seq_len=cfg.task.seq_len,
            num_keys=tuple(cfg.task.num_keys),
            class_probs=tuple(cfg.task.class_probs),
            block_size=cfg.task.block_size,
            distant_min_blocks=cfg.task.distant_min_blocks,
            max_placement_attempts=cfg.task.max_placement_attempts,
        )
    raise ValueError(f"unknown task {name!r}")


def slice_values(cfg: DictConfig) -> tuple[int, ...]:
    """The task's slice values (D for MQAR, K for overwrite)."""
    key = "num_kv_pairs" if str(cfg.task.name) == "mqar" else "num_keys"
    return tuple(int(value) for value in cfg.task[key])


def build_model(cfg: DictConfig) -> ZoologyLM:
    model_config = ZoologyLMConfig(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_layer=cfg.model.n_layer,
        layer_pattern=cfg.model.layer_pattern,
        state_mixer=cfg.model.state_mixer,
        num_heads=cfg.model.num_heads,
        head_dim=cfg.model.get("head_dim", None),
        expand_v=cfg.model.get("expand_v", 1.0),
        gate_mode=cfg.model.gate_mode,
        layout=cfg.model.layout,
        lambda_mode=cfg.model.lambda_mode,
        window=cfg.model.window,
        chunk_size=cfg.model.chunk_size,
        t_max=cfg.model.t_max,
        output_gate_activation=cfg.model.output_gate_activation,
        mixer_mode=cfg.model.mixer_mode,
    )
    return ZoologyLM(model_config)


def build_module(cfg: DictConfig, model: ZoologyLM, lr: float) -> SyntheticRecallModule:
    """The training module with the two AdamW weight-decay groups."""
    betas = tuple(cfg.optimizer.betas)
    matrix_optimizer_config = OptimizerConfig(
        torch.optim.AdamW,
        {"lr": 0.0, "weight_decay": cfg.optimizer.weight_decay, "betas": betas},
    )
    vector_optimizer_config = OptimizerConfig(
        torch.optim.AdamW,
        {"lr": 0.0, "weight_decay": 0.0, "betas": betas},
    )
    # Cosine from the peak to 0 over max_steps, no warmup (protocol §2):
    # the optimizer starts at lr=0 and the scheduler sets the peak for
    # step 0 (the 02_pretrain.py convention).
    schedule_config = ScheduleConfig(
        global_schedules=[
            WarmupStableDecaySchedule(
                param_name="lr",
                warmup_frac=0.0,
                cooldown_frac=1.0,
                min_value=0.0,
                max_value=lr,
                decay_type="cosine",
            )
        ]
    )
    return SyntheticRecallModule(
        model=model,
        slice_values=slice_values(cfg),
        with_classes=None,  # auto-detect (True for overwrite batches)
        loss_on=cfg.training.loss_on,
        collect_mixer_stats=True,
        matrix_optimizer_config=matrix_optimizer_config,
        vector_optimizer_config=vector_optimizer_config,
        matrix_schedule_config=schedule_config,
        vector_schedule_config=schedule_config,
        vector_target_modules=VECTOR_TARGET_MODULES,
        grad_clip_val=cfg.training.grad_clip_val,
    )


def make_run_id(cfg: DictConfig, lr: float) -> str:
    run_id = (
        f"{cfg.task.name}-{cfg.model.layout}-{cfg.model.gate_mode}"
        f"-{cfg.model.lambda_mode}-d{cfg.model.d_model}-lr{lr:g}-s{cfg.seed}"
    )
    tier = cfg.experiment.get("tier", None)
    if tier:
        run_id = f"{tier}_{run_id}"
    return run_id


def create_callbacks(cfg: DictConfig, run_id: str) -> list:
    """Early stopping (99% answer accuracy), the run summary, LR logging."""
    callbacks = [
        EarlyStopping(
            monitor="val/acc",
            mode="max",
            stopping_threshold=cfg.training.early_stop_threshold,
            patience=10**9,
        ),
        RunSummaryCallback(
            runs_dir=Path(cfg.output.runs_dir),
            run_id=run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            extra={"lr_resolved": lr_resolved.get("value")},
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    return callbacks


# A tiny mutable holder so create_callbacks can record the resolved LR.
lr_resolved: dict[str, float | None] = {"value": None}


def create_loggers(cfg: DictConfig) -> list | None:
    loggers = []
    if cfg.logging.csv.enabled:
        loggers.append(CSVLogger(save_dir=cfg.logging.csv.save_dir, name="metrics"))
    if cfg.logging.wandb.enabled:
        wandb_kwargs = {
            "project": cfg.logging.wandb.project,
            "name": cfg.experiment.name,
            "save_dir": "logs/wandb",
            "log_model": False,
        }
        if cfg.logging.wandb.entity is not None:
            wandb_kwargs["entity"] = cfg.logging.wandb.entity
        loggers.append(WandbLogger(**wandb_kwargs))
    return loggers if loggers else None


# =============================================================================
# Main entry point
# =============================================================================


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run one pilot training run (fit + test) and write the run JSON."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 80)
    logger.info("Resolved Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    # 1. Phase gate: >= 2 requires the frozen protocol (§8.4).
    frozen = load_frozen_protocol()
    if cfg.experiment.phase >= 2 and not bool(frozen.get("frozen", False)):
        raise RuntimeError(
            f"experiment.phase={cfg.experiment.phase} >= 2 but "
            f"conf/frozen_protocol.yaml has frozen={frozen.get('frozen')!r}; "
            f"finish calibration, fill in and commit the frozen protocol "
            f"before running experiment tiers"
        )

    # 2. Learning rate.
    lr = resolve_lr(cfg, frozen)
    lr_resolved["value"] = lr
    logger.info("Peak learning rate: %g", lr)

    # 3. The ll slot count depends on t_max: it must cover the task.
    if cfg.model.t_max < cfg.task.max_seq_len:
        raise RuntimeError(
            f"model.t_max ({cfg.model.t_max}) must be >= task.max_seq_len "
            f"({cfg.task.max_seq_len}); the ll slot count depends on it"
        )

    # 4. Seeds: the run seed fixes init AND the training stream.
    L.seed_everything(cfg.seed)

    # 5. Data.
    datamodule = SyntheticRecallDataModule(
        task=str(cfg.task.name),
        task_config=build_task_config(cfg),
        batch_size=cfg.training.batch_size,
        num_val_per_slice=cfg.data.num_val_per_slice,
        num_test_per_slice=cfg.data.num_test_per_slice,
        train_seed=cfg.seed,
        val_seed=cfg.task.val_seed,
        test_seed=cfg.task.test_seed,
        num_workers=cfg.data.num_workers,
    )

    # 6. Model + training module.
    model = build_model(cfg)
    if cfg.training.compile:
        model = torch.compile(model)
    module = build_module(cfg, model, lr)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %s parameters; recurrent state per mixer: %d bytes",
        f"{num_params:,}",
        model.state_bytes_per_mixer(),
    )

    # 7. Callbacks and loggers.
    run_id = make_run_id(cfg, lr)
    callbacks = create_callbacks(cfg, run_id)
    loggers = create_loggers(cfg)

    # 8. Trainer.
    trainer = L.Trainer(
        max_steps=cfg.training.max_steps,
        val_check_interval=cfg.training.val_check_interval,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=loggers,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    logger.info("Starting training (run_id=%s)...", run_id)
    trainer.fit(module, datamodule=datamodule)

    # 9. Test on the final weights; the callback updates the run JSON.
    trainer.test(module, datamodule=datamodule)

    # 10. Optional final checkpoint for the mechanism probes.
    if cfg.output.save_final_checkpoint:
        path = Path(cfg.output.runs_dir) / f"{run_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        logger.info("Saved final checkpoint to %s", path)

    logger.info("Run complete (run_id=%s).", run_id)


if __name__ == "__main__":
    main()
