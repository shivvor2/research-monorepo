"""
One training run of the structured-recall pilot.

Usage:
    python 01_train.py --config-name=test task=mqar        # tiny CPU smoke
    python 01_train.py task=mqar model.mixer=attention     # attention mixer
    python 01_train.py +sweep=<tier> --multirun            # a sweep tier

One process = one run = one JSON under ``runs/<task>/<run_id>.json``.
The task is picked by ``task`` (a config group whose file names its
harness via ``defaults``), the mechanism by ``model.mixer``
(``gated_delta`` / ``attention``). The run machinery is the shared
project package (``common``) on top of the general ``research_lib`` run
control; this file registers the project's harness builders and runs one
job.
"""

import logging
import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.utils import get_original_cwd
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

_PILOT = Path(__file__).resolve().parent
_PROJECT = _PILOT.parent
for _path in (_PROJECT, _PILOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common.layout import record_files, wandb_kwargs
from common.launch import begin_run
from common.registries import config_type_of_task, register_harness
from common.run_id import HASH_EXCLUDE, make_run_id

from research_lib.architectures.config import ZoologyLMConfig
from research_lib.architectures.zoology_lm import ZoologyLM, zoology_mixers
from research_lib.data.synthetic import SyntheticRecallDataModule
from research_lib.layers.attention import CausalSelfAttention, MultistateGatedDelta
from research_lib.training.callbacks import (
    Collector,
    LayerStatsCallback,
    RunSummaryCallback,
    Stage,
    config_hash,
    metrics_by_prefix,
    param_counts,
)
from research_lib.training.configs import ScheduleConfig
from research_lib.training.modules import MGDN2SyntheticRecallModule
from research_lib.training.presets import build_weight_decay_configs
from research_lib.training.run_control import RUN_DONE_STATUS, claim_run
from research_lib.training.scheduling import WarmupStableDecaySchedule
from research_lib.utils.secrets import check_auth

logger = logging.getLogger(__name__)

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


def build_zoology_model(cfg: DictConfig) -> ZoologyLM:
    """Build the Zoology harness under the ``model.mixer`` choice."""
    c = cfg.model
    head_dim = c.head_dim
    head_dim = int(head_dim) if head_dim is not None else c.d_model // c.num_heads

    if str(c.mixer) == "gated_delta":
        def make_mixer() -> MultistateGatedDelta:
            return MultistateGatedDelta(
                hidden_size=c.d_model,
                num_heads=c.num_heads,
                head_dim=head_dim,
                expand_v=c.expand_v,
                gate_mode=c.gate_mode,
                layout=c.layout,
                lambda_mode=c.lambda_mode,
                window=c.window,
                chunk_size=c.chunk_size,
                t_max=c.t_max,
                mode=c.mixer_mode,
                output_gate_activation=c.output_gate_activation,
            )
    elif str(c.mixer) == "attention":
        def make_mixer() -> CausalSelfAttention:
            return CausalSelfAttention(c.d_model, c.num_heads, rotary_dim=head_dim)
    else:
        raise ValueError(f"unknown model.mixer {str(c.mixer)!r}; expected "
                         f"'gated_delta' or 'attention'")

    mixers = zoology_mixers(
        n_layer=c.n_layer,
        d_model=c.d_model,
        make_mixer=make_mixer,
        pattern=c.layer_pattern,
        conv_kernel=c.base_conv_kernel,
    )
    model_config = ZoologyLMConfig(
        vocab_size=c.vocab_size,
        d_model=c.d_model,
        state_mixer=c.state_mixer,
        ff_mult=c.ff_mult,
    )
    return ZoologyLM(model_config, mixers)


# The harness is registered once, at import time; the task PRs add their
# own harnesses (and task config types) here and in the general task
# registry. ``register_harness`` and ``config_type_of_task`` live in common.
register_harness("zoology", build_zoology_model)


def load_frozen_protocol() -> DictConfig:
    path = _PILOT / "conf" / "frozen_protocol.yaml"
    return OmegaConf.load(path)


def build_module(cfg: DictConfig, model, lr: float) -> MGDN2SyntheticRecallModule:
    matrix_config, vector_config = build_weight_decay_configs(
        cfg.optimizer.decay_scheme,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
    )
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
    return MGDN2SyntheticRecallModule(
        model=model,
        slice_values=tuple(int(v) for v in cfg.task[slice_key(cfg.task)]),
        with_classes=None,
        class_names=cfg.model.get("class_names", None),
        monitor_key=cfg.training.get("monitor_key", "val/acc"),
        loss_on=cfg.training.loss_on,
        answer_head=cfg.training.answer_head,
        matrix_optimizer_config=matrix_config,
        vector_optimizer_config=vector_config,
        matrix_schedule_config=schedule_config,
        vector_schedule_config=schedule_config,
        vector_target_modules=VECTOR_TARGET_MODULES,
        grad_clip_val=cfg.training.grad_clip_val,
    )


def slice_key(task_cfg) -> str:
    from research_lib.data.synthetic.registry import require_task
    spec = require_task(str(task_cfg.name))
    return spec.slice_kwarg or "num_kv_pairs"


def run_job(cfg: DictConfig) -> None:
    precision_flags = begin_run(cfg)
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg))

    frozen = load_frozen_protocol()
    if cfg.experiment.phase >= 2 and not bool(frozen.get("frozen", False)):
        raise RuntimeError("phase >= 2 requires a frozen protocol")

    lr = float(cfg.optimizer.lr) if cfg.optimizer.lr is not None else _frozen_lr(cfg, frozen)
    logger.info("Peak learning rate: %g", lr)

    run_id = make_run_id(cfg, lr, include_task_keys=True)
    lock_path = Path(cfg.output.runs_dir) / f"{run_id}.lock"
    record_path = Path(cfg.output.runs_dir) / f"{run_id}.json"
    identity = (config_hash(OmegaConf.to_container(cfg, resolve=True), HASH_EXCLUDE), cfg.seed)
    skip_completed = bool(cfg.launch.skip_completed) and str(cfg.experiment.get("tier", ""))
    with claim_run(lock_path, record_path if skip_completed else None, identity) as reason:
        if reason is not None:
            logger.info("skip %s: %s", run_id, reason)
            return

        callbacks = [
            EarlyStopping(
                monitor=cfg.training.get("monitor_key", "val/acc"),
                mode="max",
                stopping_threshold=cfg.training.early_stop_threshold,
                patience=10**9,
            ),
            RunSummaryCallback(
                Path(cfg.output.runs_dir),
                run_id,
                OmegaConf.to_container(cfg, resolve=True),
                packages=("torch", "lightning", "hydra-core", "omegaconf"),
                monitor=cfg.training.get("monitor_key", "val/acc"),
                mode="max",
                hash_exclude=HASH_EXCLUDE,
                provenance_files=("uv.lock",),
                collectors=(
                    Collector("metrics", lambda t, p: {"acc": "per_sequence_mean"}, Stage.FIT_START),
                    Collector("params", param_counts({"lambda": ["lambda_mixer"]}), Stage.FIT_START),
                    Collector("state", _state_bytes_collector, Stage.FIT_START),
                    Collector("test", metrics_by_prefix("test"), Stage.TEST_END),
                ),
                extra={
                    "lr_resolved": lr,
                    "decay_scheme": str(cfg.optimizer.decay_scheme),
                    "precision_flags": precision_flags,
                    "final_checkpoint": None,
                },
            ),
            LearningRateMonitor(logging_interval="step"),
        ]
        if bool(cfg.training.collect_mixer_stats):
            callbacks.append(LayerStatsCallback(prefix="mixer", summarize=_per_slot_summary))
        loggers = _create_loggers(cfg, run_id)

        L.seed_everything(cfg.seed)
        task = str(cfg.task.name)
        task_config = _build_task_config(cfg)
        datamodule = SyntheticRecallDataModule(
            task=task,
            task_config=task_config,
            batch_size=cfg.training.batch_size,
            num_val_per_slice=cfg.data.num_val_per_slice,
            num_test_per_slice=cfg.data.num_test_per_slice,
            train_seed=cfg.seed,
            val_seed=cfg.task.val_seed,
            test_seed=cfg.task.test_seed,
            num_workers=cfg.data.num_workers,
        )

        model = build_zoology_model(cfg)
        module = build_module(cfg, model, lr)
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
        trainer.fit(module, datamodule=datamodule)
        trainer.test(module, datamodule=datamodule)
        if bool(cfg.logging.wandb.enabled) and bool(cfg.logging.wandb.upload_record):
            _upload(cfg, run_id)
        logger.info("Run complete (run_id=%s).", run_id)


def _state_bytes_collector(trainer, pl_module):
    layer = next(
        (m for m in pl_module.model.modules() if isinstance(m, MultistateGatedDelta)),
        None,
    )
    return {} if layer is None else {"bytes_per_mixer": layer.state_bytes()}


def _per_slot_summary(key, value):
    if key.startswith("lambda_"):
        per_slot = value.float().mean(dim=0)
        return {f"{key}/slot{slot}": float(v) for slot, v in enumerate(per_slot)}
    return {key: float(value.float().mean())}


def _create_loggers(cfg: DictConfig, run_id: str):
    loggers = []
    if cfg.logging.csv.enabled:
        loggers.append(CSVLogger(save_dir=cfg.logging.csv.save_dir, name=run_id, version=0))
    if cfg.logging.wandb.enabled:
        loggers.append(WandbLogger(**wandb_kwargs(cfg, run_id)))
    return loggers or None


def _build_task_config(cfg: DictConfig):
    name = str(cfg.task.name)
    cls = config_type_of_task(name)
    return cls(**task_fields(cfg))


def task_fields(cfg: DictConfig) -> dict:
    """Whichever fields the task config declares, from ``cfg.task``."""
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(cfg.task, resolve=True)
    cls = config_type_of_task(str(cfg.task.name))
    import inspect

    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in raw.items() if k in params and k != "name"}


def _frozen_lr(cfg: DictConfig, frozen: DictConfig) -> float:
    value = frozen.lr_table.get(str(cfg.task.name), {}).get(cfg.model.layout, {}).get(cfg.model.gate_mode)
    if value is None:
        raise RuntimeError("optimizer.lr is null and no frozen-protocol entry exists; fill it or pass optimizer.lr")
    return float(value)


def _upload(cfg: DictConfig, run_id: str) -> None:
    from research_lib.training.run_records import read_record, run_record_metadata
    from research_lib.training.wandb_transport import upload_run_record_artifact
    from common.layout import metadata_extras

    path = Path(cfg.output.runs_dir) / f"{run_id}.json"
    if not path.exists():
        return
    record = read_record(path)
    if record is None or record.get("status") != RUN_DONE_STATUS:
        return
    upload_run_record_artifact(
        f"run-{run_id}",
        run_record_metadata(record, metadata_extras()),
        record_files(cfg, run_id),
        is_global_zero=True,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if bool(cfg.logging.wandb.enabled):
        check_auth(["WANDB_API_KEY"])
    try:
        run_job(cfg)
    finally:
        if bool(cfg.logging.wandb.enabled):
            import wandb
            if wandb.run is not None:
                wandb.finish()


if __name__ == "__main__":
    main()