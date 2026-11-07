"""The one-run lifecycle both pilots share: callbacks, loggers, the post-fit
guard and final checkpoint, the W&B upload / re-upload, and W&B run closing.

This is the project layer of a run's body. It reads the project's config
shape (``output``, ``logging``, ``training``, ``task``, ``experiment``,
``optimizer``) through ``common.layout`` / ``common.run_id`` and builds the
project's training module (:class:`MGDN2SyntheticRecallModule`) and run
record. Pilot-specific choices (frozen-protocol checks, which tiers a run
skips) are arguments, never built in.

:func:`run_and_close` is the whole-job wrapper both ``01_train.py``
entries share: the W&B auth check up front, the job body, and
``wandb.finish(exit_code=...)`` no matter how the job ends.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from research_lib.layers.attention import MultistateGatedDelta
from research_lib.training.callbacks import (
    Collector,
    LayerStatsCallback,
    RunSummaryCallback,
    Stage,
    metrics_by_prefix,
    param_counts,
)
from research_lib.training.configs import ScheduleConfig
from research_lib.training.modules import MGDN2SyntheticRecallModule
from research_lib.training.presets import build_weight_decay_configs
from research_lib.training.run_control import RUN_DONE_STATUS, assert_fit_finished
from research_lib.training.run_records import read_record
from research_lib.training.scheduling import (
    EpochCosineSchedule,
    WarmupStableDecaySchedule,
)
from research_lib.utils.io import atomic_torch_save

from .layout import record_files, wandb_kwargs
from .run_id import HASH_EXCLUDE

logger = logging.getLogger(__name__)

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


def build_module(
    cfg: DictConfig,
    model: L.LightningModule,
    lr: float,
    slice_values: tuple[int, ...],
    *,
    monitor_key: str = "val/acc",
    class_names: tuple[str, ...] | None = None,
    extra_metrics: tuple = (),
) -> MGDN2SyntheticRecallModule:
    """The training module with the two AdamW weight-decay groups.

    Args:
        cfg: The resolved config (``optimizer.*``, ``training.*``).
        model: The built model.
        lr: The peak learning rate.
        slice_values: The task's slice values.
        monitor_key: The metric key that drives early stopping and the
            best value (``val/acc`` default).
        class_names: The task's class-name fragments for ``RecallMetrics``
            (``None`` uses the metric's default).
        extra_metrics: Additional evaluation metrics receiving the same
            answer-position predictions as ``RecallMetrics`` (the module's
            extra-metric hook).

    Returns:
        An :class:`MGDN2SyntheticRecallModule`, ready to fit.
    """
    matrix_optimizer_config, vector_optimizer_config = build_weight_decay_configs(
        cfg.optimizer.decay_scheme,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
    )
    schedule_kind = str(cfg.optimizer.get("schedule") or "step_cosine")
    if schedule_kind == "epoch_cosine":
        # CosineAnnealingLR(T_max=epochs) stepped once per epoch: the rate
        # is constant within an epoch and follows the cosine over the
        # epochs, from peak at epoch 0 to near-minimum at the last.
        import math

        steps_per_epoch = math.ceil(cfg.task.num_train / cfg.training.batch_size)
        step_schedule = EpochCosineSchedule(
            steps_per_epoch=steps_per_epoch,
            epochs=int(cfg.training.epochs),
            max_value=lr,
            min_value=float(cfg.optimizer.min_lr),
        )
    else:
        step_schedule = WarmupStableDecaySchedule(
            param_name="lr",
            warmup_frac=0.0,
            cooldown_frac=1.0,
            min_value=0.0,
            max_value=lr,
            decay_type="cosine",
        )
    schedule_config = ScheduleConfig(global_schedules=[step_schedule])
    return MGDN2SyntheticRecallModule(
        model=model,
        slice_values=slice_values,
        with_classes=None,  # auto-detect (True for overwrite batches)
        class_names=class_names,
        monitor_key=monitor_key,
        loss_on=cfg.training.loss_on,
        answer_head=cfg.training.answer_head,
        extra_metrics=extra_metrics,
        matrix_optimizer_config=matrix_optimizer_config,
        vector_optimizer_config=vector_optimizer_config,
        matrix_schedule_config=schedule_config,
        vector_schedule_config=schedule_config,
        vector_target_modules=VECTOR_TARGET_MODULES,
        grad_clip_val=cfg.training.grad_clip_val,
    )


def first_gated_delta(model) -> MultistateGatedDelta | None:
    """The first gated-delta mixer under ``model`` (``None`` if none)."""
    return next(
        (m for m in model.modules() if isinstance(m, MultistateGatedDelta)), None
    )


def state_bytes_collector(trainer, pl_module):
    """Recurrent-state bytes of one gated-delta layer, in fp32."""
    layer = first_gated_delta(pl_module.model)
    return {} if layer is None else {"bytes_per_mixer": layer.state_bytes()}


def per_slot_summary(key, value):
    """Keep the mixer metric names: one entry per lambda slot."""

    if key.startswith("lambda_"):
        per_slot = value.float().mean(dim=0)  # average over heads -> (L,)
        return {f"{key}/slot{slot}": float(v) for slot, v in enumerate(per_slot)}
    return {key: float(value.float().mean())}


def create_callbacks(
    cfg: DictConfig,
    run_id: str,
    precision_flags: dict[str, bool],
    *,
    lr: float,
    monitor_key: str = "val/acc",
) -> list:
    """Early stopping, the run summary and LR logging.

    The run-summary callback records the model shapes and the final
    metrics; ``val_final`` and ``mixer_stats_final`` collectors capture
    the last validation and mixer columns that ``common.records.flatten``
    reads. The ``monitor_key`` (default ``val/acc``) drives both the early
    stop and the best-value track.

    Args:
        cfg: The resolved config (``training``, ``output.runs_dir``).
        run_id: The run ID.
        precision_flags: The recorded precision flags.
        lr: The resolved peak learning rate.
        monitor_key: The metric key that drives early stopping and the
            best value.

    Returns:
        The trainer callbacks.
    """
    callbacks = [
        EarlyStopping(
            monitor=monitor_key,
            mode="max",
            stopping_threshold=cfg.training.early_stop_threshold,
            patience=10**9,
        ),
        RunSummaryCallback(
            Path(cfg.output.runs_dir),
            run_id,
            OmegaConf.to_container(cfg, resolve=True),
            # Distribution names of the packages the scripts import.
            packages=("torch", "lightning", "hydra-core", "omegaconf", "torchmetrics"),
            monitor=monitor_key,
            mode="max",
            # The run identity excludes operational keys (see HASH_EXCLUDE).
            hash_exclude=HASH_EXCLUDE,
            # The lockfile hash is part of the run's provenance: a tier
            # that mixed lock versions would be visible.
            provenance_files=("uv.lock",),
            collectors=(
                Collector(
                    "metrics",
                    lambda _trainer, _pl: {"acc": "per_sequence_mean"},
                    Stage.FIT_START,
                ),
                Collector(
                    "params",
                    param_counts({"lambda": ["lambda_mixer"]}),
                    Stage.FIT_START,
                ),
                Collector("state", state_bytes_collector, Stage.FIT_START),
                Collector("val_final", metrics_by_prefix("val"), Stage.FIT_END),
                Collector(
                    "mixer_stats_final", metrics_by_prefix("mixer"), Stage.FIT_END
                ),
                Collector("test", metrics_by_prefix("test"), Stage.TEST_END),
            ),
            extra={
                "lr_resolved": lr,
                # Also recorded via the config; kept top-level so analyses
                # need not dig for it (one JSON per run, rsync-able).
                "decay_scheme": str(cfg.optimizer.decay_scheme),
                "precision_flags": precision_flags,
                # Set by post_fit() after the checkpoint is saved (before
                # the test pass), so the test_done record names the file.
                "final_checkpoint": None,
            },
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if bool(cfg.training.collect_mixer_stats):
        callbacks.append(LayerStatsCallback(prefix="mixer", summarize=per_slot_summary))
    return callbacks


def create_callbacks_reset_curves(
    cfg: DictConfig,
    run_id: str,
    precision_flags: dict[str, bool],
    *,
    lr: float,
    monitor_key: str = "val/acc",
) -> list:
    """Build the callbacks (the run-identity check), then reset the curves.

    The run-summary callback's constructor refuses a path that already
    holds a **different** run's record, and that check must happen
    before the stale curves directory is deleted: a relaunch that
    edited a non-identity key (e.g. ``training.max_steps``) must be
    refused with the recorded run's curves still in place, not refused
    only after having destroyed them.

    Returns:
        The trainer callbacks; the loggers are created by the caller
        after this, so the CSV logger never appends to a stale
        ``version_0`` directory.
    """
    callbacks = create_callbacks(
        cfg, run_id, precision_flags, lr=lr, monitor_key=monitor_key
    )
    if bool(cfg.logging.csv.enabled):
        curves_dir = Path(cfg.logging.csv.save_dir) / run_id
        if curves_dir.exists():
            import shutil

            shutil.rmtree(curves_dir)
            logger.info("Removed stale curves directory %s", curves_dir)
    return callbacks


def run_summary_callback(callbacks: list) -> RunSummaryCallback | None:
    """The run-summary callback among ``callbacks`` (``None`` if absent)."""
    return next((cb for cb in callbacks if isinstance(cb, RunSummaryCallback)), None)


def post_fit(
    cfg: DictConfig,
    trainer: L.Trainer,
    module: L.LightningModule,
    callbacks: list,
    run_id: str,
) -> None:
    """The post-fit steps: the fit-finished guard and the optional final
    checkpoint.

    These run outside ``trainer.fit()`` / ``trainer.test()``, so
    Lightning's ``on_exception`` never fires for them; a failure here is
    recorded on the run summary (``status == "failed"``,
    ``failed_stage == "fit"``) and re-raised.

    Args:
        cfg: The resolved config (``output.save_final_checkpoint``,
            ``output.runs_dir``).
        trainer: The fitted trainer (its ``state.status`` must be
            ``FINISHED``).
        module: The training module; its ``.model`` state dict is saved
            as the final checkpoint.
        callbacks: The trainer's callbacks (the run-summary callback is
            updated on failure).
        run_id: The run ID (names the checkpoint file).

    Raises:
        RuntimeError: If the fit did not end normally.
        The exception raised by the checkpoint save, if any.
    """
    try:
        assert_fit_finished(trainer)
        # The final checkpoint for the mechanism probes: saved after fit
        # and before test, atomically (the test pass must never run on
        # weights that were not saved; a partial file must never be
        # observed).
        if cfg.output.save_final_checkpoint:
            path = Path(cfg.output.runs_dir) / f"{run_id}.pt"
            atomic_torch_save(module.model.state_dict(), path)
            logger.info("Saved final checkpoint to %s", path)
            summary_cb = run_summary_callback(callbacks)
            if summary_cb is not None and summary_cb.extra is not None:
                summary_cb.extra["final_checkpoint"] = path.name
    except BaseException as exc:
        summary_cb = run_summary_callback(callbacks)
        if summary_cb is not None:
            summary_cb.on_exception(trainer, module, exc)
        raise


def upload_init_kwargs(cfg: DictConfig, run_id: str) -> dict[str, Any]:
    """The ``wandb.init`` keyword arguments of a short re-upload run.

    The same project / entity / name / group / tags as the training
    logger (:func:`wandb_kwargs`), but ``job_type="upload"``: a relaunch
    that skips a finished run opens this short run to upload the record
    that finished run missed. The keyword arguments only ``WandbLogger``
    understands are mapped or dropped for ``wandb.init``: ``save_dir``
    becomes ``dir`` (so the re-upload run writes under ``logs/wandb``
    like training does), and ``log_model`` is removed.
    """
    kwargs = dict(wandb_kwargs(cfg, run_id))
    kwargs["dir"] = kwargs.pop("save_dir")
    kwargs.pop("log_model", None)
    kwargs["job_type"] = "upload"
    return kwargs


def upload_run_record_artifact(
    cfg: DictConfig, run_id: str, is_global_zero: bool
) -> None:
    """Upload a completed run to W&B as a ``run_record`` artifact.

    Runs only when W&B is enabled and ``logging.wandb.upload_record`` is
    set, and only on the global-zero process (a non-zero rank under
    ``devices="auto"`` has no live W&B run and would only log noise).
    The artifact's paths are the pull-side contract (``common.layout``);
    the metadata carries the run's identity and provenance, so the pull
    can decide skip / conflict / download without fetching the artifact.
    An upload failure is logged at ERROR level and does not fail the
    run: the local record is already done, and a --check pull reports
    what the canonical store still misses.

    Args:
        cfg: The resolved config (``logging.wandb.*``,
            ``output.runs_dir``, ``logging.csv.save_dir``).
        run_id: The run ID (also the run record's file stem).
        is_global_zero: Whether this process is rank zero (the only one
            with a live W&B run).
    """
    if not bool(cfg.logging.wandb.enabled) or not bool(cfg.logging.wandb.upload_record):
        return
    if not is_global_zero:
        return
    from research_lib.training.run_records import artifact_name
    from research_lib.training.wandb_transport import (
        upload_run_record_artifact as _transport_upload,
    )

    from .layout import record_metadata

    path = Path(cfg.output.runs_dir) / f"{run_id}.json"
    record = read_record(path)
    if record is None:
        logger.error("run record %s not found; skipping the W&B artifact upload", path)
        return
    if record.get("status") != RUN_DONE_STATUS:
        logger.error(
            "run record %s has status %r, not test_done; skipping the W&B "
            "artifact upload",
            path,
            record.get("status"),
        )
        return
    _transport_upload(
        artifact_name(run_id),
        record_metadata(record),
        record_files(cfg, run_id),
        is_global_zero=is_global_zero,
    )


def ensure_run_record_uploaded(cfg: DictConfig, run_id: str) -> None:
    """Make sure a finished run's record reaches W&B as an artifact.

    A relaunched sweep skips a run whose record is already done with the
    same identity, before a fresh job would have uploaded anything, so if
    that run's upload failed when it finished it would stay off W&B
    forever. This re-checks the ``run_record`` artifact: when it is
    absent or carries a different ``(config_hash, seed)``, a short W&B
    run (``job_type="upload"``) is opened and the record is (re)uploaded.
    Any problem is logged at ERROR level and never fails the job; a
    machine without W&B is untouched.

    Args:
        cfg: The resolved config.
        run_id: The run ID (also the run record's file stem).
    """
    if not bool(cfg.logging.wandb.enabled) or not bool(cfg.logging.wandb.upload_record):
        return
    try:
        import wandb

        from research_lib.training.run_records import artifact_name

        path = Path(cfg.output.runs_dir) / f"{run_id}.json"
        record = read_record(path)
        if record is None:
            return
        entity = cfg.logging.wandb.get("entity", None)
        entity = str(entity) if entity else None
        project = str(cfg.logging.wandb.project)
        location = f"{entity}/{project}" if entity else project
        target = f"{location}/{artifact_name(run_id)}:latest"
        api = wandb.Api()
        if api.artifact_exists(target):
            artifact = api.artifact(target)
            metadata = dict(artifact.metadata or {})
            already = (metadata.get("config_hash"), metadata.get("seed")) == (
                record.get("config_hash"),
                record.get("seed"),
            )
            if already:
                logger.info(
                    "run record %s already on W&B; nothing to (re)upload", run_id
                )
                return
        wandb.init(**upload_init_kwargs(cfg, run_id))
        upload_run_record_artifact(cfg, run_id, is_global_zero=True)
        wandb.finish()
    except Exception as exc:  # a failed check/upload never fails a skipped job
        logger.error("could not ensure run record %s is on W&B: %s", run_id, exc)


def create_loggers(cfg: DictConfig, run_id: str) -> list | None:
    """CSV (next to the run records) and optional W&B loggers.

    The CSV logger writes under ``<save_dir>/<run_id>/version_0``, so the
    curves of one run live next to its run JSON (``runs/<task>/curves/``
    in the base config; the smoke and debug configs keep their own
    ``save_dir``). The W&B logger gets one run per pilot run; the caller
    closes it at the end of the job (:func:`run_and_close`).
    """
    loggers = []
    if cfg.logging.csv.enabled:
        loggers.append(
            CSVLogger(save_dir=cfg.logging.csv.save_dir, name=run_id, version=0)
        )
    if cfg.logging.wandb.enabled:
        loggers.append(WandbLogger(**wandb_kwargs(cfg, run_id)))
    return loggers if loggers else None


def run_and_close(cfg: DictConfig, job: Callable[[DictConfig], None]) -> None:
    """Check the W&B credentials, run one job, and close the W&B run.

    W&B credentials are required before anything is built when W&B is
    enabled, so a misconfigured rented machine fails fast. The W&B run
    is closed after the job no matter how it ends: in a Hydra multirun
    all jobs share one process, and a job that does not close its run
    makes the next job reuse it (its log lines and curves land in the
    first job's run). The exit code records the outcome on the run W&B
    shows; a job that raised also propagates its exception.

    Args:
        cfg: The resolved config (``logging.wandb.enabled``).
        job: The one-job body; called with ``cfg``.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if bool(cfg.logging.wandb.enabled):
        from research_lib.utils.secrets import check_auth

        check_auth(["WANDB_API_KEY"])

    error: BaseException | None = None
    try:
        job(cfg)
    except BaseException as exc:
        error = exc
        raise
    finally:
        if bool(cfg.logging.wandb.enabled):
            import wandb

            if wandb.run is not None:
                wandb.finish(exit_code=1 if error is not None else 0)
