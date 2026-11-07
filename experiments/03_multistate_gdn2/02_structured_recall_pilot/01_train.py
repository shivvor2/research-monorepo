"""
One training run of the structured-recall pilot.

Usage:
    python 01_train.py --config-name=test task=mqar        # tiny CPU smoke
    python 01_train.py task=mqar optimizer.lr=0.003         # one run
    python 01_train.py --config-name=debug --multirun \
        task=mqar model.layout=single,ws,ll                # the layouts, default mixer
    python 01_train.py --config-name=debug task=mqar \
        model.mixer=attention                              # one attention control run

One process = one run = one JSON under ``runs/<task>/<run_id>.json``.
The task is a config group whose file names its harness via ``defaults``;
the mechanism is ``model.mixer``. The run's body is the shared project
machinery (:mod:`common.run_lifecycle`, :mod:`common.launch`,
:mod:`common.harnesses`); this file keeps the pilot's own protocol (the
phase gate and the frozen-choice check).
"""

import logging
import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

_PILOT = Path(__file__).resolve().parent
_PROJECT = _PILOT.parent
for _path in (_PROJECT, _PILOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common.launch import begin_run, claimed_run
from common.registries import config_type_of_task, require_harness
from common.run_id import HASH_EXCLUDE, make_run_id
from common.run_lifecycle import (
    build_module,
    create_callbacks_reset_curves,
    create_loggers,
    post_fit,
    run_and_close,
    upload_run_record_artifact,
)
from structured_recall_analysis.protocol import PROTOCOL_TIERS

from research_lib.data.synthetic import (
    FixedSetRecallDataModule,
    MacroTokenAccuracy,
    SyntheticRecallDataModule,
    TokenAccuracy,
    fuzzy_recall_dataset,
    mad_batch_row,
    require_task,
)
from research_lib.training.callbacks import config_hash

logger = logging.getLogger(__name__)


def load_frozen_protocol() -> DictConfig:
    """Load conf/frozen_protocol.yaml (from this experiment's conf dir)."""
    return OmegaConf.load(_PILOT / "conf" / "frozen_protocol.yaml")


def check_frozen_choices(cfg: DictConfig, frozen: DictConfig) -> None:
    """Assert the run uses the frozen width and harness choices.

    Applies to protocol tiers only (the smoke/debug tiers are tooling).
    The run's config stays the record; the frozen file is checked against
    it, never used as a second source that silently overrides.

    Raises:
        RuntimeError: Naming the key, the run's value, the frozen value
            and the fix.
    """
    tier = str(cfg.experiment.get("tier") or "")
    if tier not in PROTOCOL_TIERS:
        return
    frozen_d_model = frozen.get("d_model", None)
    if frozen_d_model is not None and int(cfg.model.d_model) != int(frozen_d_model):
        raise RuntimeError(
            f"the frozen protocol's d_model is {frozen_d_model}, but this run "
            f"uses model.d_model={cfg.model.d_model} (tier={tier!r})"
        )
    harness = frozen.get("harness", None)
    if harness is not None:
        for key in ("layer_pattern", "state_mixer"):
            frozen_value = harness.get(key, None)
            if frozen_value is None:
                continue
            if str(cfg.model[key]) != str(frozen_value):
                raise RuntimeError(
                    f"the frozen protocol's harness uses {key}={frozen_value!r}, "
                    f"but this run uses model.{key}={cfg.model[key]!r} "
                    f"(tier={tier!r})"
                )


def build_task_config(cfg: DictConfig):
    """The generator config for the selected task (from the registry)."""
    cls = config_type_of_task(str(cfg.task.name))
    field_names = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    kwargs = {}
    for key, value in OmegaConf.to_container(cfg.task, resolve=True).items():
        if key in field_names and key != "name":
            kwargs[key] = value
    return cls(**kwargs)


def run_job(cfg: DictConfig) -> None:
    """The body of one run (fit + test) and the run JSON write."""
    precision_flags = begin_run(cfg)
    if precision_flags is None:
        return  # not in this process's shard; another shard runs it
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg))

    frozen = load_frozen_protocol()
    if cfg.experiment.phase >= 2 and not bool(frozen.get("frozen", False)):
        raise RuntimeError(
            f"experiment.phase={cfg.experiment.phase} >= 2 but "
            f"conf/frozen_protocol.yaml has frozen={frozen.get('frozen')!r}"
        )
    check_frozen_choices(cfg, frozen)

    lr = (
        float(cfg.optimizer.lr)
        if cfg.optimizer.lr is not None
        else _frozen_lr(cfg, frozen)
    )
    logger.info("Peak learning rate: %g", lr)

    if int(cfg.model.t_max) < int(cfg.task.max_seq_len):
        raise RuntimeError(
            f"model.t_max ({cfg.model.t_max}) must be >= task.max_seq_len "
            f"({cfg.task.max_seq_len}); the ll slot count depends on it"
        )

    run_id = make_run_id(cfg, lr, include_task_keys=True)
    identity = (
        config_hash(OmegaConf.to_container(cfg, resolve=True), HASH_EXCLUDE),
        cfg.seed,
    )
    with claimed_run(cfg, run_id, identity, protocol_tiers=PROTOCOL_TIERS) as reason:
        if reason is not None:
            logger.info("skip %s: %s", run_id, reason)
            return
        monitor_key = str(cfg.training.monitor_key)
        callbacks = create_callbacks_reset_curves(
            cfg, run_id, precision_flags, lr=lr, monitor_key=monitor_key
        )
        loggers = create_loggers(cfg, run_id)

        L.seed_everything(cfg.seed)
        task = str(cfg.task.name)
        task_config = build_task_config(cfg)
        spec = require_task(task)
        if task == "mad_fuzzy":
            # Fixed training set, permuted per epoch (MAD's training
            # regime); validations/test are fixed test-style sets.
            def to_row(inputs, targets, is_training):
                return mad_batch_row(
                    task_config,
                    inputs,
                    targets,
                    pad_to=cfg.task.max_seq_len,
                    is_training=bool(is_training),
                )

            datamodule = FixedSetRecallDataModule(
                dataset_fn=fuzzy_recall_dataset,
                config=task_config,
                to_row=to_row,
                num_train=cfg.task.num_train,
                num_val=cfg.task.num_val,
                num_test=cfg.task.num_test,
                batch_size=cfg.training.batch_size,
                train_seed=cfg.seed,
                val_seed=cfg.task.val_seed,
                test_seed=cfg.task.test_seed,
                num_workers=cfg.data.num_workers,
            )
        else:
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

        # 0. Model through the harness registry; the mixer kind comes from
        # model.mixer (the run, not the harness).
        harness = require_harness(str(cfg.harness.name))
        model = harness(cfg)
        if bool(getattr(cfg.training, "compile", False)):
            model = torch.compile(model)
        extra_metrics = ()
        if task == "mad_fuzzy":
            # The task's reported metric (macro accuracy) and the plain
            # scored-token accuracy; both attach through the module's
            # extra-metric hook and drive nothing but their own columns.
            extra_metrics = (
                MacroTokenAccuracy(num_classes=int(cfg.task.vocab_size)),
                TokenAccuracy(),
            )
        module = build_module(
            cfg,
            model,
            lr,
            spec.slice_values(task_config),
            monitor_key=monitor_key,
            class_names=spec.class_names,
            extra_metrics=extra_metrics,
        )

        # MAD-style training runs by epochs with a per-epoch cosine LR:
        # max_steps and the validation interval are derived from the epoch
        # count and the steps-per-epoch, so the epoch count is the single
        # source of the run length.
        if str(
            cfg.optimizer.get("schedule") or ""
        ) == "epoch_cosine" and cfg.training.get("epochs"):
            import math

            steps_per_epoch = math.ceil(
                int(cfg.task.num_train) / int(cfg.training.batch_size)
            )
            max_steps = int(cfg.training.epochs) * steps_per_epoch
            val_check_interval = steps_per_epoch
        else:
            max_steps = cfg.training.max_steps
            val_check_interval = cfg.training.val_check_interval

        trainer = L.Trainer(
            max_steps=max_steps,
            val_check_interval=val_check_interval,
            precision=cfg.training.precision,
            callbacks=callbacks,
            logger=loggers,
            accelerator="auto",
            devices=cfg.training.devices,
            enable_progress_bar=True,
            log_every_n_steps=cfg.training.log_every_n_steps,
        )

        logger.info("Starting training (run_id=%s)...", run_id)
        trainer.fit(module, datamodule=datamodule)
        post_fit(cfg, trainer, module, callbacks, run_id)
        trainer.test(module, datamodule=datamodule)
        upload_run_record_artifact(cfg, run_id, trainer.is_global_zero)
        logger.info("Run complete (run_id=%s).", run_id)


def _walk(node: object, *keys: str) -> object | None:
    """The node reached by ``keys``, or ``None`` if any step is absent
    (a missing key and a present-``null`` key are treated the same, so a
    hand-edited ``lr_table`` node with no value cannot raise)."""
    current = node
    for key in keys:
        if current is None:
            return None
        current = current.get(key) if hasattr(current, "get") else None
        if current is None:
            return None
    return current


def _frozen_lr(cfg: DictConfig, frozen: DictConfig) -> float:
    """The frozen peak LR for this run, or raise.

    For the gated-delta mixer the table is
    ``lr_table.<task>.gated_delta.<layout>.<gate>``; an attention run has
    its own row ``lr_table.<task>.attention``.
    """
    mixer = str(OmegaConf.select(cfg, "model.mixer", default="gated_delta"))
    task_node = _walk(frozen.get("lr_table"), str(cfg.task.name))
    if mixer == "attention":
        value = _walk(task_node, "attention")
    else:
        value = _walk(task_node, mixer, str(cfg.model.layout), str(cfg.model.gate_mode))
    if value is None:
        raise RuntimeError(
            "optimizer.lr is null and no frozen-protocol entry exists (mixer="
            f"{mixer!r}); fill conf/frozen_protocol.yaml after calibration, or "
            "pass optimizer.lr=<value>"
        )
    return float(value)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run one pilot training run; see :func:`run_and_close`."""
    run_and_close(cfg, run_job)


if __name__ == "__main__":
    main()
