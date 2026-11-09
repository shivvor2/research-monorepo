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

from common import conf_resolvers  # noqa: E402,F401  (registers config resolvers)
from common.launch import begin_run, claimed_run  # noqa: E402
from common.registries import (  # noqa: E402
    build_extra_metrics,
    config_type_of_task,
    require_harness,
)
from common.run_id import HASH_EXCLUDE, make_run_id  # noqa: E402
from common.run_lifecycle import (  # noqa: E402
    build_module,
    create_callbacks_reset_curves,
    create_loggers,
    post_fit,
    run_and_close,
    run_length,
    upload_run_record_artifact,
)
from structured_recall_analysis.protocol import (  # noqa: E402
    PROTOCOL_TIERS,
    TASK_SETTING_BANDS,
)

from research_lib.data.synthetic import (  # noqa: E402
    FixedSetRecallDataModule,
    SyntheticRecallDataModule,
    TaskRegime,
    require_task,
)
from research_lib.training.callbacks import config_hash  # noqa: E402

logger = logging.getLogger(__name__)


def load_frozen_protocol() -> DictConfig:
    """Load conf/frozen_protocol.yaml (from this experiment's conf dir)."""
    return OmegaConf.load(_PILOT / "conf" / "frozen_protocol.yaml")


def check_frozen_choices(cfg: DictConfig, frozen: DictConfig) -> None:
    """Assert the run uses the frozen architecture choices and, once the
    protocol is frozen, that those choices are complete.

    Applies to protocol tiers only (the smoke/debug tiers are tooling).
    The run's config stays the record; the frozen file is checked against
    it, never used as a second source that silently overrides.  While
    ``frozen: false`` the check is permissive: only the ``harness_choices``
    entry for ``cfg.harness.name`` is checked, and only for the keys the
    entry sets.  While ``frozen: true`` a protocol-tier run additionally
    requires that the entry exists and that every key it lists is non-null
    (an unfilled entry would otherwise never be checked), and the same for
    the ``task_settings.<task>`` entry (the frozen settings such as the
    sequence length).

    Raises:
        RuntimeError: Naming the key, the run's value, the frozen value
            and the fix.
    """
    tier = str(cfg.experiment.get("tier") or "")
    if tier not in PROTOCOL_TIERS:
        return
    is_frozen = bool(frozen.get("frozen", False))
    harness = str(cfg.harness.name)
    entry = (frozen.get("harness_choices") or {}).get(harness)
    if is_frozen:
        if entry is None:
            raise RuntimeError(
                f"the frozen protocol is frozen but has no harness_choices "
                f"entry for harness {harness!r} (tier={tier!r}); fill "
                "conf/frozen_protocol.yaml"
            )
        for key in entry:
            if entry.get(key) is None:
                raise RuntimeError(
                    f"the frozen protocol is frozen but harness_choices."
                    f"{harness}.{key} is still null (tier={tier!r}); fill "
                    "conf/frozen_protocol.yaml"
                )
    if entry is not None:
        if entry.get("d_model") is not None and int(cfg.model.d_model) != int(
            entry["d_model"]
        ):
            raise RuntimeError(
                f"the frozen protocol for harness {harness!r} fixes d_model="
                f"{entry['d_model']}, but this run uses model.d_model="
                f"{cfg.model.d_model} (tier={tier!r})"
            )
        for key in ("layer_pattern", "state_mixer"):
            frozen_value = entry.get(key, None)
            if frozen_value is None:
                continue
            if str(cfg.model[key]) != str(frozen_value):
                raise RuntimeError(
                    f"the frozen protocol for harness {harness!r} uses "
                    f"{key}={frozen_value!r}, but this run uses "
                    f"model.{key}={cfg.model[key]!r} (tier={tier!r})"
                )

    # ---- the task's frozen settings (e.g. the sequence length) ----
    # Only a task with a setting rule (one in the band registry) needs a
    # ``task_settings`` entry once frozen; a task without one (its difficulty
    # comes from its slices, not its length) has nothing to freeze. An entry
    # that does exist is still checked against the run.
    task = str(OmegaConf.select(cfg, "task.name", default=""))
    task_settings = (frozen.get("task_settings") or {}).get(task)
    if is_frozen and task in TASK_SETTING_BANDS:
        if task_settings is None:
            raise RuntimeError(
                f"the frozen protocol is frozen but has no task_settings "
                f"entry for task {task!r} (tier={tier!r}); fill "
                "conf/frozen_protocol.yaml"
            )
        for key in task_settings:
            if task_settings.get(key) is None:
                raise RuntimeError(
                    f"the frozen protocol is frozen but task_settings."
                    f"{task}.{key} is still null (tier={tier!r}); fill "
                    "conf/frozen_protocol.yaml"
                )
    if task_settings is not None and task_settings.get("seq_len") is not None:
        frozen_seq = int(task_settings["seq_len"])
        run_seq = int(OmegaConf.select(cfg, "task.seq_len", default=-1))
        if run_seq != frozen_seq:
            raise RuntimeError(
                f"the frozen protocol fixes task_settings.{task}.seq_len="
                f"{frozen_seq}, but this run uses task.seq_len={run_seq} "
                f"(tier={tier!r}); launch the task at the frozen length"
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
    weight_decay = (
        float(cfg.optimizer.weight_decay)
        if cfg.optimizer.weight_decay is not None
        else _frozen_weight_decay(cfg, frozen)
    )
    logger.info("Peak learning rate: %g; weight decay: %g", lr, weight_decay)

    if int(cfg.model.t_max) < int(cfg.task.max_seq_len):
        raise RuntimeError(
            f"model.t_max ({cfg.model.t_max}) must be >= task.max_seq_len "
            f"({cfg.task.max_seq_len}); the ll slot count depends on it"
        )

    run_id = make_run_id(cfg, lr, weight_decay=weight_decay, include_task_keys=True)
    identity = (
        config_hash(OmegaConf.to_container(cfg, resolve=True), HASH_EXCLUDE),
        cfg.seed,
    )
    with claimed_run(cfg, run_id, identity, protocol_tiers=PROTOCOL_TIERS) as reason:
        if reason is not None:
            logger.info("skip %s: %s", run_id, reason)
            return
        L.seed_everything(cfg.seed)
        task = str(cfg.task.name)
        task_config = build_task_config(cfg)
        spec = require_task(task)
        regime = spec.regime
        length = run_length(cfg, regime)
        monitor_key = str(cfg.training.monitor_key)

        if regime is TaskRegime.FIXED_SET:
            # Fixed, chunk-padded rows: the run's max_seq_len must cover a
            # row (seq_len + 1) and line up with the chunk size.
            if int(cfg.task.max_seq_len) < int(cfg.task.seq_len) + 1:
                raise ValueError(
                    f"task.max_seq_len ({cfg.task.max_seq_len}) must be >= "
                    f"task.seq_len + 1 ({int(cfg.task.seq_len) + 1})"
                )
            if int(cfg.task.max_seq_len) % int(cfg.model.chunk_size) != 0:
                raise ValueError(
                    f"task.max_seq_len ({cfg.task.max_seq_len}) must be a "
                    f"multiple of model.chunk_size ({cfg.model.chunk_size})"
                )
            to_row = spec.row_fn(task_config, pad_to=int(cfg.task.max_seq_len))
            datamodule = FixedSetRecallDataModule(
                dataset_fn=spec.dataset_fn,
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
        extra_metrics = build_extra_metrics(
            list(cfg.task.get("extra_metrics") or ()),
            vocab_size=int(cfg.task.vocab_size),
        )
        module = build_module(
            cfg,
            model,
            lr,
            spec.slice_values(task_config),
            weight_decay=weight_decay,
            regime=regime,
            monitor_key=monitor_key,
            class_names=spec.class_names,
            extra_metrics=extra_metrics,
        )

        # The run record's caller-owned top-level fields (the model class,
        # the training regime and the derived run length).
        record_extra = {
            "model_class": type(model).__name__,
            "task_regime": regime.value,
            "epochs": int(cfg.training.get("epochs") or 0) or None,
            "steps_per_epoch": length.steps_per_epoch,
            "max_steps": length.max_steps,
            "weight_decay": weight_decay,
        }
        callbacks = create_callbacks_reset_curves(
            cfg,
            run_id,
            precision_flags,
            lr=lr,
            monitor_key=monitor_key,
            record_extra=record_extra,
        )
        loggers = create_loggers(cfg, run_id)

        trainer = L.Trainer(
            max_steps=length.max_steps,
            val_check_interval=length.val_check_interval,
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


def _frozen_value(cfg: DictConfig, frozen: DictConfig, table: str, kind: str) -> float:
    """The frozen scalar for this run from ``frozen[table]``, or raise.

    Both tables share the key path: for the gated-delta mixer
    ``<table>.<task>.gated_delta.<layout>.<gate>``; an attention run has
    its own row ``<table>.<task>.attention``.  ``kind`` names the config
    key in the error (e.g. ``optimizer.lr``).
    """
    mixer = str(OmegaConf.select(cfg, "model.mixer", default="gated_delta"))
    task_node = _walk(frozen.get(table), str(cfg.task.name))
    if mixer == "attention":
        value = _walk(task_node, "attention")
    else:
        value = _walk(task_node, mixer, str(cfg.model.layout), str(cfg.model.gate_mode))
    if value is None:
        raise RuntimeError(
            f"{kind} is null and no frozen-protocol entry exists in "
            f"{table!r} (mixer={mixer!r}); fill conf/frozen_protocol.yaml "
            f"after calibration, or pass {kind}=<value>"
        )
    return float(value)


def _frozen_lr(cfg: DictConfig, frozen: DictConfig) -> float:
    """The frozen peak LR for this run (see :func:`_frozen_value`)."""
    return _frozen_value(cfg, frozen, "lr_table", "optimizer.lr")


def _frozen_weight_decay(cfg: DictConfig, frozen: DictConfig) -> float:
    """The frozen weight decay for this run (see :func:`_frozen_value`)."""
    return _frozen_value(cfg, frozen, "weight_decay_table", "optimizer.weight_decay")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run one pilot training run; see :func:`run_and_close`."""
    run_and_close(cfg, run_job)


if __name__ == "__main__":
    main()
