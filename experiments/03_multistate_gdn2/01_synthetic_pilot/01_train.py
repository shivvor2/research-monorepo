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
(the unit of record: one JSON per run).
``experiment.phase >= 2`` refuses to run until
``conf/frozen_protocol.yaml`` has ``frozen: true``, and ``optimizer.lr``
is resolved from the frozen LR table by (task, layout, gate_mode).

Precision: fp32 throughout; ``torch.set_float32_matmul_precision`` is
left at its default (TF32 would lower the reference layers' fp32 matmuls).

The run's body is the shared project machinery
(:mod:`common.run_lifecycle` for callbacks / loggers / post-fit / the W&B
transport, :mod:`common.launch` for the head and the claim, and
:mod:`common.harnesses` for the model). This file keeps the pilot's
protocol: the frozen-LR resolution, the frozen-choice gate and the
overwrite-width check.
"""

import logging
import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

# The shared project package (common/) sits one level up; its modules
# import research_lib and the pilot's own protocol, so the pilot adds the
# project folder to ``sys.path`` the way it adds its own folder for
# ``pilot_analysis``.
_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# The shared run machinery. The names imported here are the modules'
# public surfaces; the pilot keeps only its protocol and run_job below.
from common.harnesses import build_zoology  # noqa: E402
from common.launch import job_num, stop_requested  # noqa: E402
from common.layout import wandb_kwargs as wandb_logger_kwargs  # noqa: E402
from common.run_id import HASH_EXCLUDE, make_run_id  # noqa: E402

# Re-exports the tests and the module's callers rely on; the definitions
# live in common / research_lib now, not here. Listed in __all__ so the
# linter keeps them as intentional re-exports.
from common.run_lifecycle import (  # noqa: E402; noqa: E402  (bulk re-exports)
    VECTOR_TARGET_MODULES,
    build_module,
    create_callbacks_reset_curves,
    create_loggers,
    ensure_run_record_uploaded,
    first_gated_delta,
    per_slot_summary,
    post_fit,
    run_and_close,
    run_summary_callback,
    state_bytes_collector,
    upload_init_kwargs,
    upload_run_record_artifact,
)

# The protocol's tier names, one source of truth shared with the analysis
# reports (pilot_analysis/protocol.py): a tier added there cannot
# silently skip the frozen-choice gate, and a tier added here cannot
# silently vanish from the analysis's expected-run counts.
from pilot_analysis.protocol import (  # noqa: E402
    PROTOCOL_TIERS,
    TIER_ANCHORS,
    TIER_CALIB_DMODEL,
    TIER_CALIB_HARNESS,
    TIER_CALIB_LR_ANCHORS,
)

from research_lib.architectures.zoology_lm import ZoologyLM  # noqa: E402
from research_lib.data.synthetic import (  # noqa: E402
    MQARConfig,
    OverwriteConfig,
    SyntheticRecallDataModule,
    require_task,
)
from research_lib.training.callbacks import config_hash  # noqa: E402
from research_lib.training.run_control import claim_run  # noqa: E402
from research_lib.training.run_control import assert_fit_finished
from research_lib.training.run_control import (  # noqa: E402
    disable_tf32 as set_precision_flags,
)
from research_lib.training.run_control import in_shard, should_skip
from research_lib.training.run_records import run_record_metadata  # noqa: E402

logger = logging.getLogger(__name__)

# The names this module re-exports from common / research_lib, so the
# linter treats them as public rather than unused.
__all__ = [
    "HASH_EXCLUDE",
    "VECTOR_TARGET_MODULES",
    "assert_fit_finished",
    "build_model",
    "build_module",
    "claim_run",
    "config_hash",
    "create_callbacks_reset_curves",
    "create_loggers",
    "ensure_run_record_uploaded",
    "in_shard",
    "job_num",
    "lr_source",
    "make_run_id",
    "post_fit",
    "resolve_lr",
    "run_summary_callback",
    "set_precision_flags",
    "run_record_metadata",
    "should_skip",
    "stop_requested",
    "upload_init_kwargs",
    "upload_run_record_artifact",
    "wandb_logger_kwargs",
]


def load_frozen_protocol() -> DictConfig:
    """Load conf/frozen_protocol.yaml (from this experiment's conf dir)."""
    path = Path(__file__).resolve().parent / "conf" / "frozen_protocol.yaml"
    return OmegaConf.load(path)


def lr_source(cfg: DictConfig) -> tuple[str, str, str, str | None]:
    """The frozen-table lookup key of a run, after LR inheritance.

    The core configurations look themselves up unchanged. The other
    tiers inherit (no other LR is tuned):

    - ``b_only`` / ``w_only`` use the LR of ``(layout, full)``;
    - ``fs`` uses ``(ws, same gate)``;
    - λ ``static`` / ``lin`` use the λ-``mlp`` LR of the same
      ``(layout, gate)`` (the table is keyed by gate mode only, so the
      λ-``mlp`` entry *is* the ``(layout, gate)`` entry);
    - matched bytes uses ``(single, full)`` (its own configuration);
    - R1 and the long-sequence tier run the core configurations, so they
      use the core entries unchanged;
    - the anchors use their own entries, keyed by ``d_model`` (gate ``gdn``
      is never calibrated in the main table).

    Returns:
        tuple: ``(task, layout, gate_mode, anchor_d_model)`` — the
        table key, with ``anchor_d_model`` set (to the run's ``d_model``)
        for anchor runs, else ``None`` (main LR table).

    Raises:
        RuntimeError: If an anchor run's ``tier`` is not ``tier6_anchors``
            while ``gate_mode`` is ``gdn`` (uncalibrated gate, no entry).
    """
    task = str(cfg.task.name)
    layout = str(cfg.model.layout)
    gate_mode = str(cfg.model.gate_mode)
    tier = str(cfg.experiment.get("tier", None) or "")
    if tier == TIER_ANCHORS:
        return task, layout, gate_mode, int(cfg.model.d_model)
    if gate_mode == "gdn":
        raise RuntimeError(
            f"gate_mode='gdn' outside the anchor tier (tier={tier!r}) has no "
            f"LR entry; the anchors are the only gdn runs and use the anchor "
            f"table keyed by d_model (run the anchor LR sweep "
            f"conf/sweep/calib_lr_anchors.yaml)"
        )
    # Inheritance: b_only/w_only -> full; fs -> ws (same gate). Everything
    # else (lambdas, matched bytes, R1, long-T) uses its own key.
    gate_mode = "full" if gate_mode in ("b_only", "w_only") else gate_mode
    layout = "ws" if layout == "fs" else layout
    return task, layout, gate_mode, None


def resolve_lr(cfg: DictConfig, frozen: DictConfig) -> float:
    """The peak LR: cfg.optimizer.lr, else the frozen-protocol entry.

    Non-core tiers inherit from the core entries; see :func:`lr_source`.
    Logs which entry was used. Fails loudly if no entry exists.
    """
    if cfg.optimizer.lr is not None:
        lr = float(cfg.optimizer.lr)
        logger.info("Learning rate %g from optimizer.lr (config)", lr)
        return lr
    task, layout, gate_mode, anchor_d_model = lr_source(cfg)
    requested_layout = str(cfg.model.layout)
    requested_gate = str(cfg.model.gate_mode)
    if anchor_d_model is not None:
        try:
            value = frozen.anchor_lr[anchor_d_model][layout]
        except (KeyError, AttributeError) as e:
            raise RuntimeError(
                f"optimizer.lr is null and no anchor LR entry exists for "
                f"(d_model={anchor_d_model}, layout={layout!r}); run the anchor "
                f"LR sweep (conf/sweep/calib_lr_anchors.yaml) and fill in "
                f"conf/frozen_protocol.yaml, or pass optimizer.lr=<value>"
            ) from e
        if value is None:
            raise RuntimeError(
                f"the anchor LR entry for (d_model={anchor_d_model}, "
                f"layout={layout!r}) is empty; fill in conf/frozen_protocol.yaml "
                f"at the end of calibration, or pass optimizer.lr=<value>"
            )
        logger.info(
            "Learning rate %g from the anchor table (d_model=%s, layout=%s)",
            float(value),
            anchor_d_model,
            layout,
        )
        return float(value)
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
    if (layout, gate_mode) != (requested_layout, requested_gate):
        logger.info(
            "Learning rate %g inherited from entry (task=%s, layout=%s, "
            "gate=%s) for requested (layout=%s, gate=%s)",
            float(value),
            task,
            layout,
            gate_mode,
            requested_layout,
            requested_gate,
        )
    else:
        logger.info(
            "Learning rate %g from the frozen LR table (task=%s, layout=%s, "
            "gate=%s)",
            float(value),
            task,
            layout,
            gate_mode,
        )
    return float(value)


def check_overwrite_k(cfg: DictConfig, frozen: DictConfig) -> None:
    """Assert the overwrite task's largest K equals 2 * d_k.

    The primary endpoint is defined at K = 2 * d_k, so the task file must
    match the model width. On a mismatch the error names the right task
    file (from the frozen protocol's ``overwrite_task_by_d_model`` record).

    Raises:
        RuntimeError: If the largest key count does not equal 2 * d_k.
    """
    if str(cfg.task.name) != "overwrite":
        return
    d_model = int(cfg.model.d_model)
    num_heads = int(cfg.model.num_heads)
    head_dim = cfg.model.get("head_dim", None)
    dk = int(head_dim) if head_dim is not None else d_model // num_heads
    largest_k = max(slice_values(cfg))
    if largest_k == 2 * dk:
        return
    by_d_model = frozen.get("overwrite_task_by_d_model", None)
    suggested = None
    if by_d_model is not None:
        suggested = by_d_model.get(d_model, None)
    if suggested is not None:
        hint = f"use task={suggested} (conf/task/{suggested}.yaml)"
    else:
        hint = (
            "pick the task file whose largest K equals 2 * d_k "
            "(conf/task/overwrite.yaml at d_k = 16, "
            "conf/task/overwrite_dk32.yaml at d_k = 32)"
        )
    raise RuntimeError(
        f"the overwrite task's largest K is {largest_k}, but the primary "
        f"endpoint is defined at K = 2 * d_k = {2 * dk} (d_model={d_model}, "
        f"num_heads={num_heads}, d_k={dk}); {hint}, or change "
        f"model.d_model / model.num_heads to match the task"
    )


# check_frozen_choices applies to PROTOCOL_TIERS ONLY (imported from
# pilot_analysis.protocol, one source of truth with the analysis
# reports): tooling configs (the smoke and debug configs, or no tier at
# all) are not protocol runs and must not trip the gate - an allow-list
# is safer than exempting them one by one, because the next tooling
# config would hit the same problem.


def check_frozen_choices(cfg: DictConfig, frozen: DictConfig) -> None:
    """Assert the run uses the frozen calibration choices (width, harness).

    Applies to :data:`PROTOCOL_TIERS` only - the smoke and debug configs
    and tier-less runs are tooling, not protocol runs. The run's config
    stays the record: the frozen file is checked against it, never used as
    a second source that silently overrides the config. Each check applies
    in every phase, but only once the frozen field is filled.

    Raises:
        RuntimeError: Naming the key, the run's value, the frozen value
            and the fix.
    """
    tier = str(cfg.experiment.get("tier", None) or "")
    if tier not in PROTOCOL_TIERS:
        return

    frozen_d_model = frozen.get("d_model", None)
    if frozen_d_model is not None and tier not in (
        TIER_CALIB_DMODEL,
        TIER_CALIB_LR_ANCHORS,
        TIER_ANCHORS,
    ):
        if int(cfg.model.d_model) != int(frozen_d_model):
            raise RuntimeError(
                f"the frozen protocol's d_model is {frozen_d_model}, but this "
                f"run uses model.d_model={cfg.model.d_model} (tier={tier!r}); "
                f"set `model.d_model: {frozen_d_model}` in conf/config.yaml, "
                f"or pass `model.d_model={frozen_d_model}`"
            )

    harness = frozen.get("harness", None)
    if harness is not None:
        for key in ("layer_pattern", "state_mixer"):
            frozen_value = harness.get(key, None)
            if frozen_value is None or tier == TIER_CALIB_HARNESS:
                continue
            if str(cfg.model[key]) != str(frozen_value):
                raise RuntimeError(
                    f"the frozen protocol's harness uses {key}={frozen_value!r}, "
                    f"but this run uses model.{key}={cfg.model[key]!r} "
                    f"(tier={tier!r}; the anchors are not exempt - the harness "
                    f"is shared by every tier); set `model.{key}: {frozen_value}` "
                    f"in conf/config.yaml, or pass `model.{key}={frozen_value}`"
                )


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
            reserved_filler=cfg.task.reserved_filler,
        )
    raise ValueError(f"unknown task {name!r}")


def slice_values(cfg: DictConfig) -> tuple[int, ...]:
    """The task's slice values (D for MQAR, K for overwrite)."""
    key = "num_kv_pairs" if str(cfg.task.name) == "mqar" else "num_keys"
    return tuple(int(value) for value in cfg.task[key])


def build_model(cfg: DictConfig) -> ZoologyLM:
    """Build the model: the shared Zoology harness with the gated-delta
    mixer (the first pilot's config has no ``model.mixer`` key, so the
    harness defaults to gated-delta; the identity hash is unchanged)."""
    return build_zoology(cfg)


def run_job(cfg: DictConfig) -> None:
    """The body of one pilot run (fit + test) and the run JSON write.

    Wrapped by :func:`main`, which checks the W&B credentials first and
    closes the W&B run afterwards.
    """
    # 0. Precision first: fp32 without TF32.
    precision_flags = set_precision_flags()
    logger.info("Precision flags: %s", precision_flags)

    logger.info("=" * 80)
    logger.info("Resolved Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    # 0b. STOP file: while it exists, no new run starts. In a multirun
    # this ends the whole sweep (Hydra's job wrapper catches Exception,
    # not SystemExit); in a single run it just exits. The file is never
    # deleted by the script.
    stop = stop_requested(cfg)
    if stop is not None:
        logger.info(
            "STOP file found at %s; not starting job %d. Remove it to continue.",
            stop,
            job_num(cfg),
        )
        sys.exit(0)

    # 0c. Shard: several processes on one machine split one sweep; job k
    # runs only in the process with shard_index == k % num_shards. All
    # shards must be launched with the identical sweep command.
    if not in_shard(job_num(cfg), cfg.launch.shard_index, cfg.launch.num_shards):
        logger.info(
            "job %d not in shard %d of %d; skipping",
            job_num(cfg),
            cfg.launch.shard_index,
            cfg.launch.num_shards,
        )
        return

    # 1. Phase gate: >= 2 requires the frozen protocol.
    frozen = load_frozen_protocol()
    if cfg.experiment.phase >= 2 and not bool(frozen.get("frozen", False)):
        raise RuntimeError(
            f"experiment.phase={cfg.experiment.phase} >= 2 but "
            f"conf/frozen_protocol.yaml has frozen={frozen.get('frozen')!r}; "
            f"finish calibration, fill in and commit the frozen protocol "
            f"before running experiment tiers"
        )

    # 1b. The run must use the frozen calibration choices (width and
    # harness), once they are recorded in the frozen protocol.
    check_frozen_choices(cfg, frozen)

    # 2. Learning rate.
    lr = resolve_lr(cfg, frozen)
    logger.info("Peak learning rate: %g", lr)

    # 3. The ll slot count depends on t_max: it must cover the task.
    if cfg.model.t_max < cfg.task.max_seq_len:
        raise RuntimeError(
            f"model.t_max ({cfg.model.t_max}) must be >= task.max_seq_len "
            f"({cfg.task.max_seq_len}); the ll slot count depends on it"
        )

    # 3b. The overwrite task's largest K must equal 2 * d_k (the primary
    # endpoint is defined there); the error names the right task file.
    check_overwrite_k(cfg, frozen)

    # 4. Run ID, before any data or model is built (the skip and lock
    # checks below need it).
    run_id = make_run_id(cfg, lr)

    # 5. Claim the run: acquire the lock, then decide. The record is
    # read only while this process owns the run. The skip applies to
    # protocol tiers only (the smoke and debug configs always run); a
    # different identity is refused by RunSummaryCallback, a non-done
    # record is overwritten by the re-run.
    lock_path = Path(cfg.output.runs_dir) / f"{run_id}.lock"
    record_path = Path(cfg.output.runs_dir) / f"{run_id}.json"
    identity = (
        config_hash(OmegaConf.to_container(cfg, resolve=True), HASH_EXCLUDE),
        cfg.seed,
    )
    skip_completed = (
        bool(cfg.launch.skip_completed) and str(cfg.experiment.tier) in PROTOCOL_TIERS
    )
    with claim_run(
        lock_path, record_path if skip_completed else None, identity
    ) as reason:
        if reason is not None:
            logger.info("skip %s: %s", run_id, reason)
            if reason == "already test_done":
                # A finished run must still reach W&B even if its upload
                # failed when it finished: the relaunch skips before any
                # upload would have run, so re-check the artifact now.
                ensure_run_record_uploaded(cfg, run_id)
            return

        # 6. Callbacks first, then the curves reset, then the loggers.
        callbacks = create_callbacks_reset_curves(cfg, run_id, precision_flags, lr=lr)
        loggers = create_loggers(cfg, run_id)

        # 8. Seeds. 9. Data. 10. Model + training module.
        L.seed_everything(cfg.seed)
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
        model = build_model(cfg)
        if cfg.training.compile:
            model = torch.compile(model)
        module = common_run_lifecycle_build_module(cfg, model, lr)
        num_params = sum(p.numel() for p in model.parameters())
        gd_layer = first_gated_delta(model)
        state_bytes = gd_layer.state_bytes() if gd_layer is not None else 0
        logger.info(
            "Model: %s parameters; recurrent state per mixer: %d bytes",
            f"{num_params:,}",
            state_bytes,
        )

        # 11. Trainer.
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

        # 12. Post-fit steps: the fit-finished guard and the optional
        # final checkpoint.
        post_fit(cfg, trainer, module, callbacks, run_id)

        # 13. Test on the final weights; the callback updates the run JSON.
        trainer.test(module, datamodule=datamodule)

        # 14. W&B artifact (rented machines).
        upload_run_record_artifact(cfg, run_id, trainer.is_global_zero)

        logger.info("Run complete (run_id=%s).", run_id)


def common_run_lifecycle_build_module(cfg, model, lr):
    """Build the training module via the shared lifecycle (with the
    task's class names and the default monitor key)."""
    return build_module(
        cfg,
        model,
        lr,
        slice_values(cfg),
        monitor_key="val/acc",
        class_names=require_task(str(cfg.task.name)).class_names,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run one pilot training run; see :func:`run_and_close`."""
    run_and_close(cfg, run_job)


if __name__ == "__main__":
    main()
