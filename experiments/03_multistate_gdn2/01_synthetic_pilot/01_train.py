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

import json
import logging
import shutil
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.trainer.states import TrainerStatus
from omegaconf import DictConfig, OmegaConf

# Library imports
from research_lib.architectures.config import ZoologyLMConfig
from research_lib.architectures.zoology_lm import ZoologyLM, zoology_mixers
from research_lib.data.synthetic import (
    MQARConfig,
    OverwriteConfig,
    SyntheticRecallDataModule,
)
from research_lib.layers.attention import MultistateGatedDelta
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
from research_lib.training.scheduling import WarmupStableDecaySchedule
from research_lib.utils.io import atomic_torch_save
from research_lib.utils.run_lock import exclusive_lock

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
# Run identity: config keys excluded from the hash
# =============================================================================

# Dotted config keys excluded from the run identity (config_hash). Each
# one is safe because it cannot change a result:
#   - output: runs_dir is an absolute path under the launch directory, so
#     the same run launched from another directory or machine would get a
#     different identity (and a record pulled from another machine would
#     block a relaunch);
#   - logging: logger settings (CSV/W&B on/off, save dirs) do not change
#     the training stream or the metrics;
#   - launch: execution controls (skip_completed, stop_file, shards) are
#     operational, not experimental;
#   - data.num_workers: the training stream is defined per
#     (train_seed, batch index) independently of the worker count
#     (SyntheticRecallDataModule docstring).
HASH_EXCLUDE: tuple[str, ...] = ("output", "logging", "launch", "data.num_workers")


def set_precision_flags() -> dict[str, bool]:
    """fp32 without TF32 (protocol §2): both PyTorch TF32 switches off.

    ``torch.set_float32_matmul_precision`` is left untouched: the matmul
    flag covers the reference layers' fp32 matmuls, and the cuDNN flag
    covers BaseConv / the short convs, which run through cuDNN and
    default to TF32 on.

    Returns:
        The values read back, for the run record's ``precision_flags``.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def assert_fit_finished(trainer: L.Trainer) -> None:
    """Raise unless the fit ended normally.

    With Lightning < 2.4, Ctrl-C made ``fit()`` return normally and the
    script would test a half-trained model and record ``test_done``; the
    lock pins >= 2.4, the guard keeps the invariant independent of that.
    ``status == "test_done"`` means fit ended normally (max steps or
    early stop) AND the test pass completed on the final weights.

    Raises:
        RuntimeError: Naming the status, if it is not ``FINISHED``.
    """
    if trainer.state.status != TrainerStatus.FINISHED:
        raise RuntimeError(
            f"fit() ended with trainer.state.status={trainer.state.status!r}, "
            f"not FINISHED; the model is not fully trained, refusing to test"
        )


def job_num(cfg: DictConfig) -> int:
    """The Hydra job number (0 when not in a multirun).

    ``hydra.job.num`` is set only in a multirun; a single run has no job
    number, and the shard round-robin treats it as job 0.
    """
    job = cast(Any, HydraConfig.get().job)  # a DictConfig at runtime
    return int(OmegaConf.select(job, "num", default=0))


def stop_requested(cfg: DictConfig) -> Path | None:
    """The STOP file path if it exists in the launch directory.

    The launch directory is the original working directory (the
    directory the command was run from; with ``job.chdir`` the job's cwd
    is the job dir, so the STOP file must live next to the launch
    command, not in the job dir).
    """
    path = Path(get_original_cwd()) / str(cfg.launch.stop_file)
    return path if path.exists() else None


def in_shard(job_num: int, shard_index: int, num_shards: int) -> bool:
    """Round-robin shard membership: job ``k`` runs only in the process
    with ``shard_index == k % num_shards``.

    All shards must be launched with the identical sweep command; the
    round-robin keeps every shard moving along the sweep's priority order
    (overwrite before mqar, seed-major).

    Raises:
        ValueError: If ``num_shards < 1`` or ``shard_index`` is outside
            ``[0, num_shards)``.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index {shard_index} out of range for num_shards {num_shards}"
        )
    return job_num % num_shards == shard_index


def should_skip(record_path: Path, identity: tuple[str, Any]) -> str | None:
    """Skip reason for an existing run record, or ``None`` to run.

    A record is done only when its status is ``test_done`` and its
    ``(config_hash, seed)`` equals this run's identity. Any other record
    falls through: a different identity is refused by
    ``RunSummaryCallback`` as today; a non-done record is overwritten by
    the re-run; a missing or unreadable record runs.
    """
    if not record_path.exists():
        return None
    try:
        with open(record_path, encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("status") != "test_done":
        return None
    if (record.get("config_hash"), record.get("seed")) != identity:
        return None
    return "already test_done"


@contextmanager
def claim_run(
    lock_path: Path | str,
    record_path: Path | str | None,
    identity: tuple[str, Any],
) -> Generator[str | None, None, None]:
    """Acquire the run lock and decide whether to run.

    Yields a skip reason (or ``None`` to run) while the lock is held:
    the record is read only after this process owns the run, so a run
    that another process just completed is skipped instead of re-run (a
    check before the lock would race with a process that finishes
    between the check and the lock). ``record_path=None`` skips the
    record check (the run always proceeds once the lock is acquired).

    Args:
        lock_path: The per-run lock file.
        record_path: The run's record file, or ``None`` when the skip
            check does not apply (e.g. the smoke and debug configs).
        identity: The run's ``(config_hash, seed)``.

    Yields:
        str | None: The skip reason, or ``None`` to run.
    """
    with exclusive_lock(lock_path) as acquired:
        if not acquired:
            yield "running in another process"
            return
        reason = None
        if record_path is not None:
            reason = should_skip(Path(record_path), identity)
        yield reason


# =============================================================================
# Helpers
# =============================================================================


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
    if tier == "tier6_anchors":
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
    match the model width: conf/task/overwrite.yaml at d_k = 16
    (d_model = 32), conf/task/overwrite_dk32.yaml at d_k = 32
    (d_model = 64). On a mismatch the error names the right task file
    (from the frozen protocol's ``overwrite_task_by_d_model`` record).

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


# The protocol's tiers: every calibration tier plus every tier the
# analysis's expected-runs table counts (02_analyze.py EXPECTED_RUNS).
# check_frozen_choices applies to these ONLY: tooling configs (the smoke
# and debug configs, or no tier at all) are not protocol runs and must
# not trip the gate - an allow-list is safer than exempting them one by
# one, because the next tooling config would hit the same problem.
PROTOCOL_TIERS: frozenset[str] = frozenset(
    {
        "calib_dmodel",
        "calib_harness",
        "calib_lr",
        "calib_lr_anchors",
        "tier1_core",
        "tier2_gates",
        "tier3_frozen",
        "tier4_lambda",
        "tier5_matched_bytes",
        "tier6_anchors",
        "tier_longT",
        "R1",
    }
)


def check_frozen_choices(cfg: DictConfig, frozen: DictConfig) -> None:
    """Assert the run uses the frozen calibration choices (width, harness).

    Applies to :data:`PROTOCOL_TIERS` only - the smoke and debug configs
    and tier-less runs are tooling, not protocol runs. The run's config
    stays the record: the frozen file is checked against it, never used as
    a second source that silently overrides the config. Each check applies
    in every phase, but only once the frozen field is filled (so
    ``calib_lr`` is checked against ``d_model`` as soon as
    ``calib_dmodel`` has been recorded, before ``frozen: true``).

    Exemptions: ``calib_dmodel`` (chooses the width) and
    ``calib_lr_anchors`` / ``tier6_anchors`` (the anchors set their own
    width) skip the ``d_model`` check; ``calib_harness`` skips the harness
    checks. The anchors are NOT exempt from the harness: it is shared by
    every tier, so the anchor numbers are read in the same harness as the
    main grid.

    Raises:
        RuntimeError: Naming the key, the run's value, the frozen value
            and the fix.
    """
    tier = str(cfg.experiment.get("tier", None) or "")
    if tier not in PROTOCOL_TIERS:
        return

    frozen_d_model = frozen.get("d_model", None)
    if frozen_d_model is not None and tier not in (
        "calib_dmodel",
        "calib_lr_anchors",
        "tier6_anchors",
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
            if frozen_value is None or tier == "calib_harness":
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
        )
    raise ValueError(f"unknown task {name!r}")


def slice_values(cfg: DictConfig) -> tuple[int, ...]:
    """The task's slice values (D for MQAR, K for overwrite)."""
    key = "num_kv_pairs" if str(cfg.task.name) == "mqar" else "num_keys"
    return tuple(int(value) for value in cfg.task[key])


def build_model(cfg: DictConfig) -> ZoologyLM:
    """Build the model: make_mixer from cfg.model, layout by zoology_mixers.

    The Hydra config's keys and structure are unchanged; only this
    Python that reads them is mixer-specific.
    """
    c = cfg.model
    head_dim = c.head_dim
    head_dim = int(head_dim) if head_dim is not None else c.d_model // c.num_heads

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


def build_module(
    cfg: DictConfig, model: ZoologyLM, lr: float
) -> MGDN2SyntheticRecallModule:
    """The training module with the two AdamW weight-decay groups."""
    matrix_optimizer_config, vector_optimizer_config = build_weight_decay_configs(
        cfg.optimizer.decay_scheme,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
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
    return MGDN2SyntheticRecallModule(
        model=model,
        slice_values=slice_values(cfg),
        with_classes=None,  # auto-detect (True for overwrite batches)
        loss_on=cfg.training.loss_on,
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
        f"-{cfg.model.lambda_mode}-d{cfg.model.d_model}-lr{lr:g}"
    )
    # Every non-default value that the base ID leaves out is part of the
    # run's identity too: two runs that differ only in such a key would
    # overwrite each other's run JSON and checkpoint.
    scheme = str(cfg.optimizer.decay_scheme)
    if scheme != "grouped":
        run_id += f"-wd-{scheme}"
    if str(cfg.model.layer_pattern) != "zoology":
        run_id += f"-pat-{cfg.model.layer_pattern}"
    if str(cfg.model.state_mixer) != "none":
        run_id += f"-sm-{cfg.model.state_mixer}"
    if float(cfg.model.expand_v) != 1.0:
        run_id += f"-ev{cfg.model.expand_v:g}"
    run_id += f"-s{cfg.seed}"
    tier = cfg.experiment.get("tier", None)
    if tier:
        run_id = f"{tier}_{run_id}"
    return run_id


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
    """Keep the pilot's mixer metric names: one entry per lambda slot."""

    if key.startswith("lambda_"):
        per_slot = value.float().mean(dim=0)  # average over heads -> (L,)
        return {f"{key}/slot{l}": float(v) for l, v in enumerate(per_slot)}
    return {key: float(value.float().mean())}


def create_callbacks(
    cfg: DictConfig, run_id: str, precision_flags: dict[str, bool]
) -> list:
    """Early stopping (99% answer accuracy), the run summary, LR logging."""
    callbacks = [
        EarlyStopping(
            monitor="val/acc",
            mode="max",
            stopping_threshold=cfg.training.early_stop_threshold,
            patience=10**9,
        ),
        RunSummaryCallback(
            Path(cfg.output.runs_dir),
            run_id,
            OmegaConf.to_container(cfg, resolve=True),
            # Distribution names of the packages this script imports.
            packages=("torch", "lightning", "hydra-core", "omegaconf"),
            monitor="val/acc",
            mode="max",
            # The run identity excludes operational keys (see HASH_EXCLUDE).
            hash_exclude=HASH_EXCLUDE,
            # The lockfile hash is part of the run's provenance: a tier
            # that mixed lock versions would be visible (protocol §10.1).
            provenance_files=("uv.lock",),
            collectors=(
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
                "lr_resolved": lr_resolved.get("value"),
                # Also recorded via the config; kept top-level so analyses
                # need not dig for it (one JSON per run, rsync-able).
                "decay_scheme": str(cfg.optimizer.decay_scheme),
                "precision_flags": precision_flags,
                # Set by main() after the checkpoint is saved (before the
                # test pass), so the test_done record names the file.
                "final_checkpoint": None,
            },
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if bool(cfg.training.collect_mixer_stats):
        callbacks.append(LayerStatsCallback(prefix="mixer", summarize=per_slot_summary))
    return callbacks


# A tiny mutable holder so create_callbacks can record the resolved LR.
lr_resolved: dict[str, float | None] = {"value": None}


def create_loggers(cfg: DictConfig, run_id: str) -> list | None:
    """CSV (next to the run records) and optional W&B loggers.

    The CSV logger writes under ``<save_dir>/<run_id>/version_0``, so the
    curves of one run live next to its run JSON (``runs/<task>/curves/``
    in the base config; the smoke and debug configs keep their own
    ``save_dir``).
    """
    loggers = []
    if cfg.logging.csv.enabled:
        loggers.append(
            CSVLogger(save_dir=cfg.logging.csv.save_dir, name=run_id, version=0)
        )
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

    # 0. Precision first: fp32 without TF32 (protocol §2). The cuDNN
    # switch defaults to on, and BaseConv / the short convs run through
    # cuDNN; the matmul switch covers the reference layers' fp32 matmuls.
    # torch.set_float32_matmul_precision is left untouched.
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

    # 1. Phase gate: >= 2 requires the frozen protocol (§8.4).
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
    lr_resolved["value"] = lr
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
    # read only while this process owns the run, so a run that another
    # process just completed is skipped instead of re-run (a check before
    # the lock would race with a process that finishes between the check
    # and the lock). The lock file of a completed run already exists, so
    # taking the lock for a run that is then skipped costs nothing. The
    # skip applies to protocol tiers only (the smoke and debug configs
    # always run); a different identity is refused by RunSummaryCallback,
    # a non-done record is overwritten by the re-run.
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
            return

        # 6. Curves: a relaunch is a fresh run. The CSV logger appends to
        # an existing version_0 directory, so a stale one is removed
        # before the logger is created.
        if bool(cfg.logging.csv.enabled):
            curves_dir = Path(cfg.logging.csv.save_dir) / run_id
            if curves_dir.exists():
                shutil.rmtree(curves_dir)
                logger.info("Removed stale curves directory %s", curves_dir)

        # 8. Seeds: the run seed fixes init AND the training stream.
        L.seed_everything(cfg.seed)

        # 9. Data.
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

        # 10. Model + training module.
        model = build_model(cfg)
        if cfg.training.compile:
            model = torch.compile(model)
        module = build_module(cfg, model, lr)
        num_params = sum(p.numel() for p in model.parameters())
        gd_layer = first_gated_delta(model)
        state_bytes = gd_layer.state_bytes() if gd_layer is not None else 0
        logger.info(
            "Model: %s parameters; recurrent state per mixer: %d bytes",
            f"{num_params:,}",
            state_bytes,
        )

        # 11. Callbacks and loggers.
        callbacks = create_callbacks(cfg, run_id, precision_flags)
        loggers = create_loggers(cfg, run_id)

        # 12. Trainer.
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

        # 13. Post-fit guard: with Lightning < 2.4, Ctrl-C made fit()
        # return normally and the script would test a half-trained model;
        # the lock pins >= 2.4, the guard keeps the invariant independent
        # of that. test_done means fit ended normally AND the test pass
        # completed on the final weights.
        assert_fit_finished(trainer)

        # 14. Optional final checkpoint for the mechanism probes: saved
        # after fit and before test, atomically (the test pass must never
        # run on weights that were not saved; a partial file must never
        # be observed).
        if cfg.output.save_final_checkpoint:
            path = Path(cfg.output.runs_dir) / f"{run_id}.pt"
            atomic_torch_save(model.state_dict(), path)
            logger.info("Saved final checkpoint to %s", path)
            for cb in callbacks:
                if isinstance(cb, RunSummaryCallback) and cb.extra is not None:
                    cb.extra["final_checkpoint"] = path.name

        # 15. Test on the final weights; the callback updates the run JSON.
        trainer.test(module, datamodule=datamodule)

        logger.info("Run complete (run_id=%s).", run_id)


if __name__ == "__main__":
    main()
