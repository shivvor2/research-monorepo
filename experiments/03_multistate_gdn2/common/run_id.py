"""The project's run-ID builder and its identity exclusion set.

The run ID names a run's JSON, its curves directory and its W&B run and
artifact. The base ID covers task / layout / gate / lambda / width / lr /
seed; non-default optimiser, model, harness and mixer keys are appended so
two runs that differ only in such a key never share a file name. With
``include_task_keys`` on, the non-default fields of the task's registered
config type are appended too ("the second pilot turns it on, because its
task keys are part of the run's identity"); with it off (the default) the
base ID reproduces the first pilot's builder exactly.
"""

from __future__ import annotations

import inspect
from typing import Any

from omegaconf import DictConfig

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

# Task config fields that never join the run ID (already in the base ID,
# or operational seeding / naming).
_TASK_SKIP_FIELDS = frozenset({"name", "val_seed", "test_seed", "max_seq_len"})


def _cfg_value(cfg: Any, dotted: str) -> Any:
    """The value at a dotted config key, or ``None`` for a missing path
    (a missing intermediate key yields ``None``, not an error; works for
    OmegaConf nodes and plain dicts alike)."""
    node: Any = cfg
    for part in dotted.split("."):
        if node is None:
            return None
        node = node.get(part)
    return node


def _join(value: Any) -> str:
    """One run-ID fragment for a value: sequences dot-joined, else str."""
    if isinstance(value, (tuple, list)) and value:
        return ".".join(str(item) for item in value)
    return str(value)


def _task_suffixes(cfg: DictConfig) -> list[str]:
    """The non-default task-config fields of this run's task.

    Compares each field the task's registered config type declares against
    its dataclass default; a field whose configured value differs (and is
    not a seed / already in the base ID) is appended, deterministically.
    """
    from research_lib.data.synthetic.registry import require_task

    try:
        spec = require_task(str(cfg.task.name))
    except ValueError:
        return []
    cls = spec.config_type
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return []
    suffixes = []
    for leaf in sorted(params):
        if leaf in _TASK_SKIP_FIELDS or leaf in ("self", "kwargs"):
            continue
        default = params[leaf].default
        if default is inspect.Parameter.empty:
            continue
        value = _cfg_value(cfg, f"task.{leaf}")
        try:
            differs = value != default
        except (ValueError, TypeError):
            differs = False
        if value is not None and differs:
            suffixes.append(f"-{leaf}-{_join(value)}")
    return suffixes


def make_run_id(cfg: DictConfig, lr: float, *, include_task_keys: bool = False) -> str:
    """The run ID of one run, per the base-ID rule.

    Args:
        cfg: The resolved config (``task.*``, ``model.*``,
            ``experiment.tier``, ``optimizer.*``, ``harness.name``).
        lr: The (resolved) peak learning rate.
        include_task_keys: Also append the run's non-default task-config
            fields (off reproduces the first pilot's IDs exactly).

    Returns:
        The run ID string.
    """
    run_id = (
        f"{cfg.task.name}-{cfg.model.layout}-{cfg.model.gate_mode}"
        f"-{cfg.model.lambda_mode}-d{cfg.model.d_model}-lr{lr:g}"
    )
    scheme = str(_cfg_value(cfg, "optimizer.decay_scheme") or "grouped")
    if scheme != "grouped":
        run_id += f"-wd-{scheme}"
    if str(cfg.model.layer_pattern) != "zoology":
        run_id += f"-pat-{cfg.model.layer_pattern}"
    if str(cfg.model.state_mixer) != "none":
        run_id += f"-sm-{cfg.model.state_mixer}"
    if (
        _cfg_value(cfg, "model.expand_v") is not None
        and float(cfg.model.expand_v) != 1.0
    ):
        run_id += f"-ev{cfg.model.expand_v:g}"
    mixer = str(_cfg_value(cfg, "model.mixer") or "gated_delta")
    if mixer != "gated_delta":
        run_id += f"-mix-{mixer}"
    harness = str(_cfg_value(cfg, "harness.name") or "zoology")
    if harness != "zoology":
        run_id += f"-h-{harness}"
    if include_task_keys:
        for suffix in _task_suffixes(cfg):
            run_id += suffix
    run_id += f"-s{cfg.seed}"
    tier = cfg.experiment.get("tier", None)
    if tier:
        run_id = f"{tier}_{run_id}"
    return run_id
