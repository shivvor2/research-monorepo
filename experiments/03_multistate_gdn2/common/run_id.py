"""The project's run-ID builder and its identity exclusion set.

The run ID names a run's JSON, its curves directory and its W&B run and
artifact. The base ID covers task / layout / gate / lambda / width / lr /
seed; non-default optimiser and model keys are appended so two runs that
differ only in such a key never share a file name. With ``include_task_keys``
on, non-default **task** keys are appended too (the second pilot turns it
on, because its task keys are part of the run's identity); with it off
(the default) the base ID reproduces the first pilot's builder exactly.
"""
from __future__ import annotations

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


# Non-default model / optimiser keys that are part of the run identity and
# so appended (as a short suffix) to the base run ID.
_DEFAULT_SCALARS = {
    "optimizer.decay_scheme": "grouped",
    "model.layer_pattern": "zoology",
    "model.state_mixer": "none",
    "model.expand_v": 1.0,
}

# Non-default task keys appended only when ``include_task_keys`` is set.
_TASK_KEYS = ("task.reserved_filler",)
"""Task keys that join a run ID when the option is on; task keys that the
base ID already covers (task.name) are not repeated."""


def _cfg_value(cfg: DictConfig, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        node = node.get(part)
    return node


def make_run_id(
    cfg: DictConfig, lr: float, *, include_task_keys: bool = False
) -> str:
    """The run ID of one run, per the base-ID rule.

    Args:
        cfg: The resolved config (``task.*``, ``model.*``,
            ``experiment.tier``, ``optimizer.*``).
        lr: The (resolved) peak learning rate.
        include_task_keys: Also append non-default task keys (tools run
            with it off to reproduce the first pilot's IDs exactly).

    Returns:
        The run ID string.
    """
    run_id = (
        f"{cfg.task.name}-{cfg.model.layout}-{cfg.model.gate_mode}"
        f"-{cfg.model.lambda_mode}-d{cfg.model.d_model}-lr{lr:g}"
    )
    scheme = str(_cfg_value(cfg, "optimizer.decay_scheme"))
    if scheme != "grouped":
        run_id += f"-wd-{scheme}"
    if str(cfg.model.layer_pattern) != "zoology":
        run_id += f"-pat-{cfg.model.layer_pattern}"
    if str(cfg.model.state_mixer) != "none":
        run_id += f"-sm-{cfg.model.state_mixer}"
    if float(cfg.model.expand_v) != 1.0:
        run_id += f"-ev{cfg.model.expand_v:g}"
    if include_task_keys:
        for key in _TASK_KEYS:
            value = _cfg_value(cfg, key)
            if value is not None and str(value) not in ("", "0"):
                leaf = key.rsplit(".", 1)[-1]
                run_id += f"-{leaf}-{value}"
    run_id += f"-s{cfg.seed}"
    tier = cfg.experiment.get("tier", None)
    if tier:
        run_id = f"{tier}_{run_id}"
    return run_id