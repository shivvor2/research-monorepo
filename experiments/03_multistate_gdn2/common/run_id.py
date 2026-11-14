"""The project's run-ID builder and its identity exclusion set.

The run ID names a run's JSON, its curves directory and its W&B run and
artifact. For the gated-delta mixer the base ID is
``{task}-{layout}-{gate}-{lambda}-d{d_model}-lr{lr}``; a run with any
other mixer omits the layout / gate / lambda segment (and the ``-ev``
``expand_v`` suffix) because those are gated-delta settings the other
mixers do not read, so its base ID is ``{task}-d{d_model}-lr{lr}``. In
both cases non-default optimiser, model and harness keys are appended,
along with ``-mix-{mixer}`` for a non-default mixer, so two runs that
differ only in such a key never share a file name. When
``optimizer.calibrate_weight_decay`` is set, the resolved weight decay is
appended as ``-wd<value>`` so a calibration that sweeps it gets distinct
IDs (Zoology's and the first pilot's IDs are unchanged). With
``include_task_keys`` on, the non-default fields of the task's registered
config type are appended too (a run whose task keys are part of its
identity enables it); with it off (the default) the base ID reproduces
the first pilot's builder exactly.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

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
#     (SyntheticRecallDataModule docstring);
#   - training.devices / training.accelerator: the data stream, the steps
#     per epoch, the loss and the metrics do not depend on the device
#     count or accelerator (global batches, the global-mean loss), so a
#     run relaunched on a different number of devices is the same
#     experiment; the first pilot's config gains these keys without its
#     identity changing.
HASH_EXCLUDE: tuple[str, ...] = (
    "output",
    "logging",
    "launch",
    "data.num_workers",
    "training.devices",
    "training.accelerator",
)

_MIXER_SPECIFIC_IDENTITY_KEYS: tuple[str, ...] = ("model.attention_qk_norm",)
"""Config keys that are part of a run's identity only for some mixers."""


def identity_exclude_keys(cfg: DictConfig) -> tuple[str, ...]:
    """The config keys excluded from a run's identity hash.

    ``HASH_EXCLUDE`` (the operational keys) always applies, plus the
    mixer-specific keys that a run of the current ``model.mixer`` does not
    read.  ``model.attention_qk_norm`` is set for every mixer on the MAD
    harness (the harness config carries it) but only the attention control
    reads it, so it must not change the identity of a run using another
    mixer (e.g. a gated-delta run); for an attention run it is part of the
    identity, so a run with and without query/key normalisation are
    distinct.
    """
    keys = list(HASH_EXCLUDE)
    if str(OmegaConf.select(cfg, "model.mixer", default="gated_delta")) != "attention":
        keys.extend(_MIXER_SPECIFIC_IDENTITY_KEYS)
    return tuple(keys)


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
    """One run-ID fragment for a value: sequences dot-joined, else str.

    The fragment may only carry ``[A-Za-z0-9._-]`` (it names the run's
    artifact), so a list of numbers becomes ``2.4``, not ``[2, 4]``.
    """
    value = (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (DictConfig, ListConfig))
        else value
    )
    if isinstance(value, (tuple, list)) and value:
        return ".".join(str(item) for item in value)
    return str(value)


def _normalize(value: Any) -> Any:
    """A comparable form of a config value: containers become tuples
    (recursively), everything else is unchanged."""
    value = (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, (DictConfig, ListConfig))
        else value
    )
    if isinstance(value, (list, tuple)):
        return tuple(_normalize(item) for item in value)
    return value


def _values_differ(value: Any, default: Any) -> bool:
    """Whether a configured task value differs from its dataclass default.

    Containers are compared as tuples (a Hydra ``ListConfig`` and the
    dataclass's tuple are the same value), and floats with a relative
    tolerance so a YAML decimal that rounds the default (e.g. ``1/3``)
    does not count as a change.
    """
    a = _normalize(value)
    b = _normalize(default)
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return True
        return any(_values_differ(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return not math.isclose(a, b, rel_tol=1e-9)
    return a != b


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
        if value is None:
            continue
        try:
            differs = _values_differ(value, default)
        except (ValueError, TypeError):
            differs = False
        if differs:
            suffixes.append(f"-{leaf}-{_join(value)}")
    return suffixes


def make_run_id(
    cfg: DictConfig,
    lr: float,
    *,
    weight_decay: float | None = None,
    include_task_keys: bool = False,
) -> str:
    """The run ID of one run, per the base-ID rule.

    The gated-delta base is ``{task}-{layout}-{gate}-{lambda}-d{d}-lr{lr}``;
    any other mixer drops that segment (the layout / gate / lambda keys are
    gated-delta settings it ignores), leaving ``{task}-d{d}-lr{lr}``.

    Args:
        cfg: The resolved config (``task.*``, ``model.*``,
            ``experiment.tier``, ``optimizer.*``, ``harness.name``).
        lr: The (resolved) peak learning rate.
        weight_decay: The resolved weight decay; appended to the run ID
            (for either mixer) only when ``optimizer.calibrate_weight_decay``
            is set, so a calibration that sweeps the value gets distinct
            IDs.
        include_task_keys: Also append the run's non-default task-config
            fields (off reproduces the first pilot's IDs exactly).

    Returns:
        The run ID string.
    """
    mixer = str(_cfg_value(cfg, "model.mixer") or "gated_delta")
    if mixer == "gated_delta":
        run_id = (
            f"{cfg.task.name}-{cfg.model.layout}-{cfg.model.gate_mode}"
            f"-{cfg.model.lambda_mode}-d{cfg.model.d_model}-lr{lr:g}"
        )
    else:
        # Layout, gate and lambda are gated-delta settings; the other
        # mixers are built from the model width and head count. Their base
        # ID names the model without those keys.
        run_id = f"{cfg.task.name}-d{cfg.model.d_model}-lr{lr:g}"
    if (
        bool(_cfg_value(cfg, "optimizer.calibrate_weight_decay"))
        and weight_decay is not None
    ):
        run_id += f"-wd{weight_decay:g}"
    scheme = str(_cfg_value(cfg, "optimizer.decay_scheme") or "grouped")
    if scheme != "grouped":
        run_id += f"-wd-{scheme}"
    if str(_cfg_value(cfg, "model.layer_pattern") or "zoology") != "zoology":
        run_id += f"-pat-{_cfg_value(cfg, 'model.layer_pattern')}"
    if str(_cfg_value(cfg, "model.state_mixer") or "none") != "none":
        run_id += f"-sm-{_cfg_value(cfg, 'model.state_mixer')}"
    if mixer == "gated_delta" and (
        _cfg_value(cfg, "model.expand_v") is not None
        and float(cfg.model.expand_v) != 1.0
    ):
        run_id += f"-ev{cfg.model.expand_v:g}"
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
