"""Run-summary callback: one atomic JSON file per training run.

:class:`RunSummaryCallback` writes its file (default
``<runs_dir>/<run_id>.json``) at the start and end of fitting, at the end
of testing, and when the run fails. The write is always atomic (temp file
+ ``os.replace``), so a partial summary is never observed. The JSON
carries the full resolved config, its hash (seed excluded, so all seeds
of one configuration share a hash), provenance (git, versions, device),
timing, early-stopping / best-monitored-value information, a lifecycle
``status``, and the outputs of the run's **collectors**.

Writing a collector
-------------------

A collector adds one top-level section to the JSON. It is a
:class:`Collector` triple ``(section, fn, stage)``:

- ``fn(trainer, pl_module)`` is called once, at the Lightning hook named
  by ``stage`` (see :class:`Stage`). Its return value is stored as
  ``summary[section]``.
- What is available at each stage:

  - ``FIT_START``: the model is built and on its device; no metrics yet.
    Use it for anything known before training: parameter counts, state
    sizes.
  - ``FIT_END``: ``trainer.callback_metrics`` holds the last logged value
    of every metric.
  - ``TEST_END``: ``trainer.callback_metrics`` holds the test metrics.

- The returned mapping must be JSON-serializable. Anything else is
  converted with ``default=str``.
- **Errors.** If ``fn`` raises, the section becomes
  ``{"error": "<repr of the exception>"}`` and the error is logged at
  ERROR level; the run is not stopped (a collector bug must not destroy a
  finished run). With ``strict=True`` the exception is re-raised instead.
- **Section names** must be unique across collectors and must not
  collide with a built-in section (:data:`BUILTIN_SECTIONS`) or a key of
  ``extra``. Raise ``ValueError`` at construction.

Passing ``collectors=`` **replaces** the defaults; it does not merge with
them. An experiment lists everything it wants.

Two examples::

    # A one-line collector that records a model attribute.
    Collector("model", lambda trainer, pl: {"dtype": str(pl.dtype)}, Stage.FIT_START)

    # A named function that reads several metrics.
    def losses(trainer, pl_module):
        metrics = trainer.callback_metrics
        return {"train": float(metrics["train/loss"]), "val": float(metrics["val/loss"])}
    Collector("losses", losses, Stage.FIT_END)

Run tests with::

    pytest src/research_lib/training/callbacks/tests/test_run_summary.py -v
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import string
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.trainer.states import TrainerFn

from research_lib.training.param_utils import select_parameters

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    """When a collector runs. Each value is the Lightning hook that triggers it."""

    FIT_START = "on_fit_start"
    FIT_END = "on_fit_end"
    TEST_END = "on_test_end"


CollectorFn = Callable[[L.Trainer, L.LightningModule], Mapping[str, Any]]
"""A collector body: called once with ``(trainer, pl_module)``."""


@dataclass(frozen=True)
class Collector:
    """One section of the run JSON.

    Args:
        section: Top-level key of the JSON this collector fills.
        fn: Returns the section's contents (see "Writing a collector").
        stage: When it runs (default: end of fit).
    """

    section: str
    fn: CollectorFn
    stage: Stage = Stage.FIT_END


BUILTIN_SECTIONS: tuple[str, ...] = (
    "run_id",
    "status",
    "failed_stage",
    "error",
    "config",
    "config_hash",
    "seed",
    "filename",
    "git",
    "versions",
    "device",
    "timing",
    "stopping",
)
"""Top-level JSON keys owned by the callback itself; collectors may not use them."""

_FILENAME_FIELDS: tuple[str, ...] = (
    "run_id",
    "seed",
    "config_hash",
    "hash8",
    "date",
    "datetime",
)
"""Fields a ``filename`` template may use."""


def metrics_by_prefix(prefix: str) -> CollectorFn:
    """Every numeric ``trainer.callback_metrics`` entry under ``prefix/``.

    Args:
        prefix: Metric prefix, e.g. ``"val"`` collects ``val/...`` as
            ``...``. Non-numeric values are skipped.

    Returns:
        A collector body returning the stripped mapping.
    """

    def fn(trainer: L.Trainer, pl_module: L.LightningModule) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in trainer.callback_metrics.items():
            if not key.startswith(f"{prefix}/"):
                continue
            try:
                out[key[len(prefix) + 1 :]] = float(value)
            except (TypeError, ValueError):  # non-numeric metric
                continue
        return out

    return fn


def param_counts(
    groups: Mapping[str, Sequence[str]] | None = None,
) -> CollectorFn:
    """Parameter counts, total / trainable and per pattern group.

    Args:
        groups: Optional mapping of group name to parameter-name patterns.
            Each group selects parameters with
            :func:`research_lib.training.param_utils.select_parameters`
            (PEFT-style matching), so e.g. ``{"lambda": ["lambda_mixer"]}``
            counts every parameter whose name contains a
            ``lambda_mixer`` component.

    Returns:
        A collector body returning ``{"total", "trainable",
        "groups": {name: count}}``.
    """

    def fn(trainer: L.Trainer, pl_module: L.LightningModule) -> dict[str, Any]:
        total = sum(p.numel() for p in pl_module.parameters())
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        counted: dict[str, int] = {}
        if groups:
            for name, patterns in groups.items():
                selected = select_parameters(
                    pl_module, list(patterns), require_grad=False
                )
                counted[name] = sum(p.numel() for p in selected)
        return {"total": total, "trainable": trainable, "groups": counted}

    return fn


DEFAULT_COLLECTORS: tuple[Collector, ...] = (
    Collector("params", param_counts(), Stage.FIT_START),
    Collector("val_final", metrics_by_prefix("val"), Stage.FIT_END),
    Collector("test", metrics_by_prefix("test"), Stage.TEST_END),
)
"""Default sections: parameters at fit start, val metrics at fit end, test at test end."""


def config_hash(config: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of the config without the seed.

    Args:
        config: Fully resolved config mapping.

    Returns:
        str: The hex digest. All seeds of one configuration share a hash.
    """
    plain = json.loads(json.dumps(config, default=str, sort_keys=True))
    plain.pop("seed", None)
    payload = json.dumps(plain, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _missing_stage_hooks(cls: type) -> list[str]:
    """Stage hooks not overridden by ``cls`` (still Lightning's no-ops)."""
    missing = []
    for stage in Stage:
        hook = getattr(cls, stage.value, None)
        base_hook = getattr(L.Callback, stage.value, None)
        if hook is None or hook == base_hook:
            missing.append(stage.value)
    return missing


def _check_stage_hooks(cls: type) -> None:
    """Raise if ``cls`` does not override every Stage member's hook.

    Raises:
        TypeError: Naming the missing hooks. A Stage member without its
            hook can never silently do nothing.
    """
    missing = _missing_stage_hooks(cls)
    if missing:
        raise TypeError(
            f"{cls.__name__} is missing stage hook(s) {missing}; "
            f"every Stage member needs its hook overridden"
        )


def _git_info() -> dict[str, Any]:
    """Commit hash and dirty flag of the repository containing this package."""
    root = Path(__file__).resolve()
    for parent in root.parents:
        if (parent / ".git").exists():  # dir (checkout) or file (worktree)
            root = parent
            break
    else:  # not a git checkout (e.g. installed from a wheel)
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError) as e:  # pragma: no cover
        logger.warning("RunSummaryCallback: git info unavailable: %s", e)
        return {"commit": None, "dirty": None}


class RunSummaryCallback(L.Callback):
    """Write one JSON summary per run, atomically, at every stage.

    Args:
        runs_dir: Directory the JSON is written to (created if missing).
        run_id: Identifies the run; used by the default filename and
            recorded in the JSON.
        config: The fully resolved run config (the ``seed`` is recorded
            but excluded from the hash).
        filename: ``str.format`` template for the file name, relative to
            ``runs_dir``. Fields: ``run_id``, ``seed`` (``config["seed"]``
            or ``none``), ``config_hash``, ``hash8`` (its first 8
            characters), ``date`` (``YYYY-MM-DD``) and ``datetime``
            (``YYYYMMDD-HHMMSS``), both local time. Resolved once, in
            ``__init__`` (the file is written several times per run; a
            time field must not produce several files). **Trade-off:** a
            template with ``{date}`` or ``{datetime}`` writes a new file
            each time the same run is relaunched, instead of overwriting
            the previous one; analysis code must then deduplicate.
        packages: Distribution names (not import names) whose installed
            versions are recorded; a missing distribution gives ``None``.
            The Python version and ``torch.version.cuda`` are always
            recorded.
        monitor: Metric name to track the best value of (e.g.
            ``"val/acc"``); ``None`` disables tracking.
        mode: ``"max"`` or ``"min"``: which direction improves ``monitor``.
        collectors: The sections to fill (see "Writing a collector").
            Replaces :data:`DEFAULT_COLLECTORS`.
        extra: Optional mapping merged into the JSON at every write.
        strict: Re-raise collector errors instead of recording them.

    Raises:
        TypeError: If a :class:`Stage` hook is not overridden on this class
            (so an enum member without its hook can never silently do
            nothing).
        ValueError: Invalid ``filename`` (unknown field, absolute path,
            ``..``, or not ending in ``.json``), invalid ``mode``,
            duplicate or reserved collector section names, or a
            collector section colliding with an ``extra`` key.
    """

    def __init__(
        self,
        runs_dir: Path | str,
        run_id: str,
        config: Mapping[str, Any],
        *,
        filename: str = "{run_id}.json",
        packages: Sequence[str] = ("torch", "lightning", "numpy"),
        monitor: str | None = None,
        mode: Literal["min", "max"] = "max",
        collectors: Sequence[Collector] = DEFAULT_COLLECTORS,
        extra: Mapping[str, Any] | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        _check_stage_hooks(type(self))
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.runs_dir = Path(runs_dir)
        self.run_id = str(run_id)
        self.config = json.loads(json.dumps(dict(config), default=str))
        self.packages = tuple(packages)
        self.monitor = monitor
        self.mode = mode
        self.collectors = tuple(collectors)
        self.extra = dict(extra) if extra is not None else None
        self.strict = bool(strict)
        self.summary: dict[str, Any] = {}

        hash_full = config_hash(self.config)
        seed = self.config.get("seed")
        fields = {
            "run_id": self.run_id,
            "seed": "none" if seed is None else str(seed),
            "config_hash": hash_full,
            "hash8": hash_full[:8],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "datetime": datetime.now().strftime("%Y%m%d-%H%M%S"),
        }
        self.filename = _resolve_filename(filename, fields)

        sections = [c.section for c in self.collectors]
        if len(set(sections)) != len(sections):
            raise ValueError(f"duplicate collector section names: {sections}")
        reserved = set(BUILTIN_SECTIONS)
        collisions = sorted(set(sections) & reserved)
        if collisions:
            raise ValueError(f"collector sections collide with built-ins: {collisions}")
        if self.extra is not None:
            collisions = sorted(set(sections) & set(self.extra))
            if collisions:
                raise ValueError(
                    f"collector sections collide with extra keys: {collisions}"
                )

        self._git = _git_info()
        self._fit_start: float | None = None
        self._best_value: float | None = None
        self._best_step: int | None = None

    # ------------------------------------------------------------------ #
    # File
    # ------------------------------------------------------------------ #

    @property
    def path(self) -> Path:
        """The resolved JSON path (``runs_dir`` / resolved ``filename``)."""
        return self.runs_dir / self.filename

    def _write(self) -> None:
        """Write the summary JSON atomically (temp file + rename)."""
        self.summary["run_id"] = self.run_id
        self.summary["config"] = self.config
        self.summary["config_hash"] = config_hash(self.config)
        self.summary["seed"] = self.config.get("seed")
        self.summary["filename"] = self.filename
        self.summary["git"] = self._git
        self.summary["versions"] = self._versions()
        self.summary["device"] = self._device()
        path = self.path
        # The filename may contain subdirectories; create the file's
        # parent (covers runs_dir itself).
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
        logger.info("RunSummaryCallback: wrote %s", path)

    def _versions(self) -> dict[str, Any]:
        """Python version, ``torch.version.cuda`` and the package versions."""
        versions: dict[str, Any] = {
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
        }
        for name in self.packages:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return versions

    @staticmethod
    def _device() -> dict[str, Any]:
        """GPU name and device count (``null`` / ``0`` on CPU)."""
        if torch.cuda.is_available():
            return {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            }
        return {"gpu_name": None, "gpu_count": 0}

    # ------------------------------------------------------------------ #
    # Built-in sections
    # ------------------------------------------------------------------ #

    def _update_timing(self, trainer: L.Trainer) -> None:
        """Wall clock since fit start and steps completed (fitting only)."""
        if self._fit_start is None:
            return  # this process never saw a fit; keep the loaded timing
        elapsed = time.monotonic() - self._fit_start
        steps = int(trainer.global_step)
        self.summary["timing"] = {
            "wall_clock_seconds": elapsed,
            "steps_completed": steps,
            "steps_per_second": steps / elapsed if elapsed > 0 else None,
        }

    def _update_stopping(self, trainer: L.Trainer) -> None:
        """Early-stopping flag and the best monitored value / step."""
        early_stopped = False
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                early_stopped = getattr(cb, "stopping_reason", None) is not None
                break
        self.summary["stopping"] = {
            "early_stopped": early_stopped,
            "monitor": self.monitor,
            "mode": self.mode,
            "best_value": self._best_value,
            "best_step": self._best_step,
        }

    def _merge_extra(self) -> None:
        if self.extra is not None:
            self.summary.update(self.extra)

    def _run_stage(
        self, stage: Stage, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Run the collectors tagged with ``stage``; merge their sections."""
        for collector in self.collectors:
            if collector.stage is not stage:
                continue
            try:
                section = collector.fn(trainer, pl_module)
            except Exception as e:  # noqa: BLE001 - must not kill the run
                if self.strict:
                    raise
                logger.error(
                    "RunSummaryCallback: collector for section %r failed: %r",
                    collector.section,
                    e,
                )
                section = {"error": repr(e)}
            self.summary[collector.section] = section

    def _load_existing(self) -> None:
        """Load a previously written file (test-only launch into a fresh summary)."""
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self.summary.update(json.load(f))

    # ------------------------------------------------------------------ #
    # Hooks (one per Stage, plus tracking and exception handling)
    # ------------------------------------------------------------------ #

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._fit_start = time.monotonic()
        self._best_value = None
        self._best_step = None
        self._run_stage(Stage.FIT_START, trainer, pl_module)
        self.summary["status"] = "running"
        self._merge_extra()
        self._write()

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._run_stage(Stage.FIT_END, trainer, pl_module)
        self.summary["status"] = "fit_done"
        self._update_timing(trainer)
        self._update_stopping(trainer)
        self._merge_extra()
        self._write()

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.summary:
            # A test-only launch: update the existing file instead of
            # writing one with only the test section.
            self._load_existing()
        self._run_stage(Stage.TEST_END, trainer, pl_module)
        self.summary["status"] = "test_done"
        # Nothing else: timing and stopping are fit-time sections, computed
        # at fit end only. Recomputing them here would either fold the test
        # run into the timing of a fit in the same process, or (in a
        # test-only launch, where this process tracked nothing) rewrite
        # stopping with best_value=None, destroying the loaded values.
        self._merge_extra()
        self._write()

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Track the best monitored value (fitting only, no sanity check)."""
        if trainer.sanity_checking or self.monitor is None:
            return
        from lightning.pytorch.trainer.states import TrainerFn

        if trainer.state.fn != TrainerFn.FITTING:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        current = float(value)
        improved = self._best_value is None or (
            current > self._best_value
            if self.mode == "max"
            else current < self._best_value
        )
        if improved:
            self._best_value = current
            self._best_step = trainer.global_step

    def on_exception(
        self, trainer: L.Trainer, pl_module: L.LightningModule, exception: BaseException
    ) -> None:
        if not self.summary:
            # A test-only launch (or any fresh callback) that raised:
            # keep the existing file's sections instead of replacing the
            # complete fit summary with built-ins and the error only.
            self._load_existing()
        status = "interrupted" if isinstance(exception, KeyboardInterrupt) else "failed"
        self.summary["status"] = status
        self.summary["failed_stage"] = _failed_stage(trainer)
        self.summary["error"] = repr(exception)
        self._merge_extra()
        self._write()


def _failed_stage(trainer: L.Trainer) -> str | None:
    """The stage that raised: ``"fit"`` / ``"test"`` (from ``trainer.state.fn``)."""
    fn = trainer.state.fn
    return {TrainerFn.FITTING: "fit", TrainerFn.TESTING: "test"}.get(
        fn, str(fn) if fn is not None else None
    )


def _resolve_filename(template: str, fields: Mapping[str, str]) -> str:
    """Validate and resolve a ``filename`` template, once.

    Args:
        template: The template string.
        fields: Allowed field names and their values.

    Returns:
        The resolved (relative) file name.

    Raises:
        ValueError: Unknown field, absolute path, ``..`` component, or a
            result not ending in ``.json``.
    """
    allowed = ", ".join(sorted(fields))
    for _, field_name, _, _ in string.Formatter().parse(template):
        # Formatter().parse yields (literal_text, field_name, format_spec,
        # conversion); an escaped brace has field_name None.
        if field_name is None or field_name == "":
            continue
        if field_name not in fields:
            raise ValueError(
                f"unknown filename field {field_name!r}; allowed fields: {allowed}"
            )
    if re.search(r"{(\w+)\s*:", template):
        raise ValueError(
            f"format specs are not supported in filename templates: {template!r}"
        )
    resolved = template.format(**fields)
    if not resolved.endswith(".json"):
        raise ValueError(f"filename must end in .json, got {resolved!r}")
    candidate = Path(resolved)
    if candidate.is_absolute():
        raise ValueError(f"filename must be relative, got {resolved!r}")
    if ".." in candidate.parts:
        raise ValueError(f"filename must not contain '..', got {resolved!r}")
    return resolved
