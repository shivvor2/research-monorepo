"""Run-summary callback: one JSON per training run.

:attr:`RunSummaryCallback` writes ``runs_dir/<run_id>.json`` — the full
resolved config, its hash, git / version / device provenance, parameter
and state accounting, timing, early-stopping info and every validation /
test metric — at the end of ``fit``, and updates it with the test
results at the end of ``test``.  The file is written atomically (temp
file + rename), so a partial summary is never observed.

One JSON per run is the protocol's unit of record
(``06_experiment_protocol.md`` §10): results leave the machine from day
one, one file per run, ``rsync``-able at any time.

Run tests with::

    pytest src/research_lib/training/callbacks/tests/test_run_summary.py -v
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


def _canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    """The config with the seed removed, JSON-canonicalized for hashing.

    Args:
        config: Fully resolved config dict.

    Returns:
        dict: A plain (JSON-serializable) copy without the ``seed`` key,
        so all seeds of one configuration share a hash.
    """
    plain = json.loads(json.dumps(config, default=str, sort_keys=True))
    plain.pop("seed", None)
    return plain


def config_hash(config: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of the config without the seed.

    Args:
        config: Fully resolved config dict.

    Returns:
        str: The hex digest.
    """
    payload = json.dumps(_canonical_config(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _versions() -> dict[str, Any]:
    """Python / torch / CUDA / lightning / numpy versions."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "lightning": L.__version__,
        "numpy": np.__version__,
    }


def _device_info() -> dict[str, Any]:
    """GPU name and device count (``null`` / ``0`` on CPU)."""
    if torch.cuda.is_available():
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
        }
    return {"gpu_name": None, "gpu_count": 0}


class RunSummaryCallback(L.Callback):
    """Write one JSON summary per run, atomically, at fit / test end.

    Args:
        runs_dir (Path | str): Directory the JSON is written to (created
            if missing).
        run_id (str): File name (without extension); identifies the run.
        config (dict): The fully resolved run config (the ``seed`` is
            recorded but excluded from the hash).
        extra (dict | None): Optional extra top-level entries merged into
            the JSON (e.g. the resolved learning rate).

    Writes ``runs_dir/<run_id>.json`` with:

        - ``run_id``, ``config``, ``config_hash``, ``seed``;
        - ``git`` (commit, dirty), ``versions``, ``device``;
        - ``params`` (total, ``lambda_only``, per gated-delta layer,
          ``state_bytes_per_mixer``);
        - ``timing`` (wall-clock seconds, steps completed, steps per
          second);
        - ``stopping`` (early stopping fired, stop step, best ``val/acc``
          and its step);
        - ``val_final`` and ``test`` (every metric key from
          :class:`~research_lib.data.synthetic.RecallMetrics`);
        - ``mixer_stats_final`` (the last logged λ / erase statistics).
    """

    def __init__(
        self,
        runs_dir: Path | str,
        run_id: str,
        config: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.runs_dir = Path(runs_dir)
        self.run_id = run_id
        self.config = json.loads(json.dumps(config, default=str))
        self.extra = extra
        self._fit_start: float | None = None
        self._best_val_acc: float | None = None
        self._best_val_step: int | None = None
        self.summary: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _path(self) -> Path:
        return self.runs_dir / f"{self.run_id}.json"

    @staticmethod
    def _metrics(trainer: L.Trainer, prefix: str) -> dict[str, float]:
        """Every callback metric under ``prefix/`` (e.g. ``val``)."""
        out: dict[str, float] = {}
        for key, value in trainer.callback_metrics.items():
            if key.startswith(f"{prefix}/"):
                try:
                    out[key[len(prefix) + 1 :]] = float(value)
                except (TypeError, ValueError):  # non-numeric metric
                    continue
        return out

    def _early_stopping(self, trainer: L.Trainer) -> dict[str, Any]:
        """Whether early stopping fired and the best val/acc."""
        info: dict[str, Any] = {
            "stopped": False,
            "best_val_acc": self._best_val_acc,
            "best_val_step": self._best_val_step,
        }
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                info["stopped"] = getattr(cb, "stopping_reason", None) is not None
                break
        return info

    @staticmethod
    def _params(model: torch.nn.Module) -> dict[str, Any]:
        """Total / λ-only / per-layer parameter counts and state bytes."""
        lambda_only = sum(
            p.numel() for name, p in model.named_parameters() if "lambda_mixer" in name
        )
        per_layer: list[dict[str, Any]] = []
        layers = getattr(model, "gated_delta_layers", None)
        state_bytes = None
        if callable(layers):
            gd_layers = model.gated_delta_layers()  # type: ignore[attr-defined]
            for i, layer in enumerate(gd_layers):
                lam = sum(
                    p.numel()
                    for name, p in layer.named_parameters()
                    if "lambda_mixer" in name
                )
                per_layer.append(
                    {"index": i, "total": _count(layer), "lambda_only": lam}
                )
            if gd_layers:
                state_bytes = model.state_bytes_per_mixer()  # type: ignore[attr-defined]
        return {
            "total": _count(model),
            "lambda_only": lambda_only,
            "per_gated_delta_layer": per_layer,
            "state_bytes_per_mixer": state_bytes,
        }

    def _write(self) -> None:
        """Write the summary JSON atomically (temp file + rename)."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self._path()
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
        logger.info("RunSummaryCallback: wrote %s", path)

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._fit_start = time.monotonic()
        self._best_val_acc = None
        self._best_val_step = None

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Track the best ``val/acc`` and its step (during fit only)."""
        if trainer.sanity_checking:
            return
        from lightning.pytorch.trainer.states import TrainerFn

        if trainer.state.fn != TrainerFn.FITTING:
            return
        value = trainer.callback_metrics.get("val/acc")
        if value is None:
            return
        acc = float(value)
        if self._best_val_acc is None or acc > self._best_val_acc:
            self._best_val_acc = acc
            self._best_val_step = trainer.global_step

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        elapsed = time.monotonic() - (self._fit_start or time.monotonic())
        steps = int(trainer.global_step)
        summary: dict[str, Any] = {
            "run_id": self.run_id,
            "config": self.config,
            "config_hash": config_hash(self.config),
            "seed": self.config.get("seed"),
            "git": _git_info(),
            "versions": _versions(),
            "device": _device_info(),
            "params": self._params(pl_module.model),
            "timing": {
                "wall_clock_seconds": elapsed,
                "steps_completed": steps,
                "steps_per_second": steps / elapsed if elapsed > 0 else None,
            },
            "stopping": self._early_stopping(trainer),
            "val_final": self._metrics(trainer, "val"),
            "mixer_stats_final": self._metrics(trainer, "mixer"),
            "test": None,  # filled by on_test_end
        }
        if self.extra:
            summary.update(self.extra)
        self.summary = summary
        self._write()

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.summary["test"] = self._metrics(trainer, "test")
        self.summary.setdefault("val_final", self._metrics(trainer, "val"))
        self._write()


def _count(module: torch.nn.Module) -> int:
    """Total (trainable) parameter count of a module."""
    return sum(p.numel() for p in module.parameters())
