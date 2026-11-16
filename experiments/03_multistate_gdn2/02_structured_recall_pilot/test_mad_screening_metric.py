"""The MAD fuzzy screening reading uses the per-sequence mean accuracy.

The MAD harness monitors ``val/acc`` (the per-sequence mean accuracy,
the reading every other task of the project uses) for early stopping and
the best-value track, and the analysis reads ``test/acc`` as the
screening cell and ``val/acc`` as the length-rule band's metric. MAD's
macro token accuracy and the plain token accuracy are still computed and
logged for every run (``val/acc_mad`` / ``test/acc_mad`` /
``val/acc_token`` / ``test/acc_token``); no protocol reading uses them,
because the macro average counts token classes that are never targets, so
a few stray predictions move it far more than accuracy changes do.

Run with::

    pytest experiments/03_multistate_gdn2/02_structured_recall_pilot/test_mad_screening_metric.py -v
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PILOT = Path(__file__).resolve().parent
_PROJECT = _PILOT.parent
_SRC = _PROJECT.parents[1] / "src"
_CONF = _PILOT / "conf"
_TRAIN = _PILOT / "01_train.py"


def _resolve(task: str, *overrides: str) -> str:
    """The resolved Hydra config for ``task`` with the given overrides,
    exactly as ``01_train.py`` resolves it (same store, same resolver)."""
    for path in (_PILOT, _PROJECT, _SRC):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from common import conf_resolvers  # noqa: F401  (registers the resolver)
    from common.run_id import identity_exclude_keys
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from research_lib.training.callbacks import config_hash

    with initialize_config_dir(config_dir=str(_CONF), version_base=None):
        cfg = compose(config_name="config", overrides=[f"task={task}", *overrides])
    # ``output.runs_dir`` interpolates the hydra runtime, which is not set
    # under ``initialize_config_dir``; the ``output`` subtree is excluded
    # from the identity hash anyway (HASH_EXCLUDE), so a fixed value keeps
    # the resolution deterministic.
    cfg.output.runs_dir = "/tmp/unused-runs"
    return config_hash(
        OmegaConf.to_container(cfg, resolve=True),
        identity_exclude_keys(cfg),
    )


class TestHarnessAndProtocolKeys:
    def test_mad_harness_monitors_the_per_sequence_mean_accuracy(self):
        # The regression this change makes: the MAD harness's monitor key
        # is the per-sequence mean accuracy, not the macro accuracy.
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(_CONF), version_base=None):
            cfg = compose(config_name="config", overrides=["task=mad_fuzzy"])
        assert cfg.training.monitor_key == "val/acc"

    def test_entity_field_config_keeps_the_base_monitor_key(self):
        # The MAD harness change must not leak into other tasks: an
        # entity-field run (the Zoology harness) still monitors the base
        # ``val/acc`` key. This is the config-side guard for run identity:
        # the monitor key is part of the identity hash, so keeping it at
        # the base value for entity field is what keeps that task's config
        # hashes unchanged by this change (the enforcement test is
        # ``test_mad_monitor_key_changes_the_run_identity``).
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(_CONF), version_base=None):
            cfg = compose(config_name="config", overrides=["task=entity_field"])
        assert cfg.harness.name == "zoology"
        assert cfg.training.monitor_key == "val/acc"

    def test_mad_monitor_key_changes_the_run_identity(self):
        # ``training.monitor_key`` is part of the identity hash, so the
        # switch to the per-sequence reading changes every MAD run's
        # identity. That is expected and harmless here: the change only
        # re-keys runs that have not started a protocol tier yet, so no
        # existing record's identity is moved. The test pins that the
        # change is visible to the identity machinery.
        assert _resolve("mad_fuzzy") != _resolve(
            "mad_fuzzy", "training.monitor_key=val/acc_mad"
        )


def _run_smoke(launch: Path) -> None:
    """Run the CPU smoke config for the MAD task from ``launch``."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_SRC), str(_PROJECT), env.get("PYTHONPATH", "")) if p
    )
    result = subprocess.run(
        [sys.executable, str(_TRAIN), "--config-name=test", "task=mad_fuzzy"],
        cwd=launch,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert result.returncode == 0, result.stderr[-4000:]


@pytest.mark.slow
def test_mad_smoke_record_monitors_and_logs_all_three_accuracies(tmp_path):
    # A real (tiny CPU) MAD run: the run record's stopping block names
    # the per-sequence mean accuracy as the monitor, the test block holds
    # all three test accuracies (per-sequence, macro and token), and the
    # curves log the monitored value and both extra metrics per
    # validation.
    launch = tmp_path / "launch"
    launch.mkdir()
    _run_smoke(launch)

    records = list((launch / "runs" / "mad_fuzzy").glob("*.json"))
    assert len(records) == 1, f"expected one record, got {records}"
    record = json.loads(records[0].read_text())

    assert record["stopping"]["monitor"] == "val/acc"
    test_keys = set(record["test"])
    assert {"acc", "acc_mad", "acc_token"} <= test_keys
    # A record with scores instead of mappings would silently pass every
    # key assertion; require that at least the per-sequence reading is a
    # number.
    assert isinstance(record["test"]["acc"], (int, float))

    curves = [p for p in launch.rglob("*.csv") if "metrics" in p.name]
    assert len(curves) == 1, f"expected one curves file, got {curves}"
    with curves[0].open() as f:
        columns = set(next(csv.reader(f)))
    assert {"val/acc", "val/acc_mad", "val/acc_token"} <= columns
