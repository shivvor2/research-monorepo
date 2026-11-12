"""
End-to-end tests for multi-device training runs of the structured-recall
pilot.

The train script is run as a subprocess (its Lightning-launched child
ranks re-execute the same script), from a fresh launch directory, exactly
as a user would launch it.  These tests cover:

* a fixed-set (MAD-style) run on two CPU processes versus one: same
  final global step, the same LR schedule, final weights equal within fp
  tolerance, the same early-stop decision, exactly one run record;
* the same for a streaming (MQAR) run;
* the launch gate: a two-process run holds the lock once, a relaunch
  skips it without starting child ranks, and rank-0-only decisions (STOP
  file, shard exclusion) and unsupported launchers behave correctly;
* Hydra ``--multirun`` with two jobs on two devices each: exactly two
  records, each job's child ranks running that job's configuration.

Run with: pytest experiments/03_multistate_gdn2/02_structured_recall_pilot/test_ddp_e2e.py -v
(marked slow: each test launches real training subprocesses)
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_PILOT = Path(__file__).resolve().parent
_PROJECT = _PILOT.parent
_SRC = _PROJECT.parents[1] / "src"
_TRAIN = _PILOT / "01_train.py"


def _run_train(
    cwd: Path,
    *overrides: str,
    config: str = "test",
    env_extra: dict | None = None,
):
    """Run the pilot's train script from ``cwd`` with the overrides.

    Args:
        cwd: The launch directory.
        *overrides: Hydra overrides (a ``--multirun`` flag and sweep-file
            ``+sweep=...`` go here, before any key=value overrides).
        config: The ``--config-name`` (``None`` to use the default
            ``config``).
        env_extra: Extra environment variables for the subprocess.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_SRC), str(_PROJECT), env.get("PYTHONPATH", "")) if p
    )
    env.update(env_extra or {})
    cmd = [sys.executable, str(_TRAIN)]
    if config is not None:
        cmd.append(f"--config-name={config}")
    cmd.extend(overrides)
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )


def _one_record(cwd: Path, task: str) -> dict:
    """The single run record under ``cwd/runs/<task>/``."""
    records = list((cwd / "runs" / task).glob("*.json"))
    assert len(records) == 1, f"expected one record, got {[r.name for r in records]}"
    with open(records[0]) as f:
        return json.load(f)


def _one_curves_csv(cwd: Path) -> Path:
    csvs = list(cwd.glob("**/*.csv"))
    assert len(csvs) == 1, f"expected one curves file, got {csvs}"
    return csvs[0]


def _final_checkpoint(cwd: Path, task: str) -> dict:
    pts = list((cwd / "runs" / task).glob("*.pt"))
    assert len(pts) == 1, f"expected one checkpoint, got {[p.name for p in pts]}"
    return torch.load(pts[0], weights_only=True)


def _lr_columns(curves: Path) -> list[tuple[str, str]]:
    """The per-step LR values (both optimizer groups) of a curves file.

    Raises:
        AssertionError: The LR monitor's first-optimizer column is missing
            or holds no value, so a comparison of two empty column lists
            could not pass.
    """
    with open(curves) as f:
        rows = list(csv.DictReader(f))
    first = [r.get("lr-AdamW", "") for r in rows]
    assert any(column.strip() for column in first), (
        "the LR-monitor column (lr-AdamW) is missing or empty in "
        f"{curves}; a rename of its keys would let this comparison pass "
        "vacuously"
    )
    return [(first[i], r.get("lr-AdamW-1", "")) for i, r in enumerate(rows)]


def _max_abs_diff(weights_a: dict, weights_b: dict) -> float:
    assert set(weights_a) == set(weights_b)
    return max(
        (weights_a[key] - weights_b[key]).abs().max().item() for key in weights_a
    )


def _assert_same_run(record_a: dict, record_b: dict) -> None:
    """Everything a run's identity and outcome claim must agree between a
    single-device and a multi-device run of the same configuration."""
    assert record_a["config_hash"] == record_b["config_hash"]
    assert record_a["run_id"] == record_b["run_id"]
    assert record_a["status"] == record_b["status"] == "test_done"
    assert (
        record_a["timing"]["steps_completed"] == record_b["timing"]["steps_completed"]
    )
    assert record_a["stopping"] == record_b["stopping"]
    for key in record_a["test"]:
        assert record_a["test"][key] == pytest.approx(record_b["test"][key], abs=1e-4)


@pytest.mark.slow
def test_fixed_set_run_two_devices_matches_single_device(tmp_path):
    """A tiny MAD-style run on two CPU processes versus one: same final
    global step, the same LR at every step, final weights equal within fp
    tolerance, the same (early-stop) decision on every rank, and exactly
    one run record per run."""
    overrides = (
        "task=mad_fuzzy",
        "training.devices={devices}",
        "training.accelerator=cpu",
        "output.save_final_checkpoint=true",
        "training.early_stop_threshold=0.0",  # reached at the first validation
    )
    runs = {}
    for label, devices in (("one", 1), ("two", 2)):
        launch = tmp_path / label
        launch.mkdir()
        result = _run_train(
            launch,
            overrides[0],
            *(o.format(devices=devices) for o in overrides[1:]),
        )
        assert result.returncode == 0, result.stderr[-4000:]
        if devices == 2:
            assert "Lightning child rank" in result.stdout
        runs[label] = launch

    record_one = _one_record(runs["one"], "mad_fuzzy")
    record_two = _one_record(runs["two"], "mad_fuzzy")
    _assert_same_run(record_one, record_two)
    # The threshold was chosen so the run EARLY STOPS: both the single-
    # device run and every rank of the multi-device run took the same
    # decision at the same step.
    assert record_one["stopping"]["early_stopped"] is True
    # The monitored-value history (the early-stopped metric at each
    # validation) is identical on 1 and 2 devices.
    assert set(record_one["monitored"]) == set(record_two["monitored"])
    for key in record_one["monitored"]:
        assert record_one["monitored"][key] == pytest.approx(
            record_two["monitored"][key], abs=1e-6
        )

    weights_one = _final_checkpoint(runs["one"], "mad_fuzzy")
    weights_two = _final_checkpoint(runs["two"], "mad_fuzzy")
    assert _max_abs_diff(weights_one, weights_two) < 1e-4

    lr_one = _lr_columns(_one_curves_csv(runs["one"]))
    lr_two = _lr_columns(_one_curves_csv(runs["two"]))
    assert lr_one == lr_two


@pytest.mark.slow
def test_streaming_run_two_devices_matches_single_device(tmp_path):
    """The same comparison for a tiny streaming (MQAR) run: the training
    stream, the schedule, the final weights and the test metrics do not
    depend on the device count."""
    overrides = (
        "task=mqar",
        "training.devices={devices}",
        "training.accelerator=cpu",
        "output.save_final_checkpoint=true",
    )
    launches = {}
    for label, devices in (("one", 1), ("two", 2)):
        launch = tmp_path / label
        launch.mkdir()
        result = _run_train(
            launch,
            overrides[0],
            *(o.format(devices=devices) for o in overrides[1:]),
        )
        assert result.returncode == 0, result.stderr[-4000:]
        launches[label] = launch

    record_one = _one_record(launches["one"], "mqar")
    record_two = _one_record(launches["two"], "mqar")
    _assert_same_run(record_one, record_two)

    weights_one = _final_checkpoint(launches["one"], "mqar")
    weights_two = _final_checkpoint(launches["two"], "mqar")
    assert _max_abs_diff(weights_one, weights_two) < 1e-4

    lr_one = _lr_columns(_one_curves_csv(launches["one"]))
    lr_two = _lr_columns(_one_curves_csv(launches["two"]))
    assert lr_one == lr_two


@pytest.mark.slow
def test_launch_gate_single_machine_decisions(tmp_path):
    """The launch gate: a two-process run holds the lock once and
    completes; relaunching the finished run skips it without starting
    child ranks; a STOP file or a shard exclusion stops the run before
    any child rank exists; an independent-rank launch (simulated
    torchrun) is refused at start-up."""
    overrides = (
        "task=mqar",
        "training.devices=2",
        "training.accelerator=cpu",
        # a protocol tier, so the skip-completed rule (which applies to
        # protocol tiers only) is active on the relaunch below
        "experiment.tier=calib_mad_single",
    )
    launch = tmp_path / "two"
    launch.mkdir()
    first = _run_train(launch, *overrides)
    assert first.returncode == 0, first.stderr[-4000:]
    record = _one_record(launch, "mqar")
    locks = list((launch / "runs" / "mqar").glob("*.lock"))
    assert len(locks) == 1
    # The fresh two-process run started its child rank...
    assert "Lightning child rank" in first.stdout

    # ... a relaunch skips the finished run on rank 0, so no child ranks
    # are ever started (no second fit, no child-rank lines).
    relaunch = _run_train(launch, *overrides)
    assert relaunch.returncode == 0, relaunch.stderr[-4000:]
    assert "already test_done" in relaunch.stdout
    assert "Lightning child rank" not in relaunch.stdout
    assert _one_record(launch, "mqar") == record  # untouched (one record)

    # STOP file: rank 0 stops the job before fit; no child ranks, no run.
    stopped = tmp_path / "stopped"
    stopped.mkdir()
    (stopped / "STOP").write_text("")
    result = _run_train(stopped, *overrides)
    assert result.returncode == 0
    assert "STOP file found" in result.stdout
    assert "Lightning child rank" not in result.stdout
    assert not list((stopped / "runs" / "mqar").glob("*.json"))

    # Shard exclusion: this single run is job 0, not in shard 1 of 2.
    sharded = tmp_path / "sharded"
    sharded.mkdir()
    result = _run_train(
        sharded, *overrides, "launch.num_shards=2", "launch.shard_index=1"
    )
    assert result.returncode == 0
    assert "not in shard" in result.stdout
    assert "Lightning child rank" not in result.stdout
    assert not list((sharded / "runs" / "mqar").glob("*.json"))

    # An independent-rank launch (torchrun sets RANK and LOCAL_RANK) is
    # refused at start-up, before any machine or data is touched.
    refused = tmp_path / "refused"
    refused.mkdir()
    result = _run_train(
        refused,
        "task=mqar",
        "training.devices=1",
        "training.accelerator=cpu",
        env_extra={"RANK": "0", "LOCAL_RANK": "0"},
    )
    assert result.returncode != 0
    assert "unsupported launcher" in (result.stdout + result.stderr)
    assert not list((refused / "runs" / "mqar").glob("*.json"))


@pytest.mark.slow
def test_multirun_with_several_devices_is_refused(tmp_path):
    """A Hydra multirun with ``training.devices`` other than 1 is refused
    by rank 0 before fit: Lightning's subprocess launcher re-executes the
    whole sweep in each child, so a multi-device sweep would train
    different jobs in the same process group whenever rank 0's gate skips
    a job (a shard exclusion, a completed record, a STOP file).  The
    refused run exits non-zero with the error, writes no record, starts
    no child (no child-rank line) and leaves no second sweep directory."""
    launch = tmp_path / "sweep"
    launch.mkdir()
    result = _run_train(
        launch,
        "--multirun",
        "task=mqar",
        "seed=0,1",
        "training.devices=2",
        "training.accelerator=cpu",
    )
    assert result.returncode != 0
    assert "multirun" in (result.stdout + result.stderr)
    assert "not supported" in (result.stdout + result.stderr)
    assert not list((launch / "runs" / "mqar").glob("*.json"))
    assert "Lightning child rank" not in result.stdout
    # No child sweep (children would each re-run the whole sweep and add
    # their own timestamped directory): exactly one sweep root.
    assert len(list((launch / "multirun").rglob("multirun.yaml"))) == 1


@pytest.mark.slow
def test_single_device_multirun_writes_both_records(tmp_path):
    """The same two-job sweep with one device per job (the supported
    configuration: a sweep runs one device per job) writes both
    records."""
    launch = tmp_path / "sweep"
    launch.mkdir()
    result = _run_train(
        launch,
        "--multirun",
        "task=mqar",
        "seed=0,1",
        "training.devices=1",
        "training.accelerator=cpu",
    )
    assert result.returncode == 0, result.stderr[-4000:]
    records = list((launch / "runs" / "mqar").glob("*.json"))
    assert len(records) == 2, f"expected two records, got {[r.name for r in records]}"
    seeds = set()
    for record_path in records:
        with open(record_path) as f:
            seeds.add(json.load(f)["seed"])
    assert seeds == {0, 1}


@pytest.mark.slow
def test_sweep_file_task_comes_from_the_sweeper(tmp_path):
    """A multirun launched as the pilots launch one: ``+sweep=<file>``
    with the task in the sweeper params (not the command line).  One
    record per job, the task and seed coming from the params.  This
    runs single-device: any sweep runs one device per job (a multirun
    with ``training.devices`` other than 1 is refused), so the
    device-count machinery is exercised by the single-run multi-device
    tests."""
    launch = tmp_path / "sweep"
    launch.mkdir()
    result = _run_train(launch, "--multirun", "+sweep=test_ddp_multirun", config=None)
    assert result.returncode == 0, result.stderr[-4000:]
    records = list((launch / "runs" / "mad_fuzzy").glob("*.json"))
    assert len(records) == 2, f"expected two records, got {[r.name for r in records]}"
    seeds = set()
    tasks = set()
    statuses = set()
    for record_path in records:
        with open(record_path) as f:
            record = json.load(f)
        seeds.add(record["seed"])
        tasks.add(record["config"]["task"]["name"])
        statuses.add(record["status"])
    assert seeds == {0, 1}
    assert tasks == {"mad_fuzzy"}  # the task came from the sweeper params
    assert statuses == {"test_done"}
