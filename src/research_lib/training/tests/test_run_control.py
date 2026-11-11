"""
Tests for the general run-control helpers: precision flags, the fit guard,
shard membership, the skip decision and the run claim.

Target: src/research_lib/training/run_control.py
Run with: pytest src/research_lib/training/tests/test_run_control.py -v
"""

import json
from types import SimpleNamespace

import pytest

import research_lib.training.run_control as rc
from research_lib.utils.run_lock import exclusive_lock


def _record(*, status="test_done", hash_="abc", seed=0):
    return {"status": status, "config_hash": hash_, "seed": seed}


class TestDisableTf32:
    def test_both_flags_are_off_and_read_back(self):
        flags = rc.disable_tf32()
        assert flags["cuda_matmul_allow_tf32"] is False
        assert flags["cudnn_allow_tf32"] is False


class TestLauncherRanks:
    """``lightning_launched_rank``: only Lightning's own launcher is
    recognised, so a plain single-device run (including inside a
    one-task Slurm allocation, which sets ``SLURM_PROCID`` for every
    ``sbatch`` task) is rank 0, while a multi-task or unparseable Slurm
    allocation, ``torchrun``'s ``RANK``, and a ``LOCAL_RANK`` without
    the other Lightning variables are all refused."""

    def _clear(self, monkeypatch):
        for name in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "WORLD_SIZE", "MASTER_PORT"):
            monkeypatch.delenv(name, raising=False)

    def test_no_launcher_variables_is_rank_zero(self, monkeypatch):
        self._clear(monkeypatch)
        assert rc.lightning_launched_rank() == 0

    def test_one_task_slurm_allocation_is_rank_zero(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("SLURM_PROCID", "0")
        monkeypatch.setenv("SLURM_NTASKS", "1")
        assert rc.lightning_launched_rank() == 0

    def test_multi_task_slurm_allocation_is_refused(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("SLURM_PROCID", "1")
        monkeypatch.setenv("SLURM_NTASKS", "2")
        with pytest.raises(RuntimeError, match="Slurm"):
            rc.lightning_launched_rank()

    def test_slurm_without_a_task_count_is_refused(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("SLURM_PROCID", "0")
        with pytest.raises(RuntimeError, match="Slurm"):
            rc.lightning_launched_rank()

    def test_slurm_unparseable_task_count_is_refused(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("SLURM_PROCID", "0")
        monkeypatch.setenv("SLURM_NTASKS", "many")
        with pytest.raises(RuntimeError, match="Slurm"):
            rc.lightning_launched_rank()

    def test_torchrun_rank_is_refused(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        with pytest.raises(RuntimeError, match="unsupported launcher"):
            rc.lightning_launched_rank()

    def test_local_rank_without_master_port_is_refused(self, monkeypatch):
        """A ``LOCAL_RANK`` without the launcher's ``MASTER_PORT`` is not a
        Lightning child, so it must not bypass the launch gate."""
        self._clear(monkeypatch)
        monkeypatch.setenv("LOCAL_RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "2")
        with pytest.raises(RuntimeError, match="MASTER_PORT"):
            rc.lightning_launched_rank()

    def test_lightning_child_rank_returns_its_rank(self, monkeypatch):
        """A child rank Lightning started (``LOCAL_RANK`` with
        ``MASTER_PORT``) is recognised and returned."""
        self._clear(monkeypatch)
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("MASTER_PORT", "29500")
        assert rc.lightning_launched_rank() == 1

    def test_child_rank_after_lightning_teardown_is_recognised(self, monkeypatch):
        """A ``LOCAL_RANK`` with ``MASTER_PORT`` but no ``WORLD_SIZE`` is
        still a Lightning child: Lightning's cluster environment removes
        ``WORLD_SIZE`` in its post-fit teardown, so the gate that runs
        after ``fit`` (e.g. the final-checkpoint save) must not require
        it."""
        self._clear(monkeypatch)
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("MASTER_PORT", "29500")
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        assert rc.lightning_launched_rank() == 1

    def test_multi_device_under_slurm(self):
        """``refuse_multi_device_under_slurm`` refuses a device count other
        than 1 only when ``SLURM_JOB_ID`` is set."""
        import os

        old = os.environ.get("SLURM_JOB_ID")
        os.environ["SLURM_JOB_ID"] = "123"
        try:
            rc.refuse_multi_device_under_slurm(1)  # single device passes
            with pytest.raises(RuntimeError, match="Slurm"):
                rc.refuse_multi_device_under_slurm(2)
        finally:
            if old is None:
                os.environ.pop("SLURM_JOB_ID", None)
            else:
                os.environ["SLURM_JOB_ID"] = old
        # Outside an allocation a device count other than 1 is allowed.
        rc.refuse_multi_device_under_slurm(2)


class TestInShard:
    def test_membership(self):
        assert rc.in_shard(0, 0, 2) is True
        assert rc.in_shard(1, 0, 2) is False
        assert rc.in_shard(1, 1, 2) is True
        assert rc.in_shard(3, 0, 2) is False  # 3 % 2 == 1, not 0
        assert rc.in_shard(3, 1, 2) is True

    def test_single_shard_runs_every_job(self):
        assert rc.in_shard(7, 0, 1) is True

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            rc.in_shard(0, 0, 0)
        with pytest.raises(ValueError):
            rc.in_shard(0, 2, 1)


class TestShouldSkip:
    def test_missing_record_runs(self, tmp_path):
        assert rc.should_skip(tmp_path / "nope.json", ("abc", 0)) is None

    def test_done_with_matching_identity_skips(self, tmp_path):
        path = tmp_path / "r.json"
        path.write_text(json.dumps(_record()))
        assert rc.should_skip(path, ("abc", 0)) == "already test_done"

    def test_done_with_different_identity_runs(self, tmp_path):
        path = tmp_path / "r.json"
        path.write_text(json.dumps(_record(hash_="abc", seed=0)))
        # Same hash different seed, and different hash:
        assert rc.should_skip(path, ("abc", 1)) is None
        assert rc.should_skip(path, ("xyz", 0)) is None

    def test_non_done_record_runs(self, tmp_path):
        path = tmp_path / "r.json"
        path.write_text(json.dumps(_record(status="failed")))
        assert rc.should_skip(path, ("abc", 0)) is None

    def test_unreadable_record_runs(self, tmp_path):
        path = tmp_path / "r.json"
        path.write_text("not json")
        assert rc.should_skip(path, ("abc", 0)) is None

    def test_done_status_constant_is_shared(self):
        assert rc.RUN_DONE_STATUS == "test_done"


class TestClaimRun:
    def test_no_record_claims_and_runs(self, tmp_path):
        lock = tmp_path / "r.lock"
        with rc.claim_run(lock, None, ("abc", 0)) as reason:
            assert reason is None

    def test_done_matching_record_is_skipped_after_lock(self, tmp_path):
        lock = tmp_path / "r.lock"
        record = tmp_path / "r.json"
        record.write_text(json.dumps(_record()))
        with rc.claim_run(lock, record, ("abc", 0)) as reason:
            assert reason == "already test_done"

    def test_lock_held_reports_running_in_another_process(self, tmp_path):
        lock = tmp_path / "r.lock"
        with exclusive_lock(lock) as acquired:
            assert acquired
            with rc.claim_run(lock, None, ("abc", 0)) as reason:
                assert reason == "running in another process"


class TestAssertFitFinished:
    def test_finished_passes(self):
        from lightning.pytorch.trainer.states import TrainerStatus

        trainer = SimpleNamespace(state=SimpleNamespace(status=TrainerStatus.FINISHED))
        rc.assert_fit_finished(trainer)  # no raise

    def test_not_finished_raises(self):
        from lightning.pytorch.trainer.states import TrainerStatus

        trainer = SimpleNamespace(
            state=SimpleNamespace(status=TrainerStatus.INTERRUPTED)
        )
        with pytest.raises(RuntimeError):
            rc.assert_fit_finished(trainer)
