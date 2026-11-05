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