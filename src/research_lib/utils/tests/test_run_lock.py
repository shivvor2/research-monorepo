"""Tests for the exclusive run lock.

Target: src/research_lib/utils/run_lock.py
Run with: pytest src/research_lib/utils/tests/test_run_lock.py -v
"""

import errno

import pytest

from research_lib.utils.run_lock import exclusive_lock


class TestExclusiveLock:
    """A second acquisition through a separate open of the same file
    returns False while the first is held, and True after it is released."""

    def test_second_acquisition_fails_while_held(self, tmp_path):
        path = tmp_path / "run.lock"
        with exclusive_lock(path) as first:
            assert first is True
            with exclusive_lock(path) as second:
                assert second is False
        # Released: a fresh acquisition succeeds.
        with exclusive_lock(path) as third:
            assert third is True

    def test_lock_file_is_created_and_left_on_disk(self, tmp_path):
        path = tmp_path / "run.lock"
        with exclusive_lock(path):
            pass
        assert path.exists()
        assert path.read_bytes() == b""

    def test_parent_directories_are_created(self, tmp_path):
        path = tmp_path / "nested" / "run.lock"
        with exclusive_lock(path) as acquired:
            assert acquired is True
        assert path.exists()

    def test_release_happens_on_exception(self, tmp_path):
        path = tmp_path / "run.lock"
        with pytest.raises(RuntimeError, match="boom"):
            with exclusive_lock(path) as acquired:
                assert acquired is True
                raise RuntimeError("boom")
        with exclusive_lock(path) as again:
            assert again is True


class TestLockErrorsPropagate:
    """Only lock contention yields False; every other OSError from
    flock (ENOLCK / EOPNOTSUPP on a filesystem without working flock,
    EINTR, ...) propagates instead of silently skipping every run."""

    def test_enolck_propagates(self, tmp_path, monkeypatch):
        import fcntl

        def raise_enolck(*args, **kwargs):
            raise OSError(errno.ENOLCK, "no record locks available")

        monkeypatch.setattr(fcntl, "flock", raise_enolck)
        with (
            pytest.raises(OSError, match="no record locks"),
            exclusive_lock(tmp_path / "run.lock"),
        ):
            pass  # pragma: no cover - never reached

    def test_ewouldblock_still_yields_false(self, tmp_path, monkeypatch):
        import fcntl

        def raise_ewouldblock(*args, **kwargs):
            raise OSError(errno.EWOULDBLOCK, "resource temporarily unavailable")

        monkeypatch.setattr(fcntl, "flock", raise_ewouldblock)
        with exclusive_lock(tmp_path / "run.lock") as acquired:
            assert acquired is False
