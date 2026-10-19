"""Tests for the exclusive run lock.

Target: src/research_lib/utils/run_lock.py
Run with: pytest src/research_lib/utils/tests/test_run_lock.py -v
"""

import pytest

from research_lib.utils.run_lock import exclusive_lock


class TestExclusiveLock:
    """B09: a second acquisition through a separate open of the same file
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
