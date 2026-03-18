"""
Test suite for ExperimentStateCallback and checkpoint resolution utilities.

Run with: pytest src/research_lib/training/callbacks/tests/test_experiment_state.py -v
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ..experiment_state import (
    ExperimentStateCallback,
    _find_latest_named_checkpoint,
    _get_state_file_path,
    resolve_checkpoint_path,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_dir(tmp_path):
    """Base directory for state files (simulates original working directory)."""
    return tmp_path / "project_root"


@pytest.fixture
def base_dir_created(base_dir):
    """Base directory that actually exists on disk."""
    base_dir.mkdir()
    return base_dir


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Empty checkpoint directory."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    return ckpt_dir


@pytest.fixture
def checkpoint_dir_with_named(checkpoint_dir):
    """Checkpoint directory with named checkpoints at staggered mtimes.

    Files (oldest to newest by mtime):
        step_1000.ckpt
        step_2000.ckpt
        step_3000.ckpt  <-- should be selected as latest
        last.ckpt       <-- newest mtime but should be skipped
    """
    base_time = 1_700_000_000.0

    files_and_offsets = [
        ("step_1000.ckpt", 100),
        ("step_2000.ckpt", 200),
        ("step_3000.ckpt", 300),
        ("last.ckpt", 400),
    ]

    for name, offset in files_and_offsets:
        path = checkpoint_dir / name
        path.write_bytes(b"fake checkpoint data")
        mtime = base_time + offset
        os.utime(path, (mtime, mtime))

    return checkpoint_dir


@pytest.fixture
def checkpoint_dir_only_last(checkpoint_dir):
    """Checkpoint directory containing only last.ckpt."""
    path = checkpoint_dir / "last.ckpt"
    path.write_bytes(b"fake checkpoint data")
    return checkpoint_dir


def _make_mock_trainer(callbacks=None, status="running"):
    """Create a mock Lightning Trainer with configurable callbacks and status."""
    trainer = MagicMock()
    trainer.callbacks = callbacks or []
    trainer.state.status = status
    return trainer


def _make_mock_model_checkpoint(dirpath):
    from lightning.pytorch.callbacks import ModelCheckpoint

    mock = MagicMock(spec=ModelCheckpoint)
    # These are instance attrs set in __init__, spec doesn't expose them
    mock.dirpath = str(dirpath)
    mock.best_model_path = ""
    mock.last_model_path = ""
    return mock


# =============================================================================
# Tests: _get_state_file_path
# =============================================================================


class TestGetStateFilePath:
    """Tests for state file path construction."""

    def test_default_prefix(self, tmp_path):
        """Test state file path with default prefix."""
        result = _get_state_file_path("my_experiment", tmp_path)
        assert result == tmp_path / ".latest_ckpt_my_experiment.txt"

    def test_custom_prefix(self, tmp_path):
        """Test state file path with custom prefix."""
        result = _get_state_file_path("exp", tmp_path, prefix=".run_state_")
        assert result == tmp_path / ".run_state_exp.txt"

    def test_experiment_name_in_filename(self, tmp_path):
        """Test that experiment name is embedded in the filename."""
        path_a = _get_state_file_path("alpha", tmp_path)
        path_b = _get_state_file_path("beta", tmp_path)
        assert path_a != path_b
        assert "alpha" in path_a.name
        assert "beta" in path_b.name


# =============================================================================
# Tests: _find_latest_named_checkpoint
# =============================================================================


class TestFindLatestNamedCheckpoint:
    """Tests for checkpoint directory scanning."""

    def test_nonexistent_directory(self, tmp_path):
        """Test that nonexistent directory returns None."""
        result = _find_latest_named_checkpoint(tmp_path / "does_not_exist")
        assert result is None

    def test_empty_directory(self, checkpoint_dir):
        """Test that empty directory returns None."""
        result = _find_latest_named_checkpoint(checkpoint_dir)
        assert result is None

    def test_skips_last_ckpt_when_named_exist(self, checkpoint_dir_with_named):
        """Test that last.ckpt is skipped in favor of named checkpoints."""
        result = _find_latest_named_checkpoint(checkpoint_dir_with_named)
        assert result is not None
        assert result.name != "last.ckpt"

    def test_selects_most_recent_by_mtime(self, checkpoint_dir_with_named):
        """Test that the checkpoint with the newest mtime is selected."""
        result = _find_latest_named_checkpoint(checkpoint_dir_with_named)
        assert result.name == "step_3000.ckpt"

    def test_falls_back_to_last_ckpt(self, checkpoint_dir_only_last):
        """Test fallback to last.ckpt when no named checkpoints exist."""
        result = _find_latest_named_checkpoint(checkpoint_dir_only_last)
        assert result is not None
        assert result.name == "last.ckpt"

    def test_ignores_non_ckpt_files(self, checkpoint_dir):
        """Test that non-.ckpt files are ignored."""
        (checkpoint_dir / "metrics.csv").write_text("data")
        (checkpoint_dir / "config.yaml").write_text("data")

        result = _find_latest_named_checkpoint(checkpoint_dir)
        assert result is None

    def test_single_named_checkpoint(self, checkpoint_dir):
        """Test with exactly one named checkpoint."""
        path = checkpoint_dir / "step_500.ckpt"
        path.write_bytes(b"data")

        result = _find_latest_named_checkpoint(checkpoint_dir)
        assert result == path


# =============================================================================
# Tests: resolve_checkpoint_path
# =============================================================================


class TestResolveCheckpointPath:
    """Tests for checkpoint path resolution logic."""

    def test_none_returns_none(self, base_dir_created):
        """Test that resume_from=None always starts fresh."""
        result = resolve_checkpoint_path(
            resume_from=None,
            experiment_name="test",
            base_dir=base_dir_created,
        )
        assert result is None

    def test_explicit_path_exists(self, checkpoint_dir_with_named):
        """Test that an explicit existing path is returned as-is."""
        ckpt = checkpoint_dir_with_named / "step_1000.ckpt"

        result = resolve_checkpoint_path(
            resume_from=str(ckpt),
            experiment_name="test",
            base_dir=checkpoint_dir_with_named.parent,
        )
        assert result == str(ckpt)

    def test_explicit_path_missing_raises(self, base_dir_created):
        """Test that a missing explicit path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_checkpoint_path(
                resume_from="/nonexistent/path/model.ckpt",
                experiment_name="test",
                base_dir=base_dir_created,
            )

    def test_explicit_relative_path_resolved_against_base_dir(
        self, base_dir_created, checkpoint_dir_with_named
    ):
        """Test that relative explicit paths are resolved against base_dir."""
        # Place checkpoints inside base_dir
        ckpt_dir = base_dir_created / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_file = ckpt_dir / "step_500.ckpt"
        ckpt_file.write_bytes(b"data")

        result = resolve_checkpoint_path(
            resume_from="checkpoints/step_500.ckpt",
            experiment_name="test",
            base_dir=base_dir_created,
        )
        assert result == str(ckpt_file)

    def test_auto_no_state_file(self, base_dir_created):
        """Test auto-resume with no previous run history returns None."""
        result = resolve_checkpoint_path(
            resume_from="auto",
            experiment_name="test",
            base_dir=base_dir_created,
        )
        assert result is None

    def test_auto_state_file_points_to_directory(
        self, base_dir_created, checkpoint_dir_with_named
    ):
        """Test auto-resume when state file points to a checkpoint directory."""
        # Write state file pointing to checkpoint dir
        state_file = base_dir_created / ".latest_ckpt_test.txt"
        state_file.write_text(str(checkpoint_dir_with_named))

        result = resolve_checkpoint_path(
            resume_from="auto",
            experiment_name="test",
            base_dir=base_dir_created,
            # checkpoint_subdir not used since state file points directly to ckpt dir
        )
        assert result is not None
        assert "step_3000.ckpt" in result

    def test_auto_state_file_points_to_file(
        self, base_dir_created, checkpoint_dir_with_named
    ):
        """Test auto-resume when state file points directly to a checkpoint file."""
        target = checkpoint_dir_with_named / "step_2000.ckpt"

        state_file = base_dir_created / ".latest_ckpt_test.txt"
        state_file.write_text(str(target))

        result = resolve_checkpoint_path(
            resume_from="auto",
            experiment_name="test",
            base_dir=base_dir_created,
        )
        assert result == str(target)

    def test_auto_state_file_points_to_nonexistent(self, base_dir_created):
        """Test auto-resume when state file points to deleted path returns None."""
        state_file = base_dir_created / ".latest_ckpt_test.txt"
        state_file.write_text("/long/gone/checkpoints")

        result = resolve_checkpoint_path(
            resume_from="auto",
            experiment_name="test",
            base_dir=base_dir_created,
        )
        assert result is None

    def test_auto_state_file_points_to_empty_directory(
        self, base_dir_created, checkpoint_dir
    ):
        """Test auto-resume when checkpoint directory exists but is empty."""
        state_file = base_dir_created / ".latest_ckpt_test.txt"
        state_file.write_text(str(checkpoint_dir))

        result = resolve_checkpoint_path(
            resume_from="auto",
            experiment_name="test",
            base_dir=base_dir_created,
        )
        assert result is None

    def test_auto_legacy_run_dir_format(self, base_dir_created):
        """Test auto-resume with legacy format: state file stores run dir, not ckpt dir.

        Legacy state files store the Hydra run directory. The resolver should
        look inside run_dir / checkpoint_subdir for checkpoints.
        """
        # Simulate: run_dir/checkpoints/step_5000.ckpt
        run_dir = base_dir_created / "outputs" / "2024-01-01" / "12-00-00"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "step_5000.ckpt").write_bytes(b"data")

        state_file = base_dir_created / ".latest_ckpt_test.txt"
        state_file.write_text(str(run_dir))

        result = resolve_checkpoint_path(
            resume_from="auto",
            experiment_name="test",
            base_dir=base_dir_created,
            checkpoint_subdir="checkpoints",
        )
        assert result is not None
        assert "step_5000.ckpt" in result

    def test_custom_state_file_prefix(
        self, base_dir_created, checkpoint_dir_with_named
    ):
        """Test that custom state_file_prefix is respected."""
        state_file = base_dir_created / ".custom_prefix_test.txt"
        state_file.write_text(str(checkpoint_dir_with_named))

        result = resolve_checkpoint_path(
            resume_from="auto",
            experiment_name="test",
            base_dir=base_dir_created,
            state_file_prefix=".custom_prefix_",
        )
        assert result is not None

        # Default prefix should NOT find it
        result_default = resolve_checkpoint_path(
            resume_from="auto",
            experiment_name="test",
            base_dir=base_dir_created,
        )
        assert result_default is None


# =============================================================================
# Tests: ExperimentStateCallback
# =============================================================================


class TestExperimentStateCallback:
    """Tests for the Lightning callback."""

    def test_state_file_path(self, base_dir_created):
        """Test that state_file property returns the correct path."""
        cb = ExperimentStateCallback(
            experiment_name="my_exp",
            base_dir=base_dir_created,
        )
        assert cb.state_file == base_dir_created / ".latest_ckpt_my_exp.txt"

    def test_custom_prefix(self, base_dir_created):
        """Test callback with custom state file prefix."""
        cb = ExperimentStateCallback(
            experiment_name="my_exp",
            base_dir=base_dir_created,
            state_file_prefix=".run_",
        )
        assert cb.state_file.name == ".run_my_exp.txt"

    def test_on_save_checkpoint_writes_dirpath(
        self, base_dir_created, checkpoint_dir_with_named
    ):
        """Test that on_save_checkpoint writes ModelCheckpoint.dirpath to state file."""
        mock_ckpt_callback = _make_mock_model_checkpoint(checkpoint_dir_with_named)
        trainer = _make_mock_trainer(callbacks=[mock_ckpt_callback])

        cb = ExperimentStateCallback(
            experiment_name="test",
            base_dir=base_dir_created,
        )

        cb.on_save_checkpoint(trainer, MagicMock(), {})

        assert cb.state_file.exists()
        stored = cb.state_file.read_text().strip()
        assert stored == str(checkpoint_dir_with_named)

    def test_on_save_checkpoint_no_model_checkpoint_callback(self, base_dir_created):
        """Test fallback when no ModelCheckpoint callback is present."""
        trainer = _make_mock_trainer(callbacks=[])

        cb = ExperimentStateCallback(
            experiment_name="test",
            base_dir=base_dir_created,
        )

        # Should not raise, should write some fallback (or skip)
        cb.on_save_checkpoint(trainer, MagicMock(), {})

    def test_on_save_checkpoint_overwrites_previous(
        self, base_dir_created, checkpoint_dir
    ):
        """Test that successive saves overwrite the state file."""
        cb = ExperimentStateCallback(
            experiment_name="test",
            base_dir=base_dir_created,
        )

        # First save: points to dir_a
        dir_a = checkpoint_dir / "run_a"
        dir_a.mkdir()
        trainer_a = _make_mock_trainer(callbacks=[_make_mock_model_checkpoint(dir_a)])
        cb.on_save_checkpoint(trainer_a, MagicMock(), {})
        assert cb.state_file.read_text().strip() == str(dir_a)

        # Second save: should overwrite to dir_b
        dir_b = checkpoint_dir / "run_b"
        dir_b.mkdir()
        trainer_b = _make_mock_trainer(callbacks=[_make_mock_model_checkpoint(dir_b)])
        cb.on_save_checkpoint(trainer_b, MagicMock(), {})
        assert cb.state_file.read_text().strip() == str(dir_b)

    def test_on_train_end_finished_writes_marker_preserves_pointer(
        self, base_dir_created, checkpoint_dir
    ):
        """Test that on_train_end writes a completion marker without destroying the pointer."""
        cb = ExperimentStateCallback(
            experiment_name="test",
            base_dir=base_dir_created,
        )

        # Simulate a prior save
        cb.state_file.write_text(str(checkpoint_dir))

        trainer = _make_mock_trainer(status="finished")
        cb.on_train_end(trainer, MagicMock())

        # Original state file should still exist
        assert cb.state_file.exists()

        # Completion marker should also exist
        completed_file = cb.state_file.with_suffix(".completed.txt")
        assert completed_file.exists()

    def test_on_train_end_not_finished_no_marker(
        self, base_dir_created, checkpoint_dir
    ):
        """Test that on_train_end does nothing if training didn't finish cleanly."""
        cb = ExperimentStateCallback(
            experiment_name="test",
            base_dir=base_dir_created,
        )

        cb.state_file.write_text(str(checkpoint_dir))

        trainer = _make_mock_trainer(status="interrupted")
        cb.on_train_end(trainer, MagicMock())

        # No completion marker
        completed_file = cb.state_file.with_suffix(".completed.txt")
        assert not completed_file.exists()

        # Pointer still intact
        assert cb.state_file.exists()


# =============================================================================
# Tests: Round-Trip (Callback writes, Resolver reads)
# =============================================================================


class TestRoundTrip:
    """Integration tests: callback writes state, resolver reads it back."""

    def test_save_then_resolve(self, base_dir_created, checkpoint_dir_with_named):
        """Test that a checkpoint saved by the callback can be resolved."""
        # Callback writes state
        cb = ExperimentStateCallback(
            experiment_name="roundtrip",
            base_dir=base_dir_created,
        )
        mock_ckpt_cb = _make_mock_model_checkpoint(checkpoint_dir_with_named)
        trainer = _make_mock_trainer(callbacks=[mock_ckpt_cb])

        cb.on_save_checkpoint(trainer, MagicMock(), {})

        # Resolver reads it back
        result = resolve_checkpoint_path(
            resume_from="auto",
            experiment_name="roundtrip",
            base_dir=base_dir_created,
        )

        assert result is not None
        assert Path(result).exists()
        assert Path(result).name == "step_3000.ckpt"

    def test_save_complete_then_resolve(
        self, base_dir_created, checkpoint_dir_with_named
    ):
        """Test that auto-resume still works after training completes."""
        cb = ExperimentStateCallback(
            experiment_name="roundtrip",
            base_dir=base_dir_created,
        )

        # Save checkpoint
        mock_ckpt_cb = _make_mock_model_checkpoint(checkpoint_dir_with_named)
        trainer = _make_mock_trainer(callbacks=[mock_ckpt_cb], status="finished")
        cb.on_save_checkpoint(trainer, MagicMock(), {})

        # Training ends
        cb.on_train_end(trainer, MagicMock())

        # State file should still be readable
        result = resolve_checkpoint_path(
            resume_from="auto",
            experiment_name="roundtrip",
            base_dir=base_dir_created,
        )
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
