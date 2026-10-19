"""Tests for the atomic file helpers.

Target: src/research_lib/utils/io.py
Run with: pytest src/research_lib/utils/tests/test_io.py -v
"""

import pytest
import torch

from research_lib.utils.io import atomic_torch_save


class TestAtomicTorchSave:
    """B06: the target file appears only whole (temp file + rename), and
    a failing save leaves nothing behind."""

    def test_success_leaves_exactly_the_target(self, tmp_path):
        target = tmp_path / "model.pt"
        atomic_torch_save({"a": 1, "b": [2, 3]}, target)
        assert target.exists()
        assert torch.load(target) == {"a": 1, "b": [2, 3]}
        # No temp files, no other files.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.pt"]

    def test_parent_directories_are_created(self, tmp_path):
        target = tmp_path / "nested" / "deeper" / "model.pt"
        atomic_torch_save(torch.zeros(2, 2), target)
        assert target.exists()
        assert target.parent.is_dir()

    def test_failing_save_leaves_nothing(self, tmp_path, monkeypatch):
        target = tmp_path / "model.pt"

        def boom(*args, **kwargs):
            raise RuntimeError("save boom")

        monkeypatch.setattr(torch, "save", boom)
        with pytest.raises(RuntimeError, match="save boom"):
            atomic_torch_save({"a": 1}, target)
        # No target, no temp file.
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_failing_save_keeps_a_previous_target(self, tmp_path, monkeypatch):
        target = tmp_path / "model.pt"
        atomic_torch_save({"old": True}, target)
        before = target.read_bytes()

        def boom(*args, **kwargs):
            raise RuntimeError("save boom")

        monkeypatch.setattr(torch, "save", boom)
        with pytest.raises(RuntimeError, match="save boom"):
            atomic_torch_save({"new": True}, target)
        # The previous target is untouched; no temp file remains.
        assert target.read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.pt"]
