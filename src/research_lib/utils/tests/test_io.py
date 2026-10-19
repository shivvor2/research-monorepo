"""Tests for the atomic file helpers.

Target: src/research_lib/utils/io.py
Run with: pytest src/research_lib/utils/tests/test_io.py -v
"""

import json
import os
from pathlib import Path

import pytest
import torch

from research_lib.utils.io import atomic_torch_save, atomic_write


def record_fsyncs(monkeypatch):
    """Patch ``os.fsync`` to record its calls and ``os.open`` to record
    every opened path. Returns ``(fsynced_fds, opened_by_fd)``."""
    real_open = os.open
    opened_by_fd = {}

    def recording_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        opened_by_fd[fd] = Path(path)
        return fd

    monkeypatch.setattr(os, "open", recording_open)
    fsynced = []
    monkeypatch.setattr(os, "fsync", lambda fd: fsynced.append(fd))
    return fsynced, opened_by_fd


class TestAtomicTorchSave:
    """The target file appears only whole (temp file + rename), and a
    failing save leaves nothing behind."""

    def test_success_leaves_exactly_the_target(self, tmp_path):
        target = tmp_path / "model.pt"
        atomic_torch_save({"a": 1, "b": [2, 3]}, target)
        assert target.exists()
        assert torch.load(target, weights_only=False) == {"a": 1, "b": [2, 3]}
        # No temp files, no other files.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.pt"]

    def test_parent_directories_are_created(self, tmp_path):
        target = tmp_path / "nested" / "deeper" / "model.pt"
        atomic_torch_save({"a": 1}, target)
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

    def test_fsyncs_the_file_and_the_parent_directory(self, tmp_path, monkeypatch):
        """One save fsyncs the file before the rename and the parent
        directory after it, so the checkpoint is durable, not only
        never-partial."""
        real_save = torch.save
        saved_fds = []

        def recording_save(obj, f, *args, **kwargs):
            saved_fds.append(f.fileno())
            return real_save(obj, f, *args, **kwargs)

        monkeypatch.setattr(torch, "save", recording_save)
        fsynced, opened_by_fd = record_fsyncs(monkeypatch)
        target = tmp_path / "model.pt"

        atomic_torch_save({"a": 1}, target)

        assert len(fsynced) == 2
        assert fsynced[0] == saved_fds[0]  # the file, before the rename
        assert opened_by_fd[fsynced[1]] == target.parent  # the directory, after it


class TestAtomicWrite:
    """The shared writer: one write lands whole and never loses the
    rename to a power cut (fsync of the file and of the parent
    directory), and a failing write leaves no temporary file."""

    def test_text_write_round_trip(self, tmp_path):
        target = tmp_path / "record.json"
        atomic_write(target, lambda f: json.dump({"a": 1}, f), "w", encoding="utf-8")
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
        assert sorted(p.name for p in tmp_path.iterdir()) == ["record.json"]

    def test_binary_write(self, tmp_path):
        target = tmp_path / "record.bin"
        atomic_write(target, lambda f: f.write(b"x"), "wb")
        assert target.read_bytes() == b"x"

    def test_parent_directories_are_created(self, tmp_path):
        target = tmp_path / "nested" / "record.json"
        atomic_write(target, lambda f: f.write(b"x"), "wb")
        assert target.read_bytes() == b"x"

    def test_fsyncs_the_file_and_the_parent_directory(self, tmp_path, monkeypatch):
        fsynced, opened_by_fd = record_fsyncs(monkeypatch)
        saved_fds = []

        def write(f):
            saved_fds.append(f.fileno())
            f.write(b"payload")

        target = tmp_path / "record.json"
        atomic_write(target, write, "wb")

        assert len(fsynced) == 2
        assert fsynced[0] == saved_fds[0]  # the file, before the rename
        assert opened_by_fd[fsynced[1]] == target.parent  # the directory, after it

    def test_failing_write_leaves_nothing(self, tmp_path):
        target = tmp_path / "record.json"

        def boom(f):
            raise RuntimeError("write boom")

        with pytest.raises(RuntimeError, match="write boom"):
            atomic_write(target, boom, "wb")
        # No target, no temp file.
        assert list(tmp_path.iterdir()) == []

    def test_failing_write_keeps_a_previous_target(self, tmp_path):
        target = tmp_path / "record.json"
        atomic_write(target, lambda f: f.write(b"old"), "wb")
        before = target.read_bytes()

        def boom(f):
            raise RuntimeError("write boom")

        with pytest.raises(RuntimeError, match="write boom"):
            atomic_write(target, boom, "wb")
        # The previous target is untouched; no temp file remains.
        assert target.read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["record.json"]
