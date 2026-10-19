"""Atomic file helpers (torch checkpoints, run records).

:func:`atomic_write` writes any payload through a writer-unique
temporary file in the target directory and renames it into place, so a
partial file is never observed and two writers never race on one temp
file. It also fsyncs the file and the parent directory, so the payload
and the rename survive a power loss / kernel crash (the checkpoint is
promised to be fully written before the test pass; a truncated run
record would break the next relaunch). :func:`atomic_torch_save` and the
run-summary writer
(:mod:`research_lib.training.callbacks.run_summary`) both write through
it.

Run tests with::

    pytest src/research_lib/utils/tests/test_io.py -v
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch


def atomic_write(
    path: Path | str,
    write_fn: Callable[[Any], Any],
    mode: str,
    **open_kwargs: Any,
) -> None:
    """Write a file through a writer-unique temp name and rename it into
        place, durably.

        ``write_fn(f)`` writes the payload into the open file object ``f``
        (e.g. ``lambda f: torch.save(obj, f)`` or a ``json.dump`` closure).
        The temp file lives in the target's directory (so the rename stays
        on one filesystem), is flushed and fsynced before the rename, and
        the parent directory is fsynced after it, so a power loss cannot
        turn the target into an empty or truncated file or lose the rename.

        A writer-unique name (pid + uuid) keeps two writers (a relaunch
        racing the old process, two machines sharing the directory) off
    each other's temp file.

        Args:
            path: Target file path (its parent is created if missing).
            write_fn: Called with the open file object; its return value is
                discarded. Anything it raises propagates.
            mode: The :func:`open` mode (``"wb"`` for torch, text mode for
                JSON).
            **open_kwargs: Passed on to :func:`open` (e.g.
                ``encoding="utf-8"``).

        Raises:
            The exception raised by ``write_fn``, if any. The temporary file
            is removed and the target left untouched in that case.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, mode, **open_kwargs) as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    # The rename itself is a directory entry; fsync the parent so a
    # power loss cannot lose it. (No fd context manager: os.open needs
    # a close on the error path too.)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_torch_save(obj, path: Path | str) -> None:
    """Save ``obj`` with :func:`torch.save`, atomically and durably.

    The object is written to a writer-unique temporary file in the same
    directory as ``path`` and moved into place with ``os.replace``, so a
    partial target file is never observed and the save survives a power
    loss; both fsyncs go through :func:`atomic_write`. If saving raises,
    the temporary file is removed and the target is left untouched.

    Args:
        obj: Anything :func:`torch.save` accepts.
        path: Target file path (parent directories are created).

    Raises:
        The exception raised by :func:`torch.save`, if any.
    """
    atomic_write(path, lambda f: torch.save(obj, f), "wb")
