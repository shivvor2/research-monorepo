"""Atomic file helpers (torch checkpoints, run records).

:func:`atomic_torch_save` writes a checkpoint through a writer-unique
temporary file in the target directory and renames it into place, so a
partial file is never observed and two writers never race on one temp
file. It mirrors the run-summary writer
(:mod:`research_lib.training.callbacks.run_summary`).

Run tests with::

    pytest src/research_lib/utils/tests/test_io.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import torch


def atomic_torch_save(obj, path: Path | str) -> None:
    """Save ``obj`` with :func:`torch.save`, atomically.

    The object is written to a writer-unique temporary file in the same
    directory as ``path`` and moved into place with ``os.replace``, so a
    partial target file is never observed. If saving raises, the
    temporary file is removed and the target is left untouched.

    Args:
        obj: Anything :func:`torch.save` accepts.
        path: Target file path (parent directories are created).

    Raises:
        The exception raised by :func:`torch.save`, if any.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
