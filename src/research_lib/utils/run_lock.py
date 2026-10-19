"""Exclusive run locks (``flock``).

:func:`exclusive_lock` guards one run's record against a second process
launching the same run: the lock is acquired before anything expensive
is built and held until the process exits. The lock file stays on disk
(empty); it is not a record, and analysis code reads only ``*.json``.

Run tests with::

    pytest src/research_lib/utils/tests/test_run_lock.py -v
"""

from __future__ import annotations

import errno
import fcntl
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_lock(path: Path | str) -> Generator[bool, None, None]:
    """Yield ``True`` if an exclusive ``flock`` on ``path`` was acquired,
    else ``False``.

    The lock is released when the context exits or the process dies
    (also on SIGKILL). Linux/POSIX only; per host: not reliable on
    network filesystems.

    Only lock **contention** yields ``False`` (``EWOULDBLOCK`` /
    ``EAGAIN`` from the non-blocking ``flock``): every other
    :class:`OSError` propagates, so a filesystem whose ``flock`` does not
    work at all (``ENOLCK`` / ``EOPNOTSUPP`` on some network or FUSE
    mounts) fails the run loudly instead of silently skipping it as
    "running in another process".

    Args:
        path: Lock file path (its parent is created; the file itself is
            created empty if missing and left on disk).

    Yields:
        bool: Whether the lock was acquired. When ``False``, another
        process holds it; the caller should skip the run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                raise
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
