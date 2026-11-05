"""Put the project folder and the library source on ``sys.path`` for the
common package's tests, the same way the pilots' scripts and their
conftests do: the project folder (for ``common``) and the repository's
``src`` (for ``research_lib``)."""
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]  # experiments/03_multistate_gdn2
_SRC = _PROJECT.parents[1] / "src"
for _path in (_PROJECT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))