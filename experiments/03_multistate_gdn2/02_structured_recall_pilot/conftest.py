"""Put the project folder, this pilot's folder and the library source on
``sys.path`` for the tests, the same way the scripts do when run from a
launch directory: the project folder (for ``common``), the pilot folder
(for ``structured_recall_analysis``) and the repository's ``src`` (for
``research_lib``)."""

import sys
from pathlib import Path

_PILOT = Path(__file__).resolve().parent
_PROJECT = _PILOT.parent
_SRC = _PROJECT.parents[1] / "src"
for _path in (_PILOT, _PROJECT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
