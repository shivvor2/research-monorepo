"""Put the pilot directory and the library source on ``sys.path`` for
the tests, the same way ``python 01_train.py`` / ``python 02_analyze.py``
do when run from any launch directory: the scripts' own directory (for
``pilot_analysis``) and the repository's ``src`` (for ``research_lib``)."""

import sys
from pathlib import Path

_PILOT = Path(__file__).resolve().parent
_SRC = _PILOT.parents[2] / "src"
for _path in (_PILOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
