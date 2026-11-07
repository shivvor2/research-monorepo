"""Copy run records from W&B artifacts into the local runs directory.

A thin wrapper over the shared ``03_pull.py`` machinery
(:func:`common.pull.main`): the same command-line interface and output
as the first pilot's script, with this pilot's expected run counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared project package and this pilot's own directory need to be
# importable: ``common`` adds the project folder, and argparse runs from
# any launch directory.
_PILOT = Path(__file__).resolve().parent
_PROJECT = _PILOT.parent
for _path in (_PROJECT, _PILOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common.pull import main as _pull_main  # noqa: E402
from structured_recall_analysis.protocol import EXPECTED_RUNS  # noqa: E402


def main() -> int:
    return _pull_main(
        expected_counts=EXPECTED_RUNS, conf_config_path=_PILOT / "conf" / "config.yaml"
    )


if __name__ == "__main__":
    sys.exit(main())
