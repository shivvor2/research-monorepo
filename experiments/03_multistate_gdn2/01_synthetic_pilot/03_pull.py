"""Copy run records from W&B artifacts into the local runs directory.

Rented machines upload every completed run — its JSON record, its
training curve and optionally its final checkpoint — as a W&B artifact
of type ``run_record`` named ``run-<run_id>``. This script pulls those
artifacts into the canonical local ``runs/`` store that the analysis
reads, without Hydra.

Usage:
    python 03_pull.py                           # pull everything
    python 03_pull.py --tier tier2_gates        # one tier
    python 03_pull.py --check                   # report only, no download
    python 03_pull.py --include-checkpoints     # also the final checkpoints

The decision per artifact, against the local record at
``<runs-dir>/<task>/<run_id>.json``:

- missing -> download it (into a temp dir first, then atomically);
- ``test_done`` with the same ``(config_hash, seed)`` -> skip;
- ``test_done`` with a different identity -> **conflict**: reported,
  never overwritten;
- any other status -> replaced by the artifact's record.

A download is installed only after its staged record is validated
against the artifact's metadata, and the record JSON is the last file
moved into place. A real install takes the run's lock (the same file
training holds); ``--check`` takes no lock and writes nothing. The
per-tier report counts the local ``test_done`` records against the
protocol's expected counts and lists the run IDs still missing locally.
The exit code is non-zero on any conflict (1) or check failure (2).

The run is the shared project machinery: this file is a thin wiring of
:func:`common.pull.main` to this pilot's expected counts and
``conf/config.yaml``; the decisions, install and report live in
:mod:`research_lib.training.run_records`. ``--entity`` and ``--project``
default to ``conf/config.yaml``'s ``logging.wandb`` block.
"""

import sys
from pathlib import Path

# The project folder (for ``common``) and this pilot's own folder (for
# ``pilot_analysis``) are added the way the other scripts add them.
_PILOT = Path(__file__).resolve().parent
_PROJECT = _PILOT.parent
for _path in (_PROJECT, _PILOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common.pull import main  # noqa: E402
from pilot_analysis.protocol import EXPECTED_RUNS  # noqa: E402


def run() -> None:
    """Wire this pilot's expected counts and config into the shared pull."""
    sys.exit(
        main(
            expected_counts=EXPECTED_RUNS,
            conf_config_path=_PILOT / "conf" / "config.yaml",
        )
    )


if __name__ == "__main__":
    run()
