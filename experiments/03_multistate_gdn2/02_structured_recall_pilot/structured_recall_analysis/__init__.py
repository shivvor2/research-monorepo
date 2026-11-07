"""The structured-recall pilot's analysis package.

Built on the shared :mod:`common.analysis` pieces. The pilot's own
protocol constants live in :mod:`structured_recall_analysis.protocol`
(the screening cells, decision thresholds, expected counts); the
per-report reading modules are added in :mod:`.reports`. Each task
registers its screening cell (the metric key the reports read) in
``protocol.py``.
"""

from . import protocol

__all__ = ["protocol"]
