"""Test-only helpers for the run-machinery fixtures.

The regression test computes the run IDs, identity hashes, W&B kwargs /
metadata and the pull report itself, through the migrated code, and
compares them against the committed fixture files. This module holds what
it needs: the representative configs, the small fake W&B API, and the
migrated pull path.

This is deliberately independent of ``make_run_machinery_fixture.py``,
which is reference-only: that script produced the committed files on the
``b3d162e`` checkout and must never run against the migrated code.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_PILOT = Path(__file__).resolve().parents[1]
_PROJECT = _PILOT.parent
_SRC = _PILOT.parents[2] / "src"
for _p in (_PILOT, _PROJECT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# A representative configuration per dimension the run ID and identity
# hash depend on: tier prefixes, gate / layout / lambda, non-default
# decay scheme, layer pattern, state mixer, expand_v and the anchor
# width. lr is passed as an argument (the base ID embeds it), not
# resolved from the frozen table.
REPRESENTATIVES = {
    "calib_lr_overwrite_kda_single": {
        "task": "overwrite",
        "model.gate_mode": "kda",
        "model.layout": "single",
        "experiment.tier": "calib_lr",
    },
    "calib_lr_mqar_full_ws": {
        "task": "mqar",
        "model.gate_mode": "full",
        "model.layout": "ws",
        "experiment.tier": "calib_lr",
    },
    "calib_harness_mqar_kda_ll": {
        "task": "mqar",
        "model.gate_mode": "kda",
        "model.layout": "ll",
        "experiment.tier": "calib_harness",
    },
    "tier1_core_overwrite_full_ll": {
        "task": "overwrite",
        "model.gate_mode": "full",
        "model.layout": "ll",
        "experiment.tier": "tier1_core",
    },
    "tier2_gates_overwrite_b_only_single": {
        "task": "overwrite",
        "model.gate_mode": "b_only",
        "model.layout": "single",
        "experiment.tier": "tier2_gates",
    },
    "tier3_frozen_mqar_full_fs": {
        "task": "mqar",
        "model.gate_mode": "full",
        "model.layout": "fs",
        "experiment.tier": "tier3_frozen",
    },
    "tier4_lambda_overwrite_kda_ws_static": {
        "task": "overwrite",
        "model.gate_mode": "kda",
        "model.layout": "ws",
        "model.lambda_mode": "static",
        "experiment.tier": "tier4_lambda",
    },
    "tier5_matched_overwrite_kda_single_ev10": {
        "task": "overwrite",
        "model.gate_mode": "kda",
        "model.layout": "single",
        "model.expand_v": "10.0",
        "experiment.tier": "tier5_matched_bytes",
    },
    "tier6_anchor_mqar_gdn_single_d32_1head": {
        "task": "mqar",
        "model.gate_mode": "gdn",
        "model.layout": "single",
        "model.d_model": "32",
        "model.num_heads": "1",
        "experiment.tier": "tier6_anchors",
    },
    "tier_longT_mqar_full_ll": {
        "task": "mqar",
        "model.gate_mode": "full",
        "model.layout": "ll",
        "experiment.tier": "tier_longT",
    },
    "r1_overwrite_kda_ws_uniform": {
        "task": "overwrite",
        "model.gate_mode": "kda",
        "model.layout": "ws",
        "optimizer.decay_scheme": "uniform",
        "experiment.tier": "R1",
    },
    "calib_lr_overwrite_full_single_pat_mlp_sm": {
        "task": "overwrite",
        "model.gate_mode": "full",
        "model.layout": "single",
        "model.layer_pattern": "mixer_x2",
        "model.state_mixer": "mlp",
        "experiment.tier": "calib_harness",
    },
}


def compose(overrides):
    from hydra import compose, initialize_config_dir

    # Neutralise the ${hydra:...} interpolations so the compose API can
    # resolve the config without a live HydraConfig; the affected keys are
    # excluded from the identity hash anyway.
    overrides = [
        "hydra.run.dir=outputs",
        "output.runs_dir=runs",
        "logging.csv.save_dir=runs/curves",
    ] + [f"{k}={v}" for k, v in overrides.items()]
    with initialize_config_dir(config_dir=str(_PILOT / "conf"), version_base=None):
        return compose(config_name="config", overrides=overrides)


def write_artifact_root(root, task, run_id, record, *, curves=True):
    root = Path(root)
    dir_ = root / task
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{run_id}.json").write_text(json.dumps(record))
    if curves:
        curve = dir_ / "curves" / run_id
        curve.mkdir(parents=True, exist_ok=True)
        (curve / "metrics.csv").write_text("step,val/acc\n0,0.5\n")
    return root


class FakeVersionedArtifact:
    version_field = "v1"

    def __init__(self, root, metadata, version="v1"):
        self._root = Path(root)
        self.metadata = metadata
        self.version = version

    def download(self, root=None, path_prefix=None):
        target = Path(root) if root is not None else self._root
        for src in self._root.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(self._root)
            if path_prefix is not None and not str(rel).startswith(path_prefix):
                continue
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        return str(target)


class FakeCollection:
    def __init__(self, name, artifacts):
        self.name = name
        self._artifacts = list(artifacts)

    def artifacts(self, per_page=50, start=None):
        return list(self._artifacts)


class FakeApi:
    def __init__(self, collections):
        self.collections = list(collections)

    def artifact_collections(self, project_name, type_name):
        return self.collections


def _record_of(run_id, task, tier, hash_):
    return {
        "run_id": run_id,
        "status": "test_done",
        "config_hash": hash_,
        "seed": 0,
        "config": {"task": {"name": task}, "experiment": {"tier": tier}},
    }


def pull_report(train, cfg_upsert, tmp, tier_filter):
    """The pull --check report through the migrated pull (the functions
    ``03_pull.main`` calls), wired to this project's layout and expected
    counts; the fixture-trained local records produce the skip/conflict
    report the committed files hold."""
    from common.layout import (
        LAYOUT,
        group_of,
        group_of_record,
        metadata_identity_fields,
        record_metadata,
    )
    from pilot_analysis.protocol import EXPECTED_RUNS

    from research_lib.training.run_records import (
        list_run_record_artifacts,
        pull,
        render_report,
    )

    records = {
        "tier2_gates_overwrite-ws-full-mlp-d32-lr0.003-s0": (
            "overwrite",
            "tier2_gates",
            "hashA",
        ),
        "tier3_frozen_mqar-ll-full-mlp-d32-lr0.003-s1": (
            "mqar",
            "tier3_frozen",
            "hashB",
        ),
    }
    collections = []
    for run_id, (task, tier, hash_) in records.items():
        rec = _record_of(run_id, task, tier, hash_)
        root = write_artifact_root(tmp / f"src-{run_id}", task, run_id, rec)
        collections.append(
            FakeCollection(
                f"c-{run_id}",
                [FakeVersionedArtifact(root, record_metadata(rec))],
            )
        )

    runs = Path(tmp) / "runs"
    # one matching local (skip), one different hash (conflict)
    for run_id, (task, tier, hash_) in records.items():
        local_hash = hash_ if tier == "tier2_gates" else "hashX"
        local = _record_of(run_id, task, tier, local_hash)
        pth = runs / task / f"{run_id}.json"
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_text(json.dumps(local))
    api = FakeApi(collections)
    listed = list_run_record_artifacts(
        api, "P", None, group=tier_filter, group_of=group_of
    )
    summaries = pull(
        listed,
        runs,
        LAYOUT,
        metadata_identity_fields(),
        group_of=group_of,
        group_of_record=group_of_record,
        expected_counts=EXPECTED_RUNS,
        dry_run=True,
        group_filter=tier_filter,
    )
    return render_report(summaries, len(listed))
