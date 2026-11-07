"""Regenerate the run-machinery fixtures (run IDs, identity hashes, W&B
kwargs and metadata, and the pull --check report).

Run this on a checkout where the first pilot's run machinery is in its
reference state (target: master before the shared-code move), then commit
the written fixture files unchanged. The pull fixtures were written once;
a regeneration is only ever a reference re-capture, never a fix.

Usage:
    python fixtures/make_run_machinery_fixture.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

_PILOT = Path(__file__).resolve().parents[1]
_SRC = _PILOT.parents[2] / "src"
for _p in (_PILOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERE = Path(__file__).resolve().parent

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


def load_train():
    spec = importlib.util.spec_from_file_location(
        "pilot_train_ref", _PILOT / "01_train.py"
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    return train


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


def pull_report(train, cfg_upsert, tmp, tier_filter):
    import importlib.util as _util

    spec = _util.spec_from_file_location("pilot_pull_ref", _PILOT / "03_pull.py")
    p = _util.module_from_spec(spec)
    spec.loader.exec_module(p)

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
    for run_id, (task, tier, hashes) in records.items():
        rec = {
            "run_id": run_id,
            "status": "test_done",
            "config_hash": hashes,
            "seed": 0,
            "config": {"task": {"name": task}, "experiment": {"tier": tier}},
        }
        root = write_artifact_root(tmp / f"src-{run_id}", task, run_id, rec)
        collections.append(
            FakeCollection(
                f"c-{run_id}",
                [FakeVersionedArtifact(root, train.run_record_metadata(rec))],
            )
        )

    runs = tmp / "runs"
    # one matching local (skip), one different hash (conflict)
    for run_id, (task, tier, hashes) in records.items():
        local_hash = hashes if tier == "tier2_gates" else "hashX"
        local = {
            "run_id": run_id,
            "status": "test_done",
            "config_hash": local_hash,
            "seed": 0,
            "config": {"task": {"name": task}, "experiment": {"tier": tier}},
        }
        pth = runs / task / f"{run_id}.json"
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_text(json.dumps(local))
    api = FakeApi(collections)
    listed = p.list_run_record_artifacts(api, "P", None, tier=tier_filter)
    summaries = p.pull(listed, runs, dry_run=True, tier_filter=tier_filter)
    return p.render_report(summaries, len(listed))


def main() -> None:
    train = load_train()

    # (a) run IDs and identity hashes.
    identity = {}
    for label, overrides in REPRESENTATIVES.items():
        cfg = compose(overrides)
        run_id = train.make_run_id(cfg, 0.003)
        cfg_dict = {k: v for k, v in cfg.items() if k != "hydra"}
        from omegaconf import OmegaConf

        container = OmegaConf.to_container(OmegaConf.create(cfg_dict), resolve=True)
        hash_ = train.config_hash(container, train.HASH_EXCLUDE)
        identity[label] = {"run_id": run_id, "config_hash": hash_}

    # (b) W&B logger kwargs and artifact metadata for one sample record.
    cfg = compose(
        {
            "task": "overwrite",
            "model.gate_mode": "full",
            "model.layout": "ws",
            "experiment.tier": "tier1_core",
        }
    )
    run_id = train.make_run_id(cfg, 0.003)
    kwargs = train.wandb_logger_kwargs(cfg, run_id)
    kwargs["config"] = None  # the resolved config is huge; not a run decision
    sample_record = {
        "run_id": run_id,
        "status": "test_done",
        "config_hash": identity["tier1_core_overwrite_full_ll"]["config_hash"],
        "seed": 0,
        "config": {
            "task": {"name": "overwrite"},
            "experiment": {"tier": "tier1_core"},
        },
        "git": {"commit": "beef", "dirty": False},
        "provenance": {"uv.lock": "deadbeef"},
    }
    sample = {
        "run_id": run_id,
        "wandb_logger_kwargs": kwargs,
        "artifact_metadata": train.run_record_metadata(sample_record),
    }

    # (c) the pull --check report, with and without a tier filter.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        all_report = pull_report(train, None, Path(tmp), None)
        tier_report = pull_report(train, None, Path(tmp), "tier3_frozen")
    HERE.joinpath("pull_report.txt").write_text(all_report + "\n", encoding="utf-8")
    HERE.joinpath("pull_report_tier3.txt").write_text(
        tier_report + "\n", encoding="utf-8"
    )

    HERE.joinpath("run_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    HERE.joinpath("wandb_sample.json").write_text(
        json.dumps(sample, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "wrote run_identity.json, wandb_sample.json, pull_report.txt, pull_report_tier3.txt"
    )


if __name__ == "__main__":
    main()
