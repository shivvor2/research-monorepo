"""
Tests for the W&B run-record transport: 03_pull.py and the shared
wandb_artifacts helpers.

Target: experiments/03_multistate_gdn2/01_synthetic_pilot/03_pull.py
Run with: pytest experiments/03_multistate_gdn2/01_synthetic_pilot/test_03_pull.py -v

No network: the wandb public API is replaced by a fake that serves
artifact collections whose files live in temp directories, and
01_train.py's logger kwargs are tested on a composed config.
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from wandb_artifacts import artifact_name, run_record_metadata

from research_lib.utils.run_lock import exclusive_lock

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_ROOT = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("pilot_pull", _ROOT / "03_pull.py")
pull = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pull)

# 01_train.py for wandb_logger_kwargs (imports torch/lightning; heavy).
_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "pilot_train", _ROOT / "01_train.py"
)
train = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(train)

TIER = "tier2_gates"
TASK = "overwrite"
RUN_ID = "tier2_gates_overwrite-ws-full-mlp-d32-lr0.003-s7"


def _record(
    run_id=RUN_ID,
    task=TASK,
    *,
    hash_="abc123",
    seed=0,
    status="test_done",
    tier=TIER,
):
    """A run JSON like RunSummaryCallback writes (the fields the
    artifacts and the pull decisions read)."""
    return {
        "run_id": run_id,
        "status": status,
        "config_hash": hash_,
        "seed": seed,
        "config": {
            "task": {"name": task},
            "experiment": {"tier": tier},
        },
        "git": {"commit": "beef", "dirty": False},
        "provenance": {"uv.lock": "deadbeef"},
    }


def write_artifact_root(root, task, run_id, record, *, curves=True, checkpoint=False):
    """Write one artifact's files (the 01_train.py contract) under
    ``root`` and return the root."""
    dir_ = root / task
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{run_id}.json").write_text(json.dumps(record))
    if curves:
        curve_dir = dir_ / "curves" / run_id
        curve_dir.mkdir(parents=True, exist_ok=True)
        (curve_dir / "metrics.csv").write_text("step,val/acc\n0,0.5\n")
    if checkpoint:
        (dir_ / f"{run_id}.pt").write_text("weights")
    return root


class FakeVersionedArtifact:
    """A wandb public artifact: metadata plus a file tree served from a
    temp directory; ``download`` copies the tree under the requested
    root and honours ``path_prefix`` (the API the real artifacts
    expose). ``version`` is the W&B-style ``"vN"`` string."""

    def __init__(self, root, metadata, version="v1"):
        self._root = Path(root)
        self.metadata = metadata
        self.version = version
        self.downloaded = False
        self.requested_prefixes = []

    def download(self, root=None, path_prefix=None):
        target = Path(root) if root is not None else self._root
        self.downloaded = True
        self.requested_prefixes.append(path_prefix)
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
    """A fake collection: the artifact list is stored privately and
    served by the ``artifacts()`` method, as the real
    ``ArtifactCollection.artifacts`` does."""

    def __init__(self, name, artifacts):
        self.name = name
        self._artifacts = list(artifacts)

    def artifacts(self, per_page=50, start=None):
        return list(self._artifacts)


class FakeApi:
    """A fake ``wandb.Api``: ``artifact_collections`` serves the
    collections it was built with and records every call."""

    def __init__(self, collections):
        self.collections = list(collections)
        self.calls = []

    def artifact_collections(self, project_name, type_name):
        self.calls.append((project_name, type_name))
        assert type_name == "run_record"
        return self.collections


def make_artifact(tmp_path, record, *, curves=True, checkpoint=False, version=1):
    """An artifact whose files sit in a fresh temp directory."""
    root = write_artifact_root(
        tmp_path / f"src-{record['run_id']}",
        record["config"]["task"]["name"],
        record["run_id"],
        record,
        curves=curves,
        checkpoint=checkpoint,
    )
    return FakeVersionedArtifact(
        root, run_record_metadata(record), version=f"v{version}"
    )


class TestArtifactName:
    """The collection name of a run record: run-<run_id>, with
    characters outside [A-Za-z0-9._-] replaced by '-'."""

    def test_safe_run_ids_are_kept(self):
        assert artifact_name(RUN_ID) == f"run-{RUN_ID}"

    def test_unsafe_characters_raise(self):
        # Run IDs are generated; a '+' (e.g. a %g-rendered 1e+06) would
        # collapse distinct runs into one collection, so it is a bug.
        with pytest.raises(ValueError, match="outside"):
            artifact_name("x-lr1e+06-s0")

    def test_contract_with_the_training_script(self):
        # 01_train.py names its uploads through the same helper.
        import wandb_artifacts

        assert artifact_name is wandb_artifacts.artifact_name


class TestRunRecordMetadata:
    """The artifact metadata carries the record's identity so the pull
    can decide without downloading."""

    def test_fields(self):
        record = _record()
        metadata = run_record_metadata(record)
        assert metadata == {
            "run_id": RUN_ID,
            "tier": TIER,
            "task": TASK,
            "seed": 0,
            "config_hash": "abc123",
            "git_commit": "beef",
            "lock_hash": "deadbeef",
            "status": "test_done",
        }


class TestPullDecisions:
    """download / skip / conflict against the local record."""

    def test_missing_local_record_downloads(self):
        assert pull.pull_decision(None, {}) == "download"

    def test_local_not_done_downloads(self):
        local = _record(status="interrupted")
        assert (
            pull.pull_decision(local, {"config_hash": "abc123", "seed": 0})
            == "download"
        )

    def test_same_identity_skips(self):
        local = _record()
        assert pull.pull_decision(local, {"config_hash": "abc123", "seed": 0}) == "skip"

    def test_different_identity_conflicts(self):
        local = _record()
        assert (
            pull.pull_decision(local, {"config_hash": "other", "seed": 0}) == "conflict"
        )
        assert (
            pull.pull_decision(local, {"config_hash": "abc123", "seed": 1})
            == "conflict"
        )

    def test_unprovable_identity_conflicts(self):
        # A test_done local record and a metadata without identity
        # fields: never overwrite what we cannot prove is the same run.
        local = _record()
        assert (
            pull.pull_decision(local, {"config_hash": None, "seed": None}) == "conflict"
        )


class TestStagedRecordProblem:
    """The downloaded record must be the completed run the metadata
    claims before anything is installed."""

    def test_passing_record(self):
        metadata = run_record_metadata(_record())
        assert pull.staged_record_problem(_record(), metadata) is None

    def test_missing_record(self):
        assert pull.staged_record_problem(None, {}) is not None

    def test_not_done_record(self):
        problem = pull.staged_record_problem(_record(status="failed"), {})
        assert problem is not None and "not test_done" in problem

    @pytest.mark.parametrize(
        "field,kwargs",
        [
            ("run_id", {"run_id": "different"}),
            ("config_hash", {"hash_": "different"}),
            ("seed", {"seed": "different"}),
        ],
    )
    def test_record_identity_field_differs(self, field, kwargs):
        metadata = run_record_metadata(_record())
        record = _record(**kwargs)
        problem = pull.staged_record_problem(record, metadata)
        assert problem is not None and field in problem

    def test_record_task_differs(self):
        metadata = run_record_metadata(_record())
        record = _record(task="mqar")
        problem = pull.staged_record_problem(record, metadata)
        assert problem is not None and "task" in problem

    def test_record_tier_differs(self):
        metadata = run_record_metadata(_record())
        record = _record(tier="tier3_frozen")
        problem = pull.staged_record_problem(record, metadata)
        assert problem is not None and "tier" in problem


class TestVersionIndex:
    """The newest-version key parses the W&B ``"vN"`` version strings."""

    def test_examples(self):
        cases = [("v0", 0), ("v12", 12), (3, 3), (None, -1), ("latest", -1), ("v", -1)]
        for version, expected in cases:
            assert pull.version_index(version) == expected, version


class TestListing:
    """The latest version of every collection, with the tier filter."""

    def test_latest_version_per_collection(self, tmp_path):
        older = make_artifact(tmp_path, _record(), version=1)
        newer = make_artifact(tmp_path, _record(), version=2)
        collection = FakeCollection(f"run-{RUN_ID}", [older, newer])
        api = FakeApi([collection])
        artifacts = pull.list_run_record_artifacts(api, "P", None)
        assert artifacts == [newer]

    def test_v9_and_v10_order_by_number_not_lexicographically(self, tmp_path):
        v9 = make_artifact(tmp_path, _record(), version=9)
        v10 = make_artifact(tmp_path, _record(), version=10)
        # v10 is the newest and is picked regardless of list order.
        for order in ([v10, v9], [v9, v10]):
            api = FakeApi([FakeCollection(f"run-{RUN_ID}", order)])
            assert pull.list_run_record_artifacts(api, "P", None) == [v10]

    def test_tier_filter(self, tmp_path):
        artifact = make_artifact(tmp_path, _record())
        api = FakeApi([FakeCollection(f"run-{RUN_ID}", [artifact])])
        assert pull.list_run_record_artifacts(api, "P", None, tier=TIER) == [artifact]
        assert pull.list_run_record_artifacts(api, "P", None, tier="tier3_frozen") == []

    def test_foreign_collections_are_skipped(self, tmp_path):
        foreign = FakeVersionedArtifact(tmp_path / "foreign", {}, version=1)
        api = FakeApi([FakeCollection("some-other-artifact", [foreign])])
        assert pull.list_run_record_artifacts(api, "P", None) == []

    def test_entity_path_is_forwarded(self, tmp_path):
        collection = FakeCollection(
            f"run-{RUN_ID}", [make_artifact(tmp_path, _record())]
        )
        api = FakeApi([collection])
        pull.list_run_record_artifacts(api, "P", "my-entity")
        assert api.calls[-1][0] == "my-entity/P"
        pull.list_run_record_artifacts(api, "P", None)
        assert api.calls[-1][0] == "P"


class TestInstall:
    """Downloads land at the local shapes 01_train.py writes, atomically."""

    def test_record_and_curve_land_at_the_local_shapes(self, tmp_path):
        artifact = make_artifact(tmp_path, _record())
        runs_dir = tmp_path / "runs"
        written, skipped = pull.install_artifact(artifact, runs_dir, False)
        assert sorted(written) == sorted(
            [
                runs_dir / TASK / f"{RUN_ID}.json",
                runs_dir / TASK / "curves" / RUN_ID / "version_0" / "metrics.csv",
            ]
        )
        assert skipped == []
        assert artifact.downloaded
        assert (runs_dir / TASK / f"{RUN_ID}.json").read_text() == json.dumps(_record())

    def test_checkpoint_only_with_the_flag(self, tmp_path):
        artifact = make_artifact(tmp_path, _record(), checkpoint=True)
        runs_dir = tmp_path / "runs"
        pull.install_artifact(artifact, runs_dir, include_checkpoints=False)
        assert not (runs_dir / TASK / f"{RUN_ID}.pt").exists()
        pull.install_artifact(artifact, runs_dir, include_checkpoints=True)
        assert (runs_dir / TASK / f"{RUN_ID}.pt").read_text() == "weights"

    def test_checkpoint_not_downloaded_without_the_flag(self, tmp_path):
        artifact = make_artifact(tmp_path, _record(), checkpoint=True)
        runs_dir = tmp_path / "runs"
        pull.install_artifact(artifact, runs_dir, include_checkpoints=False)
        requested = artifact.requested_prefixes
        assert f"{TASK}/{RUN_ID}.pt" not in requested
        assert f"{TASK}/curves/{RUN_ID}/" in requested
        assert f"{TASK}/{RUN_ID}.json" in requested
        pull.install_artifact(artifact, runs_dir, include_checkpoints=True)
        assert f"{TASK}/{RUN_ID}.pt" in artifact.requested_prefixes

    def test_no_curve_artifact_writes_no_curve(self, tmp_path):
        artifact = make_artifact(tmp_path, _record(), curves=False)
        runs_dir = tmp_path / "runs"
        written, skipped = pull.install_artifact(artifact, runs_dir, False)
        assert not (runs_dir / TASK / "curves").exists()


class TestPull:
    """The end-to-end decisions, the check, and the per-tier report."""

    def test_new_record_is_pulled(self, tmp_path):
        artifact = make_artifact(tmp_path, _record())
        runs_dir = tmp_path / "runs"
        summaries = pull.pull([artifact], runs_dir)
        summary = summaries[TIER]
        assert summary.pulled == [RUN_ID]
        assert summary.conflicts == []
        assert summary.failures == []
        assert summary.missing == []
        record = json.loads((runs_dir / TASK / f"{RUN_ID}.json").read_text())
        assert record["status"] == "test_done"

    def test_same_identity_is_skipped_without_downloading(self, tmp_path):
        artifact = make_artifact(tmp_path, _record())
        runs_dir = tmp_path / "runs"
        target = runs_dir / TASK / f"{RUN_ID}.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(_record()))
        summaries = pull.pull([artifact], runs_dir)
        summary = summaries[TIER]
        assert summary.skipped == [RUN_ID]
        assert summary.pulled == []
        assert not artifact.downloaded

    def test_conflict_is_reported_and_never_overwritten(self, tmp_path):
        artifact = make_artifact(tmp_path, _record(hash_="artifact-hash"))
        runs_dir = tmp_path / "runs"
        target = runs_dir / TASK / f"{RUN_ID}.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(_record(hash_="local-hash")))
        summaries = pull.pull([artifact], runs_dir)
        summary = summaries[TIER]
        assert summary.conflicts == [(RUN_ID, "artifact-hash:0", "local-hash:0")]
        # Never overwritten, and not downloaded either.
        assert json.loads(target.read_text())["config_hash"] == "local-hash"
        assert not artifact.downloaded
        assert summary.missing == [RUN_ID]

    def test_local_record_not_done_is_replaced(self, tmp_path):
        artifact = make_artifact(tmp_path, _record())
        runs_dir = tmp_path / "runs"
        target = runs_dir / TASK / f"{RUN_ID}.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(_record(status="interrupted")))
        summaries = pull.pull([artifact], runs_dir)
        assert summaries[TIER].pulled == [RUN_ID]
        assert json.loads(target.read_text())["status"] == "test_done"

    def test_downloaded_record_must_parse_and_be_done(self, tmp_path):
        record = _record()
        root = write_artifact_root(tmp_path / "src", TASK, RUN_ID, record)
        (root / TASK / f"{RUN_ID}.json").write_text("{not json")
        artifact = FakeVersionedArtifact(root, run_record_metadata(record))
        runs_dir = tmp_path / "runs"
        # A pre-existing local record that is not done: the broken
        # artifact must not be installed and must leave it untouched.
        target = runs_dir / TASK / f"{RUN_ID}.json"
        target.parent.mkdir(parents=True)
        local_bytes = b'{"status": "interrupted"}'
        target.write_bytes(local_bytes)
        summaries = pull.pull([artifact], runs_dir)
        summary = summaries[TIER]
        assert summary.pulled == []
        assert len(summary.failures) == 1
        assert summary.failures[0][0] == RUN_ID
        assert "unparseable" in summary.failures[0][1]
        assert summary.missing == [RUN_ID]
        assert target.read_bytes() == local_bytes

    @pytest.mark.parametrize("field", ["config_hash", "run_id", "task"])
    def test_record_identity_mismatch_writes_nothing(self, tmp_path, field):
        # The artifact's staged record is not the run the metadata
        # claims, so installing would put the wrong record under a run's
        # name. Nothing is installed. A metadata run_id/task mismatch
        # also points the fetch at paths the artifact does not carry, so
        # only the metadata-consistency case names the field.
        record = _record()
        metadata = run_record_metadata(record)
        metadata[field] = "claimed-by-metadata"
        root = write_artifact_root(tmp_path / "src", TASK, RUN_ID, record)
        artifact = FakeVersionedArtifact(root, metadata)
        runs_dir = tmp_path / "runs"
        summaries = pull.pull([artifact], runs_dir)
        summary = summaries[TIER]
        assert summary.pulled == []
        assert len(summary.failures) == 1
        assert summary.failures[0][0] == str(metadata["run_id"])
        assert not (runs_dir / TASK / f"{RUN_ID}.json").exists()
        assert summary.missing == [summary.failures[0][0]]

    def test_record_is_written_after_the_curves(self, tmp_path, monkeypatch):
        # The record JSON is installed last: if a companion (the curves)
        # fails to copy, no record is written, so the next pull retries.
        artifact = make_artifact(tmp_path, _record())
        runs_dir = tmp_path / "runs"
        real_copy = pull.atomic_copy

        def fail_curves_copy(src, target):
            if target.name == "metrics.csv":
                raise OSError("disk full")
            return real_copy(src, target)

        monkeypatch.setattr(pull, "atomic_copy", fail_curves_copy)
        summaries = pull.pull([artifact], runs_dir)
        summary = summaries[TIER]
        assert summary.pulled == []
        assert len(summary.failures) == 1
        assert not (runs_dir / TASK / f"{RUN_ID}.json").exists()
        assert summary.missing == [RUN_ID]

    def test_check_reports_without_downloading(self, tmp_path):
        artifact = make_artifact(tmp_path, _record())
        runs_dir = tmp_path / "runs"
        summaries = pull.pull([artifact], runs_dir, dry_run=True)
        summary = summaries[TIER]
        assert summary.would_pull == [RUN_ID]
        assert summary.pulled == []
        assert not artifact.downloaded
        assert not (runs_dir / TASK).exists()

    def test_check_leaves_no_lock_file(self, tmp_path):
        artifact = make_artifact(tmp_path, _record())
        runs_dir = tmp_path / "runs"
        pull.pull([artifact], runs_dir, dry_run=True)
        assert not (runs_dir / TASK / f"{RUN_ID}.lock").exists()

    def test_run_lock_held_by_training_fails_without_writing(self, tmp_path):
        # Training holds runs/<task>/<run_id>.lock for the whole run; a
        # pull on the same machine must not overwrite it or race it.
        artifact = make_artifact(tmp_path, _record())
        runs_dir = tmp_path / "runs"
        lock_path = runs_dir / TASK / f"{RUN_ID}.lock"
        with exclusive_lock(lock_path):
            summaries = pull.pull([artifact], runs_dir)
            summary = summaries[TIER]
            assert summary.pulled == []
            assert summary.failures == [(RUN_ID, "run lock held by another process")]
            assert summary.missing == [RUN_ID]
            assert not (runs_dir / TASK / f"{RUN_ID}.json").exists()

    def test_tier_filter(self, tmp_path):
        other_tier = "tier3_frozen"
        artifact_a = make_artifact(tmp_path, _record())
        artifact_b = make_artifact(
            tmp_path,
            _record(
                run_id=f"{other_tier}_mqar-ll-full-mlp-d32-lr0.003-s0",
                task="mqar",
                tier=other_tier,
            ),
        )
        runs_dir = tmp_path / "runs"
        summaries = pull.pull(
            [artifact_a, artifact_b], runs_dir, tier_filter=other_tier
        )
        assert set(summaries) == {other_tier}
        assert summaries[other_tier].pulled == [artifact_b.metadata["run_id"]]

    def test_expected_counts_are_reported(self, tmp_path):
        artifact = make_artifact(tmp_path, _record())
        runs_dir = tmp_path / "runs"
        summaries = pull.pull([artifact], runs_dir)
        summary = summaries[TIER]
        assert summary.expected == 60  # tier2_gates in the protocol
        assert summary.local_done == 1

    def test_report_lists_what_is_still_missing(self, tmp_path):
        artifact = make_artifact(tmp_path, _record(hash_="artifact-hash"))
        runs_dir = tmp_path / "runs"
        target = runs_dir / TASK / f"{RUN_ID}.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(_record(hash_="local-hash")))
        summaries = pull.pull([artifact], runs_dir)
        report = pull.render_report(summaries, 1)
        assert "1 run_record artifacts on W&B" in report
        assert f"local test_done: 1 / 60 expected" in report
        assert "CONFLICTS (1, never overwritten)" in report
        assert RUN_ID in report
        assert "still missing locally: 1" in report


class TestDefaults:
    """The --entity/--project defaults come from conf/config.yaml."""

    def test_defaults_from_config(self):
        entity, project = pull.default_entity_project()
        assert project == "MultistateGDN2"
        # The config's entity is empty (the personal default applies).
        assert entity is None


class TestWandbLoggerKwargs:
    """One W&B run per pilot run, named and tagged from the config."""

    def _cfg(self, **extra):
        cfg = {
            "experiment": {
                "name": "multistate_gdn2_synthetic_pilot",
                "phase": 1,
                "tier": TIER,
            },
            "task": {"name": TASK},
            "model": {
                "layout": "ws",
                "gate_mode": "full",
                "lambda_mode": "mlp",
                "d_model": 32,
            },
            "logging": {"wandb": {"project": "MultistateGDN2", "entity": None}},
        }
        cfg.update(extra)
        return OmegaConf.create(cfg)

    def test_composed_config(self):
        cfg = self._cfg()
        kwargs = train.wandb_logger_kwargs(cfg, RUN_ID)
        assert kwargs == {
            "project": "MultistateGDN2",
            "entity": None,
            "name": RUN_ID,
            "group": TIER,
            "job_type": "phase1",
            "tags": [
                "multistate_gdn2_synthetic_pilot",
                TASK,
                "ws",
                "full",
                "mlp",
                "d32",
            ],
            "config": OmegaConf.to_container(cfg, resolve=True),
            "save_dir": "logs/wandb",
            "log_model": False,
        }

    def test_tierless_run_groups_under_none(self):
        cfg = self._cfg(**{"experiment": {"name": "x", "phase": 2, "tier": None}})
        kwargs = train.wandb_logger_kwargs(cfg, "run-2")
        assert kwargs["group"] == "none"
        assert kwargs["job_type"] == "phase2"

    def test_entity_is_forwarded(self):
        cfg = self._cfg(**{"logging": {"wandb": {"project": "P", "entity": "mark"}}})
        assert train.wandb_logger_kwargs(cfg, "run-1")["entity"] == "mark"
