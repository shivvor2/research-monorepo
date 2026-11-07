"""
Tests for the general run-record machinery: artifact names, metadata,
pull decisions, and the convention-free pull over a toy layout.

Target: src/research_lib/training/run_records.py
Run with: pytest src/research_lib/training/tests/test_run_records.py -v

The pull is exercised with a flat ``runs/<run_id>.json`` toy layout and a
grouping by seed parity, over a fake W&B-like API. It must download,
skip and report conflicts with no reference to any project convention.
"""

import json
import shutil
from pathlib import Path

import pytest

from research_lib.training.run_records import (
    ARTIFACT_TYPE,
    artifact_name,
    atomic_copy,
    identity_of,
    list_run_record_artifacts,
    pull,
    pull_decision,
    render_report,
    run_record_metadata,
    staged_record_problem,
    version_index,
)
from research_lib.utils.run_lock import exclusive_lock

# --------------------------------------------------------------------------- #
# A toy layout: flat runs/<run_id>.json, a curve next to it, no checkpoint.
# --------------------------------------------------------------------------- #


class ToyLayout:
    def record_path(self, metadata):
        return f"{metadata['run_id']}.json"

    def curve_path(self, metadata):
        return f"curves/{metadata['run_id']}/metrics.csv"

    def checkpoint_path(self, metadata):
        return None

    def local_target(self, rel, metadata, include_checkpoints):
        run_id = metadata["run_id"]
        if rel == Path(f"{run_id}.json"):
            return "record", Path(f"{run_id}.json")
        if rel == Path(f"curves/{run_id}/metrics.csv"):
            return "curves", Path(f"curves/{run_id}/metrics.csv")
        return None


TOY = ToyLayout()
# Identity fields the toy records carry at the top level.
TOY_IDENTITY = {
    "run_id": lambda r: r.get("run_id"),
    "config_hash": lambda r: r.get("config_hash"),
    "seed": lambda r: r.get("seed"),
}


def toy_group_of(metadata):
    return "even" if int(metadata["seed"]) % 2 == 0 else "odd"


def toy_group_of_record(record):
    # The record's group lives in its config, a different place from the
    # metadata's (which reads the metadata ``seed``).
    return (record.get("config") or {}).get("group") or "none"


def _record(seed, hash_="h", status="test_done"):
    return {
        "run_id": f"run-{seed}",
        "config_hash": hash_,
        "seed": seed,
        "status": status,
        "config": {"group": "even" if seed % 2 == 0 else "odd"},
    }


class FakeArtifact:
    """Serves a record + curve already written into a source tree."""

    def __init__(self, run_id, seed, hash_, version="v1", source=None):
        self._run_id, self._seed, self._hash = run_id, seed, hash_
        self.version = version
        self._source = source

    @property
    def metadata(self):
        return {"run_id": self._run_id, "seed": self._seed, "config_hash": self._hash}

    def download(self, root, path_prefix=None):
        target = Path(root)
        for src in Path(self._source).rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(Path(self._source))
            if path_prefix is not None and not str(rel).startswith(path_prefix):
                continue
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        return str(target)


class FakeCollection:
    def __init__(self, artifacts):
        self._artifacts = artifacts

    def artifacts(self):
        return list(self._artifacts)


class FakeAPI:
    def __init__(self, collections):
        self._collections = collections

    def artifact_collections(self, project_path, type_name):
        yield from self._collections


def _write_source(tmp, record, with_curve=True):
    """A source tree laid out per the toy layout."""
    src = tmp / "src"
    src.mkdir(exist_ok=True)
    (src / f"{record['run_id']}.json").write_text(json.dumps(record))
    if with_curve:
        curve = src / "curves" / record["run_id"]
        curve.mkdir(parents=True, exist_ok=True)
        (curve / "metrics.csv").write_text("step,val/acc\n0,0.5\n")
    return src


# --------------------------------------------------------------------------- #
# Individual helpers
# --------------------------------------------------------------------------- #


class TestArtifactName:
    def test_safe_run_id(self):
        assert artifact_name("a-b_c.1") == "run-a-b_c.1"

    def test_unsafe_run_id_raises(self):
        with pytest.raises(ValueError):
            artifact_name("a/b")

    def test_type_constant(self):
        assert ARTIFACT_TYPE == "run_record"


class TestVersionIndex:
    def test_parsing(self):
        assert version_index("v3") == 3
        assert version_index(4) == 4
        assert version_index(None) == -1
        assert version_index("v10") > version_index("v9")


class TestRunRecordMetadata:
    def test_extra_fields_are_added(self):
        record = {
            "run_id": "r",
            "seed": 7,
            "config_hash": "h",
            "git": {"commit": "c"},
            "provenance": {"uv.lock": "l"},
            "status": "test_done",
            "config": {"task": {"name": "t"}, "extra": {"k": "v"}},
        }
        meta = run_record_metadata(record, {"task": "task.name", "extra": "extra.k"})
        assert meta["task"] == "t"
        assert meta["extra"] == "v"
        assert meta["status"] == "test_done"

    def test_missing_config_handled(self):
        assert run_record_metadata({"run_id": "r"}, {})["run_id"] == "r"


class TestPullDecision:
    def test_no_local_downloads(self):
        assert pull_decision(None, {"config_hash": "h", "seed": 0}) == "download"

    def test_non_done_local_downloads(self):
        local = {"status": "failed", "config_hash": "h", "seed": 0}
        assert pull_decision(local, {"config_hash": "h", "seed": 0}) == "download"

    def test_matching_done_skips(self):
        local = {"status": "test_done", "config_hash": "h", "seed": 0}
        assert pull_decision(local, {"config_hash": "h", "seed": 0}) == "skip"

    def test_differing_identity_conflicts(self):
        local = {"status": "test_done", "config_hash": "h", "seed": 0}
        assert pull_decision(local, {"config_hash": "h", "seed": 1}) == "conflict"


class TestStagedRecordProblem:
    def test_matching_record_passes(self):
        record = _record(0)
        assert staged_record_problem(record, record, TOY_IDENTITY) is None

    def test_mismatch_reports(self):
        record = _record(0, hash_="a")
        bad = _record(0, hash_="b")
        assert staged_record_problem(record, bad, TOY_IDENTITY) is not None

    def test_non_done_record_rejected(self):
        record = _record(0, status="failed")
        assert staged_record_problem(record, record, TOY_IDENTITY) is not None

    def test_missing_record_rejected(self):
        assert staged_record_problem(None, {}, TOY_IDENTITY) is not None


# --------------------------------------------------------------------------- #
# The convention-free pull (S04)
# --------------------------------------------------------------------------- #


class TestGeneralPull:
    def test_downloads_missing_and_reports_counts(self, tmp_path):
        runs = tmp_path / "runs"
        src = _write_source(tmp_path, _record(0))
        art = FakeArtifact("run-0", 0, "h")
        art._source = src
        summaries = pull(
            [art],
            runs,
            TOY,
            TOY_IDENTITY,
            group_of=toy_group_of,
            group_of_record=toy_group_of_record,
            expected_counts={"even": 1},
        )
        s = summaries["even"]
        assert s.pulled == ["run-0"]
        assert (runs / "run-0.json").exists()
        assert (runs / "curves" / "run-0" / "metrics.csv").exists()
        assert s.local_done == 1

    def test_skips_matching_local_without_download(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / "run-0.json").write_text(json.dumps(_record(0)))
        art = FakeArtifact("run-0", 0, "h")
        art._source = _write_source(tmp_path, _record(0))
        # A download would overwrite; make the source record a different hash
        # so an (incorrect) download would change the file.
        (art._source / "run-0.json").write_text(json.dumps(_record(0, hash_="WRONG")))
        summaries = pull(
            [art],
            runs,
            TOY,
            TOY_IDENTITY,
            group_of=toy_group_of,
            group_of_record=toy_group_of_record,
        )
        s = summaries["even"]
        assert s.skipped == ["run-0"]
        assert json.loads((runs / "run-0.json").read_text())["config_hash"] == "h"

    def test_conflict_is_reported_not_overwritten(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / "run-0.json").write_text(json.dumps(_record(0, hash_="local")))
        art = FakeArtifact("run-0", 0, "h")
        art._source = _write_source(tmp_path, _record(0, hash_="h"))
        summaries = pull(
            [art],
            runs,
            TOY,
            TOY_IDENTITY,
            group_of=toy_group_of,
            group_of_record=toy_group_of_record,
        )
        s = summaries["even"]
        assert s.conflicts and len(s.conflicts) == 1
        assert json.loads((runs / "run-0.json").read_text())["config_hash"] == "local"

    def test_dry_run_writes_nothing(self, tmp_path):
        runs = tmp_path / "runs"
        art = FakeArtifact("run-0", 0, "h")
        art._source = _write_source(tmp_path, _record(0))
        summaries = pull(
            [art],
            runs,
            TOY,
            TOY_IDENTITY,
            group_of=toy_group_of,
            group_of_record=toy_group_of_record,
            dry_run=True,
        )
        s = summaries["even"]
        assert s.would_pull == ["run-0"]
        assert not (runs / "run-0.json").exists()

    def test_grouping_by_seed_parity_partitions(self, tmp_path):
        runs = tmp_path / "runs"
        even = FakeArtifact("run-0", 0, "h")
        even._source = _write_source(tmp_path, _record(0))
        odd = FakeArtifact("run-1", 1, "h")
        odd._source = _write_source(tmp_path, _record(1))
        summaries = pull(
            [even, odd],
            runs,
            TOY,
            TOY_IDENTITY,
            group_of=toy_group_of,
            group_of_record=toy_group_of_record,
        )
        assert set(summaries) == {"even", "odd"}
        assert summaries["even"].pulled == ["run-0"]
        assert summaries["odd"].pulled == ["run-1"]

    def test_local_counts_use_the_record_not_the_metadata(self, tmp_path):
        # A local record whose group (config.group) differs from what the
        # metadata's seed would imply must be counted under the record's
        # group, not folded into the metadata-derived one.
        runs = tmp_path / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        local = {
            "run_id": "run-0",
            "config_hash": "h",
            "seed": 0,  # metadata parity would say "even"
            "status": "test_done",
            "config": {"group": "odd"},  # record itself says "odd"
        }
        (runs / "run-0.json").write_text(json.dumps(local))
        art = FakeArtifact("run-0", 0, "h")
        art._source = _write_source(tmp_path, _record(0))
        summaries = pull(
            [art],
            runs,
            TOY,
            TOY_IDENTITY,
            group_of=toy_group_of,
            group_of_record=toy_group_of_record,
            dry_run=True,
        )
        assert summaries["odd"].local_done == 1
        assert set(summaries) == {"odd", "even"}

    def test_listing_filters_by_group(self, tmp_path):
        even = FakeArtifact("run-0", 0, "h")
        even._source = _write_source(tmp_path, _record(0))
        odd = FakeArtifact("run-1", 1, "h")
        odd._source = _write_source(tmp_path, _record(1))
        api = FakeAPI([FakeCollection([even, odd])])
        listed = list_run_record_artifacts(
            api, "P", None, group="even", group_of=toy_group_of
        )
        assert [a.metadata["run_id"] for a in listed] == ["run-0"]

    def test_listing_requires_group_of_when_filtering(self, tmp_path):
        api = FakeAPI([])
        with pytest.raises(ValueError):
            list_run_record_artifacts(api, "P", None, group="even", group_of=None)

    def test_no_project_conventions_in_general_sources(self):
        """The general modules never reference groups' or tasks' nouns."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        for name in (
            "run_control.py",
            "run_records.py",
            "record_table.py",
            "wandb_transport.py",
        ):
            text = (root / "training" / name).read_text()
            for token in ("tier", "experiment."):
                assert token not in text, f"{name} mentions {token!r}"


class TestRenderReport:
    def test_renders_groups(self, tmp_path):
        runs = tmp_path / "runs"
        art = FakeArtifact("run-0", 0, "h")
        art._source = _write_source(tmp_path, _record(0))
        summaries = pull(
            [art],
            runs,
            TOY,
            TOY_IDENTITY,
            group_of=toy_group_of,
            group_of_record=toy_group_of_record,
        )
        report = render_report(summaries, 1)
        assert "== even ==" in report
        assert "no expected count" in report  # expected_counts not supplied
        assert "pull clean: no conflicts, no failures" in report


class _NullTargetLayout:
    """A layout whose local_target maps nothing (the record has no target)."""

    def record_path(self, metadata):
        return f"{metadata['run_id']}.json"

    def curve_path(self, metadata):
        return f"curves/{metadata['run_id']}/metrics.csv"

    def checkpoint_path(self, metadata):
        return None

    def local_target(self, rel, metadata, include_checkpoints):
        return None


class TestRobustness:
    def test_read_record_non_dict_json_is_none(self, tmp_path):
        from research_lib.training.run_control import read_record

        path = tmp_path / "r.json"
        path.write_text("[1, 2, 3]")
        assert read_record(path) is None

    def test_read_record_non_utf8_is_none(self, tmp_path):
        from research_lib.training.run_control import read_record

        path = tmp_path / "r.json"
        path.write_bytes(b"\xff\xfe garbage")
        assert read_record(path) is None

    def test_missing_local_target_is_a_failure_not_a_crash(self, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        art = FakeArtifact("run-0", 0, "h")
        art._source = _write_source(tmp_path, _record(0))
        summaries = pull(
            [art],
            runs,
            _NullTargetLayout(),
            TOY_IDENTITY,
            group_of=toy_group_of,
            group_of_record=toy_group_of_record,
            dry_run=True,
        )
        # The record has no local target: reported as a failure, no crash.
        assert any("no local target" in r for _, r in summaries["even"].failures)

    def test_download_honours_path_prefix(self, tmp_path):
        src = _write_source(tmp_path, _record(0))
        # Download only the record prefix into a clean target.
        target = tmp_path / "clean"
        target.mkdir()
        art = FakeArtifact("run-0", 0, "h")
        art._source = src
        art.download(root=str(target), path_prefix="run-0.json")
        assert (target / "run-0.json").exists()
        assert not (target / "curves").exists()
