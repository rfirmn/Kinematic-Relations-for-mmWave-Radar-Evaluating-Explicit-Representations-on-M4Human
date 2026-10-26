"""Synthetic CPU-only isolation checks. No dataset or prior artifacts are read."""
import json
from types import SimpleNamespace

import pytest

from relational64 import alignment, config


ACTIVE = "fresh_20261026_01"
OLD = "fixture_prior_generation"


@pytest.fixture
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path.resolve())
    return tmp_path


def registry(root, *, row_generation=ACTIVE, metadata_generation=ACTIVE, rows=None):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "qa.jsonl"
    records = rows or [
        {"qa_uid": "defined", "split": "train", "reference_status": "defined", "artifact_generation_id": row_generation},
        {"qa_uid": "unknown", "split": "val", "reference_status": "unknown", "artifact_generation_id": row_generation},
        {"qa_uid": "undefined", "split": "val", "reference_status": "undefined", "artifact_generation_id": row_generation},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    metadata = {"schema_id": "relational64_qa_registry_1", "artifact_generation_id": metadata_generation,
                "complete": True, "qa_count": len(records), "qa_hash": config.file_hash(path),
                "provenance": {"artifact_generation_id": metadata_generation}}
    save_sidecar(path, metadata)
    return path


def save_sidecar(path, metadata):
    metadata = dict(metadata)
    metadata.pop("identity", None)
    metadata["identity"] = config.identity(metadata)
    path.with_suffix(".meta.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_fresh_registry_checked_before_split_and_eligibility(isolated_root):
    path = registry(isolated_root / "active")
    assert [row["qa_uid"] for row in alignment.read_qa(path, expected_generation=ACTIVE)] == ["defined", "unknown"]
    assert [row["qa_uid"] for row in alignment.read_qa(path, "val", expected_generation=ACTIVE)] == ["unknown"]
    assert len(alignment.read_qa(path, eligible=False, expected_generation=ACTIVE)) == 3


def test_old_qa_in_current_folder_rejected_before_qa_hash_or_rows(isolated_root, monkeypatch):
    path = registry(isolated_root / "active", row_generation=OLD, metadata_generation=OLD)
    monkeypatch.setattr(alignment, "file_hash", lambda _path: pytest.fail("stale QA file opened before sidecar rejection"))
    with pytest.raises(ValueError, match="generation/state"):
        alignment.read_qa(path, expected_generation=ACTIVE)


def test_relabelling_only_sidecar_does_not_relabel_old_rows(isolated_root):
    path = registry(isolated_root / "active", row_generation=OLD, metadata_generation=ACTIVE)
    with pytest.raises(ValueError, match="QA row artifact generation"):
        alignment.read_qa(path, expected_generation=ACTIVE)


def test_old_row_in_filtered_split_is_also_rejected(isolated_root):
    rows = [{"qa_uid": "train", "split": "train", "reference_status": "defined", "artifact_generation_id": ACTIVE},
            {"qa_uid": "old_test", "split": "test", "reference_status": "undefined", "artifact_generation_id": OLD}]
    path = registry(isolated_root / "active", rows=rows)
    with pytest.raises(ValueError, match="QA row artifact generation"):
        alignment.read_qa(path, "train", expected_generation=ACTIVE)


@pytest.mark.parametrize("change", ["hash", "identity", "count", "provenance", "complete"])
def test_sidecar_and_file_tampering_rejected(isolated_root, change):
    path = registry(isolated_root / "active")
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if change == "hash":
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif change == "identity":
        metadata["qa_count"] += 1
        path.with_suffix(".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    else:
        if change == "count": metadata["qa_count"] += 1
        if change == "provenance": metadata["provenance"]["artifact_generation_id"] = OLD
        if change == "complete": metadata["complete"] = False
        save_sidecar(path, metadata)
    with pytest.raises(ValueError):
        alignment.read_qa(path, expected_generation=ACTIVE)


def test_archived_path_rejected_without_opening_any_sidecar(isolated_root, monkeypatch):
    forbidden = isolated_root / "archive" / "never_created" / "qa.jsonl"
    monkeypatch.setattr(alignment, "read_json", lambda _path: pytest.fail("archived sidecar was opened"))
    with pytest.raises(ValueError, match="Archived"):
        alignment.read_qa(forbidden, expected_generation=ACTIVE)
    assert not forbidden.parent.exists()


def test_outside_repository_path_rejected_without_opening(isolated_root, monkeypatch):
    forbidden = isolated_root.parent / "never_created_external" / "qa.jsonl"
    monkeypatch.setattr(alignment, "read_json", lambda _path: pytest.fail("external sidecar was opened"))
    with pytest.raises(ValueError, match="inside this repository"):
        alignment.read_qa(forbidden, expected_generation=ACTIVE)
    assert not forbidden.parent.exists()


def test_generationless_software_fixture_remains_usable(isolated_root):
    path = isolated_root / "fixture.jsonl"
    path.write_text(json.dumps({"qa_uid": "fixture", "split": "train", "reference_status": "defined"}) + "\n", encoding="utf-8")
    assert alignment.read_qa(path)[0]["qa_uid"] == "fixture"


def test_direct_alignment_rejects_stale_QA_before_output_or_gate_read(isolated_root, monkeypatch):
    output = isolated_root / "must_not_exist"
    fresh_cache = SimpleNamespace(meta={"source": {"artifact_generation_id": ACTIVE}})
    stale = [{"artifact_generation_id": OLD}]
    monkeypatch.setattr(alignment, "require_gate", lambda *_args: pytest.fail("gate was read before stale QA rejection"))
    with pytest.raises(ValueError, match="Alignment QA artifact generation"):
        alignment.train_alignment(fresh_cache, fresh_cache, stale, stale, condition="R", normalizer_path=None,
            config={"runtime": {"artifact_generation_id": ACTIVE}}, language_lock={}, gate_path="unused",
            output_dir=output)
    assert not output.exists()


def test_direct_alignment_rejects_stale_cache_before_output_or_gate_read(isolated_root, monkeypatch):
    output = isolated_root / "must_not_exist"
    stale_cache = SimpleNamespace(meta={"source": {"artifact_generation_id": OLD}})
    rows = [{"artifact_generation_id": ACTIVE}]
    monkeypatch.setattr(alignment, "require_gate", lambda *_args: pytest.fail("gate was read before stale cache rejection"))
    with pytest.raises(ValueError, match="Alignment cache artifact generation"):
        alignment.train_alignment(stale_cache, stale_cache, rows, rows, condition="R", normalizer_path=None,
            config={"runtime": {"artifact_generation_id": ACTIVE}}, language_lock={}, gate_path="unused",
            output_dir=output)
    assert not output.exists()
