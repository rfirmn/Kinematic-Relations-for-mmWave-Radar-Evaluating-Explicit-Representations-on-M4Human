"""Offline QA registry builder using the canonical per-frame target reader.

The reader exposes read(source_key)->(positions[22,3],valid[22]); sensor
features, RGB, activity names and model outputs are never read here. This
module does not establish G0 authority or silently finalize label policy.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re
import numpy as np

from .config import active_artifact_path, file_hash, identity, read_json, write_json
from .data import JsonlSequence
from .qa import LabelPolicy, audit_label_policy, generate_qa

PROVENANCE_HASHES = ("inventory_hash", "coordinate_audit_hash", "time_audit_hash", "split_hash", "target_hash", "target_provenance_hash")


def _validate_locked_target_recipe(test_lock_path, windows_path, provenance, policy):
    """Use the canonical study lock; array export identities are post-lock.

    The source window manifest, target recipe/certificate, source inventory,
    split and policy exist before test extraction. The resulting target array
    and QA hashes are recorded after extraction, without mutating the lock.
    """
    from .study import validate_test_lock
    if provenance.get("artifact_generation_id") is not None:
        test_lock_path=active_artifact_path(test_lock_path)
        proposed=read_json(test_lock_path)
        for artifact in proposed.get("artifacts",{}):
            active_artifact_path(artifact)
    lock = validate_test_lock(test_lock_path,expected_generation=provenance.get('artifact_generation_id'))
    if lock.get("test_evaluation_authorized") is not True:
        raise ValueError("test lock does not authorize target extraction/evaluation")
    plan = lock.get("plan", {})
    artifacts = lock.get("artifacts", {})
    def pinned_plan_file(key):
        if not plan.get(key):
            raise ValueError(f"test lock plan missing {key}")
        path = Path(plan[key]).resolve()
        digest = artifacts.get(str(path))
        if not digest or digest != file_hash(path):
            raise ValueError(f"test lock source hash mismatch: {key}")
        return path, digest
    manifest, _ = pinned_plan_file("source_manifest")
    if manifest != Path(windows_path).resolve():
        raise ValueError("test lock source_manifest differs from QA input")
    _, recipe_hash = pinned_plan_file("target_provenance")
    if recipe_hash != provenance["target_provenance_hash"]:
        raise ValueError("test target recipe/provenance hash mismatch")
    for plan_key, provenance_key in (("source_inventory", "inventory_hash"), ("split_lock", "split_hash")):
        _, digest = pinned_plan_file(plan_key)
        if digest != provenance[provenance_key]:
            raise ValueError(f"test {provenance_key} hash mismatch")
    policy_path, _ = pinned_plan_file("label_policy")
    payload = read_json(policy_path)
    values = payload.get("policy", payload)
    try:
        locked_policy = LabelPolicy(**values)
    except (TypeError, ValueError) as error:
        raise ValueError("locked label policy must contain the LabelPolicy fields") from error
    if locked_policy.hash != policy.hash:
        raise ValueError("test label policy hash mismatch")
    return lock


def build_qa_registry(windows_path, target_reader, output_path, *, provenance,
                      splits=("train", "val"), policy=None, test_lock_path=None):
    """Build three canonical QA per window and persist coverage diagnostics.

    G0 is supplied by an external authority audit. Synthetic attestations
    remain explicitly synthetic and never satisfy scientific G1. Test target
    export uses the canonical study lock, pinning the immutable source manifest,
    target recipe/certificate and label policy. Exported array identities are
    recorded afterwards; test arrays are never required to create this lock.
    """
    policy = policy or LabelPolicy()
    generation = provenance.get("artifact_generation_id")
    if generation is not None:
        from .data import validate_manifest_generation
        windows_path=active_artifact_path(windows_path)
        output_path=active_artifact_path(output_path)
        validate_manifest_generation(windows_path,
            inventory_path=provenance.get("inventory_path"), expected_generation=generation)
    if provenance.get("G0_passed") is not True:
        raise ValueError("G0 authority required before generating QA targets")
    if provenance.get("evidence") not in ("synthetic_software_fixture", "actual_M4Human"):
        raise ValueError("explicit synthetic or actual-M4Human evidence required")
    for key in PROVENANCE_HASHES:
        if not re.fullmatch(r"[0-9a-f]{64}", provenance.get(key) or ""):
            raise ValueError(f"missing SHA256 provenance: {key}")
    exported_metadata = getattr(target_reader, "metadata", None)
    if exported_metadata is not None:
        if generation is not None and exported_metadata.get("artifact_generation_id") != generation:
            raise ValueError("target-reader artifact generation mismatch")
        for metadata_key, provenance_key in (("target_provenance_hash", "target_provenance_hash"), ("inventory_hash", "inventory_hash")):
            if exported_metadata.get(metadata_key) != provenance[provenance_key]:
                raise ValueError(f"target-reader {metadata_key} mismatch")
        reader_root = getattr(target_reader, "root", None)
        if generation is not None and reader_root is not None:
            reader_root=active_artifact_path(reader_root)
        if reader_root is not None and file_hash(Path(reader_root)/"metadata.json") != provenance["target_hash"]:
            raise ValueError("target-reader array metadata identity mismatch")
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or set(splits)-{"train", "val", "test"}:
        raise ValueError("splits must be a nonempty unique subset of train,val,test")
    manifest_hash = file_hash(windows_path)
    if "test" in splits:
        if test_lock_path is None:
            raise ValueError("test QA target generation requires an explicit test lock")
        _validate_locked_target_recipe(test_lock_path, windows_path, provenance, policy)
    output = Path(output_path)
    meta_path = output.with_suffix(".meta.json")
    audit_path = output.with_suffix(".label_audit.json")
    if any(p.exists() for p in (output, meta_path, audit_path)):
        raise FileExistsError("QA registry destination must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name+".partial")
    windows = JsonlSequence(windows_path)
    counts, coverage = Counter(), Counter()
    seen, subjects, audit_rows = set(), {}, []
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        for window in windows:
            if window["split"] not in splits:
                continue
            if generation is not None and window.get("artifact_generation_id") != generation:
                raise ValueError("QA source window artifact generation mismatch")
            uid = window["window_uid"]
            if uid in seen:
                raise ValueError("duplicate window UID in QA input")
            seen.add(uid)
            subject = str(window["subject_id"])
            if subject in subjects and subjects[subject] != window["split"]:
                raise ValueError("subject split leakage in QA input")
            subjects[subject] = window["split"]
            times = np.asarray(window["time_s"], np.float64)
            if times.shape != (64,) or len(window["source_keys"]) != 64 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > .125+1e-12):
                raise ValueError("QA source window has invalid timestamp/gap/length")
            frames = [target_reader.read(key) for key in window["source_keys"]]
            positions = np.asarray([p for p, _ in frames], np.float64)
            valid = np.asarray([mask for _, mask in frames], bool)
            if positions.shape != (64, 22, 3) or valid.shape != (64, 22):
                raise ValueError("per-frame target reader must provide [22,3] positions and [22] valid")
            metadata = {key: window[key] for key in ("subject_id", "recording_id", "split", "activity_id") if key in window}
            rows = generate_qa(uid, times, positions, valid, policy=policy,
                               target_hash=provenance["target_hash"], window_manifest_hash=manifest_hash, metadata=metadata)
            counts[window["split"]] += 1
            for row in rows:
                if generation is not None:
                    row["artifact_generation_id"] = window["artifact_generation_id"]
                coverage[f"{window['split']}/{row['part']}/{row['reference_status']}"] += 1
                handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
                if window["split"] in ("train", "val"):
                    audit_rows.append(row)
        if not seen:
            raise ValueError("no selected windows for QA registry")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, output)
    diagnostics = audit_label_policy(audit_rows, policy=policy) if audit_rows else {"status": "test_only_no_label_tuning", "policy_hash": policy.hash}
    diagnostics["evidence"] = provenance["evidence"]
    if generation is not None:
        diagnostics["artifact_generation_id"] = generation
    diagnostics["actual_m4human_used"] = provenance["evidence"] == "actual_M4Human"
    write_json(audit_path, diagnostics)
    metadata = {"schema_id": "relational64_qa_registry_1", "artifact_generation_id":provenance.get('artifact_generation_id'),"complete": True,
                "evidence": provenance["evidence"], "actual_m4human_used": provenance["evidence"] == "actual_M4Human",
                "scientific_G1_accepted": False, "window_count": len(seen), "qa_count": 3*len(seen),
                "window_counts_by_split": dict(counts), "coverage": dict(coverage), "splits": list(splits),
                "window_manifest_hash": manifest_hash, "label_policy_hash": policy.hash,
                "provenance": provenance, "qa_hash": file_hash(output), "label_audit_hash": file_hash(audit_path),
                "test_lock_hash": file_hash(test_lock_path) if "test" in splits else None}
    metadata["identity"] = identity(metadata)
    write_json(meta_path, metadata)
    return metadata
