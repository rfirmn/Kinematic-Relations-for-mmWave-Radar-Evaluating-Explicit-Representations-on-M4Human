"""Fresh synthetic inputs only: no archive, external dataset or GPU is opened."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

import relational64.config as config
import relational64.reference_targets as reference
from relational64.config import file_hash, write_json
from relational64.data import JOINT_NAMES
from relational64.pipeline_targets import build_qa_registry

GENERATION = "fresh_20261026_01"


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root=tmp_path/"workspace";root.mkdir()
    monkeypatch.setattr(config,"ROOT",root)
    return root


@pytest.fixture
def reference_fixture(repository, monkeypatch):
    asset=repository/"assets"/"synthetic.npz";asset.parent.mkdir()
    asset.write_bytes(b"synthetic software fixture; not licensed SMPL-X data")
    regressor=repository/"assets"/"regressor.npy";np.save(regressor,np.eye(22))
    coordinate=repository/"meta"/"coordinate.json"
    write_json(coordinate,{"source_frame":"radar","output_unit":"meter"})
    inventory={"artifact_generation_id":GENERATION,"artifact_policy":"fresh_from_source_only","authority":{"G0_passed":False}}
    provenance={"artifact_generation_id":GENERATION,"evidence":"synthetic_software_fixture",
        "model_output_unit":"meter","parameter_translation_unit":"meter","source_frame":"radar",
        "ordered_names":JOINT_NAMES,"source_row_indices":list(range(22)),"num_betas":10,"num_expression_coeffs":10,
        "coordinate_audit_path":str(coordinate),"coordinate_audit_sha256":file_hash(coordinate),
        "default_hand_face_audit":True,"default_hand_face_values":{key:[0.]*size for key,size in
            (("left_hand_pose",45),("right_hand_pose",45),("jaw_pose",3),("leye_pose",3),("reye_pose",3),("expression",10))},
        "gender_mapping":{"0":"neutral"},"assets":{"neutral":{"path":str(asset),"sha256":file_hash(asset),
            "regressor_path":str(regressor),"regressor_sha256":file_hash(regressor),"trusted_distribution":True}}}
    # Initialization-only path checks do not open source LMDBs or a body model.
    monkeypatch.setattr(reference,"verify_snapshot",lambda *_:None)
    return inventory,provenance


def test_reference_current_generation_accepts_only_local_fresh_inputs(reference_fixture):
    inventory,provenance=reference_fixture
    reader=reference.ReferenceSMPLXTargetReader(inventory,provenance,diagnostic=True,_model_factory=lambda **_:None)
    assert reader.inventory["artifact_generation_id"]==GENERATION
    assert reader.regressors["neutral"].shape==(22,22)
    reader.close()


@pytest.mark.parametrize("field",["coordinate_audit_path","path","regressor_path"])
@pytest.mark.parametrize("location",["archive","outside"])
def test_reference_nested_forbidden_paths_rejected_before_any_hash(reference_fixture, repository, monkeypatch, field, location):
    inventory,original=reference_fixture;provenance=deepcopy(original)
    forbidden=(repository/"archive"/"never_open") if location=="archive" else (repository.parent/"external"/"never_open")
    if field=="coordinate_audit_path":provenance[field]=str(forbidden)
    else:provenance["assets"]["neutral"][field]=str(forbidden)
    calls=[]
    def forbidden_hash(path):
        calls.append(path)
        raise AssertionError("No nested input may be opened before all paths are authorized")
    monkeypatch.setattr(reference,"file_hash",forbidden_hash)
    with pytest.raises(ValueError,match="Archived|repository"):
        reference.ReferenceSMPLXTargetReader(inventory,provenance,diagnostic=True,_model_factory=lambda **_:None)
    assert calls==[]


def test_reference_rejects_legacy_inventory_before_opening_reference_inputs(reference_fixture, monkeypatch):
    inventory,provenance=reference_fixture
    inventory={**inventory,"metadata_hashes":{"forbidden":"0"*64}}
    monkeypatch.setattr(reference,"file_hash",lambda _:pytest.fail("legacy inventory reached artifact hashing"))
    with pytest.raises(ValueError,match="Legacy metadata"):
        reference.ReferenceSMPLXTargetReader(inventory,provenance,diagnostic=True)


def qa_fixture(repository, row_generation=GENERATION, reader_generation=GENERATION):
    path=repository/"derived"/"windows.jsonl";path.parent.mkdir()
    row={"artifact_generation_id":row_generation,"window_uid":"fixture-window","split":"train","subject_id":"fixture-subject",
         "recording_id":"fixture-recording","time_s":(np.arange(64)/12).tolist(),"source_keys":[f"fixture-{index}" for index in range(64)]}
    path.write_text(json.dumps(row)+"\n",encoding="utf8")
    write_json(path.parent/"prepare_summary.json",{"artifact_generation_id":GENERATION,"prior_artifacts_reused":[],"windows_sha256":file_hash(path)})
    provenance={"artifact_generation_id":GENERATION,"G0_passed":True,"evidence":"synthetic_software_fixture",**{key:"a"*64 for key in
        ("inventory_hash","coordinate_audit_hash","time_audit_hash","split_hash","target_hash","target_provenance_hash")}}
    calls=[]
    def read(key):
        calls.append(key)
        return np.zeros((22,3)),np.ones(22,bool)
    reader=SimpleNamespace(read=read,metadata={"artifact_generation_id":reader_generation,"target_provenance_hash":"a"*64,"inventory_hash":"a"*64})
    return path,provenance,reader,calls


def test_qa_generation_is_inherited_only_from_verified_fresh_rows(repository):
    path,provenance,reader,calls=qa_fixture(repository)
    output=repository/"derived"/"qa.jsonl"
    meta=build_qa_registry(path,reader,output,provenance=provenance)
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows)==3 and len(calls)==64
    assert all(row["artifact_generation_id"]==GENERATION for row in rows)
    assert meta["artifact_generation_id"]==GENERATION and meta["scientific_G1_accepted"] is False


@pytest.mark.parametrize("row_generation,reader_generation",[("prior",GENERATION),(GENERATION,"prior"),(None,GENERATION),(GENERATION,None)])
def test_qa_prior_rows_or_target_metadata_cannot_be_restamped(repository,row_generation,reader_generation):
    path,provenance,reader,calls=qa_fixture(repository,row_generation,reader_generation)
    output=repository/"derived"/"qa.jsonl"
    with pytest.raises(ValueError,match="generation mismatch"):
        build_qa_registry(path,reader,output,provenance=provenance)
    assert calls==[] and not output.exists() and not output.with_suffix(".meta.json").exists()


def test_qa_manifest_checksum_change_rejected_before_target_read(repository):
    path,provenance,reader,calls=qa_fixture(repository)
    path.write_text(path.read_text()+"\n",encoding="utf8")
    with pytest.raises(ValueError,match="checksum mismatch"):
        build_qa_registry(path,reader,repository/"qa.jsonl",provenance=provenance)
    assert calls==[]
