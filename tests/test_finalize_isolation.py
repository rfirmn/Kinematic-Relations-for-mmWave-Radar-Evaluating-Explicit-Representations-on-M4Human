"""Finalizer isolation fixtures: tiny temp files; no actual dataset, GPU or subprocess."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from relational64 import config,reporting
from relational64.config import file_hash,read_json,write_json


@pytest.fixture
def finalizer(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[1]/'tools/finalize_evidence.py'
    spec=importlib.util.spec_from_file_location('fixture_finalize_evidence',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generation='test_generation'
    reports=tmp_path/'docs/report_training'/generation
    derived=tmp_path/'derived'/generation
    meta=derived/'meta'
    reports.mkdir(parents=True)
    meta.mkdir(parents=True)
    source=tmp_path/'source';source.mkdir()
    databases={}
    for name in ('rpc','params','calib','indicator'):
        path=source/(name+'.fixture');path.write_bytes(name.encode());databases[name]=str(path)
    inventory={'artifact_generation_id':generation,'artifact_policy':'fresh_from_source_only','prior_artifacts_reused':[],
        'dataset_root':str(source),'databases':databases,'source_manifest_sha256':{name:file_hash(path) for name,path in databases.items()},
        'serializer':{'byte_order':'little'},'authority':{'G0_passed':False}}
    write_json(meta/'source_inventory.json',inventory)
    write_json(meta/'coordinate_audit.json',{'samples':[]})
    write_json(meta/'split.lock.json',{'synthetic':True})
    write_json(reports/'environment.json',{'synthetic':True})
    write_json(reports/'language.lock.json',{'artifact_generation_id':generation,'environment_report':file_hash(reports/'environment.json'),
        'model_commit':'0'*40,'tokenizer_commit':'0'*40,'chat_template_hash':'0'*64,'safetensors_manifest_hash':'0'*64,
        'precision':'bfloat16','head_mode':'canonical_fp32','dependency_lock':'0'*64})
    runtime={'artifact_generation_id':generation,'report_root':str(reports),'meta_root':str(meta),
        'source_inventory':str(meta/'source_inventory.json'),'derived_root':str(derived),'checkpoint_root':str(tmp_path/'checkpoints'/generation)}
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(config,'ROOT',tmp_path)
    monkeypatch.setattr(module,'resolve',lambda _: {'protocol_id':'m4human_relational64','runtime':runtime})
    def save_mock_status(*args,**kwargs):
        write_json(reports/'acceptance.json',{'G0':False,'synthetic':True})
        return {'actual_diagnostic_training_run':False}
    status=Mock(side_effect=save_mock_status)
    monkeypatch.setattr(reporting,'build_status',status)
    process=Mock(return_value=SimpleNamespace(returncode=3,stderr='G0 remains blocked'))
    monkeypatch.setattr(module.subprocess,'run',process)
    return SimpleNamespace(module=module,root=tmp_path,reports=reports,meta=meta,inventory=inventory,status=status,process=process)


def test_source_mismatch_stops_before_any_finalization(finalizer):
    f=finalizer
    Path(f.inventory['databases']['rpc']).write_bytes(b'changed source fixture')
    with pytest.raises(ValueError,match='Source snapshot mismatch'):
        f.module.main()
    evidence=read_json(f.reports/'source_integrity_final.json')
    assert evidence['status']=='failed'
    assert not evidence['checks'][0]['matches_new_audit']
    f.status.assert_not_called()
    f.process.assert_not_called()
    assert not (f.reports/'resolved_config.json').exists()
    assert not (f.meta/'runtime.lock.json').exists()
    assert not (f.meta/'artifact_registry.json').exists()


def test_environment_hash_mismatch_stops_before_locks(finalizer):
    f=finalizer
    write_json(f.reports/'environment.json',{'synthetic':'changed after language lock'})
    with pytest.raises(ValueError,match='environment hash mismatch'):
        f.module.main()
    f.status.assert_called_once()
    f.process.assert_not_called()
    assert not (f.meta/'runtime.lock.json').exists()
    assert not (f.meta/'artifact_registry.json').exists()


def test_blocked_stage_failure_stops_before_locks(finalizer):
    f=finalizer
    f.process.side_effect=[SimpleNamespace(returncode=code,stderr='synthetic result') for code in (3,0,3,3)]
    with pytest.raises(ValueError,match='Blocked-stage checks failed'):
        f.module.main()
    assert read_json(f.reports/'blocked_stage_checks.json')['status']=='failed'
    assert not (f.meta/'runtime.lock.json').exists()
    assert not (f.meta/'artifact_registry.json').exists()


def test_registry_includes_completed_blocked_stage_checks(finalizer):
    f=finalizer
    f.module.main()
    evidence=f.reports/'blocked_stage_checks.json'
    registry=read_json(f.meta/'artifact_registry.json')
    assert registry['fresh_outputs'][str(evidence.relative_to(f.root))]==file_hash(evidence)
    assert read_json(f.meta/'runtime.lock.json')['environment_report']==file_hash(f.reports/'environment.json')
    assert f.process.call_count==4
