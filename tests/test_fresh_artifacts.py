"""Isolation contracts: old research metadata cannot satisfy the new run."""
import json
import struct
from pathlib import Path
import lmdb
import numpy as np
import pytest
from relational64.config import ROOT,resolve,active_artifact_path
from relational64.data import RPCOnlyReader,SensorDataset,TargetArrayStore,assert_fresh_inventory,write_json
from relational64.cache import PhysicalCache
from relational64.prepare import build_windows


def test_nested_gate_artifacts_cannot_reference_archive_or_external(tmp_path,monkeypatch):
    from relational64 import config
    monkeypatch.setattr(config,'ROOT',tmp_path)
    gate={'passed':True,'actual_M4Human':True,'test_ids':['mocked_hash_contract'],
          'evidence':'mocked_authority','artifacts':{}}
    for path in (tmp_path/'archive/previous.txt',tmp_path.parent/'external_artifact.txt'):
        gate['artifacts']={str(path):'a'*64}
        write_json(tmp_path/'gate.json',{'artifact_generation_id':'current','gates':{'G0':gate}})
        with pytest.raises(ValueError,match='Archived|inside this repository'):
            config.require_gate(tmp_path/'gate.json','G0')


def test_sensor_rejects_wrong_generation_window_before_opening_rpc(tmp_path,monkeypatch):
    from relational64 import config,data
    monkeypatch.setattr(config,'ROOT',tmp_path)
    inventory={'artifact_generation_id':'current','artifact_policy':'fresh_from_source_only'}
    write_json(tmp_path/'inventory.json',inventory)
    windows=tmp_path/'windows.jsonl'
    windows.write_text(json.dumps({'artifact_generation_id':'prior','split':'train'})+'\n')
    summary={'artifact_generation_id':'current','prior_artifacts_reused':[],
             'source_inventory_sha256':data.file_hash(tmp_path/'inventory.json'),
             'windows_sha256':data.file_hash(windows)}
    write_json(tmp_path/'prepare_summary.json',summary)
    with pytest.raises(ValueError,match='Window row artifact generation mismatch'):
        SensorDataset(windows,tmp_path/'inventory.json','not-opened',expected_generation='current')
    windows.write_text(json.dumps({'artifact_generation_id':'current','split':'train'})+'\n')
    with pytest.raises(ValueError,match='checksum mismatch'):
        data.validate_manifest_generation(windows,expected_generation='current')
    summary.update(windows_sha256=data.file_hash(windows),artifact_generation_id='prior')
    write_json(tmp_path/'prepare_summary.json',summary)
    with pytest.raises(ValueError,match='Manifest artifact generation mismatch'):
        data.validate_manifest_generation(windows,expected_generation='current')


def test_prepare_rejects_other_generation_before_reading_source(tmp_path):
    from relational64.prepare import prepare_data
    write_json(tmp_path/'inventory.json',{'artifact_generation_id':'prior','artifact_policy':'fresh_from_source_only'})
    with pytest.raises(ValueError,match='generation mismatch'):
        prepare_data(tmp_path/'inventory.json',tmp_path/'new',expected_generation='current')


def test_prepare_rejects_nested_scan_in_archive_before_creating_output(tmp_path,monkeypatch):
    from relational64 import config
    from relational64.prepare import prepare_data
    monkeypatch.setattr(config,'ROOT',tmp_path)
    write_json(tmp_path/'inventory.json',{'artifact_generation_id':'current','artifact_policy':'fresh_from_source_only',
               'full_scan':True,'scan_path':str(tmp_path/'archive/previous/sensor_scan.jsonl')})
    with pytest.raises(ValueError,match='Archived'):
        prepare_data(tmp_path/'inventory.json',tmp_path/'new',expected_generation='current',diagnostic=True)
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('prior_input',['inventory','targets'])
def test_cli_build_qa_rejects_prior_generation_before_output(tmp_path,monkeypatch,prior_input):
    from relational64 import cli,config
    monkeypatch.setattr(config,'ROOT',tmp_path)
    runtime={'artifact_generation_id':'current','artifact_policy':'fresh_from_source_only',
             'derived_root':str(tmp_path/'derived'),'report_root':str(tmp_path/'reports'),
             'source_inventory':str(tmp_path/'inventory.json')}
    monkeypatch.setattr(cli,'resolve',lambda _: {'runtime':runtime})
    inventory={'artifact_generation_id':'prior' if prior_input=='inventory' else 'current',
               'artifact_policy':'fresh_from_source_only','authority':{'G0_passed':True}}
    write_json(tmp_path/'inventory.json',inventory)
    write_json(tmp_path/'derived/targets/metadata.json',{'artifact_generation_id':'prior'})
    with pytest.raises(ValueError,match='generation mismatch'):
        cli.dispatch(cli.parser().parse_args(['build-qa']))
    assert not (tmp_path/'derived/qa.jsonl').exists()


@pytest.mark.parametrize('key',['reference_asset','reference_regressor','equivalence_evidence','coordinate_audit'])
def test_direct_target_nested_artifacts_reject_archive(tmp_path,monkeypatch,key):
    from relational64 import config,data
    monkeypatch.setattr(config,'ROOT',tmp_path)
    monkeypatch.setattr(data,'verify_snapshot',lambda *args:None)
    inventory={'artifact_generation_id':'current','artifact_policy':'fresh_from_source_only','authority':{'G0_passed':True}}
    evidence=tmp_path/'current_evidence';evidence.write_text('mocked fixture only')
    provenance={'artifact_generation_id':'current','target_equivalence_verified':True}
    for name in ('reference_asset','reference_regressor','equivalence_evidence','coordinate_audit'):
        provenance.update({name+'_path':str(evidence),name+'_sha256':data.file_hash(evidence)})
    provenance[key+'_path']=str(tmp_path/'archive/old/evidence')
    with pytest.raises(ValueError,match='Archived'):
        data.DirectTargetReader(inventory,provenance)

def test_active_config_is_repository_local_and_source_only():
    runtime=resolve(ROOT/'configs/relational64/runtime.yaml')['runtime']
    assert runtime['artifact_policy']=='fresh_from_source_only'
    assert 'metadata_root' not in runtime
    assert runtime['artifact_generation_id']=='fresh_20261026_01'
    for name in ('meta_root','source_inventory','derived_root','report_root','checkpoint_root'):
        assert ROOT in active_artifact_path(runtime[name]).parents


@pytest.mark.parametrize('generation',[None,'','../old'])
def test_fresh_policy_cannot_disable_generation(generation):
    from relational64.config import validate
    config=resolve(ROOT/'configs/relational64/runtime.yaml')
    config['runtime']['artifact_generation_id']=generation
    with pytest.raises(ValueError,match='nonempty safe artifact generation'):
        validate(config)

def test_archived_and_external_artifact_paths_are_rejected():
    with pytest.raises(ValueError,match='Archived'):
        active_artifact_path(ROOT/'archive/previous/source_inventory.json')
    with pytest.raises(ValueError,match='inside this repository'):
        active_artifact_path(ROOT.parent/'external_artifacts')

@pytest.mark.parametrize('field',['metadata_root','metadata_hashes'])
def test_legacy_inventory_rejected_before_open(field):
    with pytest.raises(ValueError,match='Legacy'):
        RPCOnlyReader({field:'untrusted legacy metadata'})

def test_cache_targets_and_sensor_reject_other_generation(tmp_path,monkeypatch):
    from relational64 import config
    monkeypatch.setattr(config,'ROOT',tmp_path)
    write_json(tmp_path/'inventory.json',{'artifact_generation_id':'prior','artifact_policy':'fresh_from_source_only'})
    with pytest.raises(ValueError,match='generation mismatch'):
        SensorDataset('not-opened',tmp_path/'inventory.json','not-opened',expected_generation='current')
    write_json(tmp_path/'cache.meta.json',{'source':{'artifact_generation_id':'prior'}})
    with pytest.raises(ValueError,match='generation mismatch'):
        PhysicalCache(tmp_path,expected_generation='current')
    write_json(tmp_path/'metadata.json',{'artifact_generation_id':'prior'})
    with pytest.raises(ValueError,match='generation mismatch'):
        TargetArrayStore(tmp_path,expected_generation='current')


@pytest.mark.parametrize('filename',['../old/positions.npy','C:/old/positions.npy','archive/positions.npy'])
def test_nested_target_files_rejected_before_hashing(tmp_path,filename):
    write_json(tmp_path/'metadata.json',{'protocol_id':'m4human_relational64','complete':True,'hashes':{filename:'a'*64}})
    with pytest.raises(ValueError,match='canonical local filenames'):
        TargetArrayStore(tmp_path)

def test_same_sensor_support_has_generation_specific_window_identity():
    frames=[{'frame_uid':str(i),'source_key':f'[1, 1, {i}]','recording_id':'s1/a1','segment_id':'seg',
             'split':'train','subject_id':'1','time_s':i/12,'time_source':'nominal_frame_index','sensor_valid':True} for i in range(64)]
    first=list(build_windows(frames,'split',generation_id='first'))[0]
    second=list(build_windows(frames,'split',generation_id='second'))[0]
    assert first['window_uid']!=second['window_uid']
    assert first['source_keys']==second['source_keys'] and first['time_s']==second['time_s']

def test_full_audit_builds_metadata_without_prior_metadata(tmp_path,monkeypatch):
    from relational64 import audit,config
    monkeypatch.setattr(config,'ROOT',tmp_path)
    source=tmp_path/'source';source.mkdir()
    points=np.arange(32,dtype=np.float32).reshape(8,4)+1
    payload=struct.pack('>III',2,8,4)+points.astype('>f4').tobytes()
    names={'rpc':'radar_pc.lmdb','params':'params.lmdb','calib':'calib.lmdb','indicator':'indicator.lmdb'}
    for name,filename in names.items():
        env=lmdb.open(str(source/filename),subdir=False,map_size=1<<20)
        with env.begin(write=True) as tx:tx.put(b'[1, 1, 0]',payload if name=='rpc' else b'not-read-as-target-in-this-fixture')
        env.close()
    def diagnostic(databases,samples,output):
        result={'samples':[],'target_equivalence_verified':False}
        write_json(Path(output)/'coordinate_audit.json',result);return result
    monkeypatch.setattr(audit,'coordinate_diagnostics',diagnostic)
    policy={'artifact_generation_id':'fixture','artifact_policy':'fresh_from_source_only','prior_metadata_inputs':[],
            'database_names':names,'serializer':{'byte_order':'detect_from_payload','max_source_points':100000,'max_payload_bytes':1600012,'sentinel_policy':'none'}}
    write_json(tmp_path/'policy.json',policy)
    inventory=audit.run_audit(source,tmp_path/'new_meta',policy_path=tmp_path/'policy.json',full=True)
    assert inventory['serializer']['byte_order']=='big'
    assert inventory['prior_artifacts_reused']==[] and 'metadata_root' not in inventory
    assert inventory['statistics']['points_clean']==8
    assert inventory['immutability_evidence']['unchanged_during_audit']
    assert not inventory['authority']['G0_passed']
