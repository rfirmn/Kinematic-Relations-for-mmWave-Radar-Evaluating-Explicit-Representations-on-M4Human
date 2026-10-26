"""Temporary synthetic LMDB fixtures; no actual M4Human or CUDA is used here."""
import json
import struct
import lmdb
import msgpack
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from relational64 import config
from relational64.config import read_json,write_json,file_hash
from relational64.diagnostic_pilot import DiagnosticRun,CandidateTargets,select_panel,alignment_quality,apply_sensor_gain


def row(subject,split):
    return {'window_uid':f'fixture-{subject}','subject_id':str(subject),'split':split,
            'recording_id':f's{subject}/a1','source_keys':[f'[{subject}, 1, {i}]' for i in range(64)],
            'time_s':[i/12 for i in range(64)],'artifact_generation_id':'current'}


def test_panel_is_sensor_determined_and_excludes_test():
    rows=[row(99,'test'),row(1,'train'),row(1,'train'),row(2,'val')]
    assert [r['subject_id'] for r in select_panel(rows,1,1)]==['1','2']
    with pytest.raises(ValueError,match='subject split leakage'):
        select_panel([row(1,'train'),row(1,'val')],1,1)
    invalid=row(1,'train');invalid['source_keys'][0]='[99, 1, 0]'
    with pytest.raises(ValueError,match='source-key subject'):
        select_panel([invalid],1,1)


def test_candidate_reader_rejects_unselected_subject_before_data_access():
    reader=CandidateTargets.__new__(CandidateTargets);reader.allowed={1};reader.envs={}
    with pytest.raises(ValueError,match='outside selected'):
        reader.read('[13, 1, 0]')


def test_truncated_generation_is_not_a_passed_engineering_pilot():
    result={'M':{'metrics_against_candidate_targets':{'generation':{'schema_enum_valid_rate':0.}}},
            'R':{'metrics_against_candidate_targets':{'generation':{'schema_enum_valid_rate':1.}}}}
    quality=alignment_quality(result)
    assert quality['status']=='needs_revision' and quality['failures'][0]['condition']=='M'
    assert alignment_quality({'R':result['R']})['status']=='passed'


def test_experimental_embedding_gain_preserves_gradient_and_does_not_change_shape():
    embedding=torch.randn(64,32,requires_grad=True)
    scaled=apply_sensor_gain(embedding,.026)
    assert scaled.shape==embedding.shape
    scaled.sum().backward()
    torch.testing.assert_close(embedding.grad,torch.full_like(embedding,.026))


def test_diagnostic_normalizer_and_checkpoint_cannot_enter_scientific_paths(tmp_path,monkeypatch):
    from relational64 import data,cli
    write_json(tmp_path/'inventory.json',{'authority':{'G0_passed':True}})
    write_json(tmp_path/'normalizer.json',{'diagnostic_only':True})
    (tmp_path/'windows.jsonl').write_text('')
    monkeypatch.setattr(data,'RPCOnlyReader',lambda _:SimpleNamespace(inventory={'authority':{'G0_passed':True}}))
    monkeypatch.setattr(data,'verify_snapshot',lambda *args:None)
    with pytest.raises(ValueError,match='diagnostic normalizer'):
        data.SensorDataset(tmp_path/'windows.jsonl',tmp_path/'inventory.json',tmp_path/'normalizer.json')
    torch.save({'status':'diagnostic'},tmp_path/'diagnostic.pt')
    with pytest.raises(ValueError,match='Selected physical checkpoint'):
        cli.load_selected_physical(tmp_path/'diagnostic.pt',{},42)


@pytest.fixture
def run_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'ROOT',tmp_path)
    source=tmp_path/'source';source.mkdir();paths={name:str(source/f'{name}.lmdb') for name in ('rpc','params','calib')}
    points=np.arange(32,dtype=np.float32).reshape(8,4)/10
    rpc=struct.pack('<III',2,8,4)+points.tobytes()
    calib={'vicon_to_cam_rotmatrix':np.eye(3).tolist(),'radar_to_cam_rotmatrix':np.eye(3).tolist(),
           'vicon_to_cam_tvec':[0.,0.,0.],'radar_to_cam_tvec':[0.,0.,0.]}
    rows=[row(s,'train' if s<=8 else 'val') for s in range(1,13)]
    for name,path in paths.items():
        env=lmdb.open(path,subdir=False,map_size=4<<20)
        with env.begin(write=True) as tx:
            for r in rows:
                for i,key in enumerate(r['source_keys']):
                    joints=np.arange(66,dtype=np.float64).reshape(22,3)/100+i*.001
                    value=rpc if name=='rpc' else msgpack.packb(calib if name=='calib' else {'joints':joints.tolist()})
                    tx.put(key.encode(),value)
        env.close()
    inventory={'artifact_generation_id':'current','artifact_policy':'fresh_from_source_only','immutable_snapshot':True,
               'authority':{'G0_passed':False},'databases':paths,'source_manifest_sha256':{n:file_hash(p) for n,p in paths.items()},
               'serializer':{'byte_order':'little','max_source_points':100000,'max_payload_bytes':1600012,'sentinel_policy':'none'}}
    derived=tmp_path/'derived';inventory_path=derived/'meta/source_inventory.json';write_json(inventory_path,inventory)
    manifest=derived/'windows.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(derived/'prepare_summary.json',{'artifact_generation_id':'current','prior_artifacts_reused':[],
        'source_inventory_sha256':file_hash(inventory_path),'windows_sha256':file_hash(manifest)})
    cfg={'runtime':{'artifact_generation_id':'current','source_inventory':str(inventory_path),'derived_root':str(derived),
                   'report_root':str(tmp_path/'reports'),'checkpoint_root':str(tmp_path/'checkpoints'),'device':'cpu'}}
    run=DiagnosticRun(cfg,'fixture',steps=1,alignment_steps=1,conditions='M')
    return run,inventory_path


def test_real_reader_path_with_fixture_is_provisional_and_never_changes_g0(run_fixture):
    run,inventory_path=run_fixture;before=file_hash(inventory_path)
    report=run.execute('prepare')
    assert report['status']=='passed' and report['diagnostic_only'] and not report['scientific_acceptance']
    assert report['frame_count']==768 and report['candidate_valid_joints']==768*22
    assert len(run.batches())==12
    assert file_hash(inventory_path)==before and not read_json(inventory_path)['authority']['G0_passed']
    norm=read_json(run.derived/'intensity_normalizer.diagnostic.json')
    assert norm['diagnostic_only'] and norm['considered_train_points']==8*64*8
    assert read_json(run.derived/'qa.diagnostic.json')['schema']=='relational64_diagnostic_qa_1'
    assert run.execute('prepare')==report
    with (run.derived/'batches.diagnostic.pt').open('ab') as handle:handle.write(b'changed')
    with pytest.raises(ValueError,match='dependency hash mismatch'):
        run.batches()


def test_failure_is_journaled_and_dependents_stop(run_fixture,monkeypatch):
    run,_=run_fixture
    def fail():raise RuntimeError('deliberate software fixture failure')
    monkeypatch.setattr(run,'prepare',fail)
    with pytest.raises(RuntimeError,match='deliberate'):
        run.execute('prepare')
    assert read_json(run.stage_path('prepare'))['status']=='failed'
    assert len((run.reports/'failures.jsonl').read_text().splitlines())==1
    with pytest.raises(ValueError,match='must pass'):
        run.batches()


def test_plan_changes_require_new_run_id(run_fixture):
    run,_=run_fixture
    with pytest.raises(ValueError,match='plan changed'):
        DiagnosticRun(run.config,run.run_id,steps=2,alignment_steps=1,conditions='M')
    with pytest.raises(ValueError,match='Safe run ID'):
        DiagnosticRun(run.config,'../other')
    with pytest.raises(ValueError,match='1..100'):
        DiagnosticRun(run.config,'other',steps=0)
    with pytest.raises(ValueError,match='plan changed'):
        DiagnosticRun(run.config,run.run_id,steps=1,alignment_steps=1,conditions='M',alignment_lr=1e-5,alignment_warmup=1)
    with pytest.raises(ValueError,match='plan changed'):
        DiagnosticRun(run.config,run.run_id,steps=1,alignment_steps=1,conditions='M',sensor_gain=.026)


def test_completed_alignment_checkpoint_resumes_generation_without_extra_updates(run_fixture,monkeypatch):
    from relational64 import alignment
    from relational64.models import RadarPoseModel,SensorProjector
    from relational64.checkpoint import save_checkpoint
    from relational64.training import seed_everything
    seed_everything(42)
    run,_=run_fixture;run.execute('prepare')
    original_accept=run.accepted_stage
    monkeypatch.setattr(run,'accepted_stage',lambda name:{} if name=='physical' else original_accept(name))
    physical=run.checkpoints/'physical/diagnostic.pt';physical.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':RadarPoseModel().state_dict()},physical)
    lock=run.reports.parent.parent/'language.lock.json';write_json(lock,{})
    provenance=run.provenance('alignment',condition='M',physical_checkpoint_hash=file_hash(physical),
                             language_lock_hash=file_hash(lock),qa_hash=file_hash(run.derived/'qa.diagnostic.json'))
    projector=SensorProjector('M');optimizer=torch.optim.AdamW(projector.parameters(),lr=3e-4,eps=1e-6)
    save_checkpoint(run.checkpoints/'M/diagnostic.pt',projector,optimizer,provenance=provenance,epoch=0,step=1)
    write_json(run.reports/'alignment_M_history.json',{'history':[{'step':1}]})
    class Bridge(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(1),requires_grad=False)
        def selected_answer_loss(self,*args):raise AssertionError('Completed checkpoint must not train again')
        def greedy_generate(self,*args):return SimpleNamespace(raw_text='{"answer":"unknown"}',truncated=False)
    class Splicer:
        end_token_id=0
        def build(self,**kwargs):return SimpleNamespace(sensor_positions=() if kwargs.get('text_only') else tuple(range(64)))
    monkeypatch.setattr(alignment,'load_language',lambda *args:(Bridge(),Splicer(),None,torch.float32))
    report=run.alignment()
    assert report['conditions']['M']['optimizer_steps']==1
    assert not report['conditions']['M']['projector_changed_in_this_invocation']
    assert report['frozen_Qwen_unchanged']
