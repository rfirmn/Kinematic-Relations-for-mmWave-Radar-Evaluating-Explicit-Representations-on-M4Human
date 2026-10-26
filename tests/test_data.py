"""Synthetic fixtures: software evidence, never real data authority evidence."""
import copy
import json
import pickle
import struct

import lmdb
import msgpack
import numpy as np
import pytest
import torch

from relational64.data import (DirectTargetReader, JOINT_NAMES, RPCOnlyReader, SensorDataset, TargetArrayStore, canonical_hash, export_targets,
    clean_rpc, decode_msgpack, decode_rpc, file_hash, make_sensor_batch, parse_source_key, sample_points, verify_snapshot, write_json)
from relational64.prepare import build_segments, build_windows, fit_intensity_normalizer, make_split


SCHEMA = {'byte_order':'little','max_source_points':100000,'max_payload_bytes':1600012,'sentinel_policy':'none'}


def payload(points,endian='<'):
    return struct.pack(endian+'III',2,len(points),4)+np.asarray(points,dtype=endian+'f4').tobytes()


@pytest.mark.parametrize('endian,name',[('<','little'),('>','big')])
def test_rpc_endian_and_bounds(endian,name):
    points = np.arange(32,dtype=np.float32).reshape(8,4)
    schema = {**SCHEMA,'byte_order':name}
    assert np.array_equal(decode_rpc(payload(points,endian),schema),points)
    for bad in [payload(points,endian)[:-1],struct.pack(endian+'III',3,8,4),struct.pack(endian+'III',2,100001,4),b'']:
        with pytest.raises(ValueError):
            decode_rpc(bad,schema)


def test_keys_reject_expressions_bool_oversized():
    assert parse_source_key('[1, 2, 3]')==(1,2,3)
    for bad in ('__import__("os").system("anything")','[True,2,3]','[1,2,3,4]','[1,2,2**100]', '[1,2,2147483648]'):
        with pytest.raises(ValueError):
            parse_source_key(bad)


def test_clean_sentinel_nonfinite_and_padding():
    points = np.ones((8,4),np.float32)
    points[0] = 0
    raw = np.concatenate([points,[[np.nan,1,1,1],[1,np.inf,1,1]]])
    clean,stats = clean_rpc(raw,SCHEMA)
    assert len(clean)==8 and stats['nonfinite']==2
    assert np.array_equal(clean[0],np.zeros(4))
    clean2,stats2 = clean_rpc(raw,{**SCHEMA,'sentinel_policy':'exact_row','sentinel_row':[0,0,0,0]})
    assert len(clean2)==7 and stats2['sentinel']==1
    sampled,mask = sample_points(clean,'opaque',0)
    assert mask.sum()==8 and sampled.shape==(150,4) and np.all(sampled[~mask]==0)


def frames(count=80,split='train'):
    return [{'frame_uid':f'f{i}','source_key':f'[1, 1, {i}]','subject_id':1,'recording_id':'r1',
        'source_frame_id':i,'time_s':i/12,'time_source':'nominal_frame_index','split':split,
        'sensor_valid':True,'clean_point_count':8} for i in range(count)]


@pytest.mark.parametrize('count,train,val',[(63,0,0),(64,1,1),(80,3,2)])
def test_window_counts(count,train,val):
    assert len(list(build_windows(build_segments(frames(count)), 'split')))==train
    assert len(list(build_windows(build_segments(frames(count,'val')), 'split')))==val


def test_annotation_does_not_change_window_uid_or_phase():
    base = frames()
    changed = copy.deepcopy(base)
    changed[10]['annotation_mask'] = [False]*22
    assert list(build_windows(build_segments(base),'split'))==list(build_windows(build_segments(changed),'split'))


def test_sensor_gap_boundary_and_leakage():
    for delta,expected in [(.125,False),(.126,True)]:
        rows = frames(2)
        rows[1]['time_s']=delta
        result=list(build_segments(rows))
        assert (result[0]['segment_id']!=result[1]['segment_id'])==expected
    rows=frames(80)
    rows[40]['sensor_valid']=False
    assert not list(build_windows(build_segments(rows),'split'))
    rows=frames(2)
    rows[1]['split']='test'
    with pytest.raises(ValueError,match='leakage'):
        list(build_segments(rows))


def test_split_stable_disjoint_exact20():
    split=make_split(range(1,21))
    assert split==make_split(reversed(range(1,21)))
    assert [len(split[s]) for s in ('train','val','test')]==[12,4,4]
    assert len(set(sum([split[s] for s in ('train','val','test')],[])))==20
    with pytest.raises(ValueError):
        make_split(range(1,20))


class StubReader:
    schema=SCHEMA
    inventory={'T_source_to_radar':np.eye(4).tolist()}
    def __init__(self,cloud):
        self.cloud=cloud
    def read_rpc(self,key):
        return self.cloud.copy()


def test_centroid_all_first_frame_points_before_sampling_and_sensor_isolation():
    points=np.ones((300,4),np.float32)
    points[:,0]=np.arange(300)
    points[:,2]=2
    reader=StubReader(points)
    window=list(build_windows(build_segments(frames(64)),'split'))[0]
    result=make_sensor_batch(reader,window,{'a':1,'median':.6931471805599453,'iqr':1})
    assert set(result)=={'X','point_mask','time_s','centroid','uid'}
    assert result['X'].shape==(64,150,4)
    assert torch.equal(result['centroid'],torch.tensor([149.5,1.,0.]))
    assert torch.all(result['X'][:,:,2]==2)
    again=make_sensor_batch(reader,window,{'a':1,'median':.6931471805599453,'iqr':1})
    assert torch.equal(result['X'],again['X'])


def test_train_only_normalizer_hash_and_order():
    class Reader(StubReader):
        def read_rpc(self,key):
            assert key!='forbidden_val'
            return np.arange(32,dtype=np.float32).reshape(8,4)
    rows=frames(2)
    val={**rows[0],'split':'val','source_key':'forbidden_val'}
    norm=fit_intensity_normalizer(rows+[val],Reader(None),max_points=10)
    norm2=fit_intensity_normalizer([val]+list(reversed(rows)),Reader(None),max_points=10)
    assert norm==norm2 and norm['fit_split']=='train' and norm['considered_train_points']==16


def test_msgpack_unsafe_arrays_extensions_depth():
    for obj in [{'__nd__':True,'dtype':'O','shape':[1],'data':b'12345678'},
                {'__nd__':True,'dtype':'f4','shape':[2],'data':b'1234'},
                {'__nd__':True,'dtype':'f4','shape':[0],'data':b''}]:
        with pytest.raises(ValueError):
            decode_msgpack(msgpack.packb(obj))
    with pytest.raises((ValueError,msgpack.FormatError)):
        decode_msgpack(msgpack.packb(msgpack.ExtType(1,b'x')))


def test_real_reader_spawn_and_source_readonly(tmp_path):
    path=tmp_path/'rpc.lmdb'
    env=lmdb.open(str(path),subdir=False,map_size=1024*1024)
    with env.begin(write=True) as tx:
        for i in range(64):
            points=np.arange(32,dtype=np.float32).reshape(8,4)
            tx.put(f'[1, 1, {i}]'.encode(),payload(points))
    env.close()
    before=path.stat()
    inventory={'databases':{'rpc':str(path)},'serializer':SCHEMA,'immutable_snapshot':True,'authority':{'G0_passed':True},'source_manifest_sha256':{'rpc':file_hash(path)}}
    reader=RPCOnlyReader(inventory)
    assert reader.read_rpc('[1, 1, 0]').shape==(8,4)
    restored=pickle.loads(pickle.dumps(reader))
    assert restored._env is None
    restored.close()
    reader.close()
    write_json(tmp_path/'inventory.json',inventory)
    normalizer={'a':1,'median':0,'iqr':1,'fit_split':'train'}
    normalizer['normalizer_hash']=canonical_hash(normalizer)
    write_json(tmp_path/'normalizer.json',normalizer)
    window=list(build_windows(build_segments(frames(64)),'split'))[0]
    (tmp_path/'windows.jsonl').write_text(json.dumps(window)+'\n')
    dataset=SensorDataset(tmp_path/'windows.jsonl',tmp_path/'inventory.json',tmp_path/'normalizer.json')
    a=next(iter(torch.utils.data.DataLoader(dataset,num_workers=0,batch_size=1)))
    b=next(iter(torch.utils.data.DataLoader(dataset,num_workers=2,batch_size=1,multiprocessing_context='spawn')))
    assert torch.equal(a['X'],b['X']) and torch.equal(a['point_mask'],b['point_mask'])
    assert path.stat().st_mtime_ns==before.st_mtime_ns
    dataset.reader.close()


def test_authority_fail_closed_before_target_access():
    with pytest.raises(ValueError,match='G0'):
        DirectTargetReader({'authority':{'G0_passed':False}}, {})


def test_snapshot_hash_change_rejected(tmp_path):
    path=tmp_path/'snapshot.bin'
    path.write_bytes(b'1234')
    inventory={'databases':{'rpc':str(path)},'source_manifest_sha256':{'rpc':file_hash(path)}}
    verify_snapshot(inventory,('rpc',))
    path.write_bytes(b'54321')
    with pytest.raises(ValueError,match='checksum'):
        verify_snapshot(inventory,('rpc',))


def test_new_target_export_masks_and_test_lock(tmp_path):
    def pack(value):
        if isinstance(value,np.ndarray):
            return {'__nd__':True,'dtype':str(value.dtype),'shape':list(value.shape),'data':value.tobytes()}
        if isinstance(value,dict):
            return {key:pack(v) for key,v in value.items()}
        return value
    source=tmp_path/'source'
    source.mkdir()
    calib={'vicon_to_cam_rotmatrix':np.eye(3),'radar_to_cam_rotmatrix':np.eye(3),
        'vicon_to_cam_tvec':np.zeros(3),'radar_to_cam_tvec':np.zeros(3)}
    paths={}
    for name in ('params','calib'):
        path=source/(name+'.lmdb')
        paths[name]=str(path)
        env=lmdb.open(str(path),subdir=False,map_size=1024*1024)
        with env.begin(write=True) as tx:
            for i in range(3):
                value=calib if name=='calib' else ({'joints':np.ones((22,3))} if i!=1 else {})
                tx.put(f'[1, 1, {i}]'.encode(),msgpack.packb(pack(value)))
        env.close()
    inventory={'authority':{'G0_passed':True},'databases':paths,'dataset_root':str(source),
        'meta_root':str(tmp_path/'metadata'),'source_manifest_sha256':{name:file_hash(path) for name,path in paths.items()}}
    provenance={'target_equivalence_verified':True,'ordered_names':JOINT_NAMES,
        'source_row_indices':list(range(22)),'direct_joint_unit':'meter','source_frame':'vicon'}
    for name in ('reference_asset','reference_regressor','equivalence_evidence','coordinate_audit'):
        path=tmp_path/(name+'.fixture')
        path.write_text('synthetic fixture, not actual scientific authority')
        provenance[name+'_path']=str(path)
        provenance[name+'_sha256']=file_hash(path)
    write_json(tmp_path/'inventory.json',inventory)
    write_json(tmp_path/'provenance.json',provenance)
    rows=frames(3)
    rows[1]['split']='val'
    rows[2]['split']='test'
    manifest=tmp_path/'frames.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    meta=export_targets(manifest,tmp_path/'inventory.json',tmp_path/'provenance.json',tmp_path/'targets')
    assert meta['count']==2 and meta['splits']==['train','val']
    targets=TargetArrayStore(tmp_path/'targets')
    assert targets.read('[1, 1, 0]')[1].all()
    for key in ('[1, 1, 1]','[1, 1, 2]'):
        positions,mask=targets.read(key)
        assert not mask.any() and np.all(positions==0)
    with pytest.raises(ValueError,match='test lock'):
        export_targets(manifest,tmp_path/'inventory.json',tmp_path/'provenance.json',tmp_path/'forbidden',splits=('test',))
