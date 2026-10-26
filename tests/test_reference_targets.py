"""Synthetic body-model stub only; no licensed SMPL-X asset is available here."""
from types import SimpleNamespace

import lmdb
import msgpack
import numpy as np
import pytest
import torch

from relational64.audit import transform_to_radar
from relational64.data import JOINT_NAMES,file_hash,write_json
from relational64.reference_targets import ReferenceSMPLXTargetReader,compare_direct_reference


class StubBody(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('J_regressor',torch.eye(22))
    def forward(self,**kwargs):
        assert kwargs['body_pose'].shape==(1,63)
        assert kwargs['left_hand_pose'].shape==(1,45)
        assert kwargs['expression'].shape==(1,10)
        base=torch.arange(66,dtype=torch.float32).reshape(1,22,3)/100
        return SimpleNamespace(vertices=base+kwargs['transl'][:,None,:])


def model_factory(**kwargs):
    assert kwargs['use_pca'] is False
    assert kwargs['gender']=='female'
    assert kwargs['num_betas']==10
    return StubBody()


def packed(value):
    if isinstance(value,np.ndarray):
        return {'__nd__':True,'dtype':str(value.dtype),'shape':list(value.shape),'data':value.tobytes()}
    if isinstance(value,dict):
        return {k:packed(v) for k,v in value.items()}
    return value


@pytest.fixture
def fixture(tmp_path):
    calib={'vicon_to_cam_rotmatrix':np.array([[0.,-1,0],[1,0,0],[0,0,1]]),
        'radar_to_cam_rotmatrix':np.eye(3),'vicon_to_cam_tvec':np.array([1000.,0,0]),
        'radar_to_cam_tvec':np.array([0.,1,0])}
    params={'betas':np.zeros(10),'pose_body':np.zeros(63),'root_orient':np.zeros(3),
        'trans':np.array([1000.,2000,3000]),'gender':0,
        'joints':np.arange(66).reshape(22,3)/100+np.array([1.,2,3])}
    paths={}
    for name,obj in [('params',params),('calib',calib)]:
        path=tmp_path/(name+'.lmdb');paths[name]=str(path)
        env=lmdb.open(str(path),subdir=False,map_size=1024*1024)
        with env.begin(write=True) as tx:
            tx.put(b'[1, 2, 0]',msgpack.packb(packed(obj)))
        env.close()
    asset=tmp_path/'SMPLX_FEMALE.npz';asset.write_bytes(b'synthetic fixture asset, not a licensed model')
    reg=tmp_path/'regressor.npy';np.save(reg,np.eye(22,dtype=np.float32))
    coordinate=tmp_path/'coordinate.json'
    write_json(coordinate,{'output_unit':'meter','source_frame':'vicon','vicon_to_cam_translation_unit':'millimeter','radar_to_cam_translation_unit':'meter'})
    provenance={'evidence':'synthetic_software_fixture','model_output_unit':'meter','parameter_translation_unit':'millimeter',
        'direct_joint_unit':'meter','source_frame':'vicon','ordered_names':JOINT_NAMES,'source_row_indices':list(range(22)),
        'coordinate_audit_path':str(coordinate),'coordinate_audit_sha256':file_hash(coordinate),'num_betas':10,'num_expression_coeffs':10,
        'default_hand_face_audit':True,'default_hand_face_values':{k:[0]*n for k,n in
            [('left_hand_pose',45),('right_hand_pose',45),('jaw_pose',3),('leye_pose',3),('reye_pose',3),('expression',10)]},
        'gender_mapping':{'0':'female'},'assets':{'female':{'path':str(asset),'sha256':file_hash(asset),
            'regressor_path':str(reg),'regressor_sha256':file_hash(reg),'trusted_distribution':True}}}
    inventory={'authority':{'G0_passed':False},'subjects':list(range(1,21)),
        'databases':paths,'source_manifest_sha256':{n:file_hash(p) for n,p in paths.items()}}
    return inventory,provenance,calib,params,tmp_path


def test_reference_transform_once_units_pelvis_and_comparison(fixture):
    inventory,provenance,calib,params,path=fixture
    reader=ReferenceSMPLXTargetReader(inventory,provenance,diagnostic=True,_model_factory=model_factory)
    try:
        positions,mask=reader.read('[1, 2, 0]')
        expected=transform_to_radar(params['joints'],calib)
        np.testing.assert_allclose(positions,expected,atol=1e-6)
        assert mask.all()
        assert not np.array_equal(positions[0],transform_to_radar(params['trans'][None]*.001,calib)[0])
        report=compare_direct_reference(reader,['[1, 2, 0]'],path/'comparison.json',tolerance_m=1e-6)
        assert report['all_samples_passed'] and report['G0_passed'] is False and report['target_equivalence_verified'] is False
        with pytest.raises(ValueError,match='test target'):
            compare_direct_reference(reader,['[3, 2, 0]'],path/'forbidden.json',tolerance_m=1e-6)
    finally:
        reader.close()


def test_reference_asset_gate_and_no_silent_defaults(fixture):
    inventory,provenance,_,_,_=fixture
    with pytest.raises(ValueError,match='G0'):
        ReferenceSMPLXTargetReader(inventory,provenance,_model_factory=model_factory)
    bad={**provenance,'default_hand_face_audit':False}
    with pytest.raises(ValueError,match='hand/face'):
        ReferenceSMPLXTargetReader(inventory,bad,diagnostic=True,_model_factory=model_factory)
    provenance['assets']['female']['sha256']='0'*64
    with pytest.raises(ValueError,match='asset checksum'):
        ReferenceSMPLXTargetReader(inventory,provenance,diagnostic=True,_model_factory=model_factory)
