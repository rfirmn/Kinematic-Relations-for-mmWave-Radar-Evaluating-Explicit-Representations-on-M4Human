"""Optional offline SMPL-X reference export. No body-model import during training.

Assets are supplied locally with hashes and distribution attestation. This module
never downloads assets and never infers gender, body shape or hand/face defaults.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .data import JOINT_NAMES, assert_fresh_inventory, canonical_hash, decode_msgpack, file_hash, parse_source_key, read_json, verify_snapshot, write_json
from .config import active_artifact_path


class ReferenceSMPLXTargetReader:
    def __init__(self, inventory, provenance, *, diagnostic=False, _model_factory=None):
        self.inventory = read_json(active_artifact_path(inventory)) if isinstance(inventory,(str,Path)) else inventory
        source_provenance = read_json(active_artifact_path(provenance)) if isinstance(provenance,(str,Path)) else provenance
        self.provenance = {**source_provenance, 'assets':{gender:dict(spec) for gender,spec in source_provenance.get('assets',{}).items()}}
        p=self.provenance
        generation=self.inventory.get('artifact_generation_id')
        assert_fresh_inventory(self.inventory,expected_generation=generation)
        if generation is not None:
            if p.get('artifact_generation_id')!=generation:
                raise ValueError('Reference target provenance artifact generation mismatch')
            # Validate every nested research/asset input before hashing or
            # opening any of them. A checksum never authorizes an old path.
            if p.get('coordinate_audit_path'):
                p['coordinate_audit_path']=str(active_artifact_path(p['coordinate_audit_path']))
            for spec in p['assets'].values():
                for key in ('path','regressor_path'):
                    if spec.get(key):
                        spec[key]=str(active_artifact_path(spec[key]))
        if not diagnostic and self.inventory.get('authority',{}).get('G0_passed') is not True:
            raise ValueError('G0 authority required for canonical reference target export')
        self.diagnostic=bool(diagnostic)
        if _model_factory is not None and p.get('evidence')!='synthetic_software_fixture':
            raise ValueError('injected body-model factory is restricted to explicit synthetic fixtures')
        if p.get('model_output_unit')!='meter' or p.get('parameter_translation_unit') not in ('meter','millimeter'):
            raise ValueError('explicit model output meter and parameter translation unit required')
        if p.get('source_frame') not in ('vicon','radar') or p.get('ordered_names')!=JOINT_NAMES:
            raise ValueError('explicit source frame and canonical ordered joint map required')
        if not p.get('coordinate_audit_path') or file_hash(p['coordinate_audit_path'])!=p.get('coordinate_audit_sha256'):
            raise ValueError('coordinate evidence hash mismatch')
        coordinate=read_json(p['coordinate_audit_path'])
        if coordinate.get('output_unit')!='meter' or coordinate.get('source_frame')!=p['source_frame']:
            raise ValueError('coordinate evidence unit/source frame mismatch')
        if p['source_frame']=='vicon' and (coordinate.get('vicon_to_cam_translation_unit')!='millimeter' or coordinate.get('radar_to_cam_translation_unit')!='meter'):
            raise ValueError('reference transform requires audited Vicon-mm / radar-meter translation units')
        self.indices=p.get('source_row_indices')
        if not isinstance(self.indices,list) or len(self.indices)!=22 or len(set(self.indices))!=22 or any(type(i)is not int or i<0 for i in self.indices):
            raise ValueError('explicit22 distinct regressor row indices required')
        for name in ('num_betas','num_expression_coeffs'):
            if type(p.get(name))is not int or p[name]<1:
                raise ValueError('positive explicit body model dimension required: '+name)
        defaults=p.get('default_hand_face_values',{})
        dimensions={'left_hand_pose':45,'right_hand_pose':45,'jaw_pose':3,'leye_pose':3,'reye_pose':3,'expression':p['num_expression_coeffs']}
        if p.get('default_hand_face_audit') is not True:
            raise ValueError('audited hand/face parameter conventions required')
        self.defaults={}
        for name,size in dimensions.items():
            value=np.asarray(defaults.get(name),np.float32)
            if value.shape!=(size,) or not np.isfinite(value).all():
                raise ValueError('explicit finite hand/face default required: '+name)
            self.defaults[name]=value
        if not isinstance(p.get('gender_mapping'),dict) or not p['gender_mapping']:
            raise ValueError('explicit source-gender to asset mapping required')
        assets=p.get('assets',{})
        if not assets:
            raise ValueError('licensed local SMPL-X assets required; no automatic download')
        self.regressors={}
        for gender,spec in assets.items():
            if gender not in ('male','female','neutral') or spec.get('trusted_distribution') is not True:
                raise ValueError('trusted male/female/neutral asset required')
            if not spec.get('path') or file_hash(spec['path'])!=spec.get('sha256'):
                raise ValueError('SMPL-X asset checksum mismatch: '+gender)
            if not spec.get('regressor_path') or file_hash(spec['regressor_path'])!=spec.get('regressor_sha256'):
                raise ValueError('SMPL-X regressor checksum mismatch: '+gender)
            reg=np.load(spec['regressor_path'],allow_pickle=False)
            if reg.ndim!=2 or reg.shape[0]!=22 or reg.dtype.kind not in 'fiu' or not np.isfinite(reg).all() or not np.allclose(reg.sum(1),1,atol=1e-5,rtol=0):
                raise ValueError('reference22-row regressor must be finite and affine-normalized')
            self.regressors[gender]=reg.astype(np.float64)
        if not set(p['gender_mapping'].values())<=set(assets):
            raise ValueError('gender mapping refers to unavailable asset')
        verify_snapshot(self.inventory,('params','calib'))
        self._factory=_model_factory
        self._models={}
        self._envs={}
        self._pid=None

    def _environment(self):
        if self._pid!=os.getpid():
            from .audit import _readonly
            self._envs={name:_readonly(self.inventory['databases'][name]) for name in ('params','calib')}
            self._models={}
            self._pid=os.getpid()
        return self._envs

    def _model(self,gender):
        if gender not in self._models:
            if self._factory is None:
                try:
                    import smplx
                except ImportError as exc:
                    raise ValueError('optional smplx package required only for offline reference export') from exc
                factory=smplx.create
            else:
                factory=self._factory
            spec=self.provenance['assets'][gender]
            model=factory(model_path=spec['path'],model_type='smplx',gender=gender,use_pca=False,
                num_betas=self.provenance['num_betas'],num_expression_coeffs=self.provenance['num_expression_coeffs'],
                ext=Path(spec['path']).suffix.lstrip('.'))
            model.eval().requires_grad_(False)
            reg=model.J_regressor.detach().cpu().numpy()
            if max(self.indices)>=len(reg) or reg[self.indices].shape!=self.regressors[gender].shape or not np.allclose(reg[self.indices],self.regressors[gender],atol=1e-7,rtol=0):
                raise ValueError('supplied regressor does not match the pinned model asset and row map')
            self._models[gender]=model
        return self._models[gender]

    def read(self,source_key):
        import torch
        from .audit import transform_to_radar
        parse_source_key(source_key)
        payloads={}
        for name,env in self._environment().items():
            with env.begin(write=False) as tx:
                payloads[name]=tx.get(source_key.encode())
        if payloads['calib'] is None:
            raise ValueError('missing calibration is a systematic reference export error')
        calib=decode_msgpack(payloads['calib'])
        if self.provenance['source_frame']=='vicon':
            transform_to_radar(np.zeros((1,3)),calib)
        if payloads['params'] is None:
            return np.zeros((22,3),np.float32),np.zeros(22,bool)
        params=decode_msgpack(payloads['params'])
        if not isinstance(params,dict):
            raise ValueError('body parameters must be mapping')
        fields={'betas':(self.provenance['num_betas'],),'pose_body':(63,),'root_orient':(3,),'trans':(3,)}
        if 'gender' not in params or any(name not in params for name in fields):
            return np.zeros((22,3),np.float32),np.zeros(22,bool)
        gender=self.provenance['gender_mapping'].get(str(params['gender']))
        if gender not in self.provenance['assets']:
            raise ValueError('source gender has no explicit asset mapping')
        arrays={name:np.asarray(params[name],np.float32).copy() for name in fields}
        if any(arrays[name].shape!=shape for name,shape in fields.items()):
            raise ValueError('systematic body parameter shape mismatch')
        if any(not np.isfinite(value).all() for value in arrays.values()):
            return np.zeros((22,3),np.float32),np.zeros(22,bool)
        arrays['trans']*=1 if self.provenance['parameter_translation_unit']=='meter' else .001
        kwargs={target:torch.from_numpy(arrays[source][None]) for target,source in
            [('betas','betas'),('body_pose','pose_body'),('global_orient','root_orient'),('transl','trans')]}
        for name,default in self.defaults.items():
            value=np.asarray(params.get(name,default),np.float32)
            if value.shape!=default.shape:
                raise ValueError('hand/face parameter shape mismatch: '+name)
            if not np.isfinite(value).all():
                return np.zeros((22,3),np.float32),np.zeros(22,bool)
            kwargs[name]=torch.from_numpy(value[None].copy())
        model=self._model(gender)
        with torch.no_grad():
            vertices=model(**kwargs).vertices.detach().cpu().numpy()[0].astype(np.float64)
        reg=self.regressors[gender]
        if vertices.shape!=(reg.shape[1],3) or not np.isfinite(vertices).all():
            raise ValueError('reference vertices invalid shape/finite')
        if self.provenance['source_frame']=='vicon':
            vertices=transform_to_radar(vertices,calib)
        result=(reg@vertices).astype(np.float32)
        return result,np.ones(22,bool)

    def close(self):
        for env in self._envs.values():
            env.close()
        self._envs,self._models,self._pid={},{},None


def compare_direct_reference(reference_reader, source_keys, output_path, *, tolerance_m):
    """Diagnostic sample comparison; does not set G0 or bless global equivalence."""
    from .audit import transform_to_radar
    if not reference_reader.diagnostic:
        raise ValueError('equivalence audit requires explicit diagnostic reference reader')
    if reference_reader.provenance.get('direct_joint_unit')!='meter':
        raise ValueError('comparison requires explicitly audited direct joint unit meter')
    if not np.isfinite(tolerance_m) or tolerance_m<=0:
        raise ValueError('predeclared positive equivalence tolerance required')
    from .prepare import make_split
    subjects=make_split(reference_reader.inventory['subjects'])
    allowed=set(map(int,subjects['train']+subjects['val']))
    records=[]
    for key in source_keys:
        if parse_source_key(key)[0] not in allowed:
            raise ValueError('equivalence diagnostic cannot open test target labels')
        reference,valid=reference_reader.read(key)
        envs=reference_reader._environment()
        with envs['params'].begin(write=False) as tx:
            params=decode_msgpack(tx.get(key.encode()))
        with envs['calib'].begin(write=False) as tx:
            calib=decode_msgpack(tx.get(key.encode()))
        direct=np.asarray(params['joints'],np.float64)[reference_reader.indices]
        if reference_reader.provenance['source_frame']=='vicon':
            direct=transform_to_radar(direct,calib)
        valid &= np.isfinite(direct).all(1)
        distances=np.linalg.norm(direct-reference,axis=-1)
        records.append({'source_key':key,'valid_joint_count':int(valid.sum()),
            'per_joint_error_m':[float(v) if ok else None for v,ok in zip(distances,valid)],
            'max_error_m':float(distances[valid].max()) if valid.any() else None,
            'sample_passed':bool(valid.all() and np.all(distances<=tolerance_m))})
    result={'evidence':'numerical_samples_only','G0_passed':False,'target_equivalence_verified':False,
        'reference_provenance_hash':canonical_hash(reference_reader.provenance),'tolerance_m':tolerance_m,
        'samples':records,'all_samples_passed':bool(records and all(r['sample_passed'] for r in records)),
        'limitation':'Numerical sample agreement needs separate provenance/coverage review before a global equivalence certificate.'}
    write_json(output_path,result)
    return result
