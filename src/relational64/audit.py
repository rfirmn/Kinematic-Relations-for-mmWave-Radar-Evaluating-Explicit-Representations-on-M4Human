"""Read-only inventory and numerical coordinate diagnostics, with fail-closed G0."""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
import os
import struct
import time
from pathlib import Path

import numpy as np

from .data import clean_rpc, decode_msgpack, decode_rpc, file_hash, parse_source_key, read_json, write_json

UPSTREAM_COMMIT = 'e01b8fed6132306c3553ed6680be912c21594dce'
UPSTREAM = f'https://raw.githubusercontent.com/FanJunqiao/M4Human/{UPSTREAM_COMMIT}/dataset/'
PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]


def _readonly(path):
    import lmdb
    return lmdb.open(str(path), subdir=Path(path).is_dir(), readonly=True, create=False,
                     lock=False, writemap=False, readahead=False, max_readers=256)


def _stat(path):
    st = Path(path).stat()
    return {'size_bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}


def transform_to_radar(points, calib, *, strict=True):
    """Published column-vector convention, represented as NumPy row vectors.

    Numeric consistency cannot establish the correctness of source unit claims.
    """
    r1 = np.asarray(calib['vicon_to_cam_rotmatrix'], np.float64)
    r2 = np.asarray(calib['radar_to_cam_rotmatrix'], np.float64)
    t1 = np.asarray(calib['vicon_to_cam_tvec'], np.float64).reshape(3) * .001
    t2 = np.asarray(calib['radar_to_cam_tvec'], np.float64).reshape(3)
    for r in (r1, r2):
        if r.shape != (3,3) or not np.isfinite(r).all():
            raise ValueError('invalid calibration rotation shape/finite')
        if strict and (not np.allclose(r.T@r, np.eye(3), atol=1e-5, rtol=0) or not np.isclose(np.linalg.det(r),1,atol=1e-5,rtol=0)):
            raise ValueError('calibration rotation violates SO(3) tolerance1e-5')
    if not np.isfinite(np.concatenate((t1,t2))).all():
        raise ValueError('nonfinite calibration translation')
    points = np.asarray(points, np.float64)
    return (points @ r1.T + t1 - t2) @ r2


def coordinate_diagnostics(databases, samples, output_dir):
    """Only train/val samples, chosen by sensor keys before target access."""
    output = Path(output_dir)
    envs = {n: _readonly(databases[n]) for n in ('params','calib','indicator')}
    rows, plots = [], []
    try:
        for sample in samples:
            key, cloud = sample['key'], sample['cloud']
            values = {}
            for name, env in envs.items():
                with env.begin(write=False) as tx:
                    payload = tx.get(key.encode())
                values[name] = None if payload is None else decode_msgpack(payload)
            if values['params'] is None or values['calib'] is None:
                rows.append({'key':key, 'status':'missing_metadata'})
                continue
            param, calib = values['params'], values['calib']
            joints = np.asarray(param['joints'], np.float64)
            r1 = np.asarray(calib['vicon_to_cam_rotmatrix'],np.float64)
            r2 = np.asarray(calib['radar_to_cam_rotmatrix'],np.float64)
            rotation_error = [float(np.max(np.abs(r.T@r-np.eye(3)))) for r in (r1,r2)]
            determinants = [float(np.linalg.det(r)) for r in (r1,r2)]
            mapped = transform_to_radar(joints[:22], calib, strict=False)
            t1 = np.asarray(calib['vicon_to_cam_tvec'],np.float64).reshape(3)*.001
            t2 = np.asarray(calib['radar_to_cam_tvec'],np.float64).reshape(3)
            restored = ((mapped @ r2.T+t2)-t1) @ r1
            exact_inverse = ((mapped @ np.linalg.inv(r2) + t2)-t1) @ np.linalg.inv(r1.T)
            nearest = np.sqrt(((mapped[:,None,:]-cloud[None,:,:3])**2).sum(-1)).min(1) if len(cloud) else np.full(22,np.nan)
            row = {'key':key, 'split':sample['split'], 'params_fields':sorted(param), 'joint_shape':list(joints.shape),
                'indicator':values['indicator'], 'indicator_matches_key':tuple(values['indicator'])==parse_source_key(key),
                'rotation_orthogonality_max_abs':rotation_error, 'rotation_determinants':determinants,
                'rotation_tolerance_passed':max(rotation_error)<1e-5 and max(abs(x-1) for x in determinants)<1e-5,
                'roundtrip_transpose_max_abs_m':float(np.max(np.abs(restored-joints[:22]))),
                'roundtrip_inverse_max_abs_m':float(np.max(np.abs(exact_inverse-joints[:22]))),
                'vicon_to_cam_translation_raw':np.asarray(calib['vicon_to_cam_tvec']).tolist(),
                'radar_to_cam_translation_raw':np.asarray(calib['radar_to_cam_tvec']).tolist(),
                'rpc_centroid_m':cloud[:,:3].mean(0).tolist(), 'mapped_pelvis_m':mapped[0].tolist(),
                'source_pelvis_minus_trans_m':(joints[0]-np.asarray(param['trans'])).tolist(),
                'joint_nearest_rpc_distance_m':nearest.tolist(), 'source_joints_first22':joints[:22].tolist(),
                'mapped_joints_candidate_m':mapped.tolist(), 'calibration':{k:np.asarray(v).tolist() for k,v in calib.items()},
                'mapping_status':'candidate numerically checked; SMPL-X regressor equivalence not established'}
            rows.append(row)
            if sum(1 for x in plots if x[0] == sample['split'])<3:
                plots.append((sample['split'],key,cloud.copy(),mapped))
    finally:
        for env in envs.values():
            env.close()
    result = {'samples':rows, 'target_equivalence_verified':False,
        'T_source_to_radar_candidate':np.eye(4).tolist(), 'canonical_z_vertical_verified':False,
        'axis_handedness_verified':False, 'unit_evidence':UPSTREAM+'data_loader_camera_calibration.py',
        'unit_evidence_caveat':'Published root mapping uses vicon translation /1000 and radar translation unscaled; mesh utility has inconsistent conversion. Numeric plausibility alone is not unit proof.',
        'missing_artifacts':['licensed matching SMPL-X model assets and hashed J_regressor/row map, or independent direct-joint equivalence certificate',
            'source RawPoints generation, coordinate axes/handedness and any cropping/offset definition',
            'raw frame filename-to-key mapping or synchronization metadata validating nominal time after producer reindexing']}
    write_json(output/'coordinate_audit.json',result)
    if plots:
        os.environ.setdefault('MPLCONFIGDIR',str(output/'.matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(plots),3,figsize=(13,3.3*len(plots)),squeeze=False)
        for row,(split,key,cloud,joints) in zip(axes,plots):
            for ax,(a,b) in zip(row,[(0,1),(0,2),(1,2)]):
                ax.scatter(cloud[:,a],cloud[:,b],s=3,c=cloud[:,3],cmap='viridis',alpha=.45)
                for j,p in enumerate(PARENTS):
                    if p>=0:
                        ax.plot(joints[[p,j],a],joints[[p,j],b],color='crimson',linewidth=1)
                ax.scatter(joints[:,a],joints[:,b],s=9,c='crimson')
                ax.set(xlabel='XYZ'[a]+' (candidate m)',ylabel='XYZ'[b]+' (candidate m)',title=split+' '+key)
                ax.set_aspect('equal',adjustable='datalim')
        fig.suptitle('Train/validation only: RPC and candidate transformed joints. Visual agreement is NOT regressor equivalence.',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.98))
        fig.savefig(output/'coordinate_overlay.png',dpi=140)
        plt.close(fig)
    return result


def run_audit(dataset_root, output_dir, *, policy_path, full=False):
    from .prepare import make_split
    source, output = Path(dataset_root).resolve(), Path(output_dir).resolve()
    from .config import ROOT
    if source == output or source in output.parents or ROOT not in output.parents:
        raise ValueError('audit output must be inside this repository and outside the read-only source dataset')
    policy=read_json(policy_path)
    if policy.get('artifact_policy')!='fresh_from_source_only' or policy.get('prior_metadata_inputs')!=[]:
        raise ValueError('Fresh audit cannot consume any prior metadata artifacts')
    if (output/'source_inventory.json').exists():
        raise FileExistsError('Fresh audit destination already contains an inventory; use a new generation')
    output.mkdir(parents=True,exist_ok=True)
    database_names = policy['database_names']
    databases = {name:str(source/file) for name,file in database_names.items()}
    before = {name:_stat(path) for name,path in databases.items()}
    schema=dict(policy['serializer'])
    split = make_split(range(1,21))
    # Persist split before opening any label data.
    write_json(output/'split.lock.json',split)
    split_by_subject = {int(s):name for name in ('train','val','test') for s in split[name]}
    envs = {name:_readonly(path) for name,path in databases.items()}
    counts = {name:env.stat()['entries'] for name,env in envs.items()}
    with envs['rpc'].begin(write=False) as probe:
        first=probe.cursor()
        if not first.first():raise ValueError('RPC source is empty')
        payload=bytes(first.value());candidates=[]
        for order in ('little','big'):
            try:decode_rpc(payload,{**schema,'byte_order':order});candidates.append(order)
            except ValueError:pass
        if len(candidates)!=1:raise ValueError('Source endian is not unambiguously determined by bounded payload decoding')
        schema['byte_order']=candidates[0]
    groups = collections.defaultdict(list)
    nvalues, samples, sample_groups = [], [], set()
    stats = collections.Counter()
    channel_min, channel_max = np.full(4,np.inf), np.full(4,-np.inf)
    channel_sum, channel_sumsq = np.zeros(4), np.zeros(4)
    metadata_missing = collections.Counter()
    start = time.monotonic()
    try:
        with envs['rpc'].begin(write=False) as tx, (output/'sensor_scan.jsonl').open('w',encoding='utf-8') as out:
            other_tx = {n:e.begin(write=False) for n,e in envs.items() if n!='rpc'}
            try:
                for idx,(key,payload) in enumerate(tx.cursor()):
                    if not full and idx>=4096:
                        break
                    key = key.decode('utf-8')
                    subject,action,frame = parse_source_key(key)
                    points = decode_rpc(payload,schema)
                    clean, count = clean_rpc(points,schema)
                    nvalues.append(len(clean))
                    stats.update({'frames':1,'points_raw':len(points),'points_clean':len(clean),'nonfinite_rows':count['nonfinite'],
                                  'zero_rows_kept':int(np.all(clean==0,axis=1).sum()),'sensor_invalid':int(len(clean)<8)})
                    if len(clean):
                        values = clean.astype(np.float64)
                        channel_min = np.minimum(channel_min,values.min(0))
                        channel_max = np.maximum(channel_max,values.max(0))
                        channel_sum += values.sum(0)
                        channel_sumsq += np.square(values).sum(0)
                    groups[(subject,action)].append(frame)
                    for n,t in other_tx.items():
                        metadata_missing[n] += int(t.get(key.encode()) is None)
                    out.write(json.dumps({'source_key':key,'subject_id':subject,'action_id':action,'source_frame_id':frame,
                        'clean_point_count':len(clean),'sensor_valid':len(clean)>=8,'nonfinite':count['nonfinite'],
                        'exclusion_reason':None if len(clean)>=8 else 'fewer_than_8_clean_points'},separators=(',',':'))+'\n')
                    split_name = split_by_subject.get(subject)
                    if split_name in ('train','val') and (subject,action) not in sample_groups and sum(s==subject for s,a in sample_groups)<2 and len(clean)>=8:
                        samples.append({'key':key,'split':split_name,'cloud':clean})
                        sample_groups.add((subject,action))
                    if idx and idx%100000==0:
                        print(json.dumps({'audit_frames':idx,'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
            finally:
                for t in other_tx.values():
                    t.abort()
    finally:
        for env in envs.values():
            env.close()
    recording_rows = []
    for (subject,action),frames in sorted(groups.items()):
        ordered = sorted(frames)
        differences = np.diff(ordered)
        recording_rows.append({'subject_id':subject,'action_id':action,'recording_id':f's{subject}/a{action}',
            'count':len(frames),'first':ordered[0],'last':ordered[-1], 'duplicate_count':len(frames)-len(set(frames)),
            'missing_index_count':int(np.maximum(differences-1,0).sum()),'gap_count':int((differences>1).sum()),
            'boundary_authority':'producer subject/action directories; hidden repeated takes cannot be inferred from keys'})
    write_json(output/'recordings.json',recording_rows)
    coord = coordinate_diagnostics(databases,samples,output)
    mean = channel_sum/max(stats['points_clean'],1)
    std = np.sqrt(np.maximum(channel_sumsq/max(stats['points_clean'],1)-mean**2,0))
    hashes = {}
    if full:
        for name,path in databases.items():
            print(json.dumps({'hashing_source':name}),flush=True)
            hashes[name] = file_hash(path)
    after = {name:_stat(path) for name,path in databases.items()}
    immutable_during_audit = before==after
    blockers = ['Direct joints have no independently verified equivalence to hashed SMPL-X mesh/regressor reference.',
        'RawPoints producer axes/handedness, source offsets and upstream crop privilege not established.',
        'Producer renumbers union of radar/RGB filenames; contiguous output indices cannot prove absent original-frame gaps. Need raw filename/synchronization mapping.']
    if not full:
        blockers.append('Only sampled inventory; full scan/hash pending.')
    if not immutable_during_audit:
        blockers.append('Source file stat changed during read-only audit.')
    if any(not r.get('rotation_tolerance_passed',False) for r in coord['samples']):
        blockers.append('One or more sampled source calibration matrices violate strict SO(3) tolerance1e-5.')
    inventory = {'protocol_id':'m4human_relational64','inventory_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'schema_id':'relational64_source_inventory_2','artifact_generation_id':policy['artifact_generation_id'],
        'artifact_policy':'fresh_from_source_only','prior_artifacts_reused':[],
        'dataset_root':str(source),'meta_root':str(output),'databases':databases,'serializer':schema,
        'channels':['x','y','z','intensity'],'channel4_is_doppler':False,'dtype':'float32',
        'immutable_snapshot':immutable_during_audit and full,
        'immutability_evidence':{'before':before,'after':after,'unchanged_during_audit':immutable_during_audit,
            'policy_path':str(Path(policy_path).resolve()),'policy_sha256':file_hash(policy_path),
            'limitation':'No writer exclusion guaranteed beyond audited interval; revalidate hashes before runs.'},
        'source_manifest_sha256':hashes,
        'entries':counts,'full_scan':full,'statistics':dict(stats),'missing_metadata_keys':dict(metadata_missing),
        'subjects':sorted({s for s,a in groups}),'recording_count':len(groups),'scan_path':str(output/'sensor_scan.jsonl'),
        'absent_subject_action_pairs':[[s,a] for s in range(1,21) for a in range(1,51) if (s,a) not in groups],
        'source_index_gaps':sum(r['gap_count'] for r in recording_rows),
        'missing_source_indices':sum(r['missing_index_count'] for r in recording_rows),
        'scan_sha256':file_hash(output/'sensor_scan.jsonl'),'point_count_quantiles':dict(zip(['min','p25','median','p75','p95','max'],np.quantile(nvalues,[0,.25,.5,.75,.95,1]).tolist())),
        'channel_statistics':{'min':channel_min.tolist(),'max':channel_max.tolist(),'mean':mean.tolist(),'std':std.tolist()},
        'T_source_to_radar':np.eye(4).tolist(),'coordinate_mapping_status':'candidate_identity_only_not_authorized',
        'time_source':'nominal_frame_index','nominal_hz':12.,'nominal_time_authorized':False,
        'time_evidence':{'paper':'https://arxiv.org/html/2512.12378v3','producer':UPSTREAM+'m4human_utils.py',
            'measured_timestamps_available':False,'caveat':'Nominal hardware rate plus reindexed cache keys does not prove original contiguous temporal grid.'},
        'producer_evidence':{'commit':UPSTREAM_COMMIT,'serializer':UPSTREAM+'lmdb_utils.py','packing':UPSTREAM+'m4human_utils.py',
            'byte_order':'little observed by full payload decode; native producer endian not assumed from host',
            'sentinel':'none; raw saved before inference padding; zero rows retained',
            'rpc_preprocessing_privilege':'Cache writes RawPoints unchanged; source RawPoints generation unavailable, cropping privilege unresolved'},
        'source_policy_sha256':file_hash(policy_path),
        'authority':{'G0_passed':False,'blockers':blockers},'elapsed_s':time.monotonic()-start}
    write_json(output/'source_inventory.json',inventory)
    write_json(output/'acceptance.json',{'G0':False,'status':'audit_completed_authority_blocked','blockers':blockers,
        'evidence_hashes':{n:file_hash(output/n) for n in ('source_inventory.json','coordinate_audit.json','split.lock.json','recordings.json')}})
    (output/'report.md').write_text('# Stage2: source audit\n\nRead-only numerical inventory completed. G0 remains blocked.\n\n'+
        '\n'.join('- '+b for b in blockers)+'\n\nThe coordinate overlay is a diagnostic candidate, not proof of target equivalence. No test target labels were opened.\n',encoding='utf-8')
    return inventory
