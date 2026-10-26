"""Deterministic sensor-only manifest preparation and train-only statistics."""
from __future__ import annotations

import collections
import hashlib
import heapq
import itertools
import json
from pathlib import Path

import numpy as np

from .data import RPCOnlyReader, canonical_hash, clean_rpc, file_hash, read_json, write_json


def make_split(subjects):
    subjects = sorted(set(int(x) for x in subjects))
    if len(subjects)!=20:
        raise ValueError(f'expected20 subjects; found {len(subjects)}; completeness audit required')
    ranked = sorted(subjects,key=lambda s:(hashlib.sha256(f'm4human_relational64/split/42/{s}'.encode()).hexdigest(),s))
    result = {'protocol_id':'m4human_relational64','seed':42,'test':[str(x) for x in ranked[:4]],
        'val':[str(x) for x in ranked[4:8]],'train':[str(x) for x in ranked[8:]],
        'algorithm':'SHA256 UTF8 m4human_relational64/split/42/<decimal_id>, ascending hexadecimal'}
    result['split_hash'] = canonical_hash(result)
    return result


def build_segments(frames, max_gap_s=.125):
    """Preserve given temporal order; never inspect target/annotation fields."""
    previous, segment_num = None, -1
    subjects, seen = {}, set()
    for original in frames:
        row = dict(original)
        subject = str(row['subject_id'])
        if subject in subjects and subjects[subject]!=row['split']:
            raise ValueError('subject split leakage')
        subjects[subject] = row['split']
        identity = (row['recording_id'],row['source_frame_id'])
        if identity in seen:
            raise ValueError('duplicate recording frame')
        seen.add(identity)
        if not np.isfinite(row['time_s']):
            row['sensor_valid'] = False
            row['exclusion_reason'] = 'nonfinite_time'
        boundary = previous is None or any(row[k]!=previous[k] for k in ('subject_id','recording_id','split','time_source'))
        if not boundary:
            delta = row['time_s']-previous['time_s']
            boundary = delta<=0 or delta>max_gap_s+1e-12 or row['source_frame_id']!=previous['source_frame_id']+1
            boundary |= not previous['sensor_valid'] or not row['sensor_valid']
        if boundary:
            segment_num += 1
        row['segment_id'] = f"{row['recording_id']}/segment{segment_num}"
        yield row
        previous = row


def build_windows(frames, split_hash, strides=None, *, generation_id=None):
    strides = strides or {'train':8,'val':16,'test':16}
    sampling_policy = {'seed':42,'max_points':150,'train_seed':'train/seed/epoch/window_uid/frame_index',
        'eval_seed':'sampling/42/window_uid/frame_index','replacement':False,'center':'all_clean_first_frame_xy_before_sampling'}
    for _, group in itertools.groupby(frames,key=lambda r:(r['recording_id'],r['segment_id'])):
        segment = list(group)
        if not segment or not all(row['sensor_valid'] for row in segment):
            continue
        for start in range(0,len(segment)-63,strides[segment[0]['split']]):
            rows = segment[start:start+64]
            times = [float(row['time_s']) for row in rows]
            identity = {'protocol_id':'m4human_relational64','split_hash':split_hash,'recording_id':rows[0]['recording_id'],
                'segment_id':rows[0]['segment_id'],'frame_uids':[r['frame_uid'] for r in rows],
                'timestamp_hash':canonical_hash(times),'sampling_policy_hash':canonical_hash(sampling_policy)}
            if generation_id is not None:identity['artifact_generation_id']=generation_id
            uid = canonical_hash(identity)
            yield {'window_uid':uid,'artifact_generation_id':generation_id,'split':rows[0]['split'],'subject_id':rows[0]['subject_id'],
                'recording_id':rows[0]['recording_id'],'segment_id':rows[0]['segment_id'],
                'frame_uids':identity['frame_uids'],'source_keys':[r['source_key'] for r in rows],
                'time_s':times,'duration_s':times[-1]-times[0],'time_source':rows[0]['time_source'],
                'sampling_seed_digest':canonical_hash([f'sampling/42/{uid}/{i}' for i in range(64)]),
                'sampling_policy_hash':identity['sampling_policy_hash']}


def fit_intensity_normalizer(frames, reader, max_points=1000000):
    """Bottom-k SHA256 sample of original clean-row identities, independent of iteration order."""
    reservoir = []
    considered = 0
    for frame in frames:
        if frame['split']!='train' or not frame['sensor_valid']:
            continue
        raw = reader.read_rpc(frame['source_key'])
        finite = np.isfinite(raw).all(1)
        if reader.schema.get('sentinel_policy')=='exact_row':
            finite &= ~np.all(raw==np.asarray(reader.schema['sentinel_row']),axis=1)
        for row_index in np.flatnonzero(finite):
            identity = f"{frame['frame_uid']}/{int(row_index)}"
            digest = hashlib.sha256(identity.encode()).digest()
            rank = int.from_bytes(digest,'big')
            item = (-rank,identity,float(raw[row_index,3]))
            considered += 1
            if len(reservoir)<max_points:
                heapq.heappush(reservoir,item)
            elif rank < -reservoir[0][0]:
                heapq.heapreplace(reservoir,item)
    if not reservoir:
        raise ValueError('no valid train points for intensity normalization')
    selected = sorted(reservoir,key=lambda t:(-t[0],t[1]))
    intensity = np.array([x[2] for x in selected],np.float64)
    a = max(float(np.median(np.abs(intensity))),1e-6)
    u = np.sign(intensity)*np.log1p(np.abs(intensity)/a)
    q1,median,q3 = np.quantile(u,[.25,.5,.75])
    result = {'fit_split':'train','a':a,'median':float(median),'iqr':max(float(q3-q1),1e-6),
        'selected_points':len(selected),'considered_train_points':considered,
        'sample_digest':canonical_hash([x[1] for x in selected]),'sampling':'smallest SHA256(frame_uid/row_index)',
        'clip':[-10,10],'xyz_standardized':False}
    result['normalizer_hash'] = canonical_hash(result)
    return result


def prepare_data(inventory_path, output_dir, *, diagnostic=False, fit_normalizer=True,expected_generation=None):
    inventory = read_json(inventory_path)
    from .data import assert_fresh_inventory
    expected_generation=expected_generation or inventory.get('artifact_generation_id')
    assert_fresh_inventory(inventory,expected_generation)
    if expected_generation is not None:
        from .config import active_artifact_path
        active_artifact_path(inventory_path)
        active_artifact_path(output_dir)
    if not inventory.get('full_scan'):
        raise ValueError('full source inventory required before preparation')
    authority = inventory.get('authority',{}).get('G0_passed') is True
    if not authority and not diagnostic:
        raise ValueError('G0 unresolved: only explicit diagnostic manifests permitted; training remains blocked')
    if expected_generation is not None:
        active_artifact_path(inventory['scan_path'])
    output = Path(output_dir).resolve()
    for root in ('dataset_root',):
        path = Path(inventory[root]).resolve()
        if output==path or path in output.parents:
            raise ValueError('prepared outputs must be outside read-only source')
    if (output/'windows.jsonl').exists():raise FileExistsError('Manifest already exists; use a new artifact generation')
    output.mkdir(parents=True,exist_ok=True)
    split = make_split(inventory['subjects'])
    write_json(output/'split.lock.json',split)
    if file_hash(inventory['scan_path'])!=inventory['scan_sha256']:
        raise ValueError('sensor scan checksum mismatch')
    split_lookup = {s:name for name in ('train','val','test') for s in split[name]}
    # Numeric ordering is an explicit producer-index interpretation, recorded as provisional when G0 fails.
    groups = collections.defaultdict(list)
    with open(inventory['scan_path'],encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            groups[(row['subject_id'],row['action_id'])].append(row)
    frame_count, window_count, segment_count = collections.Counter(),collections.Counter(),0
    with (output/'frames.jsonl').open('w',encoding='utf-8') as ff, (output/'windows.jsonl').open('w',encoding='utf-8') as wf:
        for (subject,action),group in sorted(groups.items()):
            group.sort(key=lambda row:row['source_frame_id'])
            first = group[0]['source_frame_id']
            for row in group:
                row.update({'frame_uid':canonical_hash(['m4human_relational64',row['source_key']]),
                    'recording_id':f's{subject}/a{action}','split':split_lookup[str(subject)],
                    'time_s':(row['source_frame_id']-first)/12.,'time_source':'nominal_frame_index',
                    'time_authority_status':'verified' if authority else 'provisional_unverified_reindex_mapping'})
            segmented = list(build_segments(group))
            segment_count += len(set(row['segment_id'] for row in segmented))
            for row in segmented:
                ff.write(json.dumps(row,separators=(',',':'))+'\n')
                frame_count[row['split']] += 1
            for window in build_windows(segmented,split['split_hash'],generation_id=inventory.get('artifact_generation_id')):
                wf.write(json.dumps(window,separators=(',',':'))+'\n')
                window_count[window['split']] += 1
    result = {'protocol_id':'m4human_relational64','artifact_generation_id':inventory.get('artifact_generation_id'),'prior_artifacts_reused':[],'diagnostic_only':not authority,'training_authorized':authority,
        'frame_counts':dict(frame_count),'window_counts':dict(window_count),'segment_count':segment_count,
        'split_hash':split['split_hash'],'source_inventory_sha256':file_hash(inventory_path),
        'frames_sha256':file_hash(output/'frames.jsonl'),'windows_sha256':file_hash(output/'windows.jsonl'),
        'normalizer_created':False,'target_exported':False,
        'notes':['Window membership uses sensor quality only; annotation missingness cannot alter boundaries.',
            'Diagnostic timestamps use assumed nominal12Hz; no actual time measurements or physical velocity claims.']}
    if authority and fit_normalizer:
        reader = RPCOnlyReader(inventory)
        try:
            with (output/'frames.jsonl').open(encoding='utf-8') as f:
                norm = fit_intensity_normalizer((json.loads(line) for line in f),reader)
            norm['artifact_generation_id']=inventory.get('artifact_generation_id')
            norm.pop('normalizer_hash',None);norm['normalizer_hash']=canonical_hash(norm)
            write_json(output/'intensity_normalizer.json',norm)
            result['normalizer_created'] = True
        finally:
            reader.close()
    write_json(output/'prepare_summary.json',result)
    return result
