"""Physical, counterfactual donor, and multi-condition evaluation orchestration."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from .config import file_hash,identity,write_json
from .data import JsonlSequence,parse_source_key
from .kinematics import estimate_velocities
from .qa import parse_answer
from .evaluation import physical_metrics,build_donor_pairs,evaluate_donors,condition_contrasts,paired_subject_bootstrap,evaluate_qa
from .alignment import project_window
from .language import MARKER,prompt_messages

def evaluate_physical_cache(cache,windows_path,targets,output_dir):
    if cache.meta['source'].get('windows_hash')!=file_hash(windows_path):raise ValueError('Physical evaluation window manifest mismatch')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    windows=JsonlSequence(windows_path,cache.meta['split']);window_index={r['window_uid']:r for r in windows}
    totals={'all_frames':{},'task_endpoints':{}};failure=False
    with (output/'physical_per_window.jsonl').open('w',encoding='utf-8') as stream:
        for item in cache:
            row=window_index[item['uid']];frames=[targets.read(key) for key in row['source_keys']]
            target=np.stack([p for p,m in frames]);mask=np.stack([m for p,m in frames])
            velocity=estimate_velocities(target,mask,item['time_s'].numpy())
            global_prediction=item['P'].numpy()+item['centroid'].numpy()
            record={'window_uid':item['uid'],'subject_id':row['subject_id']}
            for panel,interval in (('all_frames',None),('task_endpoints',(6,63))):
                metric=physical_metrics(global_prediction,target,mask,item['V'].numpy(),velocity.velocity,velocity.valid,interval=interval)
                record[panel]=metric
                failure |= metric['status']!='valid'
                for name,value in metric['metrics'].items():
                    total=totals[panel].setdefault(name,{'numerator':0.,'count':0,'nonfinite_prediction_count':0,'unit':value['unit']})
                    total['count']+=value['count'];total['nonfinite_prediction_count']+=value['nonfinite_prediction_count']
                    if value['numerator'] is not None:total['numerator']+=value['numerator']
            stream.write(json.dumps(record,allow_nan=False)+'\n')
    for panel in totals.values():
        for metric in panel.values():
            bad=metric['nonfinite_prediction_count']>0
            metric['mean']=None if bad or not metric['count'] else metric['numerator']/metric['count']
            if bad:metric['numerator']=None
    report={'status':'failed_nonfinite_prediction' if failure else 'valid','split':cache.meta['split'],'window_count':len(cache),
            'aggregation':'valid window/endpoint/joint; overlap retained','cache_identity':cache.meta['identity'],
            'target_metadata_hash':file_hash(targets.root/'metadata.json'),'panels':totals}
    write_json(output/'physical_metrics.json',report);return report

def match_donors(records,windows_path,output,*,joint_map_hash,coordinate_convention,seed=42):
    expected={r['window_uid'] for r in records}
    windows={r['window_uid']:r for r in JsonlSequence(windows_path) if r['window_uid'] in expected}
    candidates=[]
    for r in records:
        if r['reference_status']!='defined':continue
        w=windows[r['window_uid']]
        activities={parse_source_key(key)[1] for key in w['source_keys']}
        if len(activities)!=1:raise ValueError('A donor window crosses source activity/recording')
        candidates.append({**r,'activity_id':next(iter(activities)),'joint_map_hash':joint_map_hash,
                           'coordinate_convention':coordinate_convention,'time_source':w['time_source'],
                           'timestamps':w['time_s'],'sensor_frame_mask':[True]*64})
    matched=build_donor_pairs(candidates,seed=seed)
    matched['source_manifest_hash']=file_hash(windows_path);matched['qa_panel_hash']=identity(records)
    matched['policy_hash']=identity({'seed':seed,'activity':True,'subject_disjoint_blocks':True,'duration_tolerance':.05,'grid_dt_fraction':.25,'no_reuse':True})
    write_json(output,matched);return matched

def text_token_hash(splicer,task,elapsed,*,text_only=False):
    rendered=splicer.tokenizer.apply_chat_template(prompt_messages(task),tokenize=False,add_generation_prompt=True,**splicer.template_kwargs)
    if rendered.count(MARKER)!=1:raise ValueError('Invalid donor radar marker count')
    before,after=rendered.split(MARKER)
    encode=lambda s:splicer.tokenizer.encode(s,add_special_tokens=False)
    blocks=[encode(before)]
    if text_only:blocks.append(encode('No radar evidence is provided.'))
    else:
        for frame,seconds in enumerate(elapsed):blocks.extend([encode(f'Frame {frame:02d} at {seconds:.4f}s: '),{'sensor_frame':frame},encode('\n')])
    blocks.append(encode(after));return identity(blocks)

@torch.no_grad()
def generate_donors(matched,cache,projector,normalizer,bridge,splicer,tokenizer,*,device,condition,output_dir):
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True);predictions={}
    if projector:projector.eval()
    with (output/'donor_predictions.jsonl').open('w',encoding='utf-8') as stream:
        for pair in matched['pairs']:
            grid=torch.as_tensor(pair['canonical_elapsed_times'],dtype=torch.float64,device=device)
            result={'pair_uid':pair['pair_uid'],'condition':condition,'policy_hash':matched['policy_hash'],
                    'canonical_grid_hash':identity(pair['canonical_elapsed_times'])}
            for side in ('a','b'):
                uid=pair['window_uid_'+side];started=time.perf_counter()
                fingerprint=text_token_hash(splicer,pair['part'],pair['canonical_elapsed_times'],text_only=condition=='T')
                result['text_token_hash_'+side]=fingerprint
                try:
                    item=cache[uid]
                    embedding=None if condition=='T' else project_window(projector,item,normalizer,device).sensor_embeddings[0]
                    example=splicer.build(uid=uid,task=pair['part'],time_s=None if condition=='T' else grid,
                                          sensor_embeddings=embedding,text_only=condition=='T')
                    generated=bridge.greedy_generate(example,tokenizer,splicer.end_token_id)
                    parsed=parse_answer(generated.raw_text,pair['part'],truncated=generated.truncated)
                    result[side]={**asdict(parsed),'window_uid':uid,'token_ids':generated.token_ids,'input_tokens':example.prefix_length,
                                  'sensor_embeddings':len(example.sensor_positions),'stop_reason':'eos' if generated.terminated else 'token_limit'}
                except (RuntimeError,ValueError) as error:
                    result[side]={**asdict(parse_answer(None,pair['part'],runtime_error=str(error))),'window_uid':uid,'stop_reason':'runtime_error'}
                result[side]['latency_ms']=(time.perf_counter()-started)*1000
            if result['text_token_hash_a']!=result['text_token_hash_b']:raise ValueError('Donor text divergence')
            predictions[pair['pair_uid']]=result;stream.write(json.dumps(result,allow_nan=False)+'\n')
    report=evaluate_donors(matched['pairs'],predictions);report['policy_hash']=matched['policy_hash']
    write_json(output/'donor_metrics.json',report);return report

def compare_conditions(records,prediction_maps,output,*,test_lock_path=None):
    if any(r.get('split')=='test' for r in records):
        from .study import validate_test_lock
        if test_lock_path is None:raise ValueError('Test contrasts require experiment lock')
        validate_test_lock(test_lock_path,expected_generation=next((r.get('artifact_generation_id') for r in records if r.get('artifact_generation_id') is not None),None))
    per_seed={name:[evaluate_qa(records,pred)['S_atomic'] for pred in (value if isinstance(value,list) else [value])] for name,value in prediction_maps.items()}
    scores={name:float(np.mean(values)) if values and all(v is not None for v in values) else None for name,values in per_seed.items()}
    report={'scores':scores,'per_seed_scores':per_seed,'seed_sample_std':{name:float(np.std(values,ddof=1)) if len(values)>1 and all(v is not None for v in values) else None for name,values in per_seed.items()},
            'contrasts':condition_contrasts(scores),'primary_metric':'S_atomic','primary_contrast':'R_minus_S',
            'interpretation':'Means of per-seed metrics; seed repetitions do not add subject clusters.'}
    if 'R' in prediction_maps and 'S' in prediction_maps:
        report['primary_paired_subject_bootstrap']=paired_subject_bootstrap(records,prediction_maps['R'],prediction_maps['S'])
    write_json(output,report);return report
