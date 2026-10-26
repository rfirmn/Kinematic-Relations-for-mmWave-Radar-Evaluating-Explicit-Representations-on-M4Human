"""Complete a bounded diagnostic pilot review; never opens scientific test labels.

The supervised targets and timing remain provisional. This module evaluates
authored diagnostic checkpoints and cannot approve any scientific data gate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import re
import time
import traceback

import numpy as np
import torch

from .config import active_artifact_path, resolve, read_json, write_json, file_hash, identity
from .data import RPCOnlyReader, make_sensor_batch, parse_source_key, verify_snapshot, JOINT_NAMES, assert_fresh_inventory, validate_manifest_generation
from .diagnostic_pilot import CandidateTargets, diagnostic_flags, apply_sensor_gain
from .checkpoint import load_checkpoint, save_checkpoint
from .models import RadarPoseModel, SensorProjector, DescriptorNormalizer, TaskProbe, build_descriptors, freeze_for_inference, parameter_hash
from .physical import window_operators
from .kinematics import differentiate_torch, estimate_velocities
from .training import seed_everything
from .qa import TASKS, generate_qa, rule_predictions, parse_answer, canonical_answer, fit_train_majority
from .evaluation import physical_metrics, evaluate_qa, build_donor_pairs, evaluate_donors, condition_contrasts
from .resources import ResourceMonitor, environment_report


def select_validation_panel(rows, subjects, activities=12):
    """Choose common activities and windows using sensor identities, before labels."""
    subjects=set(map(str, subjects)); best={}
    for row in rows:
        if row['split']!='val':continue
        subject=str(row['subject_id'])
        if subject not in subjects:raise ValueError('Unexpected validation subject')
        keys=[parse_source_key(key) for key in row['source_keys']]
        if len(keys)!=64 or {str(key[0]) for key in keys}!={subject} or len({key[1] for key in keys})!=1:
            raise ValueError('Invalid sensor subject/activity/window boundary')
        key=(subject,keys[0][1]); rank=identity(['diagnostic-review/42',row['window_uid']])
        if key not in best or rank<best[key][0]:best[key]=(rank,row)
    common=set.intersection(*({a for s,a in best if s==subject} for subject in subjects)) if subjects else set()
    chosen=sorted(common,key=lambda a:identity(['diagnostic-review/activity/42',a]))[:activities]
    if len(chosen)!=activities:raise ValueError('Insufficient common validation activities')
    return [best[(subject,action)][1] for action in chosen for subject in sorted(subjects)]


def aggregate_physical(rows):
    totals={}
    for row in rows:
        for name,value in row['metrics'].items():
            total=totals.setdefault(name,{'numerator':0.,'count':0,'nonfinite_prediction_count':0,'unit':value['unit']})
            total['count']+=value['count'];total['nonfinite_prediction_count']+=value['nonfinite_prediction_count']
            if value['numerator'] is not None:total['numerator']+=value['numerator']
    for value in totals.values():
        if value['nonfinite_prediction_count']:value['numerator']=None
        value['mean']=value['numerator']/value['count'] if value['numerator'] is not None and value['count'] else None
        if value['unit']=='m':value['mean_mm']=1000*value['mean'] if value['mean'] is not None else None
    return totals


def review_run(config,run_id,*,review_id='validation12_v1',activities=12,probe_steps=50):
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',value) for value in (run_id,review_id)):
        raise ValueError('Safe local run/review IDs required')
    if not 1<=activities<=20 or not 1<=probe_steps<=200:raise ValueError('Bounded review budget required')
    rt=config['runtime'];generation=rt['artifact_generation_id'];device=rt['device']
    reports=active_artifact_path(Path(rt['report_root'])/'diagnostic_runs'/run_id)
    derived=active_artifact_path(Path(rt['derived_root'])/'diagnostic_runs'/run_id)
    checkpoints=active_artifact_path(Path(rt['checkpoint_root'])/'diagnostic_runs'/run_id)
    output=reports/'reviews'/review_id;data_output=derived/'reviews'/review_id;ckpt_output=checkpoints/'reviews'/review_id
    for path in (output,data_output,ckpt_output):path.mkdir(parents=True,exist_ok=True)
    plan=read_json(reports/'plan.json');dependencies={str(reports/'plan.json'):file_hash(reports/'plan.json')}
    if plan['artifact_generation_id']!=generation or plan['config_hash']!=identity(config) or plan.get('test_target_access') is not False:
        raise ValueError('Run generation/config/test boundary mismatch')
    if set(plan['conditions'])!=set('MSFER'):raise ValueError('All five sensor conditions must finish before complete review')
    for name in ('prepare','physical','alignment'):
        path=reports/f'{name}.json';stage=read_json(path)
        if stage['status']!='passed' or stage['plan_hash']!=identity(plan) or stage.get('scientific_acceptance') is not False:
            raise ValueError('Diagnostic dependency must pass: '+name)
        source=active_artifact_path(stage['source_snapshot'])
        if file_hash(source)!=stage['module_sha256']:raise ValueError('Training source snapshot mismatch')
        dependencies[str(source)]=stage['module_sha256']
        dependencies[str(path)]=file_hash(path)
        for member,digest in stage['files'].items():
            path=active_artifact_path(member)
            if file_hash(path)!=digest:raise ValueError('Diagnostic dependency hash mismatch: '+str(path))
            dependencies[str(path)]=digest
    language_path=active_artifact_path(Path(rt['report_root'])/'language.lock.json')
    dependencies[str(language_path)]=file_hash(language_path)
    inventory_path=active_artifact_path(rt['source_inventory'])
    manifest=active_artifact_path(Path(rt['derived_root'])/'windows.jsonl')
    inventory=read_json(inventory_path);assert_fresh_inventory(inventory,generation)
    validate_manifest_generation(manifest,inventory_path,expected_generation=generation)
    dependencies.update({str(path):file_hash(path) for path in (inventory_path,manifest)})
    review_plan={'schema':'relational64_diagnostic_review_plan_1',**diagnostic_flags(),
                 'run_id':run_id,'review_id':review_id,'artifact_generation_id':generation,
                 'base_plan_hash':identity(plan),'activities_per_validation_subject':activities,'probe_steps':probe_steps,
                 'selection':'common activities and minimum SHA256 window, sensor identities only, seed42',
                 'module_sha256':file_hash(__file__),'dependencies':dependencies}
    plan_path=output/'plan.json'
    if plan_path.exists() and read_json(plan_path)!=review_plan:raise ValueError('Review changed: use a new review ID')
    write_json(plan_path,review_plan)
    checks={}
    if inventory['artifact_generation_id']!=generation:raise ValueError('Source generation mismatch')
    verify_snapshot(inventory,('rpc','params','calib'))
    seed_everything(42)
    base=torch.load(derived/'batches.diagnostic.pt',map_location='cpu',weights_only=False)
    val_subjects={r['subject_id'] for r in base if r['split']=='val'}
    train_subjects={r['subject_id'] for r in base if r['split']=='train'}
    checks['subject_split_disjoint']=not (val_subjects & train_subjects)
    import json
    with manifest.open(encoding='utf-8') as stream:
        selected=select_validation_panel((json.loads(line) for line in stream if line.strip()),val_subjects,activities)
    windows={r['window_uid']:r for r in read_json(derived/'panel.diagnostic.json')['windows']}
    windows.update({r['window_uid']:r for r in selected})
    write_json(data_output/'selection.diagnostic.json',{'windows':selected,'before_target_access':True,'manifest_sha256':file_hash(manifest),**diagnostic_flags()})
    # The train-panel normalizer is reused, never fitted on the new validation windows.
    normalizer=read_json(derived/'intensity_normalizer.diagnostic.json');normalizer_before=file_hash(derived/'intensity_normalizer.diagnostic.json')
    samples={r['uid']:r for r in base};reader=RPCOnlyReader(inventory);targets=CandidateTargets(inventory,val_subjects)
    try:
        for row in selected:
            if row['window_uid'] in samples:continue
            sensor=make_sensor_batch(reader,row,normalizer)
            pairs=[targets.read(key) for key in row['source_keys']]
            mask=np.stack([m for p,m in pairs]);positions=np.stack([p for p,m in pairs])-sensor['centroid'].numpy();positions[~mask]=0
            samples[row['window_uid']]={**sensor,'positions':torch.from_numpy(positions),'target_mask':torch.from_numpy(mask),
                                      'split':'val','subject_id':str(row['subject_id']),'recording_id':row['recording_id']}
    finally:reader.close();targets.close()
    checks['train_only_intensity_unchanged']=file_hash(derived/'intensity_normalizer.diagnostic.json')==normalizer_before
    checks['test_labels_not_opened']=all(r['split'] in ('train','val') for r in samples.values())
    torch.save(list(samples.values()),data_output/'batches.diagnostic.pt')
    records=[]
    for sample in samples.values():
        records.extend(generate_qa(sample['uid'],sample['time_s'].numpy(),sample['positions'].numpy(),sample['target_mask'].numpy(),
                                  protocol_id='relational64_provisional_diagnostic',metadata=sample))
    write_json(data_output/'qa.diagnostic.json',{'records':records,**diagnostic_flags()})
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val']
    model=freeze_for_inference(RadarPoseModel().to(device));physical_path=checkpoints/'physical/diagnostic.pt'
    saved=torch.load(physical_path,map_location='cpu',weights_only=False)
    load_checkpoint(physical_path,model,provenance=saved['provenance'],restore_random=False)
    physical_hash=parameter_hash(model);checks['physical_checkpoint_is_diagnostic']=saved['status']=='diagnostic'
    states={};physical_rows=[];monitor=ResourceMonitor(device)
    old_cache=torch.load(derived/'features.diagnostic.pt',map_location='cpu',weights_only=False)['states'];max_cache_error=0.
    with torch.no_grad():
        for uid,sample in samples.items():
            times=sample['time_s'][None].to(device);prediction=model(sample['X'][None].to(device),sample['point_mask'][None].to(device),times)
            operator,available=window_operators(times)
            state={'Y':prediction.features.cpu(),'P':prediction.positions.cpu(),
                   'V':differentiate_torch(prediction.positions,operator).cpu(),'available':available.cpu(),
                   'time_s':sample['time_s'],'split':sample['split']};states[uid]=state
            if uid in old_cache:
                for key in ('Y','P','V'):max_cache_error=max(max_cache_error,float((state[key]-old_cache[uid][key]).abs().max()))
            velocity=estimate_velocities(sample['positions'].numpy(),sample['target_mask'].numpy(),sample['time_s'].numpy())
            metrics=physical_metrics(state['P'][0].numpy(),sample['positions'].numpy(),sample['target_mask'].numpy(),
                                     state['V'][0].numpy(),velocity.velocity,velocity.valid)
            physical_rows.append({'window_uid':uid,'split':sample['split'],'subject_id':sample['subject_id'],**metrics});monitor.sample()
    physical_resource=monitor.report();checks['cache_forward_equivalent']=max_cache_error<=1e-6
    checks['physical_predictions_finite']=all(r['status']=='valid' for r in physical_rows)
    checks['physical_frozen_unchanged']=physical_hash==parameter_hash(model)
    checks['physical_resource_guard']=physical_resource['resource_guard_passed']
    del model,saved,prediction;torch.cuda.empty_cache() if torch.device(device).type=='cuda' else None
    write_json(output/'physical_metrics.json',{'panels':{s:aggregate_physical([r for r in physical_rows if r['split']==s]) for s in ('train','val')},
               'per_window':physical_rows,'cache_max_absolute_error':max_cache_error,'resource':physical_resource,**diagnostic_flags()})
    donor_candidates=[]
    for record in val:
        row=windows[record['window_uid']]
        donor_candidates.append({**record,'activity_id':parse_source_key(row['source_keys'][0])[1],
            'joint_map_hash':identity(JOINT_NAMES),'coordinate_convention':'candidate_vicon_cam_radar',
            'time_source':row['time_source'],'timestamps':row['time_s'],'sensor_frame_mask':[True]*64})
    matched=build_donor_pairs(donor_candidates,seed=42)
    write_json(output/'donor_matching.json',{**matched,**diagnostic_flags()})
    checks['donor_has_actual_pairs']=bool(matched['pairs'])
    from .alignment import load_language
    from .evaluation_pipeline import text_token_hash
    bridge,splicer,tokenizer,dtype=load_language(config,read_json(language_path));qwen_hash=parameter_hash(bridge)
    monitor=ResourceMonitor(device);qa_metrics={};donor_metrics={};prediction_maps={};probe_sources={'Y':{uid:s['Y'] for uid,s in states.items()}}
    checks['donor_actual_text_embeddings_identical']=True
    checks['donor_sensor_embeddings_change']=True
    checks['projectors_frozen_during_evaluation']=True
    for condition in (*plan['conditions'],'T'):
        projector=norm=None
        if condition!='T':
            projector=freeze_for_inference(SensorProjector(condition).to(device));path=checkpoints/condition/'diagnostic.pt'
            saved=torch.load(path,map_location='cpu',weights_only=False);load_checkpoint(path,projector,provenance=saved['provenance'],restore_random=False)
            if saved['status']!='diagnostic':raise ValueError('Diagnostic projector required')
            if condition!='M':
                artifact=read_json(derived/f'descriptor_{condition}.diagnostic.json')
                if artifact['source_checkpoint_hash']!=physical_hash or artifact.get('diagnostic_only') is not True:raise ValueError('Descriptor source mismatch')
                norm=DescriptorNormalizer(condition,physical_hash,artifact['train_manifest_hash'])
                norm.load_state_dict({k:torch.as_tensor(artifact[k],dtype=v.dtype) for k,v in norm.state_dict().items()});norm.to(device)
        projector_before=None if projector is None else parameter_hash(projector)
        def embedding(uid):
            state=states[uid];descriptor=None if norm is None else norm(build_descriptors(state['P'].to(device),state['V'].to(device),state['available'].to(device),condition))
            result=projector(state['Y'].to(device),descriptor)
            if condition=='R':probe_sources.setdefault('Z_R',{})[uid]=result.fused.detach().cpu()
            return apply_sensor_gain(result.sensor_embeddings[0],plan.get('sensor_embedding_gain',1.))
        def generate(uid,part,grid=None):
            started=time.perf_counter()
            example=splicer.build(uid=uid,task=part,time_s=None if condition=='T' else (states[uid]['time_s'] if grid is None else grid).to(device),
                                  sensor_embeddings=None if condition=='T' else embedding(uid),text_only=condition=='T')
            if len(example.sensor_positions)!=(0 if condition=='T' else 64):raise ValueError('Sensor splicing count mismatch')
            output_text=bridge.greedy_generate(example,tokenizer,splicer.end_token_id)
            parsed=parse_answer(output_text.raw_text,part,truncated=output_text.truncated);monitor.sample()
            row={'raw_text':output_text.raw_text,'parse':asdict(parsed),'sensor_embeddings':len(example.sensor_positions),
                 'input_tokens':example.prefix_length,'latency_ms':(time.perf_counter()-started)*1000}
            if grid is not None:
                mask=torch.ones(example.prefix_length,dtype=torch.bool,device=example.inputs_embeds.device)
                mask[list(example.sensor_positions)]=False
                fingerprint=lambda x:hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                row['actual_text_embedding_hash']=fingerprint(example.inputs_embeds[:example.prefix_length][mask])
                row['actual_sensor_embedding_hash']=fingerprint(example.inputs_embeds[list(example.sensor_positions)])
            return row
        predictions={};generated=[];donor_predictions={}
        with torch.no_grad():
            for record in val:
                row=generate(record['window_uid'],record['part']);generated.append({'qa_uid':record['qa_uid'],**row});predictions[record['qa_uid']]=row['parse']
            if condition=='R':
                for uid in states:
                    if uid not in probe_sources['Z_R']:embedding(uid)
            for pair in matched['pairs']:
                grid=torch.as_tensor(pair['canonical_elapsed_times'],dtype=torch.float64)
                rows={side:generate(pair['window_uid_'+side],pair['part'],grid) for side in ('a','b')}
                token_hash=text_token_hash(splicer,pair['part'],pair['canonical_elapsed_times'],text_only=condition=='T')
                checks['donor_actual_text_embeddings_identical'] &= rows['a']['actual_text_embedding_hash']==rows['b']['actual_text_embedding_hash']
                if condition!='T':checks['donor_sensor_embeddings_change'] &= rows['a']['actual_sensor_embedding_hash']!=rows['b']['actual_sensor_embedding_hash']
                donor_predictions[pair['pair_uid']]={**{side:rows[side]['parse'] for side in rows},
                    'text_token_hash_a':token_hash,'text_token_hash_b':token_hash,'generations':rows}
        prediction_maps[condition]=predictions;qa_metrics[condition]=evaluate_qa(val,predictions)
        donor_metrics[condition]=evaluate_donors(matched['pairs'],donor_predictions)
        checks['projectors_frozen_during_evaluation'] &= projector is None or projector_before==parameter_hash(projector)
        write_json(output/f'predictions_{condition}.json',{'qa':generated,'donors':donor_predictions,**diagnostic_flags()})
        print(f'review {condition}: QA={len(generated)}, donor_pairs={len(matched["pairs"])}',flush=True)
        del projector,norm
        if torch.device(device).type=='cuda':torch.cuda.empty_cache()
    checks['frozen_Qwen_unchanged']=qwen_hash==parameter_hash(bridge)
    checks['alignment_resource_guard']=monitor.report()['resource_guard_passed'];alignment_resource=monitor.report()
    checks['all_QA_generation_schema_valid']=all(v['generation']['schema_enum_valid_rate']==1 for v in qa_metrics.values())
    checks['donor_text_and_generation_valid']=all(v['pair_count']>0 and v['missing_text_provenance_count']==0 and v['metrics']['both_parse_valid']==1 for v in donor_metrics.values())
    del bridge,tokenizer,splicer;torch.cuda.empty_cache() if torch.device(device).type=='cuda' else None
    majority=fit_train_majority(train)
    prediction_maps['Maj']={r['qa_uid']:canonical_answer(majority[r['part']]) for r in val}
    rules={uid:{r['part']:r for r in rule_predictions(uid,s['time_s'].numpy(),s['P'][0].numpy())} for uid,s in states.items()}
    prediction_maps['Rule']={r['qa_uid']:canonical_answer(rules[r['window_uid']][r['part']]['answer']) if rules[r['window_uid']][r['part']]['answer'] else None for r in val}
    for key in ('Maj','Rule'):qa_metrics[key]=evaluate_qa(val,prediction_maps[key])
    # Fixed-budget probes diagnose readability only; no scientific gate bypass.
    probe_report={};task_index={p:i for i,p in enumerate(TASKS)}
    for source,features in probe_sources.items():
        seed_everything(42);probe=TaskProbe();optimizer=torch.optim.AdamW(probe.parameters(),lr=1e-3,weight_decay=.01)
        fitting=[r for r in train if r['reference_status'] in ('defined','unknown')];losses=[]
        for step in range(probe_steps):
            optimizer.zero_grad();loss=sum(torch.nn.functional.cross_entropy(probe(features[r['window_uid']])[:,task_index[r['part']]],
                torch.tensor([(*TASKS[r['part']]['classes'],'unknown').index(r['answer'])])) for r in fitting)/len(fitting)
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        probe.eval();predictions={}
        with torch.no_grad():
            for r in val:
                label=int(probe(features[r['window_uid']])[0,task_index[r['part']]].argmax())
                predictions[r['qa_uid']]=canonical_answer((*TASKS[r['part']]['classes'],'unknown')[label])
        probe_report[source]={'optimizer_steps':probe_steps,'loss_history':losses,'metrics':evaluate_qa(val,predictions)}
        save_checkpoint(ckpt_output/source/'diagnostic.pt',probe,optimizer,provenance={**diagnostic_flags(),'review_plan_hash':identity(review_plan),'source':source,'artifact_generation_id':generation},epoch=0,step=probe_steps)
    physical=read_json(reports/'physical.json');reduction=1-physical['final']['train']['L_phys']/physical['initial']['train']['L_phys']
    checks['small_sample_loss_reduction_at_least_50_percent']=reduction>=.5
    checks['checkpoint_reload']=physical['checkpoint_reload_passed']
    checks['finite_gradients_and_losses']=all(np.isfinite(row['gradient_norm']) and np.isfinite(row['train']['L_phys']) for row in read_json(reports/'physical_history.json')['history'])
    checks['alignment_backward_and_loss_finite']=all(
        row['gradient_norm']>0 and np.isfinite(row['gradient_norm']) and np.isfinite(row['losses']).all()
        for condition in plan['conditions'] for row in read_json(reports/f'alignment_{condition}_history.json')['history'])
    checks['source_snapshot_unchanged']=True
    verify_snapshot(inventory,('rpc','params','calib'))
    for path,digest in dependencies.items():
        if file_hash(path)!=digest:raise ValueError('Input artifact changed during review: '+path)
    report={'schema':'relational64_diagnostic_review_1',**diagnostic_flags(),'run_id':run_id,'review_id':review_id,
            'status':'passed' if all(checks.values()) else 'needs_revision','execution_complete':True,'checks':checks,
            'train_windows':sum(r['split']=='train' for r in samples.values()),'val_windows':sum(r['split']=='val' for r in samples.values()),
            'train_QA_count':len(train),'validation_QA_count':len(val),'small_sample_train_loss_reduction':reduction,
            'QA_metrics_against_candidate_targets':qa_metrics,'contrasts_diagnostic_only':condition_contrasts({k:v['S_atomic'] for k,v in qa_metrics.items()}),
            'donor_metrics_against_candidate_targets':donor_metrics,'probes_diagnostic_only':probe_report,
            'alignment_resource':alignment_resource,'environment':environment_report(),'plan_hash':identity(review_plan)}
    write_json(output/'summary.json',report)
    lines=['# Kelengkapan pilot diagnostik aktual','',f'Run `{run_id}`, review `{review_id}`: **{report["status"]}**.',
           '', 'Seluruh hasil menggunakan calon target dan waktu nominal; scientific_acceptance=false, G0 tetap false.',
           '', '| Pemeriksaan | Hasil |','|---|---|',*[f'| {k} | {v} |' for k,v in checks.items()],
           '', f'Panel: {report["train_windows"]} train dan {report["val_windows"]} validation windows; test tidak dibuka.',
           f'Penurunan loss train: {100*reduction:.2f}%.', '', '| Kondisi | Schema valid | S_atomic provisional |','|---|---|---|',
           *[f'| {k} | {v["generation"]["schema_enum_valid_rate"]} | {v["S_atomic"]} |' for k,v in qa_metrics.items()],
           '', f'Donor aktual: {len(matched["pairs"])} pasangan, hasil deskriptif. Validitas format bukan ketepatan jawaban.',
           '', 'Ringkasan JSON memuat galat fisik, coverage/kelas, probe, kontrol donor, resource, lingkungan, dan hashes input.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'REVIEW {report["status"]}: {output}',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/relational64/runtime.yaml')
    parser.add_argument('--run-id',required=True);parser.add_argument('--review-id',default='validation12_v1')
    parser.add_argument('--activities',type=int,default=12);parser.add_argument('--probe-steps',type=int,default=50)
    args=parser.parse_args();config=resolve(args.config)
    try:report=review_run(config,args.run_id,review_id=args.review_id,activities=args.activities,probe_steps=args.probe_steps)
    except Exception as error:
        path=active_artifact_path(Path(config['runtime']['report_root'])/'diagnostic_runs'/args.run_id/'reviews'/args.review_id/'failure.json')
        write_json(path,{'status':'failed','error':str(error),'traceback':traceback.format_exc(),**diagnostic_flags()});raise
    if report['status']!='passed':raise SystemExit('Diagnostic execution complete with findings; see checks and raw generations.')


if __name__=='__main__':main()
