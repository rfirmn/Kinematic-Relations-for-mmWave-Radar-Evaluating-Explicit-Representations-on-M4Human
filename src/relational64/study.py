"""Experiment locks, non-language controls, and auditable prediction scoring."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from .config import ROOT,active_artifact_path,file_hash,identity,write_json,read_json,require_gate,resolve
from .evaluation import evaluate_qa,build_donor_pairs,evaluate_donors,paired_subject_bootstrap,condition_contrasts
from .qa import fit_train_majority,rule_predictions,parse_answer,TASKS,canonical_answer
from .models import TaskProbe,adamw_parameter_groups
from .alignment import qa_draws,read_qa,project_window
from .training import seed_everything
from .checkpoint import save_checkpoint

def fit_majority(train_qa,output):
    majority=fit_train_majority(train_qa)
    write_json(output,{'fit_split':'train','labels':majority,'train_registry_hash':identity(train_qa)})
    return majority

def evaluate_controls(records,cache,majority):
    majority_predictions={r['qa_uid']:canonical_answer(majority[r['part']]) for r in records}
    rule={};by_window={}
    for r in records:by_window.setdefault(r['window_uid'],[]).append(r)
    for uid,rows in by_window.items():
        item=cache[uid]
        generated={r['part']:r for r in rule_predictions(uid,item['time_s'].numpy(),item['P'].numpy())}
        for r in rows:
            label=generated[r['part']]['answer']
            rule[r['qa_uid']]=canonical_answer(label) if label is not None else None
    return {'Maj':evaluate_qa(records,majority_predictions),'Rule':evaluate_qa(records,rule)}, {'Maj':majority_predictions,'Rule':rule}

_PLAN_ARTIFACTS=('source_manifest','source_inventory','split_lock','target_provenance','label_policy','exposure_ledger','language_lock','resolved_config')


def _study_config(config=None):
    return config if config is not None else resolve(ROOT/'configs/relational64/runtime.yaml')


def _plan_paths(plan, *, isolated):
    """Preflight every nested path before opening or hashing any plan artifact."""
    values={key:plan[key] for key in _PLAN_ARTIFACTS}
    for seed,group in plan.get('runs',{}).items():
        values[f'runs/{seed}/physical']=group['physical']
        for condition,path in group.get('projectors',{}).items():
            values[f'runs/{seed}/{condition}']=path
    if isolated:
        values['prepare_summary']=Path(plan['source_manifest']).parent/'prepare_summary.json'
    return {name:active_artifact_path(path) if isolated else Path(path).resolve() for name,path in values.items()}


def _validate_plan_generation(plan,paths,generation,config):
    """Generation tags plus manifest lineage; no source database or model loads."""
    from .data import assert_fresh_inventory,validate_manifest_generation
    if plan.get('artifact_generation_id',generation)!=generation:
        raise ValueError('Experiment plan artifact generation mismatch')
    saved_config=read_json(paths['resolved_config'])
    if saved_config!=config:
        raise ValueError('Experiment plan resolved config differs from active runtime')
    inventory=read_json(paths['source_inventory'])
    assert_fresh_inventory(inventory,generation)
    for key in ('target_provenance','label_policy','exposure_ledger','language_lock'):
        if read_json(paths[key]).get('artifact_generation_id')!=generation:
            raise ValueError(f'Test plan {key} artifact generation mismatch')
    validate_manifest_generation(paths['source_manifest'],paths['source_inventory'],expected_generation=generation)
    summary=read_json(paths['source_manifest'].parent/'prepare_summary.json')
    split=read_json(paths['split_lock'])
    if split.get('artifact_generation_id',generation)!=generation or split.get('split_hash')!=summary.get('split_hash'):
        raise ValueError('Test plan split lock generation/manifest mismatch')
    if identity({k:v for k,v in split.items() if k not in ('split_hash','artifact_generation_id')})!=split.get('split_hash'):
        raise ValueError('Test plan split lock content hash mismatch')


def validate_test_lock(path, *, expected_generation=None, config=None):
    if expected_generation is not None:
        path=active_artifact_path(path)
    lock=read_json(path);payload=dict(lock);expected=payload.pop('identity',None)
    if expected!=identity(payload) or lock.get('status')!='locked':raise ValueError('Invalid test lock')
    generation=lock.get('artifact_generation_id')
    if generation is not None or expected_generation is not None:
        config=_study_config(config)
        active_generation=config['runtime'].get('artifact_generation_id')
        if generation!=active_generation or (expected_generation is not None and generation!=expected_generation):
            raise ValueError('Test lock artifact generation mismatch with active runtime')
        active_artifact_path(path)
        paths=_plan_paths(lock['plan'],isolated=True)
        # Validate all paths before a single nested hash/read, including artifacts
        # not otherwise named in the final checkpoint plan (e.g. the gate report).
        for artifact in lock['artifacts']:
            active_artifact_path(artifact)
        for artifact in paths.values():
            if str(artifact) not in lock['artifacts']:
                raise ValueError(f'Test plan artifact is not pinned in lock: {artifact}')
        if lock.get('config_hash')!=identity(config):
            raise ValueError('Test lock active configuration hash mismatch')
    for artifact,digest in lock['artifacts'].items():
        if file_hash(artifact)!=digest:raise ValueError(f'Test lock artifact mismatch: {artifact}')
    if generation is not None:
        _validate_plan_generation(lock['plan'],paths,generation,config)
    return lock

def require_locked_checkpoint(lock_path,checkpoint_path,*,expected_generation=None,config=None):
    lock=validate_test_lock(lock_path,expected_generation=expected_generation,config=config)
    if lock.get('artifact_generation_id') is not None:
        checkpoint_path=active_artifact_path(checkpoint_path)
    resolved=str(Path(checkpoint_path).resolve())
    if lock['artifacts'].get(resolved)!=file_hash(checkpoint_path):raise ValueError('Checkpoint is not part of the frozen test experiment plan')
    return lock

def lock_test(plan_path,output,*,gate_path,config=None):
    config=_study_config(config)
    generation=config['runtime'].get('artifact_generation_id')
    if generation is not None:
        plan_path,output,gate_path=map(active_artifact_path,(plan_path,output,gate_path))
    plan=read_json(plan_path)
    if plan.get('seeds')!=[42,43,44] or plan.get('conditions')!=['M','S','F','E','R']:
        raise ValueError('Primary study requires seeds42/43/44 and M/S/F/E/R')
    paths=_plan_paths(plan,isolated=generation is not None)
    if generation is not None:
        if read_json(gate_path).get('artifact_generation_id')!=generation:
            raise ValueError('Test lock gate report artifact generation mismatch')
        _validate_plan_generation(plan,paths,generation,config)
    for gate in ('G0','G1','G2','G3','G4'):require_gate(gate_path,gate)
    artifacts={str(Path(plan_path).resolve()):file_hash(plan_path)}
    artifacts[str(Path(gate_path).resolve())]=file_hash(gate_path)
    for seed in plan['seeds']:
        group=plan['runs'].get(str(seed))
        if not group or set(group.get('projectors',{}))!=set(plan['conditions']):raise ValueError(f'Incomplete final checkpoints seed{seed}')
        entries=[(group['physical'],None)]+[(group['projectors'][c],c) for c in plan['conditions']]
        for path,condition in entries:
            path=active_artifact_path(path) if generation is not None else Path(path).resolve()
            state=torch.load(path,map_location='cpu',weights_only=False)
            if state.get('schema')!='relational64_checkpoint_1' or state.get('status')!='best' or state.get('step',0)<=0:
                raise ValueError(f'Final selected checkpoint required: {path}')
            p=state['provenance']
            if p['seed']!=seed or p.get('condition')!=condition:raise ValueError(f'Checkpoint seed/condition mismatch: {path}')
            if generation is not None:
                if state.get('artifact_generation_id')!=generation or p.get('config')!=config:
                    raise ValueError(f'Checkpoint artifact generation/config mismatch: {path}')
                if state.get('provenance_hash')!=identity(p):
                    raise ValueError(f'Checkpoint provenance hash mismatch: {path}')
            artifacts[str(path)]=file_hash(path)
    for key in (*_PLAN_ARTIFACTS, *(('prepare_summary',) if generation is not None else ())):
        path=paths[key]
        artifacts[str(path)]=file_hash(path)
    if read_json(plan['source_inventory']).get('authority',{}).get('G0_passed') is not True:raise ValueError('Data authority not verified')
    exposure=read_json(plan['exposure_ledger'])
    if exposure.get('test_exposure_status')!='documented':raise ValueError('Exposure history must be explicitly documented')
    lock={'protocol_id':'m4human_relational64','status':'locked','plan':plan,'artifacts':artifacts,
          'test_evaluation_authorized':True,'source_inventory_sha256':file_hash(plan['source_inventory']),
          'gate_report_hash':file_hash(gate_path),'primary_contrast':'R_minus_S','primary_metric':'S_atomic',
          'bootstrap_unit':'subject','bootstrap_draws':5000,'test_time_access':'First target extraction/evaluation only after this lock'}
    if generation is not None:
        lock['artifact_generation_id']=generation
        lock['config_hash']=identity(config)
    lock['identity']=identity(lock);write_json(output,lock);return lock

def save_score(records,predictions_path,output,*,test_lock_path=None):
    if any(r.get('split')=='test' for r in records):
        if not test_lock_path:raise ValueError('Test scores require immutable experiment lock')
        generations={r.get('artifact_generation_id') for r in records}
        if len(generations)>1:
            raise ValueError('Score QA artifact generation mismatch')
        validate_test_lock(test_lock_path,expected_generation=next(iter(generations),None))
    rows=[json.loads(line) for line in Path(predictions_path).read_text().splitlines() if line.strip()]
    if len({r['qa_uid'] for r in rows})!=len(rows):raise ValueError('Duplicate prediction UID')
    predictions={}
    reference_by_uid={r['qa_uid']:r for r in records}
    for row in rows:
        # Parse original text again, preserving failures/truncation. Do not trust
        # a claimed predicted_label column as a valid output schema.
        record=reference_by_uid.get(row['qa_uid'])
        if record is None:raise ValueError(f'Unexpected prediction UID {row["qa_uid"]}')
        predictions[row['qa_uid']]=parse_answer(row.get('raw_text'),record['part'],truncated=row.get('stop_reason')=='token_limit',
                                               runtime_error=row.get('reason') if row.get('stop_reason')=='runtime_error' else None)
    score=evaluate_qa(records,predictions);score['predictions_hash']=file_hash(predictions_path);score['qa_panel_hash']=identity(records)
    write_json(output,score);return score

def train_probe(train_cache,val_cache,train_records,val_records,*,output_dir,seed=42,source='Y',projector=None,normalizer=None,gate_path):
    generation=train_cache.meta.get('source',{}).get('artifact_generation_id')
    val_generation=val_cache.meta.get('source',{}).get('artifact_generation_id')
    if generation is not None or val_generation is not None:
        if generation is None or val_generation!=generation:
            raise ValueError('Probe train/validation cache artifact generation mismatch')
        if any(r.get('artifact_generation_id')!=generation for r in (*train_records,*val_records)):
            raise ValueError('Probe QA artifact generation mismatch')
        output_dir=active_artifact_path(output_dir)
        gate_path=active_artifact_path(gate_path)
        if read_json(gate_path).get('artifact_generation_id')!=generation:
            raise ValueError('Probe gate report artifact generation mismatch')
    for gate in ('G0','G1','G2','G3'):require_gate(gate_path,gate)
    if source not in ('Y','Z_R'):raise ValueError('Probe source must be Y or Z_R')
    if source=='Z_R' and (projector is None or projector.condition!='R'):raise ValueError('Z_R needs selected frozen R projector')
    if projector:projector.requires_grad_(False).eval()
    if any(r['split']!='train' for r in train_records) or any(r['split']!='val' for r in val_records):raise ValueError('Probe split mismatch')
    seed_everything(seed);model=TaskProbe();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    def feature(cache,record):
        item=cache[record['window_uid']]
        if source=='Y':return item['Y'][None]
        with torch.no_grad():return project_window(projector,item,normalizer,next(projector.parameters()).device).fused.detach().cpu()
    def label(r):return (*TASKS[r['part']]['classes'],'unknown').index(r['answer'])
    task_index={p:i for i,p in enumerate(TASKS)}
    provenance={'seed':seed,'source':source,'train_cache':train_cache.meta['identity'],'val_cache':val_cache.meta['identity'],
                'train_qa_hash':identity(train_records),'val_qa_hash':identity(val_records),'projector_hash':None if projector is None else __import__('relational64.models',fromlist=['parameter_hash']).parameter_hash(projector)}
    if generation is not None:
        provenance['artifact_generation_id']=generation
    best=None;patience=step=0;history=[]
    for epoch in range(50):
        model.train();draws=list(qa_draws(train_records,seed,epoch))
        for offset in range(0,len(draws),128):
            group=draws[offset:offset+128];x=torch.cat([feature(train_cache,r) for r in group])
            logits=model(x)[torch.arange(len(group)),torch.tensor([task_index[r['part']] for r in group])]
            loss=torch.nn.functional.cross_entropy(logits,torch.tensor([label(r) for r in group]))
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();step+=1
        model.eval();predictions={}
        with torch.no_grad():
            for r in val_records:
                predicted=int(model(feature(val_cache,r))[0,task_index[r['part']]].argmax())
                predictions[r['qa_uid']]=canonical_answer((*TASKS[r['part']]['classes'],'unknown')[predicted])
        metric=evaluate_qa(val_records,predictions)
        if metric['S_atomic'] is None:raise ValueError('Probe validation panel lacks a defined task')
        improved=best is None or metric['S_atomic']>best+1e-8
        if improved:best=metric['S_atomic'];patience=0
        else:patience+=1
        kwargs=dict(provenance=provenance,epoch=epoch,step=step,best=best,patience=patience)
        save_checkpoint(output_dir/'last.pt',model,optimizer,**kwargs)
        if improved:save_checkpoint(output_dir/'best.pt',model,optimizer,**kwargs)
        history.append({'epoch':epoch,'optimizer_steps':step,'validation':metric})
        write_json(output_dir/'history.json',{'provenance':provenance,'history':history})
        if patience>=8:break
    return {'best_S_atomic':best,'epochs':len(history),'source':source}
