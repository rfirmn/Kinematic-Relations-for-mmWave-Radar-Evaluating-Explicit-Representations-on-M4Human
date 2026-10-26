"""Projector-only alignment and generation over frozen physical predictions."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .cache import load_descriptor_normalizer
from .checkpoint import save_checkpoint,load_checkpoint
from .config import active_artifact_path,read_json,write_json,identity,file_hash,require_gate,validate_language_lock
from .evaluation import evaluate_qa
from .language import FrozenQwenBridge,RadarSplicer,canonical_answer,collate_spliced
from .models import SensorProjector,build_descriptors,adamw_parameter_groups,parameter_hash
from .qa import parse_answer,TASKS
from .resources import ResourceMonitor
from .training import seed_everything,choose_precision,autocast,schedule_factor

def load_language(config,language_lock):
    validate_language_lock(language_lock)
    generation=config['runtime'].get('artifact_generation_id')
    if generation is not None and language_lock.get('artifact_generation_id')!=generation:
        raise ValueError('Language metadata artifact generation mismatch; rebuild the language lock')
    from transformers import AutoTokenizer
    runtime=config['runtime']; snapshot=runtime['model_path']
    if language_lock['model_commit']!=runtime['model_commit'] or language_lock['tokenizer_commit']!=runtime['tokenizer_commit']:
        raise ValueError('Language lock/config commit mismatch')
    expected_files=language_lock.get('snapshot_file_hashes',{})
    if not expected_files or not any(n.endswith('.safetensors') for n in expected_files):raise ValueError('Missing model weight manifest')
    if identity({n:h for n,h in expected_files.items() if n.endswith('.safetensors')})!=language_lock['safetensors_manifest_hash']:
        raise ValueError('Safetensors manifest identity mismatch')
    for name,digest in expected_files.items():
        if file_hash(Path(snapshot)/name)!=digest:raise ValueError(f'Language snapshot mismatch: {name}')
    dtype=getattr(torch,language_lock['precision'])
    if dtype not in (torch.bfloat16,torch.float16):raise ValueError('Unverified language precision')
    tokenizer=AutoTokenizer.from_pretrained(snapshot,revision=runtime['tokenizer_commit'],local_files_only=True)
    bridge=FrozenQwenBridge.from_local_snapshot(snapshot,model_commit=runtime['model_commit'],tokenizer_commit=runtime['tokenizer_commit'],dtype=dtype,head_mode=language_lock['head_mode']).to(runtime['device'])
    splicer=RadarSplicer(tokenizer,bridge.causal_lm.get_input_embeddings(),expected_template_hash=language_lock['chat_template_hash'])
    return bridge,splicer,tokenizer,dtype

def read_qa(path,split=None,eligible=True,*,expected_generation=None):
    """Load only a complete, hash-bound registry for the active generation.

    Generation-less use is reserved for explicitly synthetic software fixtures.
    Validate the path and sidecar before opening the potentially stale QA rows.
    """
    path=Path(path)
    metadata=None
    if expected_generation is not None:
        path=active_artifact_path(path)
        sidecar=active_artifact_path(path.with_suffix('.meta.json'))
        metadata=read_json(sidecar)
        if (metadata.get('schema_id')!='relational64_qa_registry_1' or metadata.get('complete') is not True
                or metadata.get('artifact_generation_id')!=expected_generation):
            raise ValueError('QA registry artifact generation/state mismatch')
        payload=dict(metadata);stored_identity=payload.pop('identity',None)
        if not stored_identity or identity(payload)!=stored_identity:
            raise ValueError('QA registry sidecar identity mismatch')
        if metadata.get('provenance',{}).get('artifact_generation_id')!=expected_generation:
            raise ValueError('QA registry provenance artifact generation mismatch')
        # qa_hash is the canonical key emitted by build_qa_registry.
        if metadata.get('qa_hash')!=file_hash(path):
            raise ValueError('QA registry checksum mismatch')
    records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if metadata is not None:
        if metadata.get('qa_count')!=len(records):raise ValueError('QA registry row count mismatch')
        if any(record.get('artifact_generation_id')!=expected_generation for record in records):
            raise ValueError('QA row artifact generation mismatch')
    if len({r['qa_uid'] for r in records})!=len(records):raise ValueError('Duplicate QA registry UID')
    return [r for r in records if (split is None or r['split']==split) and (not eligible or r['reference_status'] in ('defined','unknown'))]

def qa_draws(records,seed,epoch):
    pools={task:[r for r in records if r['part']==task] for task in TASKS}
    if any(not p for p in pools.values()):raise ValueError('All three QA tasks need eligible train support')
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch]))
    tasks=list(TASKS)
    for _ in range(len(records)):
        task=tasks[int(rng.choice(3,p=[.5,.25,.25]))]
        yield pools[task][int(rng.integers(len(pools[task])))]

def project_window(projector,cache_item,normalizer,device):
    y=cache_item['Y'][None].to(device)
    if projector.condition=='M': return projector(y)
    descriptor=build_descriptors(cache_item['P'][None].to(device),cache_item['V'][None].to(device),
                                 cache_item['velocity_available'][None].to(device),projector.condition)
    return projector(y,normalizer(descriptor))

@torch.no_grad()
def generate_panel(records,cache,projector,normalizer,bridge,splicer,tokenizer,*,device,output=None,condition=None,seed=42,checkpoint_hash=None):
    if projector:projector.eval()
    results=[]
    for record in sorted(records,key=lambda r:r['qa_uid']):
        # Only opaque UID selects sensor evidence; reference answer/status/mask are
        # not arguments to model/projector/splicer inference.
        uid,task=record['window_uid'],record['part']
        start=time.perf_counter();row={'qa_uid':record['qa_uid'],'window_uid':uid,'condition':condition or projector.condition,
                                     'seed':seed,'checkpoint_hash':checkpoint_hash,'prompt_hash':record['prompt_hash']}
        try:
            if condition=='T':
                example=splicer.build(uid=uid,task=task,time_s=None,sensor_embeddings=None,text_only=True)
            else:
                item=cache[uid];embedding=project_window(projector,item,normalizer,device).sensor_embeddings[0]
                example=splicer.build(uid=uid,task=task,time_s=item['time_s'].to(device),sensor_embeddings=embedding)
            generated=bridge.greedy_generate(example,tokenizer,splicer.end_token_id)
            parsed=parse_answer(generated.raw_text,task,truncated=generated.truncated)
            row.update(asdict(parsed),predicted_label=parsed.answer,parse_status=parsed.status,
                       stop_reason='eos' if generated.terminated else 'token_limit',input_tokens=example.prefix_length,
                       sensor_embeddings=len(example.sensor_positions),generated_tokens=len(generated.token_ids),token_ids=generated.token_ids)
        except (RuntimeError,ValueError) as error:
            row.update(raw_text=None,answer=None,status='runtime_error',predicted_label=None,parse_status='runtime_error',
                       reason=str(error),json_parse_valid=False,stop_reason='runtime_error',input_tokens=None,sensor_embeddings=None,generated_tokens=0)
        row['latency_ms']=(time.perf_counter()-start)*1000;results.append(row)
    if output:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in results),encoding='utf-8')
    return results

def train_alignment(train_cache,val_cache,train_records,val_records,*,condition,normalizer_path,config,language_lock,gate_path,output_dir,seed=42,resume=None):
    generation=config['runtime'].get('artifact_generation_id')
    if generation is not None:
        for records in (train_records,val_records):
            if any(record.get('artifact_generation_id')!=generation for record in records):
                raise ValueError('Alignment QA artifact generation mismatch')
        for cache in (train_cache,val_cache):
            if cache.meta.get('source',{}).get('artifact_generation_id')!=generation:
                raise ValueError('Alignment cache artifact generation mismatch')
    for gate in ('G0','G1','G2','G3','G4'):require_gate(gate_path,gate)
    if train_cache.meta['split']!='train' or val_cache.meta['split']!='val':raise ValueError('Alignment requires train/val cache')
    if any(r['split']!='train' for r in train_records) or any(r['split']!='val' for r in val_records):raise ValueError('QA split mismatch')
    seed_everything(seed);device=config['runtime']['device'];output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    bridge,splicer,tokenizer,dtype=load_language(config,language_lock)
    projector=SensorProjector(condition).to(device)
    initial_projector_hash=parameter_hash(projector)
    normalizer=None if condition=='M' else load_descriptor_normalizer(normalizer_path,train_cache).to(device)
    optimizer=torch.optim.AdamW(adamw_parameter_groups(projector),lr=3e-4,betas=(.9,.999),eps=1e-6)
    total=math.ceil(len(train_records)/16)*20;warmup=math.ceil(total*.05)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:schedule_factor(step,total,warmup,3e-4,3e-6))
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    provenance={'protocol':'m4human_relational64','condition':condition,'seed':seed,'config':config,'language_lock':language_lock,'initial_projector_hash':initial_projector_hash,
                'train_cache':train_cache.meta['identity'],'val_cache':val_cache.meta['identity'],
                'train_qa':identity(train_records),'val_qa':identity(val_records),'normalizer':file_hash(normalizer_path) if normalizer_path else None,
                'effective_batch':16,'precision':str(dtype)}
    step=patience=start_epoch=0;best=None;history=[]
    if resume:
        state=load_checkpoint(resume,projector,provenance=provenance,optimizer=optimizer,scheduler=scheduler,scaler=scaler)
        start_epoch=state['epoch']+1;step=state['step'];best=state['best'];patience=state['patience']
        if (output_dir/'history.json').exists():history=read_json(output_dir/'history.json')['history']
    for epoch in range(start_epoch,20):
        projector.train();draws=list(qa_draws(train_records,seed,epoch))
        write_json(output_dir/f'draws_{epoch:02d}.json',[{'qa_uid':r['qa_uid'],'task':r['part']} for r in draws])
        loss_sum=0.
        for offset in range(0,len(draws),16):
            group=draws[offset:offset+16];optimizer.zero_grad(set_to_none=True)
            for record in group:
                item=train_cache[record['window_uid']]
                with autocast(device,dtype):embedding=project_window(projector,item,normalizer,device).sensor_embeddings[0]
                example=splicer.build(uid=record['qa_uid'],task=record['part'],time_s=item['time_s'].to(device),
                                      sensor_embeddings=embedding,answer=canonical_answer(record['part'],record['answer']))
                loss=bridge.selected_answer_loss(collate_spliced([example]))
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite language loss {record["qa_uid"]}')
                scaler.scale(loss/len(group)).backward();loss_sum+=float(loss.detach())
            if any(p.grad is not None for p in bridge.parameters()):raise RuntimeError('Frozen Qwen received gradients')
            scaler.unscale_(optimizer)
            norm=torch.stack([p.grad.detach().float().norm() for p in projector.parameters() if p.grad is not None]).norm()
            if torch.isfinite(norm):torch.nn.utils.clip_grad_norm_(projector.parameters(),1.,error_if_nonfinite=True)
            elif not scaler.is_enabled():raise FloatingPointError('Nonfinite projector gradient')
            previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=previous:scheduler.step();step+=1
        predictions=generate_panel(val_records,val_cache,projector,normalizer,bridge,splicer,tokenizer,device=device,
                                   output=output_dir/f'validation_{epoch:02d}.jsonl',condition=condition,seed=seed,checkpoint_hash=parameter_hash(projector))
        score=evaluate_qa(val_records,{r['qa_uid']:r for r in predictions})
        if score['S_atomic'] is None:raise ValueError('Three-task validation S_atomic undefined')
        selection=(score['S_atomic'],score['generation']['schema_enum_valid_rate'])
        improved=best is None or selection[0]>best[0]+1e-8 or abs(selection[0]-best[0])<=1e-8 and selection[1]>best[1]+1e-8
        if improved:best=selection;patience=0
        else:patience+=1
        args=dict(provenance=provenance,epoch=epoch,step=step,scheduler=scheduler,scaler=scaler,best=best,patience=patience)
        save_checkpoint(output_dir/'last.pt',projector,optimizer,**args)
        if improved:save_checkpoint(output_dir/'best.pt',projector,optimizer,**args)
        history.append({'epoch':epoch,'mean_loss':loss_sum/len(draws),'draw_count':len(draws),'unique_qa_seen':len({r['qa_uid'] for r in draws}),
                        'optimizer_steps':step,'selection':list(selection),'validation':score})
        write_json(output_dir/'history.json',{'provenance':provenance,'history':history})
        if patience>=5:break
    return {'best_selection':best,'epochs':len(history),'optimizer_steps':step}

def profile_alignment(*,config,language_lock,condition,output_dir,warmup=10,steps=30,accumulation=16,generations=30):
    """Worst-point cache fixture profile; does not establish real-data G4."""
    seed_everything(42);device=config['runtime']['device']
    bridge,splicer,tokenizer,dtype=load_language(config,language_lock)
    projector=SensorProjector(condition).to(device)
    item={'Y':torch.randn(64,256),'P':torch.randn(64,22,3),'V':torch.randn(64,22,3),
          'time_s':torch.arange(64,dtype=torch.float64)/12,'velocity_available':torch.arange(64)>=6}
    normalizer=None
    if condition!='M':
        from .models import DescriptorNormalizer
        descriptor=build_descriptors(item['P'][None],item['V'][None],item['velocity_available'][None],condition)
        normalizer=DescriptorNormalizer(condition,'synthetic','synthetic').fit([('train',descriptor)]).to(device)
    optimizer=torch.optim.AdamW(adamw_parameter_groups(projector),lr=3e-4,eps=1e-6)
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    def update(monitor=None):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            with autocast(device,dtype):embedding=project_window(projector,item,normalizer,device).sensor_embeddings[0]
            example=splicer.build(uid='profile_synthetic',task='root',time_s=item['time_s'].to(device),sensor_embeddings=embedding,
                                  answer=canonical_answer('root','no_overall_trend'))
            loss=bridge.selected_answer_loss(collate_spliced([example]))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite alignment profile loss')
            scaler.scale(loss/accumulation).backward()
            if monitor:monitor.sample()
        scaler.unscale_(optimizer)
        norm=torch.stack([p.grad.detach().float().norm() for p in projector.parameters() if p.grad is not None]).norm()
        if torch.isfinite(norm):torch.nn.utils.clip_grad_norm_(projector.parameters(),1.,error_if_nonfinite=True)
        elif not scaler.is_enabled():raise FloatingPointError('Nonfinite profile gradient')
        previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<previous:raise FloatingPointError('Profile optimizer update skipped by GradScaler')
    for _ in range(warmup):update()
    torch.cuda.synchronize();monitor=ResourceMonitor(device);latencies=[]
    for _ in range(steps):
        start=time.perf_counter();update(monitor);torch.cuda.synchronize();latencies.append(time.perf_counter()-start)
    generation_latencies=[]
    projector.eval()
    with torch.no_grad():
        embedding=project_window(projector,item,normalizer,device).sensor_embeddings[0]
        example=splicer.build(uid='profile_synthetic',task='root',time_s=item['time_s'].to(device),sensor_embeddings=embedding)
        for _ in range(generations):
            start=time.perf_counter();bridge.greedy_generate(example,tokenizer,splicer.end_token_id);torch.cuda.synchronize()
            generation_latencies.append(time.perf_counter()-start);monitor.sample()
    report={'evidence':'pretrained_Qwen_synthetic_cached_sensors','actual_M4Human':False,'condition':condition,
            'warmup_steps':warmup,'measured_optimizer_steps':steps,'accumulation':accumulation,'microbatch':1,
            'generation_samples':generations,'resource':monitor.report(),'step_latency_s':latencies,
            'generation_latency_s':generation_latencies,'sequence_length':example.prefix_length,'sensor_embeddings':64,
            'acceptance_eligible':False,'note':'Synthetic resource diagnostic; real-data/prompt range and cold/warm I/O profile remain required.'}
    write_json(Path(output_dir)/f'profile_alignment_{condition}.json',report)
    return report
