"""Physical training, fixed-sample overfit and measured device profiles.

Full study launches are separate jobs and require accepted data/resource gates.
"""
from __future__ import annotations
from contextlib import nullcontext
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from .checkpoint import save_checkpoint,load_checkpoint
from .config import identity,write_json,require_gate
from .models import RadarPoseModel,adamw_parameter_groups,parameter_hash
from .physical import physical_loss,window_operators,effective_denominators,LOSS_WEIGHTS
from .resources import ResourceMonitor,environment_report

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)

def choose_precision(device):
    if torch.device(device).type == 'cuda':
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def locked_precision(config):
    name=config['runtime'].get('precision')
    if name not in ('bfloat16','float16'):raise ValueError('A verified runtime.precision lock is required before training')
    dtype=getattr(torch,name)
    if config['runtime']['device']=='cuda' and dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('Locked BF16 is unsupported on this CUDA device')
    return dtype

def autocast(device,dtype):
    return torch.autocast(device_type=torch.device(device).type,dtype=dtype) if dtype!=torch.float32 else nullcontext()

def move(batch,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}

def aggregate_terms(accum,terms):
    for k,values in terms.detached().items():
        row = accum.setdefault(k,{'numerator':0.,'denominator':0.})
        for name,value in values.items(): row[name]+=value

def summarize_terms(accum):
    result = {k:{**v,'mean':v['numerator']/v['denominator'] if v['denominator'] else None} for k,v in accum.items()}
    total = sum(LOSS_WEIGHTS[k]*(v['mean'] or 0) for k,v in result.items())
    return {'components':result,'L_phys':total,'no_supervision':not any(v['denominator'] for v in result.values())}

def forward_loss(model,batch,dtype):
    w,a = window_operators(batch['time_s'])
    with autocast(batch['X'].device,dtype):
        out = model(batch['X'],batch['point_mask'],batch['time_s'])
    terms = physical_loss(out.positions,batch['positions'],batch['target_mask'],w,a)
    return out,terms

@torch.no_grad()
def validate_physical(model,loader,device,dtype):
    model.eval(); accum={}
    for batch in loader:
        _,terms=forward_loss(model,move(batch,device),dtype); aggregate_terms(accum,terms)
    return summarize_terms(accum)

def grouped(loader,size):
    group=[]
    for item in loader:
        group.append(item)
        if len(group)==size:
            yield group; group=[]
    if group: yield group

def optimizer_update(model,optimizer,batches,device,dtype,scaler,monitor=None):
    denominators=effective_denominators(batches,device)
    if denominators is None or not any(float(v)>0 for v in denominators.values()):
        return {'skipped':'no_supervision'}
    optimizer.zero_grad(set_to_none=True); accum={}
    for batch in batches:
        batch=move(batch,device)
        _,terms=forward_loss(model,batch,dtype)
        scaler.scale(terms.total(denominators)).backward()
        aggregate_terms(accum,terms)
        if monitor: monitor.sample()
    scaler.unscale_(optimizer)
    norm=torch.stack([p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]).norm()
    if torch.isfinite(norm):torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    elif not scaler.is_enabled():raise FloatingPointError('Nonfinite physical gradient')
    previous_scale=scaler.get_scale()
    scaler.step(optimizer); scaler.update()
    updated=scaler.get_scale()>=previous_scale
    if monitor: monitor.sample()
    return {**summarize_terms(accum),'gradient_norm':float(norm),'optimizer_updated':updated}

def schedule_factor(step,total,warmup,maximum,minimum):
    if step<warmup: return (step+1)/max(warmup,1)
    fraction=min(1.,max(0.,(step-warmup)/max(total-warmup,1)))
    return (minimum+(maximum-minimum)*.5*(1+math.cos(math.pi*fraction)))/maximum

def train_physical(train_dataset,val_dataset,*,output_dir,config,gate_path,seed=42,resume=None):
    require_gate(gate_path,'G0'); require_gate(gate_path,'G1'); require_gate(gate_path,'G2')
    seed_everything(seed); recipe=config['training']['physical']; device=config['runtime']['device']; dtype=locked_precision(config)
    if not len(train_dataset) or not len(val_dataset): raise ValueError('Empty train/validation panel')
    model=RadarPoseModel().to(device)
    generator=torch.Generator().manual_seed(seed)
    train=DataLoader(train_dataset,batch_size=1,shuffle=True,generator=generator,num_workers=0)
    windows=getattr(getattr(val_dataset,'sensor_dataset',val_dataset),'windows',None)
    val_order=sorted(range(len(val_dataset)),key=lambda i:windows[i]['window_uid']) if windows is not None else list(range(len(val_dataset)))
    val=DataLoader(Subset(val_dataset,val_order),batch_size=1,shuffle=False,num_workers=0)
    optimizer=torch.optim.AdamW(adamw_parameter_groups(model),lr=recipe['learning_rate'],betas=(.9,.999),eps=1e-6)
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    steps_per_epoch=math.ceil(len(train)/16); total=steps_per_epoch*recipe['epochs_max']; warmup=steps_per_epoch*3
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:schedule_factor(step,total,warmup,6e-4,1e-6))
    provenance={'protocol':config['protocol_id'],'config':config,'seed':seed,'precision':str(dtype),
                'effective_batch':16,'gate_hash':identity(__import__('json').loads(Path(gate_path).read_text()))}
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    best=float('inf'); patience=step=start_epoch=0; history=[]
    if resume:
        state=load_checkpoint(resume,model,provenance=provenance,optimizer=optimizer,scheduler=scheduler,scaler=scaler)
        start_epoch=state['epoch']+1;step=state['step'];best=state['best'];patience=state['patience']
        if state['sampler'] is not None:generator.set_state(state['sampler'])
        if (output_dir/'history.json').exists():
            history=__import__('json').loads((output_dir/'history.json').read_text())['history']
    for epoch in range(start_epoch,recipe['epochs_max']):
        if hasattr(train_dataset,'set_epoch'): train_dataset.set_epoch(epoch)
        elif hasattr(train_dataset,'sensor_dataset'): train_dataset.sensor_dataset.set_epoch(epoch)
        model.train(); accum={}; skipped=0
        for batches in grouped(train,16):
            record=optimizer_update(model,optimizer,batches,device,dtype,scaler)
            if record.get('optimizer_updated'):
                scheduler.step(); step+=1
            else: skipped+=1
            for k,values in record.get('components',{}).items():
                row=accum.setdefault(k,{'numerator':0.,'denominator':0.})
                for name in row: row[name]+=values[name]
        validation=validate_physical(model,val,device,dtype)
        if validation['no_supervision']: raise ValueError('Validation has no physical supervision')
        improved=validation['L_phys'] < best-1e-8
        if improved: best=validation['L_phys']; patience=0
        else: patience+=1
        checkpoint_args=dict(provenance=provenance,epoch=epoch,step=step,scheduler=scheduler,scaler=scaler,best=best,patience=patience,sampler_state=generator.get_state())
        save_checkpoint(output_dir/'last.pt',model,optimizer,**checkpoint_args)
        if improved: save_checkpoint(output_dir/'best.pt',model,optimizer,**checkpoint_args)
        history.append({'epoch':epoch,'optimizer_step':step,'skipped_groups':skipped,'train':summarize_terms(accum),'validation':validation})
        write_json(output_dir/'history.json',{'provenance':provenance,'history':history})
        if patience>=15: break
    return {'selected_L_phys':best,'optimizer_steps':step,'epochs':len(history)}

def synthetic_windows(count=8,points=150):
    """Software fixture; never described as measured motion or M4Human."""
    generator=torch.Generator().manual_seed(19)
    data=[]
    for index in range(count):
        cloud=torch.randn(1,points,4,generator=generator)*.1
        cloud[...,1]+=2.+.1*index; cloud[...,2]+=.9
        x=cloud.expand(64,-1,-1).clone()
        joint=torch.randn(22,3,generator=generator)*.03
        joint[:,2]+=.9; joint[:,1]+=.1*index
        data.append({'X':x,'point_mask':torch.ones(64,points,dtype=torch.bool),'time_s':torch.arange(64,dtype=torch.float64)/12,
                     'positions':joint.expand(64,-1,-1).clone(),'target_mask':torch.ones(64,22,dtype=torch.bool),'uid':f'synthetic-{index}'})
    return data

def overfit_small(dataset=None,*,output_dir,device='cuda',max_steps=500,synthetic=False):
    if max_steps>500: raise ValueError('Overfit budget <=500')
    seed_everything(42)
    if synthetic: dataset=synthetic_windows()
    if dataset is None or len(dataset)<8: raise ValueError('Exactly8 fixed train windows required')
    if not synthetic: dataset=Subset(dataset,list(range(8)))
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    batches=list(loader); model=RadarPoseModel().to(device); dtype=choose_precision(device)
    optimizer=torch.optim.AdamW(adamw_parameter_groups(model),lr=6e-4,eps=1e-6)
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    initial=validate_physical(model,batches,device,dtype)['L_phys']; history=[]
    monitor=ResourceMonitor(device)
    for step in range(max_steps):
        # Fixed clouds and eval dropout expose optimization/pairing, without resampling.
        model.eval()
        train=optimizer_update(model,optimizer,batches,device,dtype,scaler,monitor)
        loss=validate_physical(model,batches,device,dtype)['L_phys']
        history.append({'step':step+1,'loss':loss,'gradient_norm':train.get('gradient_norm')})
        if loss<=initial*.5: break
    report={'evidence':'synthetic_software_fixture' if synthetic else 'actual_M4Human_train',
            'native_windows':__import__('platform').system()=='Windows','cuda':torch.device(device).type=='cuda',
            'window_count':8,'initial_loss':initial,'final_loss':loss,'reduction_fraction':1-loss/initial,
            'passed':loss<=initial*.5,'optimizer_steps':len(history),'history':history,'resource':monitor.report(),
            'dropout_mode':'eval; gradients enabled; fixed sample optimization diagnostic',
            'scientific_G2_accepted':False if synthetic else loss<=initial*.5,'environment':environment_report()}
    write_json(Path(output_dir)/'overfit.json',report)
    return report

def profile_physical(dataset=None,*,output_dir,device='cuda',warmup=10,steps=30,accumulation=16,synthetic=False):
    seed_everything(42)
    if synthetic: dataset=synthetic_windows(1)
    if dataset is None: raise ValueError('Train/validation dataset required')
    loader=DataLoader(Subset(dataset,[0]),batch_size=1,num_workers=0)
    batch=next(iter(loader)); batches=[batch]*accumulation
    model=RadarPoseModel().to(device);model.train();dtype=choose_precision(device)
    optimizer=torch.optim.AdamW(adamw_parameter_groups(model),lr=6e-4,eps=1e-6)
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    for _ in range(warmup): optimizer_update(model,optimizer,batches,device,dtype,scaler)
    if torch.device(device).type=='cuda': torch.cuda.synchronize()
    monitor=ResourceMonitor(device);latencies=[]
    for step in range(steps):
        start=time.perf_counter()
        result=optimizer_update(model,optimizer,batches,device,dtype,scaler,monitor)
        if not result.get('optimizer_updated'): raise FloatingPointError(f'Profile skipped optimizer step{step}')
        if torch.device(device).type=='cuda': torch.cuda.synchronize()
        latencies.append(time.perf_counter()-start)
    resource=monitor.report()
    report={'evidence':'synthetic_software_fixture' if synthetic else 'actual_M4Human_train_or_validation',
            'stage':'physical','points':batch['X'].shape[-2],'frames':64,'microbatch':1,'accumulation':accumulation,
            'warmup_optimizer_steps':warmup,'measured_optimizer_steps':steps,'precision':str(dtype),'resource':resource,
            'seconds_per_step_p50':float(np.median(latencies)),'seconds_per_step_p95':float(np.quantile(latencies,.95)),
            'windows_per_second':steps*accumulation/sum(latencies),'latencies_s':latencies,'environment':environment_report(),
            'acceptance_eligible':not synthetic and warmup>=10 and steps>=30 and accumulation==16 and resource['resource_guard_passed'],
            'not_measured':['cold/warm source I/O comparison','generation (alignment profile is separate)','continuous system/GPU telemetry']}
    write_json(Path(output_dir)/'profile_physical.json',report)
    return report
