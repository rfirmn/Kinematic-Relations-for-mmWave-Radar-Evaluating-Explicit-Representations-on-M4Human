"""Bounded synthetic raw-RPC to Qwen integration pilot; no scientific scores."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import torch
from .config import identity,write_json
from .training import synthetic_windows,seed_everything
from .models import RadarPoseModel,freeze_for_inference,SensorProjector,DescriptorNormalizer,build_descriptors,parameter_hash
from .physical import window_operators
from .kinematics import differentiate_torch
from .alignment import load_language
from .language import canonical_answer,collate_spliced
from .qa import generate_qa,parse_answer
from .checkpoint import save_checkpoint
from .resources import ResourceMonitor

def synthetic_pilot(config,language_lock,output_dir):
    seed_everything(42);device=config['runtime']['device'];output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    fixtures=synthetic_windows(8);physical=freeze_for_inference(RadarPoseModel().to(device));physical_hash=parameter_hash(physical)
    states=[]
    with torch.no_grad():
        for fixture in fixtures:
            times=fixture['time_s'][None].to(device)
            prediction=physical(fixture['X'][None].to(device),fixture['point_mask'][None].to(device),times)
            w,available=window_operators(times)
            states.append({'Y':prediction.features.detach().cpu(),'P':prediction.positions.detach().cpu(),
                           'V':differentiate_torch(prediction.positions,w).detach().cpu(),'available':available.cpu(),'time_s':times.cpu()})
    physical.to('cpu');torch.cuda.empty_cache() if torch.device(device).type=='cuda' else None
    references=generate_qa('synthetic-0',fixtures[0]['time_s'].numpy(),fixtures[0]['positions'].numpy(),fixtures[0]['target_mask'].numpy())
    if any(r['reference_status']!='defined' for r in references):raise ValueError('Pilot fixture must have all three defined tasks')
    bridge,splicer,tokenizer,dtype=load_language(config,language_lock)
    qwen_hash=parameter_hash(bridge);monitor=ResourceMonitor(device);results={}
    for condition in ('M','S','R'):
        seed_everything(42);projector=SensorProjector(condition).to(device);normalizer=None
        if condition!='M':
            normalizer=DescriptorNormalizer(condition,physical_hash,identity(['synthetic',8]))
            normalizer.fit([('train',build_descriptors(s['P'],s['V'],s['available'],condition)) for s in states]).to(device)
        original=parameter_hash(projector);optimizer=torch.optim.AdamW(projector.parameters(),lr=3e-4,eps=1e-6)
        optimizer.zero_grad(set_to_none=True);loss_values=[]
        state=states[0]
        def embedding():
            descriptor=None if condition=='M' else normalizer(build_descriptors(state['P'].to(device),state['V'].to(device),state['available'].to(device),condition))
            return projector(state['Y'].to(device),descriptor).sensor_embeddings[0]
        for reference in references:
            example=splicer.build(uid=reference['qa_uid'],task=reference['part'],time_s=state['time_s'][0].to(device),
                                  sensor_embeddings=embedding(),answer=canonical_answer(reference['part'],reference['answer']))
            loss=bridge.selected_answer_loss(collate_spliced([example]))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite synthetic pilot loss')
            (loss/3).backward();loss_values.append(float(loss.detach()));monitor.sample()
        norm=float(torch.nn.utils.clip_grad_norm_(projector.parameters(),1.,error_if_nonfinite=True));optimizer.step()
        if any(p.grad is not None for p in bridge.parameters()):raise RuntimeError('Frozen Qwen gradient in pilot')
        generations=[];projector.eval()
        with torch.no_grad():
            for reference in references:
                example=splicer.build(uid=reference['qa_uid'],task=reference['part'],time_s=state['time_s'][0].to(device),sensor_embeddings=embedding())
                generated=bridge.greedy_generate(example,tokenizer,splicer.end_token_id)
                generations.append({'part':reference['part'],'tokens':example.prefix_length,'sensor_embeddings':len(example.sensor_positions),
                                    'raw_text':generated.raw_text,'parse':asdict(parse_answer(generated.raw_text,reference['part'],truncated=generated.truncated))})
        provenance={'evidence':'synthetic_software_fixture','artifact_generation_id':config['runtime'].get('artifact_generation_id'),'condition':condition,'seed':42,'physical_hash':physical_hash,'language_lock':language_lock,'diagnostic_accumulation':3}
        save_checkpoint(output/condition/'last.pt',projector,optimizer,provenance=provenance,epoch=0,step=1)
        results[condition]={'loss_per_task':loss_values,'gradient_norm':norm,'projector_changed':original!=parameter_hash(projector),
                            'trainable_parameters':sum(p.numel() for p in projector.parameters()),'generations':generations}
        monitor.sample();del projector,normalizer,optimizer,example,loss
    qwen_unchanged=qwen_hash==parameter_hash(bridge);physical_unchanged=physical_hash==parameter_hash(physical)
    report={'protocol':'m4human_relational64','evidence':'synthetic_software_fixture','actual_M4Human':False,
            'native_windows':__import__('platform').system()=='Windows','cuda_executed':torch.device(device).type=='cuda',
            'conditions':results,'frozen_Qwen_unchanged':qwen_unchanged,'frozen_physical_unchanged':physical_unchanged,
            'physical_initialization':'scratch untrained; software pilot only','train_steps_per_condition':1,
            'diagnostic_accumulation':3,'scientific_G4_accepted':False,'resource':monitor.report(),
            'status':'passed' if qwen_unchanged and physical_unchanged and all(v['projector_changed'] for v in results.values()) else 'failed',
            'interpretation':'Raw synthetic points -> physical predictions -> descriptor -> 64 embeddings -> frozen pretrained Qwen. No accuracy or dataset validation claim.'}
    write_json(output/'pilot.json',report);return report
