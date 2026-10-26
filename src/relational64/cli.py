"""Fail-closed stage CLI. Use python -m relational64.cli --help."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .config import ROOT,resolve,write_json,read_json,file_hash,require_gate

def parser():
    root=argparse.ArgumentParser(description=__doc__)
    subs=root.add_subparsers(dest='command',required=True)
    for name in ('audit','prepare','export-targets','build-qa','verify','overfit','profile','train-physical','extract','fit-descriptors',
                 'train-alignment','train-probe','lock-test','evaluate','evaluate-physical','evaluate-controls','match-donors','evaluate-donors','compare','pilot','report'):
        p=subs.add_parser(name)
        p.add_argument('--config',default='configs/relational64/runtime.yaml')
        p.add_argument('--output');p.add_argument('--gates')
        p.add_argument('--inventory')
        p.add_argument('--derived');p.add_argument('--target-provenance',default='configs/data_authority.template.json')
        p.add_argument('--targets',help='Canonical target array directory; test exports normally use a separate directory')
        p.add_argument('--test-lock')
        if name=='audit':p.add_argument('--full',action='store_true')
        if name=='export-targets':p.add_argument('--target-method',default='direct',choices=['direct','reference'])
        if name=='prepare':p.add_argument('--diagnostic',action='store_true')
        if name=='verify':p.add_argument('--suite',default='all',choices=['all','data','targets','models','pipeline'])
        if name in ('overfit','profile','pilot'):p.add_argument('--synthetic',action='store_true')
        if name=='pilot':p.add_argument('--language-lock')
        if name=='overfit':p.add_argument('--max-steps',type=int,default=500)
        if name=='profile':
            p.add_argument('--stage',choices=['physical','alignment'],required=True)
            p.add_argument('--warmup',type=int,default=10);p.add_argument('--steps',type=int,default=30)
            p.add_argument('--accumulation',type=int,default=16)
        if name in ('train-physical','extract','fit-descriptors','train-alignment','train-probe','evaluate','evaluate-physical','evaluate-controls','match-donors','evaluate-donors','compare'):
            p.add_argument('--seed',type=int,default=42)
        if name in ('train-alignment','fit-descriptors','train-probe','profile','evaluate','evaluate-donors'):
            p.add_argument('--condition',default='R',choices=['M','S','F','E','R','T'])
            p.add_argument('--normalizer');p.add_argument('--language-lock')
        if name in ('train-physical','train-alignment'):p.add_argument('--resume')
        if name in ('extract','export-targets','build-qa','evaluate','evaluate-physical','evaluate-controls','match-donors','evaluate-donors','compare'):p.add_argument('--split',default='train,val')
        if name in ('extract','evaluate','train-probe','evaluate-donors'):p.add_argument('--checkpoint')
        if name in ('train-probe','evaluate'):p.add_argument('--source',choices=['Y','Z_R'],default='Y')
        if name in ('train-alignment','train-probe','evaluate','evaluate-controls','match-donors','evaluate-donors','compare'):p.add_argument('--qa')
        if name=='evaluate-donors':p.add_argument('--pairs',required=True)
        if name=='evaluate-controls':p.add_argument('--train-qa',help='Train/val QA registry for the train-only majority fit')
        if name=='compare':p.add_argument('--prediction-plan',required=True,help='JSON map condition to raw prediction JSONL')
        if name=='evaluate':p.add_argument('--predictions',help='Score existing raw predictions against pinned QA')
        if name=='lock-test':p.add_argument('--plan',required=True)
    return root

def sensor_dataset(args,config,split,training=False):
    from .data import SensorDataset
    derived=Path(args.derived or config['runtime']['derived_root'])
    return SensorDataset(derived/'windows.jsonl',args.inventory,derived/'intensity_normalizer.json',split,training,test_lock=args.test_lock,expected_generation=config['runtime'].get('artifact_generation_id'))

def supervised_dataset(args,config,split,training=False):
    from .data import SupervisedDataset,TargetArrayStore
    derived=Path(args.derived or config['runtime']['derived_root'])
    return SupervisedDataset(sensor_dataset(args,config,split,training),TargetArrayStore(args.targets or derived/'targets',expected_generation=config['runtime'].get('artifact_generation_id')))

def load_selected_physical(path,config,seed):
    import torch
    from .models import RadarPoseModel
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('status')!='best' or state.get('step',0)<=0 or state['provenance']['seed']!=seed or state['provenance']['config']!=config:
        raise ValueError('Selected physical checkpoint identity does not match resolved config/seed')
    model=RadarPoseModel();model.load_state_dict(state['model']);return model

def dispatch(args):
    config=resolve(args.config);derived=Path(args.derived or config['runtime']['derived_root']);reports=Path(config['runtime']['report_root'])
    args.inventory=args.inventory or config['runtime'].get('source_inventory',derived/'meta/source_inventory.json')
    args.gates=args.gates or reports/'acceptance.json'
    if hasattr(args,'language_lock'):args.language_lock=args.language_lock or reports/'language.lock.json'
    generation=config['runtime'].get('artifact_generation_id')
    if not generation or config['runtime'].get('artifact_policy')!='fresh_from_source_only':
        raise ValueError('CLI requires the active fresh artifact policy and generation ID')
    if generation:
        from .config import active_artifact_path
        for key in ('output','derived','inventory','gates','language_lock','normalizer','targets','checkpoint','qa','target_provenance','test_lock','resume','plan','pairs','prediction_plan','train_qa','predictions'):
            value=getattr(args,key,None)
            if value:active_artifact_path(value)
        if Path(args.gates).exists() and read_json(args.gates).get('artifact_generation_id')!=generation:
            raise ValueError('Gate metadata artifact generation mismatch')
    checkpoints=Path(config['runtime'].get('checkpoint_root','checkpoints'))
    def fresh_cache(path):
        from .cache import PhysicalCache
        return PhysicalCache(path,expected_generation=generation)
    def check_generation(state):
        if generation is not None and state.get('artifact_generation_id')!=generation:
            raise ValueError('Checkpoint artifact generation mismatch')
    command=args.command
    if command=='audit':
        from .audit import run_audit
        if read_json(config['runtime']['source_policy']).get('artifact_generation_id')!=generation:
            raise ValueError('Source policy artifact generation mismatch')
        return run_audit(config['runtime']['dataset_root'],args.output or config['runtime']['meta_root'],policy_path=config['runtime']['source_policy'],full=args.full)
    if command=='prepare':
        from .prepare import prepare_data
        return prepare_data(args.inventory,args.output or derived,diagnostic=args.diagnostic,expected_generation=generation)
    if command=='export-targets':
        from .data import export_targets
        return export_targets(derived/'frames.jsonl',args.inventory,args.target_provenance,args.output or derived/'targets',
                              splits=tuple(args.split.split(',')),test_lock=args.test_lock,target_method=args.target_method,expected_generation=generation)
    if command=='build-qa':
        from .data import TargetArrayStore,assert_fresh_inventory
        from .pipeline_targets import build_qa_registry
        inventory=read_json(args.inventory)
        assert_fresh_inventory(inventory,generation)
        if inventory.get('authority',{}).get('G0_passed') is not True:raise ValueError('G0 data authority unresolved; QA target generation blocked')
        target_root=Path(args.targets or derived/'targets')
        targets=TargetArrayStore(target_root,expected_generation=generation)
        provenance={'G0_passed':inventory['authority']['G0_passed'],'inventory_hash':file_hash(args.inventory),
                    'coordinate_audit_hash':file_hash(Path(args.inventory).parent/'coordinate_audit.json'),
                    'time_audit_hash':file_hash(args.target_provenance),'split_hash':file_hash(derived/'split.lock.json'),
                    'target_hash':file_hash(target_root/'metadata.json'),'evidence':'actual_M4Human'}
        provenance['target_provenance_hash']=file_hash(args.target_provenance)
        provenance['artifact_generation_id']=generation
        return build_qa_registry(derived/'windows.jsonl',targets,args.output or derived/'qa.jsonl',
                                 provenance=provenance,splits=tuple(args.split.split(',')),test_lock_path=args.test_lock)
    if command=='verify':
        from .reporting import verify
        result=verify(args.suite,args.output or reports)
        if result['status']!='passed':raise RuntimeError('Verification failed; see saved log and JUnit report')
        return result
    if command in ('overfit','profile'):
        from .training import overfit_small,profile_physical
        if not args.synthetic:
            require_gate(args.gates,'G0');require_gate(args.gates,'G1')
        if command=='overfit':return overfit_small(None if args.synthetic else supervised_dataset(args,config,'train'),
                 output_dir=args.output or reports/('synthetic_overfit' if args.synthetic else 'overfit'),device=config['runtime']['device'],max_steps=args.max_steps,synthetic=args.synthetic)
        if args.stage=='physical':return profile_physical(None if args.synthetic else supervised_dataset(args,config,'train'),
                 output_dir=args.output or reports/('synthetic_profile' if args.synthetic else 'profile'),device=config['runtime']['device'],warmup=args.warmup,steps=args.steps,accumulation=args.accumulation,synthetic=args.synthetic)
        if not args.synthetic:raise ValueError('Alignment profile currently requires explicit synthetic fixture; actual-cache acceptance not implemented')
        from .alignment import profile_alignment
        return profile_alignment(config=config,language_lock=read_json(args.language_lock),condition=args.condition,
                 output_dir=args.output or reports/'synthetic_alignment_profile',warmup=args.warmup,steps=args.steps,accumulation=args.accumulation)
    if command=='train-physical':
        from .training import train_physical
        for gate in ('G0','G1','G2'):require_gate(args.gates,gate)
        return train_physical(supervised_dataset(args,config,'train',True),supervised_dataset(args,config,'val'),
                 output_dir=args.output or checkpoints/'physical'/str(args.seed),config=config,gate_path=args.gates,seed=args.seed,resume=args.resume)
    if command=='extract':
        from .cache import extract_cache
        for gate in ('G0','G1','G2'):require_gate(args.gates,gate)
        if not args.checkpoint:raise ValueError('--checkpoint best physical checkpoint required')
        if 'test' in args.split.split(','):
            from .study import require_locked_checkpoint
            if not args.test_lock:raise ValueError('Test extraction requires --test-lock')
            require_locked_checkpoint(args.test_lock,args.checkpoint,expected_generation=generation,config=config)
        model=load_selected_physical(args.checkpoint,config,args.seed);result={}
        for split in args.split.split(','):
            dataset=sensor_dataset(args,config,split)
            provenance={'checkpoint_hash':file_hash(args.checkpoint),'windows_hash':file_hash(derived/'windows.jsonl'),
                        'normalizer_hash':file_hash(derived/'intensity_normalizer.json'),'inventory_hash':file_hash(args.inventory),'seed':args.seed,'artifact_generation_id':generation}
            result[split]=extract_cache(model,dataset,(Path(args.output) if args.output else derived/'cache'/str(args.seed))/split,
                       device=config['runtime']['device'],provenance=provenance,split=split)
        return result
    if command=='fit-descriptors':
        from .cache import PhysicalCache,fit_descriptor_normalizer
        require_gate(args.gates,'G2')
        return fit_descriptor_normalizer(fresh_cache(derived/'cache'/str(args.seed)/'train'),args.condition,
                 args.output or derived/'normalizers'/f'{args.condition}_{args.seed}.json',file_hash(derived/'windows.jsonl'))
    if command in ('train-alignment','train-probe'):
        for gate in ('G0','G1','G2','G3'):require_gate(args.gates,gate)
        if command=='train-alignment':require_gate(args.gates,'G4')
        from .cache import PhysicalCache,load_descriptor_normalizer
        from .alignment import train_alignment,read_qa
        qa=args.qa or derived/'qa.jsonl';train=read_qa(qa,'train',expected_generation=generation);val=read_qa(qa,'val',expected_generation=generation)
        tc=fresh_cache(derived/'cache'/str(args.seed)/'train');vc=fresh_cache(derived/'cache'/str(args.seed)/'val')
        normalizer=args.normalizer or (None if args.condition=='M' else derived/'normalizers'/f'{args.condition}_{args.seed}.json')
        if command=='train-alignment':return train_alignment(tc,vc,train,val,condition=args.condition,normalizer_path=normalizer,
                 config=config,language_lock=read_json(args.language_lock),gate_path=args.gates,
                 output_dir=args.output or checkpoints/'alignment'/str(args.seed)/args.condition,seed=args.seed,resume=args.resume)
        from .study import train_probe
        projector=norm=None
        if args.source=='Z_R':
            import torch
            from .models import SensorProjector
            if not args.checkpoint:raise ValueError('Z_R requires selected R --checkpoint')
            state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            check_generation(state)
            if state.get('status')!='best' or state['provenance'].get('condition')!='R':raise ValueError('Selected R projector required')
            projector=SensorProjector('R');projector.load_state_dict(state['model']);norm=load_descriptor_normalizer(normalizer,tc)
        return train_probe(tc,vc,train,val,output_dir=args.output or checkpoints/'probe'/str(args.seed)/args.source,
                 seed=args.seed,source=args.source,projector=projector,normalizer=norm,gate_path=args.gates)
    if command=='lock-test':
        from .study import lock_test
        return lock_test(args.plan,args.output or derived/'test.lock.json',gate_path=args.gates,config=config)
    if command=='evaluate':
        from .study import save_score,validate_test_lock
        from .alignment import read_qa,generate_panel,load_language
        if args.split=='train,val':raise ValueError('Evaluation requires one explicit --split val or test')
        if args.split=='test':
            if not args.test_lock:raise ValueError('Test evaluation requires --test-lock')
            validate_test_lock(args.test_lock,expected_generation=generation,config=config)
            if args.checkpoint:
                from .study import require_locked_checkpoint
                require_locked_checkpoint(args.test_lock,args.checkpoint,expected_generation=generation,config=config)
        records=read_qa(args.qa or derived/'qa.jsonl',args.split,expected_generation=generation)
        output=Path(args.output or reports/'evaluation'/str(args.seed)/args.condition);output.mkdir(parents=True,exist_ok=True)
        predictions=args.predictions
        if not predictions:
            import torch
            from .cache import PhysicalCache,load_descriptor_normalizer
            from .models import SensorProjector
            require_gate(args.gates,'G4');cache=fresh_cache(derived/'cache'/str(args.seed)/args.split)
            bridge,splicer,tokenizer,dtype=load_language(config,read_json(args.language_lock));projector=normalizer=None
            if args.condition!='T':
                if not args.checkpoint:raise ValueError('Selected projector --checkpoint required')
                state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
                check_generation(state)
                if state.get('status')!='best' or state['provenance'].get('condition')!=args.condition or state['provenance']['seed']!=args.seed:raise ValueError('Selected projector condition/seed mismatch')
                projector=SensorProjector(args.condition).to(config['runtime']['device']);projector.load_state_dict(state['model'])
                if args.condition!='M':normalizer=load_descriptor_normalizer(args.normalizer or derived/'normalizers'/f'{args.condition}_{args.seed}.json',cache).to(config['runtime']['device'])
            predictions=output/'predictions.jsonl'
            generate_panel(records,cache,projector,normalizer,bridge,splicer,tokenizer,device=config['runtime']['device'],
                           output=predictions,condition=args.condition,seed=args.seed,checkpoint_hash=file_hash(args.checkpoint) if args.checkpoint else None)
        return save_score(records,predictions,output/'metrics.json',test_lock_path=args.test_lock)
    if command in ('evaluate-physical','evaluate-controls','match-donors','evaluate-donors','compare'):
        from .study import validate_test_lock
        from .cache import PhysicalCache,load_descriptor_normalizer
        from .alignment import read_qa
        if args.split not in ('train','val','test'):raise ValueError('Choose one explicit --split train, val, or test')
        if args.split=='test':
            if not args.test_lock:raise ValueError('Test evaluation requires --test-lock')
            validate_test_lock(args.test_lock,expected_generation=generation,config=config)
        cache=fresh_cache(derived/'cache'/str(args.seed)/args.split)
        output=Path(args.output or reports/command/str(args.seed));output.mkdir(parents=True,exist_ok=True)
        if command=='evaluate-physical':
            from .data import TargetArrayStore
            from .evaluation_pipeline import evaluate_physical_cache
            return evaluate_physical_cache(cache,derived/'windows.jsonl',TargetArrayStore(args.targets or derived/'targets',expected_generation=generation),output)
        records=read_qa(args.qa or derived/'qa.jsonl',args.split,expected_generation=generation)
        if command=='evaluate-controls':
            from .study import fit_majority,evaluate_controls
            majority=fit_majority(read_qa(args.train_qa or derived/'qa.jsonl','train',expected_generation=generation),output/'majority.json')
            scores,predictions=evaluate_controls(records,cache,majority)
            write_json(output/'controls_metrics.json',scores);write_json(output/'controls_predictions.json',predictions);return {'status':'completed','metrics':str(output/'controls_metrics.json')}
        if command=='match-donors':
            from .evaluation_pipeline import match_donors
            provenance=read_json(args.target_provenance)
            return match_donors(records,derived/'windows.jsonl',output/'pairs.json',joint_map_hash=__import__('relational64.config',fromlist=['identity']).identity(provenance['source_row_indices']),
                                coordinate_convention=file_hash(provenance['coordinate_audit_path']),seed=args.seed)
        if command=='compare':
            from .evaluation_pipeline import compare_conditions
            from .qa import parse_answer
            plan=read_json(args.prediction_plan);maps={}
            by_uid={r['qa_uid']:r for r in records}
            for condition,paths in plan.items():
                values=[]
                for path in paths if isinstance(paths,list) else [paths]:
                    if generation:active_artifact_path(path)
                    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
                    if len({r['qa_uid'] for r in rows})!=len(rows):raise ValueError('Duplicate prediction UID')
                    values.append({r['qa_uid']:parse_answer(r.get('raw_text'),by_uid[r['qa_uid']]['part'],truncated=r.get('stop_reason')=='token_limit',
                                                          runtime_error=r.get('reason') if r.get('stop_reason')=='runtime_error' else None) for r in rows})
                maps[condition]=values if isinstance(paths,list) else values[0]
            return compare_conditions(records,maps,output/'contrasts.json',test_lock_path=args.test_lock)
        if command=='evaluate-donors':
            import torch
            from .alignment import load_language
            from .models import SensorProjector
            from .evaluation_pipeline import generate_donors
            require_gate(args.gates,'G4');projector=normalizer=None
            if args.condition!='T':
                if not args.checkpoint:raise ValueError('Selected donor projector --checkpoint required')
                if args.split=='test':
                    from .study import require_locked_checkpoint
                    require_locked_checkpoint(args.test_lock,args.checkpoint,expected_generation=generation,config=config)
                state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
                check_generation(state)
                if state.get('status')!='best' or state['provenance'].get('condition')!=args.condition or state['provenance']['seed']!=args.seed:raise ValueError('Donor checkpoint mismatch')
                projector=SensorProjector(args.condition).to(config['runtime']['device']);projector.load_state_dict(state['model'])
                if args.condition!='M':normalizer=load_descriptor_normalizer(args.normalizer or derived/'normalizers'/f'{args.condition}_{args.seed}.json',cache).to(config['runtime']['device'])
            bridge,splicer,tokenizer,dtype=load_language(config,read_json(args.language_lock))
            return generate_donors(read_json(args.pairs),cache,projector,normalizer,bridge,splicer,tokenizer,device=config['runtime']['device'],condition=args.condition,output_dir=output)
    if command=='report':
        from .reporting import build_status
        return build_status(args.output or reports,inventory_path=args.inventory)
    if command=='pilot':
        if not args.synthetic:raise ValueError('Actual-data pilot requires accepted G0-G4 and a selected physical cache; use staged training/alignment commands')
        from .pilot import synthetic_pilot
        return synthetic_pilot(config,read_json(args.language_lock),args.output or derived/'synthetic_pilot')
    raise ValueError(f'Unhandled command {command}')

def main(argv=None):
    args=parser().parse_args(argv)
    try:
        result=dispatch(args)
        # Detailed evidence remains in files; never print huge per-frame reports.
        print(json.dumps({'command':args.command,'status':'completed','summary':{k:v for k,v in result.items() if isinstance(v,(str,int,float,bool,type(None)))}},indent=2,allow_nan=False))
    except (FileNotFoundError,ValueError) as error:
        print(f'{args.command}: {error}',file=sys.stderr)
        sys.exit(3 if any(x in str(error).lower() for x in ('g0','authority','provenance','missing','required','lock')) else 2)
    except (RuntimeError,FloatingPointError) as error:
        print(f'{args.command}: {error}',file=sys.stderr);sys.exit(4 if 'memory' in str(error).lower() else 5)

if __name__=='__main__':main()
