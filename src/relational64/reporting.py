"""Saved verification evidence and automatic implementation reports."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from .config import ROOT,file_hash,identity,read_json,write_json,resolve,require_gate
from .resources import save_environment

def verify(suite='all',output_dir=None):
    runtime=resolve(ROOT/'configs/relational64/runtime.yaml')['runtime']
    output_dir=output_dir or runtime['report_root']
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    selection={'data':['tests/test_data.py'],'targets':['tests/test_targets.py','tests/test_evaluation.py'],
               'models':['tests/test_models.py','tests/test_language.py'],'pipeline':['tests/test_pipeline.py','tests/test_physical.py'],
               'all':['tests']}
    if suite not in selection:raise ValueError('Unknown verification suite')
    xml=output/f'verify_{suite}.xml'
    temporary=ROOT/'.cache/pytest'/runtime.get('artifact_generation_id','software_fixture')/suite
    temporary.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pytest',*selection[suite],'-q',f'--junitxml={xml}',f'--basetemp={temporary}']
    completed=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    (output/f'verify_{suite}.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
    cases=[]
    if xml.exists():
        for case in ET.parse(xml).getroot().iter('testcase'):
            cases.append({'id':case.get('classname','')+'::'+case.get('name',''),
                          'status':'failed' if case.find('failure') is not None or case.find('error') is not None else 'skipped' if case.find('skipped') is not None else 'passed'})
    report={'suite':suite,'status':'passed' if completed.returncode==0 else 'failed','command':command,
            'artifact_generation_id':runtime.get('artifact_generation_id'),
            'prior_artifacts_reused':[],
            'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'returncode':completed.returncode,
            'data_source':'synthetic software fixtures, including temporary LMDBs','actual_M4Human_used':False,
            'native_windows':sys.platform=='win32','cuda_executed':False,
            'test_count':len(cases),'tests':cases,'junit_sha256':file_hash(xml) if xml.exists() else None,
            'source_hashes':{str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'src').rglob('*.py'))},
            'test_hashes':{str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'tests').glob('test_*.py'))},
            'tool_hashes':{str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'tools').glob('*.py'))},
            'config_hashes':{str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'configs').rglob('*')) if p.is_file()}}
    write_json(output/f'verify_{suite}.json',report)
    return report

def build_status(output_dir=None,*,inventory_path=None):
    runtime=resolve(ROOT/'configs/relational64/runtime.yaml')['runtime']
    output_dir=output_dir or runtime['report_root']
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    def optional(path):
        p=output/path
        return read_json(p) if p.exists() else None
    inventory_path=Path(inventory_path or runtime['source_inventory'])
    inventory=read_json(inventory_path) if inventory_path.exists() else None
    if inventory:
        from .data import assert_fresh_inventory
        assert_fresh_inventory(inventory,runtime.get('artifact_generation_id'))
    checks=optional('verify_all.json');qwen=optional('qwen_native_check.json')
    overfit=optional('synthetic_overfit/overfit.json');profile=optional('synthetic_profile/profile_physical.json')
    alignment_profile=optional('synthetic_alignment_profile/profile_alignment_R.json');pilot=optional('synthetic_pilot.json')
    diagnostic=[]
    for path in sorted((output/'diagnostic_runs').glob('*/summary.json')):
        row=read_json(path)
        if row.get('artifact_generation_id')==runtime.get('artifact_generation_id') and row.get('diagnostic_only') is True:
            diagnostic.append({'run_id':row['run_id'],'status':row['status'],'stages':row['stages'],'summary_path':str(path.resolve())})
    diagnostic_physical=any(row['stages'].get('physical',{}).get('status')=='passed' for row in diagnostic)
    diagnostic_alignment=any(row['stages'].get('alignment',{}).get('status')=='passed' for row in diagnostic)
    environment=save_environment(output)
    artifacts={}
    for name in ('verify_all.json',
                 'qwen_native_check.json','synthetic_overfit/overfit.json','synthetic_profile/profile_physical.json',
                 'synthetic_alignment_profile/profile_alignment_R.json','synthetic_pilot.json'):
        p=output/name
        if p.exists():artifacts[str(p.resolve())]=file_hash(p)
    if inventory:
        for p in (inventory_path,inventory_path.parent/'coordinate_audit.json'):
            if p.exists():artifacts[str(p.resolve())]=file_hash(p)
    blocker=inventory.get('authority',{}).get('blockers',[]) if inventory else ['Source inventory missing']
    gates={name:{'passed':False,'test_ids':[],'artifacts':{},'evidence':'missing','reason':reason} for name,reason in {
        'G0':'Data unit/joint/frame/time authority unresolved','G1':'Actual train/val targets and label audit pending G0',
        'G2':'Actual8-window overfit and native resource acceptance pending G0/G1',
        'G3':'Selected physical checkpoint, cache equivalence and train descriptor statistics pending',
        'G4':'Native Qwen software checks do not establish actual-cache alignment resource acceptance',
        'G5':'15 projectors/3 physical checkpoints and experiment choices have not been selected',
        'G6':'Scientific test evaluation not run'}.items()}
    if inventory:
        gates['G0'].update(passed=inventory['authority']['G0_passed'],actual_M4Human=True,test_ids=['full_rpc_inventory','coordinate_numeric','time_provenance'],
                           evidence='actual_M4Human',artifacts={str(p.resolve()):file_hash(p) for p in (inventory_path,inventory_path.parent/'coordinate_audit.json')},blockers=blocker)
    prior_path=output/'acceptance.json'
    if prior_path.exists() and read_json(prior_path).get('artifact_generation_id')==runtime.get('artifact_generation_id'):
        # Reporting never revokes a separately reviewed, still hash-valid gate.
        for name in gates:
            try:gates[name]=require_gate(prior_path,name)
            except (ValueError,FileNotFoundError):pass
    report={'protocol_id':'m4human_relational64','artifact_generation_id':runtime.get('artifact_generation_id'),'prior_artifacts_reused':[],'status':'implementation_with_data_authority_blocked',
            'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'gates':gates,'artifacts':artifacts,
            'fixture_tests':checks,'actual_source_audited':inventory is not None,'native_qwen_software_passed':bool(qwen and qwen['status']=='passed'),
            'synthetic_overfit_passed':bool(overfit and overfit['passed']),'synthetic_physical_profile_complete':profile is not None,
            'synthetic_alignment_profile_complete':alignment_profile is not None,'synthetic_pilot_passed':bool(pilot and pilot['status']=='passed'),
            'actual_diagnostic_training_run':diagnostic_physical,'actual_diagnostic_alignment_run':diagnostic_alignment,'diagnostic_runs':diagnostic,
            'scientific_training_run':False,'pilot_actual_data_run':diagnostic_physical,
            'scientifically_accepted_pilot_run':False,'full_training_run':False,'test_evaluation_run':False,
            'dependency_lock_hash':environment['dependency_lock_sha256']}
    write_json(output/'acceptance.json',report)
    lines=['# Status implementasi relational64','',
           'Implementasi perangkat lunak tersedia. Gerbang data G0 belum lulus; training dan pilot ilmiah yang disahkan belum dijalankan. '
           'Training diagnostik dengan RPC aktual dan calon target/waktu provisional dilaporkan terpisah.','',
           '| Bukti | Status |','|---|---|',
           f'| Audit M4Human aktual | {inventory["entries"]["rpc"] if inventory else 0:,} frame; read-only |',
           f'| Fixture perangkat lunak | {checks["test_count"] if checks else 0} tests; {checks["status"] if checks else "belum dijalankan"} |',
           f'| Qwen pretrained Windows/CUDA | {qwen["status"] if qwen else "belum dijalankan"}; sensor sintetis |',
           f'| Overfit8 sintetis | {"lulus" if overfit and overfit["passed"] else "belum lulus"} |',
           f'| Profil fisik sintetis | {"tersimpan" if profile else "belum dijalankan"} |',
           f'| Profil alignment sintetis | {"10 warm-up +30 optimizer steps +30 generations tersimpan" if alignment_profile else "belum dijalankan"} |',
           f'| Pilot M/S/R sintetis | {pilot["status"] if pilot else "belum dijalankan"} |',
           f'| Training fisik diagnostik aktual | {"selesai; provisional, G0 tetap false" if diagnostic_physical else "belum selesai"} |',
           f'| Alignment diagnostik aktual | {"selesai; provisional, G0 tetap false" if diagnostic_alignment else "belum selesai"} |','',
           *['- Run diagnostik ['+row['run_id']+']('+__import__('os').path.relpath(row['summary_path'],output.resolve()).replace('\\','/')+') — '+row['status'] for row in diagnostic],'',
           '## Penghambat data','',*['- '+b for b in blocker],'',
           '## Gerbang','', '| Gate | Lulus | Alasan |','|---|---|---|',
           *[f'| {k} | {v["passed"]} | {v["reason"]} |' for k,v in gates.items()],'',
           '## Artefak','',*['- ['+str(Path(k).name)+']('+__import__('os').path.relpath(k,output.resolve()).replace('\\','/')+')' for k in artifacts],'',
           'Kecepatan/durasi belum dinyatakan terukur; manifest diagnostik memakai waktu nominal provisional. '
           'Fixture tidak membuktikan pairing RPC–target aktual atau hasil ilmiah. Training penuh adalah pekerjaan terpisah.','']
    (output/'implementation_status.md').write_text('\n'.join(lines),encoding='utf-8')
    return report
