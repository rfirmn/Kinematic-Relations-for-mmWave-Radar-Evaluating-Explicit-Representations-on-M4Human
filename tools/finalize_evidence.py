"""Finalize only the active source-only generation; never consumes archived results."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
from relational64.config import ROOT,resolve,file_hash,identity,read_json,write_json,active_artifact_path

def main():
    config=resolve(ROOT/'configs/relational64/runtime.yaml');runtime=config['runtime'];generation=runtime['artifact_generation_id']
    reports=active_artifact_path(runtime['report_root']);meta=active_artifact_path(runtime['meta_root'])
    inventory_path=active_artifact_path(runtime['source_inventory']);inventory=read_json(inventory_path)
    from relational64.data import assert_fresh_inventory
    assert_fresh_inventory(inventory,generation)
    coordinate=read_json(meta/'coordinate_audit.json')
    checks=[]
    for name,path in inventory['databases'].items():
        digest=file_hash(path)
        checks.append({'database':name,'path':path,'sha256':digest,'matches_new_audit':digest==inventory['source_manifest_sha256'][name]})
    source_passed=bool(checks) and all(r['matches_new_audit'] for r in checks)
    write_json(reports/'source_integrity_final.json',{'artifact_generation_id':generation,'status':'passed' if source_passed else 'failed','checks':checks,'prior_metadata_read':False})
    if not source_passed:
        mismatched=', '.join(r['database'] for r in checks if not r['matches_new_audit']) or 'no source files'
        raise ValueError(f'Source snapshot mismatch; finalization stopped: {mismatched}')
    write_json(reports/'resolved_config.json',config)
    pilot_path=Path(runtime['derived_root'])/'synthetic_pilot/pilot.json'
    if pilot_path.exists():write_json(reports/'synthetic_pilot.json',read_json(pilot_path))
    from relational64.reporting import build_status
    status_report=build_status(reports,inventory_path=inventory_path)
    language=read_json(reports/'language.lock.json')
    environment_hash=file_hash(reports/'environment.json')
    if language.get('artifact_generation_id')!=generation or language.get('environment_report')!=environment_hash:
        raise ValueError('Language lock generation/environment hash mismatch; rebuild the active language lock')
    attempted=[]
    for command in ('prepare','build-qa','train-physical','train-alignment'):
        result=subprocess.run([sys.executable,'-m','relational64.cli',command],cwd=ROOT,capture_output=True,text=True)
        attempted.append({'command':command,'exit_code':result.returncode,'stderr':result.stderr.strip()})
    blocked_passed=all(r['exit_code']==3 for r in attempted)
    write_json(reports/'blocked_stage_checks.json',{'artifact_generation_id':generation,'status':'passed' if blocked_passed else 'failed','checks':attempted})
    if not blocked_passed:
        raise ValueError('Blocked-stage checks failed; runtime lock and artifact registry were not finalized')
    runtime_lock={'protocol_id':config['protocol_id'],'artifact_generation_id':generation,'artifact_policy':'fresh_from_source_only','prior_artifacts_reused':[],
                  'status':'data_authority_blocked','dataset_root':inventory['dataset_root'],'database_mapping':inventory['databases'],
                  'source_inventory_sha256':file_hash(inventory_path),'source_manifest_sha256':inventory['source_manifest_sha256'],
                  'serializer':inventory['serializer'],'coordinate_audit_sha256':file_hash(meta/'coordinate_audit.json'),
                  'coordinate_authorized':False,'time_authorized':False,'target_provenance':None,
                  'split_lock_sha256':file_hash(meta/'split.lock.json'),'model_commit':language['model_commit'],
                  'tokenizer_commit':language['tokenizer_commit'],'chat_template_hash':language['chat_template_hash'],
                  'safetensors_manifest_hash':language['safetensors_manifest_hash'],'precision':language['precision'],'head_mode':language['head_mode'],
                  'dependency_lock':language['dependency_lock'],'environment_report':environment_hash,
                  'config_hash':identity(config),'acceptance_report_hash':file_hash(reports/'acceptance.json'),
                  'G0_passed':False,'actual_training_started':bool(status_report.get('actual_diagnostic_training_run')),
                  'actual_diagnostic_training_started':bool(status_report.get('actual_diagnostic_training_run')),
                  'scientific_training_started':False,'meta_root':str(meta),'report_root':str(reports),'checkpoint_root':runtime['checkpoint_root']}
    write_json(meta/'runtime.lock.json',runtime_lock)
    pending={'artifact_generation_id':generation,'status':'not_fabricated','artifacts':{name:{'status':'not_built','reason':'G0 target/coordinate/time authority unresolved'} for name in
             ('intensity_normalizer_for_training','canonical_targets','actual_QA','selected_physical_checkpoint','actual_feature_cache','descriptor_normalizers','selected_projector_checkpoints','test_predictions')}}
    write_json(meta/'pending_artifacts.json',pending)
    index={'artifact_generation_id':generation,'prior_artifacts_reused':[],'raw_source_files':[str(p) for p in inventory['databases'].values()],
           'allowed_shared_inputs':['immutable source dataset','specification and pinned official source code','pinned pretrained Qwen weights','installed software libraries'],
           'fresh_outputs':{},'validity_limitation':'Fresh generation and software correctness do not replace missing scientific authority.'}
    for base in (Path(runtime['derived_root']),reports,Path(runtime['checkpoint_root'])):
        if base.exists():
            for path in sorted(base.rglob('*')):
                if path.is_file() and path.name!='artifact_registry.json' and path.suffix in ('.json','.yaml','.xml','.txt','.md','.jsonl','.npy','.pt','.png','.log'):
                    index['fresh_outputs'][str(path.relative_to(ROOT) if path.is_absolute() else path)]=file_hash(path)
    write_json(meta/'artifact_registry.json',index)
    (ROOT/'docs/report_training/implementation_status.md').write_text(f'# Generasi aktif\n\nSemua artefak aktif dibangun baru: `{generation}`.\n\n[Laporan lengkap]({generation}/implementation_status.md)\n\nArsip sebelumnya tidak boleh dipakai untuk penerimaan penelitian.\n',encoding='utf-8')
    print(json.dumps({'generation':generation,'source_hashes_valid':all(r['matches_new_audit'] for r in checks),'coordinate_samples':len(coordinate['samples']),'indexed_new_files':len(index['fresh_outputs'])}))

if __name__=='__main__':main()
