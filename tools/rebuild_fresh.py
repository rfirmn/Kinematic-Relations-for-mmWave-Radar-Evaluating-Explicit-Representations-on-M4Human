"""Re-execute software evidence for the active generation with no old artifacts."""
from pathlib import Path
import subprocess
import sys
from relational64.config import ROOT,resolve,read_json,write_json,file_hash,identity

def main():
    config=resolve(ROOT/'configs/relational64/runtime.yaml');runtime=config['runtime'];generation=runtime['artifact_generation_id']
    reports=Path(runtime['report_root']);meta=Path(runtime['meta_root'])
    inventory=read_json(runtime['source_inventory'])
    if inventory.get('artifact_generation_id')!=generation or inventory.get('prior_artifacts_reused')!=[]:
        raise ValueError('Fresh source inventory required')
    qwen=read_json(reports/'qwen_native_check.json')
    if qwen['status']!='passed':raise ValueError('New native Qwen check must pass first')
    environment=read_json(reports/'environment.json')
    language={'artifact_generation_id':generation,'prior_artifacts_reused':[],'model_commit':qwen['model_commit'],'tokenizer_commit':qwen['tokenizer_commit'],
              'chat_template_hash':qwen['chat_template_hash'],'snapshot_file_hashes':qwen['snapshot_file_hashes'],
              'safetensors_manifest_hash':identity({n:h for n,h in qwen['snapshot_file_hashes'].items() if n.endswith('.safetensors')}),
              'dependency_lock':environment['dependency_lock_sha256'],'environment_report':file_hash(reports/'environment.json'),
              'precision':'bfloat16','head_mode':'canonical_fp32','verified_qwen_report_sha256':file_hash(reports/'qwen_native_check.json'),
              'actual_M4Human':False,'note':'Freshly measured language software lock, not G4 scientific acceptance'}
    write_json(reports/'language.lock.json',language)
    commands=[['verify','--suite','all'],['overfit','--synthetic'],['profile','--stage','physical','--synthetic'],
              ['profile','--stage','alignment','--condition','R','--synthetic'],['pilot','--synthetic']]
    checks=[]
    for arguments in commands:
        print('RUN '+ ' '.join(arguments),flush=True)
        result=subprocess.run([sys.executable,'-B','-u','-m','relational64.cli',*arguments],cwd=ROOT,text=True,capture_output=True)
        stem=arguments[0]+('_'+arguments[arguments.index('--stage')+1] if '--stage' in arguments else '')
        (reports/f'fresh_{stem}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        checks.append({'command':arguments,'exit_code':result.returncode,'log':str(reports/f'fresh_{stem}.log')})
        write_json(reports/'fresh_execution.json',{'artifact_generation_id':generation,'prior_artifacts_reused':[],'checks':checks,'status':'running' if result.returncode==0 else 'failed'})
        if result.returncode:raise RuntimeError(f'Fresh stage failed: {arguments}: {result.stderr[-1000:]}')
        print('PASS '+stem,flush=True)
    write_json(reports/'fresh_execution.json',{'artifact_generation_id':generation,'prior_artifacts_reused':[],'checks':checks,'status':'passed'})
    result=subprocess.run([sys.executable,'-B','tools/finalize_evidence.py'],cwd=ROOT)
    if result.returncode:raise RuntimeError('Final evidence collection failed')

if __name__=='__main__':main()
