"""Recompute train/validation coordinate diagnostics for the active generation."""
from pathlib import Path
from relational64.config import ROOT,resolve,active_artifact_path
from relational64.audit import coordinate_diagnostics
from relational64.data import RPCOnlyReader,clean_rpc,read_json,assert_fresh_inventory

def main():
    runtime=resolve(ROOT/'configs/relational64/runtime.yaml')['runtime']
    inventory=read_json(active_artifact_path(runtime['source_inventory']));root=active_artifact_path(runtime['meta_root'])
    assert_fresh_inventory(inventory,runtime['artifact_generation_id'])
    split=read_json(root/'split.lock.json');lookup={int(s):name for name in ('train','val') for s in split[name]}
    counts={};samples=[];reader=RPCOnlyReader(inventory)
    try:
        for group in read_json(root/'recordings.json'):
            subject=group['subject_id']
            if subject not in lookup or counts.get(subject,0)>=2:continue
            key=f"[{subject}, {group['action_id']}, {group['first']}]"
            cloud,_=clean_rpc(reader.read_rpc(key),reader.schema)
            samples.append({'key':key,'split':lookup[subject],'cloud':cloud});counts[subject]=counts.get(subject,0)+1
    finally:reader.close()
    result=coordinate_diagnostics(inventory['databases'],samples,root)
    print({'artifact_generation_id':runtime['artifact_generation_id'],'new_samples':len(result['samples'])})

if __name__=='__main__':main()
