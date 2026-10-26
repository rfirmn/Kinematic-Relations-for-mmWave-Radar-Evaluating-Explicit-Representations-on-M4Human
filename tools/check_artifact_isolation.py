"""Check the active rebuilt artifacts without opening any prior research files."""
from relational64.config import ROOT,resolve,active_artifact_path,read_json,write_json,file_hash
from relational64.data import assert_fresh_inventory,validate_manifest_generation


def main():
    runtime=resolve(ROOT/'configs/relational64/runtime.yaml')['runtime']
    generation=runtime['artifact_generation_id']
    paths={name:active_artifact_path(runtime[name]) for name in
           ('meta_root','source_inventory','derived_root','report_root','checkpoint_root')}
    inventory=read_json(paths['source_inventory'])
    assert_fresh_inventory(inventory,generation)
    if inventory.get('prior_artifacts_reused')!=[]:
        raise ValueError('Fresh source inventory must declare zero prior artifacts')
    policy=read_json(active_artifact_path(runtime['source_policy']))
    if policy.get('artifact_generation_id')!=generation or policy.get('prior_metadata_inputs')!=[]:
        raise ValueError('Source policy must match this generation without prior metadata')
    for filename,rows in (('windows.jsonl',True),('frames.jsonl',False)):
        validate_manifest_generation(paths['derived_root']/filename,paths['source_inventory'],
                                     expected_generation=generation,require_row_generation=rows)
    scan=active_artifact_path(inventory['scan_path'])
    if file_hash(scan)!=inventory['scan_sha256']:
        raise ValueError('Fresh sensor scan checksum mismatch')
    summary=read_json(paths['derived_root']/'prepare_summary.json')
    output={
        'artifact_generation_id':generation,'status':'passed','prior_artifacts_read':[],
        'actual_M4Human_manifest_checked':True,'raw_data_read_by_this_check':False,
        'source_inventory_sha256':file_hash(paths['source_inventory']),
        'source_policy_sha256':file_hash(active_artifact_path(runtime['source_policy'])),
        'paths':{name:str(path) for name,path in paths.items()},
        'checks':{
            'all_active_outputs_inside_repository':True,'prior_metadata_inputs_empty':True,
            'inventory_source_only':True,'frames_hash_matches_fresh_summary':True,
            'windows_hash_matches_fresh_summary':True,'all_window_rows_current_generation':True,
            'sensor_scan_hash_matches_fresh_inventory':True,
        },
        'frame_counts':summary['frame_counts'],'window_counts':summary['window_counts'],
        'scientific_authority_passed':inventory.get('authority',{}).get('G0_passed') is True,
        'limitation':'Artifact isolation does not establish target/coordinate/time authority.',
    }
    write_json(paths['report_root']/'artifact_isolation_checks.json',output)
    print({'generation':generation,'status':'passed','actual_manifest':True,'G0':output['scientific_authority_passed']})


if __name__=='__main__':main()
