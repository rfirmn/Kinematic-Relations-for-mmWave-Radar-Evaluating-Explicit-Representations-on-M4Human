"""Canonical identities and fail-closed runtime gates."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'docs/planning_relasi_kinematik/spesifikasi.yaml'

def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')

def identity(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()

def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def active_artifact_path(path):
    resolved=(ROOT/Path(path)).resolve()
    if resolved!=ROOT and ROOT not in resolved.parents:
        raise ValueError('Artifact path must stay on C: inside this repository')
    archive=(ROOT/'archive').resolve()
    if resolved==archive or archive in resolved.parents:
        raise ValueError('Archived research artifacts cannot enter the active pipeline')
    return resolved

def resolve(path: str | Path | None = None) -> dict:
    base = yaml.safe_load(SPEC.read_text(encoding='utf-8'))
    if path:
        patch = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        if set(patch) - {'runtime'}:
            raise ValueError('Runtime patch may only contain runtime; scientific variants require a separate specification.')
        base['runtime'].update(patch.get('runtime', {}))
    validate(base)
    return base

def validate(config: dict) -> None:
    runtime=config['runtime']
    if 'metadata_root' in runtime:
        raise ValueError('Legacy metadata_root is forbidden; generate a new meta_root from the source dataset')
    if runtime.get('artifact_policy') not in (None,'fresh_from_source_only'):
        raise ValueError('Research artifacts must be rebuilt from source')
    if runtime.get('artifact_policy')=='fresh_from_source_only':
        import re
        if not isinstance(runtime.get('artifact_generation_id'),str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',runtime['artifact_generation_id']):
            raise ValueError('Fresh artifact policy requires a nonempty safe artifact generation ID')
    for key in ('derived_root','meta_root','report_root','checkpoint_root','source_inventory'):
        if runtime.get(key):
            active_artifact_path(runtime[key])
    checks = {
        'protocol': config['protocol_id'] == 'm4human_relational64',
        'input': config['data']['channels'] == ['x','y','z','intensity'],
        'frames': config['data']['window_frames'] == 64,
        'joint map': len(config['kinematics']['parents']) == config['data']['joint_count'] == 22,
        'root': config['data']['root_joint'] == 0,
        'edges': 21 + len(config['kinematics']['bilateral_edges']) == config['kinematics']['edge_count'] == 29,
        'embeddings': config['projector']['output_sensor_embeddings'] == 64,
        'freeze': config['language']['frozen_all_weights'] and not config['language']['lora'],
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError('Specification conflict: ' + ', '.join(failed))
    canonical(config)

def require_gate(report_path: str | Path, name: str) -> dict:
    report = read_json(report_path)
    if report.get('artifact_generation_id') is not None:
        active_artifact_path(report_path)
    gate = report.get('gates', {}).get(name)
    if not isinstance(gate, dict) or gate.get('passed') is not True:
        raise ValueError(f'{name} not accepted in {report_path}; dependent execution blocked')
    if not gate.get('test_ids') or not gate.get('artifacts') or not gate.get('evidence'):
        raise ValueError(f'{name} is missing evidence/test IDs/artifact hashes')
    if gate['evidence'] == 'synthetic_software_fixture' or gate.get('actual_M4Human') is not True:
        raise ValueError(f'{name} cannot be accepted from synthetic evidence alone')
    if name in ('G2','G4') and gate.get('native_windows_cuda') is not True:
        raise ValueError(f'{name} requires native Windows/CUDA evidence')
    for path, expected in gate.get('artifacts', {}).items():
        import re
        if report.get('artifact_generation_id') is not None:
            active_artifact_path(path)
        if not re.fullmatch(r'[0-9a-f]{64}', expected or ''):
            raise ValueError(f'Invalid artifact digest in {name}: {path}')
        if file_hash(path) != expected:
            raise ValueError(f'{name} artifact hash mismatch: {path}')
    return gate

def validate_language_lock(lock: dict) -> None:
    import re
    for key in ('model_commit','tokenizer_commit'):
        if not re.fullmatch(r'[0-9a-f]{40}', lock.get(key) or ''):
            raise ValueError(f'Missing pinned {key}')
    for key in ('chat_template_hash','safetensors_manifest_hash','dependency_lock','environment_report'):
        if not re.fullmatch(r'[0-9a-f]{64}', lock.get(key) or ''):
            raise ValueError(f'Missing language provenance: {key}')
    if lock.get('precision') not in ('bfloat16','float16') or lock.get('head_mode')!='canonical_fp32':
        raise ValueError('Language precision/head_mode must match the accepted native numerical check')
