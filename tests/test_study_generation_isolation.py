"""Synthetic software checks only; no source dataset or GPU access."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relational64 import study
from relational64 import config as config_module
from relational64.config import ROOT, file_hash, identity, read_json, write_json
from relational64.prepare import make_split


@pytest.fixture
def plan_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module,'ROOT',tmp_path)
    generation = 'synthetic_study_generation'
    config = {'runtime': {'artifact_generation_id': generation}}
    monkeypatch.setattr(study, 'resolve', lambda _: config)
    monkeypatch.setattr(study, 'require_gate', lambda *_: None)
    paths = {key: tmp_path / (key + '.json') for key in study._PLAN_ARTIFACTS}
    paths['source_manifest'] = tmp_path / 'windows.jsonl'
    for key, path in paths.items():
        write_json(path, {'artifact_generation_id': generation})
    write_json(paths['resolved_config'], config)
    write_json(paths['source_inventory'], {
        'artifact_generation_id': generation, 'artifact_policy': 'fresh_from_source_only',
        'authority': {'G0_passed': True}})
    write_json(paths['exposure_ledger'], {
        'artifact_generation_id': generation, 'test_exposure_status': 'documented'})
    paths['source_manifest'].write_text('{"artifact_generation_id":"' + generation + '"}\n')
    split = make_split(range(1, 21))
    write_json(paths['split_lock'], split)
    write_json(tmp_path / 'prepare_summary.json', {
        'artifact_generation_id': generation, 'prior_artifacts_reused': [],
        'windows_sha256': file_hash(paths['source_manifest']),
        'source_inventory_sha256': file_hash(paths['source_inventory']),
        'split_hash': split['split_hash']})
    gate = tmp_path / 'gates.json'
    write_json(gate, {'artifact_generation_id': generation})
    plan = {key: str(value) for key, value in paths.items()}
    plan.update(seeds=[42, 43, 44], conditions=['M', 'S', 'F', 'E', 'R'], runs={})
    for seed in plan['seeds']:
        group = {'projectors': {}}
        for condition in (None, *plan['conditions']):
            path = tmp_path / f'{seed}_{condition}.pt'
            provenance = {'seed': seed, 'condition': condition, 'config': config}
            torch.save({'schema': 'relational64_checkpoint_1', 'status': 'best', 'step': 1,
                        'artifact_generation_id': generation, 'provenance': provenance,
                        'provenance_hash': identity(provenance)}, path)
            if condition is None:
                group['physical'] = str(path)
            else:
                group['projectors'][condition] = str(path)
        plan['runs'][str(seed)] = group
    plan_path = tmp_path / 'plan.json'
    write_json(plan_path, plan)
    return SimpleNamespace(generation=generation, config=config, plan=plan, plan_path=plan_path,
                           gate=gate, output=tmp_path / 'test.lock.json', paths=paths)


def create_lock(f):
    return study.lock_test(f.plan_path, f.output, gate_path=f.gate, config=f.config)


def test_valid_versioned_lock_pins_all_lineage(plan_fixture):
    f = plan_fixture
    lock = create_lock(f)
    assert lock['artifact_generation_id'] == f.generation
    assert str(f.gate) in lock['artifacts']
    assert str(f.output.parent / 'prepare_summary.json') in lock['artifacts']
    assert study.validate_test_lock(f.output, expected_generation=f.generation, config=f.config) == lock
    study.require_locked_checkpoint(f.output, f.plan['runs']['42']['physical'],
                                    expected_generation=f.generation, config=f.config)


@pytest.mark.parametrize('kind', ['checkpoint', 'inventory'])
def test_nested_forbidden_paths_rejected_before_hash_or_checkpoint_load(plan_fixture, monkeypatch, kind):
    f = plan_fixture
    if kind == 'checkpoint':
        f.plan['runs']['42']['physical'] = str(f.output.parent / 'archive' / 'must_not_read.pt')
    else:
        f.plan['source_inventory'] = str(f.output.parent.parent / 'must_not_read.json')
    write_json(f.plan_path, f.plan)
    def forbidden(*args, **kwargs):
        pytest.fail('Nested hash/load must not execute before path preflight')
    monkeypatch.setattr(study, 'file_hash', forbidden)
    monkeypatch.setattr(study.torch, 'load', forbidden)
    with pytest.raises(ValueError, match='Archived|inside this repository'):
        create_lock(f)


@pytest.mark.parametrize('kind', ['metadata', 'checkpoint', 'checkpoint_config', 'gate'])
def test_generation_or_config_mismatch_rejected(plan_fixture, kind):
    f = plan_fixture
    if kind == 'metadata':
        write_json(f.paths['label_policy'], {'artifact_generation_id': 'prior'})
    elif kind == 'gate':
        write_json(f.gate, {'artifact_generation_id': 'prior'})
    else:
        path = f.plan['runs']['42']['physical']
        state = torch.load(path, weights_only=False)
        if kind == 'checkpoint':
            state['artifact_generation_id'] = 'prior'
        else:
            state['provenance']['config'] = {'runtime': {'artifact_generation_id': 'prior'}}
            state['provenance_hash'] = identity(state['provenance'])
        torch.save(state, path)
    with pytest.raises(ValueError, match='generation'):
        create_lock(f)


def test_lock_missing_generation_cannot_bypass_explicit_expectation(plan_fixture, monkeypatch):
    f = plan_fixture
    lock = create_lock(f)
    del lock['artifact_generation_id']
    lock.pop('identity')
    lock['identity'] = identity(lock)
    write_json(f.output, lock)
    monkeypatch.setattr(study, 'file_hash', lambda *_: pytest.fail('Must reject before hash'))
    with pytest.raises(ValueError, match='generation'):
        study.validate_test_lock(f.output, expected_generation=f.generation, config=f.config)


def test_lock_rejects_extra_archived_artifact_before_hash(plan_fixture, monkeypatch):
    f = plan_fixture
    lock = create_lock(f)
    lock['artifacts'][str(f.output.parent / 'archive' / 'must_not_read.json')] = '0' * 64
    lock.pop('identity')
    lock['identity'] = identity(lock)
    write_json(f.output, lock)
    monkeypatch.setattr(study, 'file_hash', lambda *_: pytest.fail('Must reject before hash'))
    with pytest.raises(ValueError, match='Archived'):
        study.validate_test_lock(f.output, config=f.config)


def test_unversioned_synthetic_lock_stays_compatible(tmp_path):
    artifact = tmp_path / 'fixture.txt'
    artifact.write_text('synthetic')
    lock = {'status': 'locked', 'artifacts': {str(artifact): file_hash(artifact)}}
    lock['identity'] = identity(lock)
    path = tmp_path / 'fixture.lock.json'
    write_json(path, lock)
    assert study.validate_test_lock(path) == lock


@pytest.mark.parametrize('mismatch', ['cache', 'qa'])
def test_probe_rejects_generation_mixing_before_training(tmp_path, mismatch):
    train = SimpleNamespace(meta={'source': {'artifact_generation_id': 'active'}})
    val = SimpleNamespace(meta={'source': {'artifact_generation_id': 'prior' if mismatch == 'cache' else 'active'}})
    with pytest.raises(ValueError, match='generation'):
        study.train_probe(train, val, [{'artifact_generation_id': 'active'}],
                          [{'artifact_generation_id': 'prior' if mismatch == 'qa' else 'active'}],
                          output_dir=tmp_path, gate_path=tmp_path / 'unused.json')
