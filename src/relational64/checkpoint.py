"""Atomic stateful checkpoints; frozen language weights are not duplicated."""
from __future__ import annotations
import os
import random
from pathlib import Path
import numpy as np
import torch
from .config import identity, file_hash

def rng_state() -> dict:
    return {'python':random.getstate(), 'numpy':np.random.get_state(), 'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state: dict) -> None:
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])

def save_checkpoint(path, model, optimizer, *, provenance, epoch, step, scheduler=None, scaler=None, best=None, patience=0, sampler_state=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    state = {'schema':'relational64_checkpoint_1','artifact_generation_id':provenance.get('artifact_generation_id',provenance.get('config',{}).get('runtime',{}).get('artifact_generation_id')),'status':path.stem,'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
             'scheduler':scheduler.state_dict() if scheduler else None, 'scaler':scaler.state_dict() if scaler else None,
             'provenance':provenance, 'provenance_hash':identity(provenance), 'epoch':epoch, 'step':step,
             'rng':rng_state(), 'best':best,'patience':patience,'sampler':sampler_state}
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(state, handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_hash(path)

def load_checkpoint(path, model, *, provenance, optimizer=None, scheduler=None, scaler=None, restore_random=True):
    # Local authored checkpoints contain Python/NumPy RNG state; never load untrusted files.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('schema') != 'relational64_checkpoint_1' or state['provenance_hash'] != identity(provenance):
        raise ValueError(f'Checkpoint provenance mismatch: {path}')
    if identity(state['provenance']) != state['provenance_hash']:
        raise ValueError('Corrupt checkpoint provenance')
    model.load_state_dict(state['model'], strict=True)
    for obj,key in ((optimizer,'optimizer'),(scheduler,'scheduler'),(scaler,'scaler')):
        if obj is not None and state[key] is not None:
            obj.load_state_dict(state[key])
    if restore_random:
        restore_rng(state['rng'])
    return state
