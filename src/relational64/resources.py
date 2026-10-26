"""Measured environment identity and sampled resource limits."""
from __future__ import annotations
import importlib.metadata
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
import psutil
import torch
from .config import ROOT, SPEC, file_hash, write_json

def environment_report() -> dict:
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    cuda = torch.cuda.is_available()
    gpu = None
    if cuda:
        p = torch.cuda.get_device_properties(0)
        gpu = {'name':p.name,'total_memory':p.total_memory,'capability':list(torch.cuda.get_device_capability()),
               'bf16_supported':torch.cuda.is_bf16_supported()}
    driver = subprocess.run(['nvidia-smi','--query-gpu=driver_version,name,memory.total','--format=csv,noheader'],
                            text=True,capture_output=True).stdout.strip() if shutil.which('nvidia-smi') else None
    return {'platform':platform.platform(),'os':platform.system(),'python':sys.version,'python_executable':sys.executable,
            'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':cuda,'gpu':gpu,'nvidia_smi':driver,
            'physical_ram':psutil.virtual_memory().total,'available_ram':psutil.virtual_memory().available,
            'cpu_count':psutil.cpu_count(),'free_disk':shutil.disk_usage(ROOT).free,
            'packages':dict(sorted(packages.items())),'specification_sha256':file_hash(SPEC),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark':torch.backends.cudnn.benchmark,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32}

def save_environment(directory: str | Path) -> dict:
    directory = Path(directory)
    report = environment_report()
    directory.mkdir(parents=True,exist_ok=True)
    environment_path=directory/'environment.json'
    if environment_path.exists():
        import json
        previous=json.loads(environment_path.read_text(encoding='utf-8'))
        if previous.get('packages')==report['packages'] and previous.get('python_executable')==report['python_executable']:
            return previous  # Preserve the identity already pinned in language/run locks.
    lock = directory/'requirements.windows.lock.txt'
    lock.write_text('--extra-index-url https://download.pytorch.org/whl/cu124\n'+''.join(f'{k}=={v}\n' for k,v in report['packages'].items() if k.lower()!='m4human-relational64'),encoding='utf-8')
    report['dependency_lock_sha256'] = file_hash(lock)
    report['environment_origin'] = 'New CPython3.12.14 venv. Existing installed dependency bytes copied read-only from local reference environment; uv/pytest installed from index. No reference source code imported.'
    write_json(environment_path,report)
    return report

class ResourceMonitor:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.min_free_cuda = None
        self.min_available_ram = psutil.virtual_memory().available
        self.peak_rss = 0
        self.start = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def sample(self):
        self.min_available_ram = min(self.min_available_ram,psutil.virtual_memory().available)
        self.peak_rss = max(self.peak_rss,psutil.Process().memory_info().rss)
        if self.device.type == 'cuda':
            free,_ = torch.cuda.mem_get_info(self.device)
            self.min_free_cuda = min(self.min_free_cuda or free,free)

    def report(self):
        self.sample()
        cuda = self.device.type == 'cuda'
        return {'wall_seconds':time.perf_counter()-self.start,'device':str(self.device),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(self.device) if cuda else None,
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(self.device) if cuda else None,
                'minimum_free_cuda_bytes':self.min_free_cuda,'minimum_available_ram_bytes':self.min_available_ram,
                'peak_process_rss_bytes_sampled':self.peak_rss,
                'resource_guard_passed':(not cuda or self.min_free_cuda >= 1024*1024**2) and self.min_available_ram>=2048*1024**2,
                'sampling':'At each microbatch/optimizer boundary; not continuous OS telemetry.'}
