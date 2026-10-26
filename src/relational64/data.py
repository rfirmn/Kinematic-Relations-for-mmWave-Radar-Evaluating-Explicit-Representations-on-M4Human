"""Bounded RPC-only reader and explicitly separated supervised targets.

No old experiment modules are imported. Source databases are never created or
written. Scientific authority is checked separately from serializer validity.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def assert_fresh_inventory(inventory, expected_generation=None):
    if 'metadata_root' in inventory or 'metadata_hashes' in inventory:
        raise ValueError('Legacy metadata inventory is prohibited; rebuild it from source')
    expected_generation=expected_generation or inventory.get('artifact_generation_id')
    if expected_generation is not None and (inventory.get('artifact_generation_id')!=expected_generation or inventory.get('artifact_policy')!='fresh_from_source_only'):
        raise ValueError('Inventory artifact generation mismatch; prior artifacts cannot be reused')


def validate_manifest_generation(path, inventory_path=None, *, expected_generation=None, require_row_generation=True):
    """Bind rebuilt manifests to their generation, source inventory and contents.

    Frame identities deliberately describe source facts. Their file is pinned by
    the fresh preparation summary; window identities additionally carry a run ID.
    """
    if expected_generation is None:
        return
    from .config import active_artifact_path
    path=active_artifact_path(path)
    summary=read_json(path.parent/'prepare_summary.json')
    if summary.get('artifact_generation_id')!=expected_generation or summary.get('prior_artifacts_reused')!=[]:
        raise ValueError('Manifest artifact generation mismatch')
    digest_key={'windows.jsonl':'windows_sha256','frames.jsonl':'frames_sha256'}.get(path.name)
    if digest_key is None or summary.get(digest_key)!=file_hash(path):
        raise ValueError('Manifest checksum mismatch with fresh preparation summary')
    if inventory_path is not None:
        active_artifact_path(inventory_path)
        assert_fresh_inventory(read_json(inventory_path),expected_generation)
        if summary.get('source_inventory_sha256')!=file_hash(inventory_path):
            raise ValueError('Manifest source inventory mismatch')
    if require_row_generation:
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip() and json.loads(line).get('artifact_generation_id')!=expected_generation:
                    raise ValueError('Window row artifact generation mismatch')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


_VERIFIED_SNAPSHOTS = {}


def verify_snapshot(inventory, names):
    assert_fresh_inventory(inventory)
    """Hash source files once per process; rehash on observed identity/size/mtime change."""
    for name in names:
        path = Path(inventory['databases'][name]).resolve()
        digest = inventory.get('source_manifest_sha256',{}).get(name)
        if not isinstance(digest,str) or len(digest)!=64:
            raise ValueError('source manifest SHA256 required: '+name)
        actual = path/'data.mdb' if path.is_dir() else path
        before = actual.stat()
        signature = (os.getpid(),str(actual),before.st_size,before.st_mtime_ns,before.st_ctime_ns,digest)
        if signature not in _VERIFIED_SNAPSHOTS:
            observed = file_hash(actual)
            after = actual.stat()
            if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                raise ValueError('source changed while hashing: '+name)
            if observed!=digest:
                raise ValueError('source snapshot checksum mismatch: '+name)
            _VERIFIED_SNAPSHOTS[signature] = True


def parse_source_key(key):
    if isinstance(key, bytes):
        key = key.decode('utf-8', errors='strict')
    if not isinstance(key, str) or len(key) > 128:
        raise ValueError('source key must be bounded integer triple')
    try:
        obj = ast.literal_eval(key)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError('source key must be literal integer triple') from exc
    if not isinstance(obj, (list, tuple)) or len(obj) != 3 or any(type(v) is not int or abs(v) > 2**31 - 1 for v in obj):
        raise ValueError('source key must be bounded integer triple')
    return tuple(obj)


def decode_rpc(payload, schema):
    if schema.get('byte_order') not in ('little', 'big'):
        raise ValueError('explicit source byte order required')
    if payload is None or not 12 <= len(payload) <= schema.get('max_payload_bytes', 1600012):
        raise ValueError('missing/truncated/oversized RPC payload')
    endian = '<' if schema['byte_order'] == 'little' else '>'
    ndim, n, channels = struct.unpack_from(endian + 'III', payload)
    if ndim != 2 or channels != 4 or n > schema.get('max_source_points', 100000) or len(payload) != 12 + 16 * n:
        raise ValueError('RPC header/shape/length violates bounded Nx4 float32 schema')
    return np.frombuffer(payload, dtype=endian + 'f4', offset=12).astype(np.float32, copy=True).reshape(n, 4)


def decode_msgpack(payload, max_bytes=16 * 1024 * 1024, max_depth=12):
    import msgpack
    if payload is None or len(payload) > max_bytes:
        raise ValueError('missing/oversized metadata')
    def reject_extension(code, data):
        raise ValueError('metadata extension forbidden')
    obj = msgpack.unpackb(payload, raw=False, strict_map_key=True, ext_hook=reject_extension,
                          max_str_len=max_bytes, max_bin_len=max_bytes, max_array_len=100000,
                          max_map_len=10000, max_ext_len=0)
    def checked(x, depth=0):
        if depth > max_depth:
            raise ValueError('metadata depth exceeded')
        if isinstance(x, dict):
            if '__nd__' in x:
                if set(x) != {'__nd__', 'dtype', 'shape', 'data'} or x['__nd__'] is not True:
                    raise ValueError('invalid array tag')
                dtype, shape = np.dtype(x['dtype']), x['shape']
                if dtype.kind not in 'biuf' or dtype.itemsize > 8 or not isinstance(shape, list) or len(shape) > 5 or any(type(s) is not int or s <= 0 for s in shape):
                    raise ValueError('unsafe metadata dtype/shape')
                size = math.prod(shape) * dtype.itemsize
                if size > max_bytes or not isinstance(x['data'], bytes) or len(x['data']) != size:
                    raise ValueError('metadata array length mismatch')
                return np.frombuffer(x['data'], dtype=dtype).copy().reshape(shape)
            if any(not isinstance(k, str) for k in x):
                raise ValueError('metadata keys must be strings')
            return {k: checked(v, depth+1) for k, v in x.items()}
        if isinstance(x, list):
            return [checked(v, depth+1) for v in x]
        if x is None or isinstance(x, (str, bytes, bool, int, float)):
            return x
        raise ValueError('unsupported metadata type')
    return checked(obj)


def clean_rpc(points, schema):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError('expected Nx4 RPC')
    finite = np.isfinite(points).all(1)
    keep = finite.copy()
    policy = schema.get('sentinel_policy', 'none')
    if policy == 'exact_row':
        sentinel = np.asarray(schema.get('sentinel_row'), dtype=np.float32)
        if sentinel.shape != (4,) or not np.isfinite(sentinel).all():
            raise ValueError('explicit finite sentinel row required')
        keep &= ~np.all(points == sentinel, axis=1)
    elif policy != 'none':
        raise ValueError('unrecognized sentinel policy')
    return points[keep].copy(), {'raw': len(points), 'clean': int(keep.sum()), 'nonfinite': int((~finite).sum()), 'sentinel': int((finite & ~keep).sum())}


class RPCOnlyReader:
    """Only inventory['databases']['rpc'] can be opened by this reader."""
    _pool = {}
    def __init__(self, inventory):
        self.inventory = read_json(inventory) if isinstance(inventory, (str, Path)) else inventory
        assert_fresh_inventory(self.inventory)
        self.schema = self.inventory['serializer']
        self.path = Path(self.inventory['databases']['rpc'])
        if self.inventory.get('immutable_snapshot') is not True:
            raise ValueError('lock=False requires explicit immutable snapshot attestation')
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._env = None
        self._pid = None

    def environment(self):
        pid = os.getpid()
        if self._env is None or self._pid != pid:
            import lmdb
            key = (pid, str(self.path.resolve()))
            if key not in self._pool:
                self._pool[key] = [lmdb.open(str(self.path), subdir=self.path.is_dir(), readonly=True,
                    create=False, lock=False, writemap=False, readahead=False, max_readers=256), 0]
            self._pool[key][1] += 1
            self._env, self._pid = self._pool[key][0], pid
        return self._env

    def read_rpc(self, key):
        parse_source_key(key)
        with self.environment().begin(write=False, buffers=False) as tx:
            payload = tx.get(key.encode('utf-8') if isinstance(key, str) else key)
        return decode_rpc(payload, self.schema)

    read = read_rpc

    def close(self):
        if self._env is not None and self._pid == os.getpid():
            key = (self._pid, str(self.path.resolve()))
            if key in self._pool:
                self._pool[key][1] -= 1
                if self._pool[key][1] == 0:
                    self._pool.pop(key)[0].close()
        self._env = self._pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_env'] = state['_pid'] = None
        return state


class JsonlSequence:
    """Offset index, bounded per-row loading, and process-local file handles."""
    def __init__(self, path, split=None):
        self.path = str(Path(path).resolve())
        offsets = []
        with open(path, 'rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip() and (split is None or json.loads(line)['split'] == split):
                    offsets.append(offset)
        self.offsets = np.array(offsets, dtype=np.int64)
        self._handle = None
        self._pid = None

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if self._handle is None or self._pid != os.getpid():
            self._handle = open(self.path, 'rb')
            self._pid = os.getpid()
        self._handle.seek(int(self.offsets[index]))
        return json.loads(self._handle.readline())

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_handle'] = state['_pid'] = None
        return state


def sample_points(points, window_uid, frame_index, training=False, epoch=0, seed=42, max_points=150):
    material = f'sampling/{seed}/{window_uid}/{frame_index}'
    if training:
        material = f'train/{seed}/{epoch}/{window_uid}/{frame_index}'
    digest = hashlib.sha256(material.encode()).digest()
    if len(points) > max_points:
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
        points = points[rng.choice(len(points), max_points, replace=False)]
    out, mask = np.zeros((max_points, 4), np.float32), np.zeros(max_points, bool)
    out[:len(points)], mask[:len(points)] = points, True
    return out, mask


def make_sensor_batch(reader, window, normalizer, training=False, epoch=0, seed=42):
    if any(not isinstance(normalizer.get(name),(int,float)) or not math.isfinite(normalizer[name]) for name in ('a','median','iqr')) or normalizer['a']<1e-6 or normalizer['iqr']<1e-6:
        raise ValueError('finite intensity normalization with positive floored scale required')
    if len(window['source_keys']) != 64 or len(window['time_s']) != 64:
        raise ValueError('window must contain exactly64 frames')
    times = np.asarray(window['time_s'], np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > .125 + 1e-12):
        raise ValueError('invalid window timestamp/gap')
    clouds, masks, centroid = [], [], None
    transform = np.asarray(reader.inventory.get('T_source_to_radar', np.eye(4)), np.float64)
    for i, key in enumerate(window['source_keys']):
        points, stats = clean_rpc(reader.read_rpc(key), reader.schema)
        if stats['clean'] < 8:
            raise ValueError(f'sensor changed after audit: {key}')
        points[:, :3] = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        if centroid is None:
            centroid = np.array([points[:, 0].mean(dtype=np.float64), points[:, 1].mean(dtype=np.float64), 0], np.float32)
        points[:, :3] -= centroid
        cloud, mask = sample_points(points, window['window_uid'], i, training, epoch, seed)
        intensity = cloud[:, 3].astype(np.float64)
        u = np.sign(intensity) * np.log1p(np.abs(intensity) / normalizer['a'])
        cloud[:, 3] = np.clip((u-normalizer['median']) / normalizer['iqr'], -10, 10)
        cloud[~mask] = 0
        clouds.append(cloud)
        masks.append(mask)
    return {'X': torch.from_numpy(np.stack(clouds)), 'point_mask': torch.from_numpy(np.stack(masks)),
            'time_s': torch.from_numpy(times), 'centroid': torch.from_numpy(centroid), 'uid': window['window_uid']}


class SensorDataset(Dataset):
    def __init__(self, windows_path, inventory_path, normalizer_path, split='train', training=False, *, test_lock=None, expected_generation=None):
        inventory=read_json(inventory_path)
        expected_generation=expected_generation or inventory.get('artifact_generation_id')
        assert_fresh_inventory(inventory,expected_generation)
        validate_manifest_generation(windows_path,inventory_path,expected_generation=expected_generation)
        self.reader = RPCOnlyReader(inventory_path)
        if self.reader.inventory.get('authority', {}).get('G0_passed') is not True:
            raise ValueError('G0 authority unresolved; real-data model input is blocked')
        if split not in ('train','val','test') or (training and split!='train'):
            raise ValueError('valid split and train-only stochastic sampling required')
        if split=='test':
            from .study import validate_test_lock
            if not test_lock:
                raise ValueError('test sensor access requires final test lock')
            lock = validate_test_lock(test_lock,expected_generation=expected_generation)
            if not isinstance(inventory_path,(str,Path)) or lock.get('source_inventory_sha256')!=file_hash(inventory_path):
                raise ValueError('test lock source inventory mismatch')
        verify_snapshot(self.reader.inventory,('rpc',))
        self.windows = JsonlSequence(windows_path, split)
        self.normalizer = read_json(normalizer_path)
        if self.normalizer.get('diagnostic_only') is True:
            raise ValueError('Provisional diagnostic normalizer cannot enter scientific training')
        if expected_generation is not None and self.normalizer.get('artifact_generation_id')!=expected_generation:
            raise ValueError('Intensity normalizer artifact generation mismatch')
        if self.normalizer.get('fit_split') != 'train':
            raise ValueError('train-only intensity normalizer required')
        expected = self.normalizer.get('normalizer_hash')
        if expected != canonical_hash({k:v for k,v in self.normalizer.items() if k!='normalizer_hash'}):
            raise ValueError('intensity normalizer hash mismatch')
        self.training, self.epoch = training, 0

    def __len__(self):
        return len(self.windows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        return make_sensor_batch(self.reader, self.windows[index], self.normalizer, self.training, self.epoch)


class SupervisedDataset(Dataset):
    def __init__(self, sensor_dataset, targets):
        self.sensor_dataset, self.targets = sensor_dataset, targets

    def __len__(self):
        return len(self.sensor_dataset)

    def __getitem__(self, index):
        sensor = self.sensor_dataset[index]
        targets = [self.targets.read(key) for key in self.sensor_dataset.windows[index]['source_keys']]
        positions = np.stack([p for p, _ in targets]) - sensor['centroid'].numpy()
        mask = np.stack([m for _, m in targets])
        positions[~mask] = 0
        return {**sensor, 'positions': torch.from_numpy(positions), 'target_mask': torch.from_numpy(mask)}


JOINT_NAMES = ['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle',
    'right_ankle','spine3','left_foot','right_foot','neck','left_collar','right_collar','head',
    'left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist']


class DirectTargetReader:
    """Independent params/calib target exporter; never loads a legacy target cache.

    The direct route requires a separately audited equivalence certificate. A
    field named 'joints' or an old audited=True flag does not establish this.
    """
    def __init__(self, inventory, provenance):
        self.inventory = read_json(inventory) if isinstance(inventory,(str,Path)) else inventory
        self.provenance = read_json(provenance) if isinstance(provenance,(str,Path)) else provenance
        assert_fresh_inventory(self.inventory)
        if self.inventory.get('artifact_generation_id') is not None and self.provenance.get('artifact_generation_id')!=self.inventory['artifact_generation_id']:
            raise ValueError('Target provenance artifact generation mismatch')
        if self.inventory.get('authority',{}).get('G0_passed') is not True:
            raise ValueError('G0 authority required for canonical target export')
        verify_snapshot(self.inventory,('params','calib'))
        p = self.provenance
        required = ('target_equivalence_verified','reference_asset_sha256','reference_regressor_sha256',
                    'equivalence_evidence_sha256','coordinate_audit_sha256')
        if p.get('target_equivalence_verified') is not True or any(not p.get(name) for name in required):
            raise ValueError('target equivalence certificate, reference assets and coordinate hashes required')
        for name in ('reference_asset','reference_regressor','equivalence_evidence','coordinate_audit'):
            if self.inventory.get('artifact_generation_id') is not None and p.get(name+'_path'):
                from .config import active_artifact_path
                active_artifact_path(p[name+'_path'])
            if not p.get(name+'_path') or file_hash(p[name+'_path'])!=p[name+'_sha256']:
                raise ValueError('target authority artifact hash mismatch: '+name)
        if p.get('ordered_names')!=JOINT_NAMES or p.get('direct_joint_unit')!='meter' or p.get('source_frame')!='vicon':
            raise ValueError('canonical joint order, unit and source frame required')
        self.indices = p.get('source_row_indices')
        if not isinstance(self.indices,list) or len(self.indices)!=22 or len(set(self.indices))!=22 or any(type(i)is not int or i<0 for i in self.indices):
            raise ValueError('explicit22 distinct source row indices required')
        self._envs, self._pid = {}, None

    def _environment(self):
        if self._pid != os.getpid():
            from .audit import _readonly
            self._envs = {name:_readonly(self.inventory['databases'][name]) for name in ('params','calib')}
            self._pid = os.getpid()
        return self._envs

    def read(self, source_key):
        from .audit import transform_to_radar
        parse_source_key(source_key)
        payloads = {}
        for name,env in self._environment().items():
            with env.begin(write=False) as tx:
                payloads[name] = tx.get(source_key.encode())
        if payloads['calib'] is None:
            raise ValueError('missing calibration is a systematic export error: '+source_key)
        calib = decode_msgpack(payloads['calib'])
        transform_to_radar(np.zeros((1,3)),calib)  # calibration gate even if annotation absent
        if payloads['params'] is None:
            return np.zeros((22,3),np.float32),np.zeros(22,bool)
        params = decode_msgpack(payloads['params'])
        if not isinstance(params,dict):
            raise ValueError('params must be mapping')
        if 'joints' not in params:
            return np.zeros((22,3),np.float32),np.zeros(22,bool)
        joints = np.asarray(params['joints'],np.float64)
        if joints.ndim!=2 or joints.shape[1]!=3 or max(self.indices)>=len(joints):
            raise ValueError('source joint shape/map mismatch')
        points = joints[self.indices]
        mask = np.isfinite(points).all(-1)
        result = np.zeros((22,3),np.float32)
        result[mask] = transform_to_radar(points[mask],calib).astype(np.float32)
        mask &= np.isfinite(result).all(-1)
        result[~mask] = 0
        return result,mask

    def close(self):
        for env in self._envs.values():
            env.close()
        self._envs, self._pid = {}, None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_envs'],state['_pid'] = {},None
        return state


class TargetArrayStore:
    """Canonical new-repository target exports, mmap and numeric source-key index."""
    def __init__(self, root, *, expected_generation=None):
        self.root = Path(root)
        if expected_generation is not None:
            from .config import active_artifact_path
            active_artifact_path(self.root)
        self.metadata = read_json(self.root/'metadata.json')
        if expected_generation is not None and self.metadata.get('artifact_generation_id')!=expected_generation:
            raise ValueError('Target artifact generation mismatch')
        if self.metadata.get('protocol_id')!='m4human_relational64' or self.metadata.get('complete') is not True:
            raise ValueError('canonical complete target export required')
        if self.metadata.get('artifact_generation_id') is not None:
            from .config import active_artifact_path
            active_artifact_path(self.root)
        allowed={'positions.npy','valid.npy','index.npy','index.jsonl'}
        if set(self.metadata.get('hashes',{}))!=allowed:
            raise ValueError('Target files must use the canonical local filenames')
        for filename in self.metadata['hashes']:
            if (self.root/filename).resolve().parent!=self.root.resolve():
                raise ValueError('Target file path escapes its export directory')
        for filename,digest in self.metadata['hashes'].items():
            if file_hash(self.root/filename)!=digest:
                raise ValueError('target artifact checksum mismatch: '+filename)
        self._arrays,self._pid = None,None

    def _open(self):
        if self._arrays is None or self._pid!=os.getpid():
            self._arrays = {name:np.load(self.root/(name+'.npy'),mmap_mode='r',allow_pickle=False)
                for name in ('index','positions','valid')}
            count = self.metadata['count']
            for name,shape,dtype in [('index',(count,3),np.int64),('positions',(count,22,3),np.float32),('valid',(count,22),bool)]:
                if self._arrays[name].shape!=shape or self._arrays[name].dtype!=dtype:
                    raise ValueError('canonical target array shape/dtype: '+name)
            self._pid = os.getpid()

    def read(self, key):
        self._open()
        identity = parse_source_key(key)
        index = self._arrays['index']
        lo,hi = 0,len(index)
        while lo<hi:
            middle = (lo+hi)//2
            if tuple(index[middle])<identity:
                lo=middle+1
            else:
                hi=middle
        if lo==len(index) or tuple(index[lo])!=identity:
            return np.zeros((22,3),np.float32),np.zeros(22,bool)
        pos,mask = np.array(self._arrays['positions'][lo],copy=True),np.array(self._arrays['valid'][lo],copy=True)
        if np.any(~np.isfinite(pos)) or np.any(pos[~mask]!=0):
            raise ValueError('canonical target must store finite invalid zeros')
        return pos,mask

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_arrays'],state['_pid'] = None,None
        return state


def export_targets(frames_path, inventory_path, provenance_path, output_dir, *, splits=('train','val'), test_lock=None, target_method='direct',expected_generation=None):
    """Writes own arrays only after authority validation; no silent legacy reuse."""
    inventory=read_json(inventory_path)
    expected_generation=expected_generation or inventory.get('artifact_generation_id')
    assert_fresh_inventory(inventory,expected_generation)
    if inventory.get('authority',{}).get('G0_passed') is not True:
        raise ValueError('G0 authority required before opening target export inputs')
    validate_manifest_generation(frames_path,inventory_path,expected_generation=expected_generation,require_row_generation=False)
    if target_method=='direct':
        reader = DirectTargetReader(inventory_path,provenance_path)
    elif target_method=='reference':
        from .reference_targets import ReferenceSMPLXTargetReader
        reader = ReferenceSMPLXTargetReader(inventory_path,provenance_path)
    else:
        raise ValueError('explicit target method direct or reference required')
    if not splits or not set(splits)<= {'train','val','test'}:
        raise ValueError('explicit valid target export splits required')
    if 'test' in splits:
        from .study import validate_test_lock
        lock = validate_test_lock(test_lock,expected_generation=expected_generation) if test_lock else {}
        if lock.get('test_evaluation_authorized') is not True or lock.get('source_inventory_sha256')!=file_hash(inventory_path):
            raise ValueError('test target export requires matching final test lock')
    output = Path(output_dir).resolve()
    for root in ('dataset_root',):
        source = Path(reader.inventory[root]).resolve()
        if output==source or source in output.parents:
            raise ValueError('target output must be outside read-only source')
    if output.exists():
        raise FileExistsError(output)
    rows = JsonlSequence(frames_path)
    order = sorted((i for i in range(len(rows)) if rows[i]['split'] in splits),key=lambda i:parse_source_key(rows[i]['source_key']))
    incomplete = output.with_name(output.name+'.incomplete')
    incomplete.mkdir(parents=True,exist_ok=False)
    arrays = {}
    try:
        arrays = {name:np.lib.format.open_memmap(incomplete/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
            for name,shape,dtype in [('index',(len(order),3),np.int64),('positions',(len(order),22,3),np.float32),('valid',(len(order),22),bool)]}
        valid_count = 0
        previous = None
        with (incomplete/'index.jsonl').open('w',encoding='utf-8') as f:
            for output_row,source_row in enumerate(order):
                row = rows[source_row]
                key = parse_source_key(row['source_key'])
                if key==previous:
                    raise ValueError('duplicate source key in target export')
                previous = key
                positions,mask = reader.read(row['source_key'])
                arrays['index'][output_row],arrays['positions'][output_row],arrays['valid'][output_row] = key,positions,mask
                valid_count += int(mask.sum())
                f.write(json.dumps({'frame_uid':row['frame_uid'],'source_key':row['source_key'],'row':output_row,
                    'shard':'positions.npy','annotation_source':'params.lmdb '+target_method,
                    'coordinate_hash':reader.provenance['coordinate_audit_sha256'],
                    'joint_map_hash':canonical_hash(reader.indices)},separators=(',',':'))+'\n')
        for a in arrays.values():
            a.flush()
            a._mmap.close()
        arrays.clear()
        meta = {'protocol_id':'m4human_relational64','artifact_generation_id':reader.inventory.get('artifact_generation_id'),'complete':True,'count':len(order),'splits':list(splits),'valid_joint_count':valid_count,
            'inventory_hash':file_hash(inventory_path),'target_provenance_hash':file_hash(provenance_path),
            'hashes':{n:file_hash(incomplete/n) for n in ('positions.npy','valid.npy','index.npy','index.jsonl')}}
        write_json(incomplete/'metadata.json',meta)
        incomplete.rename(output)
    finally:
        for a in arrays.values():
            a._mmap.close()
        reader.close()
    return meta
