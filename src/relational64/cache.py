"""Condition-independent mmap physical cache, shared across QA and projectors."""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from .config import active_artifact_path,file_hash,write_json,read_json,identity
from .models import build_descriptors,DescriptorNormalizer,parameter_hash,freeze_for_inference
from .physical import window_operators
from .kinematics import differentiate_torch

ARRAYS={'Y':((64,256),np.float32),'P':((64,22,3),np.float32),'V':((64,22,3),np.float32),
        'time_s':((64,),np.float64),'centroid':((3,),np.float32),'velocity_available':((64,),np.bool_)}

@torch.no_grad()
def extract_cache(model,dataset,output_dir,*,device,provenance,split):
    if split not in ('train','val','test'): raise ValueError('Unknown cache split')
    if not len(dataset): raise ValueError('Empty extraction panel')
    if getattr(dataset,'training',False):raise ValueError('Cache extraction requires deterministic evaluation sampling')
    output=Path(output_dir)
    if output.exists(): raise FileExistsError(f'Cache destination must be new: {output}')
    temporary=output.with_name(output.name+'.partial');temporary.mkdir(parents=True,exist_ok=False)
    freeze_for_inference(model);model.to(device)
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    hashes={};arrays=None;shard=-1;count=0
    index_file=temporary/'index.jsonl'
    with index_file.open('w',encoding='utf-8') as index:
        for batch in loader:
            if count%1024==0:
                if arrays is not None:
                    for a in arrays.values(): a.flush(); a._mmap.close()
                    arrays.clear()
                shard+=1;size=min(1024,len(dataset)-count)
                arrays={name:np.lib.format.open_memmap(temporary/f'{name}_{shard:05d}.npy',mode='w+',dtype=dtype,shape=(size,*shape)) for name,(shape,dtype) in ARRAYS.items()}
            x,mask,times=(batch[k].to(device) for k in ('X','point_mask','time_s'))
            predicted=model(x,mask,times);w,available=window_operators(times)
            velocity=differentiate_torch(predicted.positions.float(),w)
            values={'Y':predicted.features.float(),'P':predicted.positions.float(),'V':velocity,'time_s':times,
                    'centroid':batch['centroid'],'velocity_available':available}
            for name,value in values.items():
                value=value[0].detach().cpu().numpy()
                if not np.isfinite(value).all(): raise FloatingPointError(f'Nonfinite cache {batch["uid"][0]} {name}')
                arrays[name][count%1024]=value
            index.write(json.dumps({'window_uid':batch['uid'][0],'split':split,'shard':shard,'row':count%1024})+'\n');count+=1
    if arrays is not None:
        for a in arrays.values():a.flush();a._mmap.close()
        arrays.clear()
    for path in sorted(temporary.glob('*.npy')): hashes[path.name]=file_hash(path)
    hashes['index.jsonl']=file_hash(index_file)
    metadata={'protocol_id':'m4human_relational64','count':count,'split':split,'complete':True,'source':provenance,
              'encoder_pose_hash':parameter_hash(model),'sampling':'deterministic_sampling/42/window/frame',
              'module_eval':True,'format':'npy_mmap_shards','max_windows_per_shard':1024,'files':hashes,'feature_dtype':'float32'}
    metadata['identity']=identity(metadata)
    write_json(temporary/'cache.meta.json',metadata)
    temporary.rename(output)
    return metadata

class PhysicalCache(Dataset):
    def __init__(self,root,*,expected_provenance=None,verify_files=True,expected_generation=None):
        self.root=active_artifact_path(root) if expected_generation is not None else Path(root).resolve()
        self.meta=read_json(self._member('cache.meta.json'))
        source_generation=self.meta.get('source',{}).get('artifact_generation_id')
        metadata_generation=self.meta.get('artifact_generation_id')
        if source_generation is not None or metadata_generation is not None:
            self.root=active_artifact_path(self.root)
        if metadata_generation is not None and metadata_generation!=source_generation:
            raise ValueError('Cache metadata/source artifact generation mismatch')
        if expected_generation is not None and source_generation!=expected_generation:
            raise ValueError('Cache artifact generation mismatch')
        original=dict(self.meta); stored=original.pop('identity')
        if identity(original)!=stored or not original['complete'] or not original['module_eval']:
            raise ValueError('Cache identity/state mismatch')
        if expected_provenance is not None and original['source']!=expected_provenance: raise ValueError('Stale cache provenance')
        count=original.get('count')
        if type(count)is not int or count<1 or original.get('max_windows_per_shard')!=1024:
            raise ValueError('Cache count/shard contract mismatch')
        self.shard_count=(count+1023)//1024
        expected_names={'index.jsonl'} | {f'{name}_{shard:05d}.npy' for name in ARRAYS for shard in range(self.shard_count)}
        if not isinstance(self.meta.get('files'),dict):
            raise ValueError('Cache files must be a pinned filename/hash map')
        # Preflight the entire map before any file hash or index read. Neither
        # an absolute key nor a symlink may escape this cache directory.
        self.files={name:self._member(name) for name in self.meta['files']}
        if set(self.files)!=expected_names:
            raise ValueError('Cache pinned files do not match canonical shards')
        if any(not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest) for digest in self.meta['files'].values()):
            raise ValueError('Invalid cache artifact digest')
        if verify_files:
            for name,digest in self.meta['files'].items():
                if file_hash(self.files[name])!=digest: raise ValueError(f'Cache hash mismatch: {name}')
        self.index=[json.loads(line) for line in self.files['index.jsonl'].read_text().splitlines()]
        if len(self.index)!=count:
            raise ValueError('Cache index count mismatch')
        for row in self.index:
            shard,offset=row.get('shard'),row.get('row')
            if type(shard)is not int or not 0<=shard<self.shard_count or type(offset)is not int or not 0<=offset<min(1024,count-shard*1024) or row.get('split')!=self.meta['split']:
                raise ValueError('Invalid cache shard/index entry')
        self.by_uid={row['window_uid']:i for i,row in enumerate(self.index)}
        if len(self.by_uid)!=len(self.index): raise ValueError('Duplicate window cache UID')
        self.arrays={}

    def _member(self,name):
        if not isinstance(name,str) or not re.fullmatch(r'(?:cache\.meta\.json|index\.jsonl|(?:Y|P|V|time_s|centroid|velocity_available)_[0-9]{5}\.npy)',name):
            raise ValueError('Cache artifact filename must be canonical and local')
        path=(self.root/name).resolve()
        if path.parent!=self.root:
            raise ValueError('Cache artifact path escapes cache directory')
        return path

    def __len__(self):return len(self.index)

    def __getitem__(self,index):
        if isinstance(index,str):index=self.by_uid[index]
        row=self.index[index];shard=row['shard']
        if shard not in self.arrays:
            # Keep at most one shard resident as mappings, release old views.
            for old in self.arrays.values():
                for array in old.values():array._mmap.close()
            self.arrays={shard:{name:np.load(self._member(f'{name}_{shard:05d}.npy'),mmap_mode='r',allow_pickle=False) for name in ARRAYS}}
            size=min(1024,self.meta['count']-shard*1024)
            if any(array.shape!=(size,*ARRAYS[name][0]) or array.dtype!=ARRAYS[name][1] for name,array in self.arrays[shard].items()):
                raise ValueError('Cache shard shape/dtype mismatch')
        return {name:torch.from_numpy(np.array(a[row['row']],copy=True)) for name,a in self.arrays[shard].items()}|{'uid':row['window_uid']}

def fit_descriptor_normalizer(cache,condition,output,train_manifest_hash):
    if cache.meta['split']!='train': raise ValueError('Descriptor fit requires train cache')
    normalizer=DescriptorNormalizer(condition,cache.meta['encoder_pose_hash'],train_manifest_hash)
    def batches():
        for batch in DataLoader(cache,batch_size=1,shuffle=False,num_workers=0):
            yield 'train',build_descriptors(batch['P'],batch['V'],batch['velocity_available'],condition)
    normalizer.fit(batches());artifact=normalizer.artifact()
    artifact['source_cache_identity']=cache.meta['identity'];artifact['fit_split']='train'
    artifact['artifact_generation_id']=cache.meta['source'].get('artifact_generation_id')
    write_json(output,artifact)
    return artifact

def load_descriptor_normalizer(path,cache=None):
    artifact=read_json(path)
    if artifact.get('fit_split')!='train':raise ValueError('Normalizer must be fit on train')
    if cache and artifact['source_checkpoint_hash']!=cache.meta['encoder_pose_hash']:raise ValueError('Normalizer/cache physical checkpoint mismatch')
    if cache:
        if cache.meta['source'].get('artifact_generation_id') is not None and artifact.get('artifact_generation_id')!=cache.meta['source']['artifact_generation_id']:
            raise ValueError('Descriptor normalizer artifact generation mismatch')
        if artifact['train_manifest_hash']!=cache.meta['source'].get('windows_hash'):
            raise ValueError('Normalizer/cache source window manifest mismatch')
        if cache.meta['split']=='train' and artifact['source_cache_identity']!=cache.meta['identity']:
            raise ValueError('Normalizer/train cache identity mismatch')
    normalizer=DescriptorNormalizer(artifact['condition'],artifact['source_checkpoint_hash'],artifact['train_manifest_hash'])
    state={name:torch.as_tensor(artifact[name],dtype=value.dtype) for name,value in normalizer.state_dict().items()}
    normalizer.load_state_dict(state,strict=True)
    return normalizer
