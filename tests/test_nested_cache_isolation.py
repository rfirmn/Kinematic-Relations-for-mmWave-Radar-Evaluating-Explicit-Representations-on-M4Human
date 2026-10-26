"""New synthetic cache files only; no old cache or source dataset is opened."""
from pathlib import Path
import numpy as np
import pytest

import relational64.cache as cache_module
import relational64.config as config
from relational64.cache import ARRAYS, PhysicalCache
from relational64.config import file_hash, identity, write_json

GENERATION="fresh_20261026_01"


@pytest.fixture
def cache_fixture(tmp_path,monkeypatch):
    repository=tmp_path/"workspace";root=repository/"active"/"cache";root.mkdir(parents=True)
    monkeypatch.setattr(config,"ROOT",repository)
    for name,(shape,dtype) in ARRAYS.items():
        np.save(root/f"{name}_00000.npy",np.zeros((1,*shape),dtype=dtype))
    (root/"index.jsonl").write_text('{"window_uid":"synthetic-1","split":"train","shard":0,"row":0}\n',encoding="utf8")
    meta={"count":1,"split":"train","source":{"artifact_generation_id":GENERATION,"evidence":"synthetic_software_fixture"},
          "complete":True,"module_eval":True,"max_windows_per_shard":1024,
          "files":{path.name:file_hash(path) for path in root.iterdir()}}
    return repository,root,meta


def publish(root,meta):
    meta={**meta,"identity":identity(meta)}
    write_json(root/"cache.meta.json",meta)


def test_current_cache_generation_and_canonical_members_load(cache_fixture):
    _,root,meta=cache_fixture;publish(root,meta)
    cache=PhysicalCache(root,expected_generation=GENERATION)
    assert cache[0]["Y"].shape==(64,256)
    assert cache[0]["uid"]=="synthetic-1"
    cache.arrays.clear()
    with pytest.raises(ValueError,match="generation mismatch"):
        PhysicalCache(root,expected_generation="different")


@pytest.mark.parametrize("name_kind",["parent","absolute","archive"])
def test_nested_cache_paths_rejected_before_any_hash(cache_fixture,monkeypatch,name_kind):
    repository,root,meta=cache_fixture
    name={"parent":"../never_open.npy","absolute":str(repository.parent/"external"/"never_open.npy"),"archive":str(repository/"archive"/"never_open.npy")}[name_kind]
    meta["files"][name]="a"*64;publish(root,meta)
    calls=[]
    def no_hash(path):
        calls.append(path)
        raise AssertionError("invalid nested path reached file hashing")
    monkeypatch.setattr(cache_module,"file_hash",no_hash)
    with pytest.raises(ValueError,match="canonical and local"):
        PhysicalCache(root,expected_generation=GENERATION)
    assert calls==[]


@pytest.mark.parametrize("location",["external","archive"])
def test_versioned_cache_root_rejected_before_metadata_open(cache_fixture,monkeypatch,location):
    repository,_,_=cache_fixture
    path=repository.parent/"never_open" if location=="external" else repository/"archive"/"never_open"
    monkeypatch.setattr(cache_module,"read_json",lambda _:pytest.fail("forbidden root metadata was opened"))
    with pytest.raises(ValueError,match="repository|Archived"):
        PhysicalCache(path,expected_generation=GENERATION)


def test_versioned_metadata_cannot_enable_unversioned_external_cache(cache_fixture,monkeypatch):
    repository,_,meta=cache_fixture
    # The explicit generic unversioned-fixture API reads metadata first. Once
    # it declares a real generation, the active path guard must run before
    # opening any listed nested files.
    external=repository.parent/"generic-fixture";external.mkdir()
    publish(external,meta)
    monkeypatch.setattr(cache_module,"file_hash",lambda _:pytest.fail("versioned external cache reached nested hashing"))
    with pytest.raises(ValueError,match="repository"):
        PhysicalCache(external)


def test_nested_cache_preflight_runs_even_when_checksum_verification_disabled(cache_fixture):
    _,root,meta=cache_fixture
    meta["files"]["../never_open.npy"]="a"*64;publish(root,meta)
    with pytest.raises(ValueError,match="canonical and local"):
        PhysicalCache(root,expected_generation=GENERATION,verify_files=False)
