"""Synthetic cache/checkpoint/target pipeline contracts; no dataset evidence."""
import json
from dataclasses import asdict
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from relational64.cache import extract_cache, PhysicalCache
from relational64.checkpoint import save_checkpoint, load_checkpoint
from relational64.config import resolve, require_gate, file_hash, write_json, validate_language_lock, identity
from relational64.physical import effective_denominators, physical_loss, window_operators
from relational64.pipeline_targets import build_qa_registry
from relational64.qa import LabelPolicy


class SensorFixture(torch.utils.data.Dataset):
    training = False
    def __init__(self, count=2):
        self.count = count
    def __len__(self):
        return self.count
    @property
    def targets(self):
        raise AssertionError("target access forbidden during cache extraction")
    def __getitem__(self, index):
        cloud = torch.full((64, 8, 4), .1+index*.001)
        return {"X": cloud, "point_mask": torch.ones(64,8,dtype=torch.bool),
                "time_s": torch.arange(64,dtype=torch.float64)/12,
                "centroid": torch.tensor([0.,1.,0.]), "uid": f"synthetic-{index:04d}"}


class SmallSensorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))
        self.dropout = torch.nn.Dropout(.9)
    def forward(self, points, mask, time_s):
        values = self.dropout(points.mean(-2))*self.scale
        return SimpleNamespace(features=values.mean(-1,keepdim=True).expand(-1,-1,256),
                               positions=values[...,:3,None].transpose(-2,-1).expand(-1,-1,22,-1))


def test_cache_is_deterministic_eval_and_sensor_only(tmp_path):
    model = SmallSensorModel().train()
    data = SensorFixture()
    source = {"sensor_hash":"a"*64,"checkpoint":"b"*64}
    first = extract_cache(model,data,tmp_path/"first",device="cpu",provenance=source,split="train")
    second = extract_cache(model,data,tmp_path/"second",device="cpu",provenance=source,split="train")
    assert first["files"] == second["files"]
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    cache = PhysicalCache(tmp_path/"first",expected_provenance=source)
    assert len(cache)==2
    assert cache[0]["Y"].shape == (64,256)
    assert cache["synthetic-0000"]["P"].shape == (64,22,3)
    assert not cache[0]["velocity_available"][:6].any()
    with pytest.raises(ValueError,match="provenance"):
        PhysicalCache(tmp_path/"first",expected_provenance={"sensor_hash":"changed"})
    cache.arrays.clear()


def test_cache_tamper_is_rejected(tmp_path):
    extract_cache(SmallSensorModel(),SensorFixture(),tmp_path/"cache",device="cpu",provenance={},split="val")
    path=tmp_path/"cache"/"Y_00000.npy"
    with path.open("r+b") as handle:
        handle.seek(-4,2)
        handle.write(b"abcd")
    with pytest.raises(ValueError,match="hash mismatch"):
        PhysicalCache(tmp_path/"cache")


def test_cache_shards_1024_windows_and_closes_before_publish(tmp_path,monkeypatch):
    # Exercise actual fixed array dimensions at the shard boundary. Replace
    # only repeated SVD operator construction, whose math is tested separately.
    import relational64.cache as module
    times=torch.arange(64,dtype=torch.float64)[None]/12
    weights,available=window_operators(times)
    monkeypatch.setattr(module,"window_operators",lambda t:(weights.to(t.device),available.to(t.device)))
    root=tmp_path/"shards"
    meta=extract_cache(SmallSensorModel(),SensorFixture(1025),root,device="cpu",provenance={},split="train")
    assert meta["count"]==1025 and meta["max_windows_per_shard"]==1024
    cache=PhysicalCache(root)
    assert np.load(root/"Y_00000.npy",mmap_mode="r").shape==(1024,64,256)
    assert np.load(root/"Y_00001.npy",mmap_mode="r").shape==(1,64,256)
    assert cache[1023]["uid"]=="synthetic-1023"
    assert cache[1024]["uid"]=="synthetic-1024"
    cache.arrays.clear()


def test_cache_rejects_stochastic_training_sampling(tmp_path):
    data=SensorFixture()
    data.training=True
    with pytest.raises(ValueError,match="deterministic|sampling|training"):
        extract_cache(SmallSensorModel(),data,tmp_path/"cache",device="cpu",provenance={},split="train")


def test_effective_denominators_ignore_microbatch_partition_and_missing_gt():
    torch.manual_seed(7)
    target=torch.randn(3,64,22,3)
    mask=torch.ones(3,64,22,dtype=torch.bool)
    mask[0,:,1:]=False
    mask[1,6:15,7]=False
    mask[2,::2]=False
    target[~mask]=float("nan")
    time_s=torch.arange(64,dtype=torch.float64)[None].expand(3,-1)/12
    predicted=torch.randn_like(target,requires_grad=True)
    w,a=window_operators(time_s)
    full=physical_loss(predicted,target,mask,w,a)
    batches=[{"positions":target[i:i+1],"target_mask":mask[i:i+1],"time_s":time_s[i:i+1]} for i in range(3)]
    den=effective_denominators(batches,"cpu")
    for key in den:
        torch.testing.assert_close(den[key],full.denominator[key])
    full.total().backward()
    grad=predicted.grad.clone(); predicted.grad=None
    for i in range(3):
        physical_loss(predicted[i:i+1],target[i:i+1],mask[i:i+1],w[i:i+1],a[i:i+1]).total(den).backward()
    torch.testing.assert_close(predicted.grad,grad,rtol=1e-5,atol=1e-5)
    assert torch.isfinite(predicted.grad).all()


def test_resume_restores_rng_optimizer_scheduler_and_sampler(tmp_path):
    model=torch.nn.Linear(3,3);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    model(torch.ones(1,3)).sum().backward();optimizer.step();scheduler.step()
    random.seed(17);np.random.seed(17);torch.manual_seed(17)
    generator=torch.Generator().manual_seed(9)
    path=tmp_path/"last.pt"
    save_checkpoint(path,model,optimizer,provenance={"seed":17,"split":"split_hash","effective_batch":16},epoch=2,step=9,scheduler=scheduler,best=.2,patience=3,sampler_state=generator.get_state())
    expected=(random.random(),np.random.random(),torch.rand(4))
    model(torch.ones(1,3)).sum().backward();optimizer.step();scheduler.step()
    loaded=load_checkpoint(path,model,provenance={"seed":17,"split":"split_hash","effective_batch":16},optimizer=optimizer,scheduler=scheduler)
    actual=(random.random(),np.random.random(),torch.rand(4))
    assert actual[:2]==expected[:2]
    torch.testing.assert_close(actual[2],expected[2],rtol=0,atol=0)
    assert scheduler.last_epoch==1
    assert optimizer.param_groups[0]["lr"]==pytest.approx(.009)
    assert loaded["epoch"]==2 and loaded["step"]==9 and loaded["patience"]==3
    assert torch.equal(loaded["sampler"],generator.get_state())
    assert not path.with_suffix(".pt.tmp").exists()
    with pytest.raises(ValueError,match="provenance"):
        load_checkpoint(path,model,provenance={"seed":18,"split":"split_hash","effective_batch":16})


def test_null_runtime_is_preparation_only_and_gates_fail_closed(tmp_path):
    config=resolve()
    assert config["runtime"]["dataset_root"] is None
    gate_path=tmp_path/"gates.json"
    write_json(gate_path,{"gates":{"G0":{"passed":False}}})
    with pytest.raises(ValueError,match="G0"):
        require_gate(gate_path,"G0")
    artifact=tmp_path/"artifact.txt";artifact.write_text("original")
    # A mocked authority payload tests schema/hash enforcement only. It is not
    # an acceptance report and must never be used to launch a scientific run.
    write_json(gate_path,{"gates":{"G0":{"passed":True,"actual_M4Human":True,"test_ids":["mock_authority"],"evidence":"mock_authority_payload", "artifacts":{str(artifact):file_hash(artifact)}}}})
    assert require_gate(gate_path,"G0")["passed"]
    artifact.write_text("changed")
    with pytest.raises(ValueError,match="hash mismatch"):
        require_gate(gate_path,"G0")
    with pytest.raises(ValueError,match="model_commit"):
        validate_language_lock({})


class TargetFixture:
    def __init__(self):self.calls=[]
    def read(self,key):
        self.calls.append(key)
        if key.startswith("test-"):raise AssertionError("test target read before lock")
        p=np.zeros((22,3),np.float32)
        mask=np.ones(22,bool)
        if int(key.rsplit("-",1)[1]) < 20:mask[18]=False
        return p,mask


def qa_fixture(tmp_path):
    path=tmp_path/"windows.jsonl"
    rows=[{"window_uid":split,"split":split,"subject_id":str(index),"recording_id":f"r{index}","time_s":(np.arange(64)/12).tolist(),"source_keys":[f"{split}-{i}" for i in range(64)]} for index,split in enumerate(("train","val","test"))]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf8")
    provenance={"G0_passed":True,"evidence":"synthetic_software_fixture",**{key:"a"*64 for key in ("inventory_hash","coordinate_audit_hash","time_audit_hash","split_hash","target_hash","target_provenance_hash")}}
    return path,provenance


def test_build_qa_registry_uses_canonical_reader_and_preserves_undefined(tmp_path):
    windows,provenance=qa_fixture(tmp_path);reader=TargetFixture()
    output=tmp_path/"qa.jsonl"
    meta=build_qa_registry(windows,reader,output,provenance=provenance)
    records=[json.loads(line) for line in output.read_text().splitlines()]
    assert len(records)==6 and len(reader.calls)==128
    assert meta["window_count"]==2 and meta["complete"]
    assert meta["evidence"]=="synthetic_software_fixture" and not meta["scientific_G1_accepted"]
    assert all(r["window_manifest_hash"]==file_hash(windows) for r in records)
    assert sum(r["reference_status"]=="undefined" for r in records)==2
    assert all(r["answer"] is None for r in records if r["reference_status"]=="undefined")
    assert not any(key.startswith("test-") for key in reader.calls)
    diagnostic=json.loads(output.with_suffix(".label_audit.json").read_text())
    assert diagnostic["status"]=="diagnostic_not_locked"
    assert not diagnostic["actual_m4human_used"]


def test_build_qa_registry_g0_and_test_lock_are_required(tmp_path):
    windows,provenance=qa_fixture(tmp_path);reader=TargetFixture()
    provenance["G0_passed"]=False
    with pytest.raises(ValueError,match="G0"):
        build_qa_registry(windows,reader,tmp_path/"blocked.jsonl",provenance=provenance)
    provenance["G0_passed"]=True
    with pytest.raises(ValueError,match="test lock"):
        build_qa_registry(windows,reader,tmp_path/"blocked.jsonl",provenance=provenance,splits=("test",))
    assert reader.calls==[]


def test_qa_registry_rejects_duplicate_windows_and_timestamp_gaps(tmp_path):
    windows,provenance=qa_fixture(tmp_path)
    first=json.loads(windows.read_text().splitlines()[0])
    first["time_s"][32:]=[v+.05 for v in first["time_s"][32:]]
    windows.write_text(json.dumps(first)+"\n",encoding="utf8")
    reader=TargetFixture()
    with pytest.raises(ValueError,match="timestamp"):
        build_qa_registry(windows,reader,tmp_path/"gap.jsonl",provenance=provenance)
    assert reader.calls==[]
    first["time_s"]=(np.arange(64)/12).tolist()
    windows.write_text((json.dumps(first)+"\n")*2,encoding="utf8")
    with pytest.raises(ValueError,match="duplicate"):
        build_qa_registry(windows,reader,tmp_path/"duplicate.jsonl",provenance=provenance)


def test_qa_registry_locked_test_keeps_label_diagnostics_disabled(tmp_path):
    windows,provenance=qa_fixture(tmp_path)
    lock=tmp_path/"test.lock.json"
    recipe=tmp_path/"target_recipe.json";write_json(recipe,{"evidence":"synthetic_software_fixture","coordinate":"fixture"})
    inventory=tmp_path/"inventory.json";write_json(inventory,{"evidence":"synthetic_software_fixture"})
    split=tmp_path/"split.json";write_json(split,{"train":["0"],"val":["1"],"test":["2"]})
    policy=tmp_path/"policy.json";write_json(policy,{"policy":asdict(LabelPolicy())})
    plan={"source_manifest":str(windows),"target_provenance":str(recipe),"source_inventory":str(inventory),"split_lock":str(split),"label_policy":str(policy)}
    payload={"protocol_id":"m4human_relational64","status":"locked","test_evaluation_authorized":True,"plan":plan,"artifacts":{str(Path(value).resolve()):file_hash(value) for value in plan.values()}}
    payload["identity"]=identity(payload)
    write_json(lock,payload)
    provenance.update(target_provenance_hash=file_hash(recipe),inventory_hash=file_hash(inventory),split_hash=file_hash(split))
    wrong={**provenance,"target_provenance_hash":"b"*64}
    with pytest.raises(ValueError,match="recipe/provenance hash mismatch"):
        build_qa_registry(windows,TargetFixture(),tmp_path/"test_qa.jsonl",provenance=wrong,splits=("test",),test_lock_path=lock)
    reader=SimpleNamespace(read=lambda key:(np.zeros((22,3)),np.ones(22,bool)))
    output=tmp_path/"test_qa.jsonl"
    meta=build_qa_registry(windows,reader,output,provenance=provenance,splits=("test",),test_lock_path=lock)
    assert meta["qa_count"]==3
    assert meta["test_lock_hash"]==file_hash(lock)
    diagnostic=json.loads(output.with_suffix(".label_audit.json").read_text())
    assert diagnostic["status"]=="test_only_no_label_tuning"
    # A fresh export identity does not change the immutable target recipe.
    provenance["target_hash"]="c"*64
    second=build_qa_registry(windows,reader,tmp_path/"test_qa_new_arrays.jsonl",provenance=provenance,splits=("test",),test_lock_path=lock)
    assert second["provenance"]["target_hash"]=="c"*64
    assert second["provenance"]["target_provenance_hash"]==file_hash(recipe)
    with pytest.raises(ValueError,match="label policy hash mismatch"):
        build_qa_registry(windows,reader,tmp_path/"wrong_policy.jsonl",provenance=provenance,splits=("test",),test_lock_path=lock,policy=LabelPolicy(limb_delta_mps=.1))
    reader.metadata={"target_provenance_hash":"d"*64,"inventory_hash":provenance["inventory_hash"]}
    with pytest.raises(ValueError,match="target-reader target_provenance_hash mismatch"):
        build_qa_registry(windows,reader,tmp_path/"wrong_export_recipe.jsonl",provenance=provenance,splits=("test",),test_lock_path=lock)
    del reader.metadata
    recipe.write_text('{"changed":true}',encoding="utf8")
    with pytest.raises(ValueError,match="artifact mismatch"):
        build_qa_registry(windows,reader,tmp_path/"changed_recipe.jsonl",provenance=provenance,splits=("test",),test_lock_path=lock)
