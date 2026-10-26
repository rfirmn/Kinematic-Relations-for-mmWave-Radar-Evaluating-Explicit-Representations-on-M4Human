"""CPU-only integration fixtures for frozen-cache alignment and locked loading."""
from collections import Counter
import hashlib
import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from relational64 import alignment
from relational64.config import identity, validate_language_lock
from relational64.language import FrozenQwenBridge, SplicedExample, answer_loss_from_logits, collate_spliced
from relational64.models import DescriptorNormalizer, SensorProjector, build_descriptors, parameter_hash


def records(counts=(4000, 1000, 1000)):
    result = []
    for task, count in zip(("root", "arms", "legs"), counts):
        for index in range(count):
            result.append({"qa_uid": f"{task}-{index}", "part": task, "split": "train",
                           "window_uid": f"window-{index}", "reference_status": "unknown" if index % 3 == 0 else "defined"})
    return result


def cache_item():
    torch.manual_seed(4)
    return {"Y": torch.randn(64, 256), "P": torch.randn(64, 22, 3), "V": torch.randn(64, 22, 3),
            "time_s": torch.arange(64, dtype=torch.float64) / 12,
            "velocity_available": torch.arange(64) >= 6, "uid": "window"}


def tiny_qwen():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        attention_dropout=0, bos_token_id=0, eos_token_id=1)
    return transformers.Qwen2ForCausalLM(config).eval()


def valid_lock():
    files = {"model.safetensors": "e" * 64}
    return {"model_commit": "a" * 40, "tokenizer_commit": "b" * 40,
            "chat_template_hash": hashlib.sha256(b"fixture <|im_end|>").hexdigest(),
            "safetensors_manifest_hash": identity(files), "dependency_lock": "c" * 64,
            "environment_report": "d" * 64, "precision": "bfloat16",
            "head_mode": "canonical_fp32", "snapshot_file_hashes": files}


def test_weighted_sampler_same_seed_stream_and_task_probabilities():
    registry = records()
    stream = list(alignment.qa_draws(registry, 42, 0))
    assert len(stream) == len(registry)
    ids = [(record["qa_uid"], record["part"]) for record in stream]
    for _condition in ("M", "S", "F", "E", "R"):
        assert [(r["qa_uid"], r["part"]) for r in alignment.qa_draws(registry, 42, 0)] == ids
    assert [r["qa_uid"] for r in alignment.qa_draws(registry, 42, 1)] != [r["qa_uid"] for r in stream]
    assert [r["qa_uid"] for r in alignment.qa_draws(registry, 43, 0)] != [r["qa_uid"] for r in stream]
    counts = Counter(r["part"] for r in stream)
    for task, probability in (("root", .5), ("arms", .25), ("legs", .25)):
        assert abs(counts[task] / len(stream) - probability) < .025
    assert any(r["reference_status"] == "unknown" for r in stream)
    assert len({r["qa_uid"] for r in stream}) < len(stream)  # replacement is intentional


def test_sampler_rejects_missing_task_instead_of_renormalizing():
    with pytest.raises(ValueError, match="three QA tasks"):
        list(alignment.qa_draws(records((3, 3, 0)), 42, 0))


@pytest.mark.parametrize("condition", list("MSFER"))
def test_cached_window_projector_matches_direct_path_and_is_deterministic(condition):
    item = cache_item()
    original = {name: value.clone() for name, value in item.items() if isinstance(value, torch.Tensor)}
    projector = SensorProjector(condition, language_dim=32)
    normalizer = None
    descriptor = None
    if condition != "M":
        descriptor = build_descriptors(item["P"][None], item["V"][None], item["velocity_available"][None], condition)
        normalizer = DescriptorNormalizer(condition, "physical_checkpoint", "train_manifest").fit([("train", descriptor)])
        descriptor = normalizer(descriptor)
    first = alignment.project_window(projector, item, normalizer, "cpu")
    second = alignment.project_window(projector, item, normalizer, "cpu")
    direct = projector(item["Y"][None], descriptor)
    assert first.sensor_embeddings.shape == (1, 64, 32)
    torch.testing.assert_close(first.sensor_embeddings, second.sensor_embeddings, atol=0, rtol=0)
    torch.testing.assert_close(first.sensor_embeddings, direct.sensor_embeddings, atol=0, rtol=0)
    for name, before in original.items():
        assert torch.equal(item[name], before)


def test_M_cached_window_path_does_not_open_pose_or_velocity():
    item = {"Y": cache_item()["Y"]}
    output = alignment.project_window(SensorProjector("M", language_dim=32), item, None, "cpu")
    assert output.sensor_embeddings.shape == (1, 64, 32)


def test_cached_alignment_gradient_changes_projector_and_keeps_qwen_frozen():
    item = cache_item()
    projector = SensorProjector("R", language_dim=32)
    descriptor = build_descriptors(item["P"][None], item["V"][None], item["velocity_available"][None], "R")
    normalizer = DescriptorNormalizer("R", "physical", "train").fit([("train", descriptor)])
    bridge = FrozenQwenBridge(tiny_qwen(), expected_hidden_size=32)
    frozen_hash, projector_hash = parameter_hash(bridge), parameter_hash(projector)
    embeddings = alignment.project_window(projector, item, normalizer, "cpu").sensor_embeddings[0]
    example = SplicedExample(embeddings, torch.tensor([-100] * 60 + [3, 4, 5, 1]), tuple(range(64)), 60, "qa")
    loss = bridge.selected_answer_loss(collate_spliced([example]))
    loss.backward()
    assert all(parameter.grad is None for parameter in bridge.parameters())
    assert projector.node_mlp[0].weight.grad.norm() > 0
    assert projector.edge_mlp[0].weight.grad.norm() > 0
    optimizer = torch.optim.AdamW(projector.parameters(), lr=3e-4)
    optimizer.step()
    assert parameter_hash(bridge) == frozen_hash
    assert parameter_hash(projector) != projector_hash
    assert all(not value.requires_grad for value in item.values() if isinstance(value, torch.Tensor))


def test_effective_language_batch_is_mean_of_example_means():
    torch.manual_seed(8)
    logits = torch.randn(2, 9, 64, requires_grad=True)
    labels = torch.full((2, 9), -100)
    labels[0, -2:] = torch.tensor([2, 1])
    labels[1, -5:] = torch.tensor([3, 4, 5, 6, 1])
    combined = answer_loss_from_logits(logits, labels)
    accumulated = sum(answer_loss_from_logits(logits[i:i + 1], labels[i:i + 1]) / 2 for i in range(2))
    combined_gradient = torch.autograd.grad(combined, logits, retain_graph=True)[0]
    accumulated_gradient = torch.autograd.grad(accumulated, logits)[0]
    torch.testing.assert_close(combined, accumulated)
    torch.testing.assert_close(combined_gradient, accumulated_gradient)
    targets = labels[:, 1:]
    all_token_mean = nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), targets, ignore_index=-100)
    assert not torch.allclose(combined, all_token_mean)


@pytest.mark.parametrize("key,value", [("head_mode", None), ("head_mode", "unverified"),
                                       ("precision", None), ("precision", "float32")])
def test_language_lock_rejects_missing_or_unverified_compute_mode(key, value):
    lock = valid_lock()
    if value is None:
        lock.pop(key)
    else:
        lock[key] = value
    with pytest.raises(ValueError):
        validate_language_lock(lock)


def test_language_load_passes_pinned_head_mode_and_precision(monkeypatch, tmp_path):
    transformers = pytest.importorskip("transformers")
    lock = valid_lock()
    received = {}
    embedding = nn.Embedding(8, 32)

    class Tokenizer:
        chat_template = "fixture <|im_end|>"
        eos_token = "<|im_end|>"
        all_special_ids = [1]
        def encode(self, _text, add_special_tokens=False):
            return [1]

    fake_bridge = SimpleNamespace(causal_lm=SimpleNamespace(get_input_embeddings=lambda: embedding))
    fake_bridge.to = lambda _device: fake_bridge
    def load(_path, **kwargs):
        received.update(kwargs)
        return fake_bridge
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *args, **kwargs: Tokenizer()))
    monkeypatch.setattr(FrozenQwenBridge, "from_local_snapshot", staticmethod(load))
    monkeypatch.setattr(alignment, "file_hash", lambda _path: "e" * 64)
    config = {"runtime": {"model_path": str(tmp_path), "model_commit": lock["model_commit"],
                           "tokenizer_commit": lock["tokenizer_commit"], "device": "cpu"}}
    alignment.load_language(config, lock)
    assert received["dtype"] == torch.bfloat16
    assert received["head_mode"] == "canonical_fp32"


def test_required_profile_defaults_match_protocol():
    params = inspect.signature(alignment.profile_alignment).parameters
    assert params["warmup"].default == 10
    assert params["steps"].default == 30
    assert params["accumulation"].default == 16
    assert params["generations"].default == 30


def test_QA_registry_keeps_unknown_excludes_undefined_and_rejects_duplicate_uid(tmp_path):
    registry = [{"qa_uid": str(index), "split": "train", "reference_status": status}
                for index, status in enumerate(("defined", "unknown", "undefined"))]
    path = tmp_path / "qa.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in registry), encoding="utf-8")
    assert [record["reference_status"] for record in alignment.read_qa(path)] == ["defined", "unknown"]
    assert len(alignment.read_qa(path, eligible=False)) == 3
    registry.append(dict(registry[0]))
    path.write_text("".join(json.dumps(record) + "\n" for record in registry), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate QA"):
        alignment.read_qa(path)


def test_generation_receives_sensor_only_and_preserves_runtime_failures(tmp_path):
    seen = []
    class Splicer:
        end_token_id = 1
        def build(self, **kwargs):
            seen.append(kwargs)
            if kwargs["uid"] == "broken":
                raise ValueError("fixture failed")
            return SimpleNamespace(prefix_length=12, sensor_positions=tuple(range(64)))
    class Bridge:
        def greedy_generate(self, *_args, **_kwargs):
            return SimpleNamespace(raw_text='{"answer":"tie"}', terminated=True, truncated=False, token_ids=[3, 1])
    registry = [{"qa_uid": uid, "window_uid": uid, "part": "arms", "prompt_hash": "prompt",
                 "answer": "faster_left", "reference_status": "unknown", "support_count": 44}
                for uid in ("okay", "broken")]
    cache = {uid: cache_item() for uid in ("okay", "broken")}
    output = alignment.generate_panel(registry, cache, SensorProjector("M", language_dim=32), None,
                 Bridge(), Splicer(), None, device="cpu", output=tmp_path / "predictions.jsonl")
    rows = {row["window_uid"]: row for row in output}
    assert rows["okay"]["predicted_label"] == "tie" and rows["okay"]["parse_status"] == "valid"
    assert rows["broken"]["parse_status"] == "runtime_error"
    assert rows["broken"]["generated_tokens"] == 0
    assert len((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert all(set(kwargs) == {"uid", "task", "time_s", "sensor_embeddings"} for kwargs in seen)
