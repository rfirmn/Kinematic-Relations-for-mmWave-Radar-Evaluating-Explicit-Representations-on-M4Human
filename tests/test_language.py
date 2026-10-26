"""Language software fixtures. Native pretrained verification is a separate report."""
import hashlib

import pytest
import torch
from torch import nn

from relational64.language import (FrozenQwenBridge, MARKER, RadarSplicer, SplicedExample,
    answer_loss_from_logits, canonical_answer, collate_spliced)
from relational64.models import SensorProjector, parameter_hash


class CharacterTokenizer:
    """Deliberately simple test vocabulary; not a substitute for Qwen's tokenizer."""
    chat_template = "fixture template <|im_end|>"
    eos_token = "<|im_end|>"
    all_special_ids = [1]

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            if text.startswith(self.eos_token):
                ids.append(1)
                text = text[len(self.eos_token):]
            else:
                ids.append(ord(text[0]) + 2)
                text = text[1:]
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.eos_token if item == 1 else chr(item - 2) for item in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False and add_generation_prompt is True
        return "".join("<" + message["role"] + ">" + message["content"] for message in messages) + "<assistant>"


def splicer_fixture():
    tokenizer = CharacterTokenizer()
    embedding = nn.Embedding(256, 32).requires_grad_(False)
    splicer = RadarSplicer(tokenizer, embedding,
                          expected_template_hash=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest())
    sensor = torch.randn(64, 32, requires_grad=True)
    time = torch.arange(64, dtype=torch.float64) / 12
    return tokenizer, splicer, sensor, time


def tiny_qwen():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                     max_position_embeddings=512, attention_dropout=0,
                                     bos_token_id=0, eos_token_id=1)
    torch.manual_seed(9)
    return transformers.Qwen2ForCausalLM(config).eval()


def short_example(uid="example", length=12, target_count=3, train=True):
    embeddings = torch.randn(length, 32, requires_grad=True)
    labels = torch.full((length,), -100, dtype=torch.long)
    if train:
        labels[-target_count:] = torch.arange(3, 3 + target_count)
    return SplicedExample(embeddings, labels, (2, 3), length - target_count if train else length, uid)


def test_splice_exact_sensor_positions_prefix_labels_and_gradients():
    tokenizer, splicer, sensor, time = splicer_fixture()
    generation = splicer.build(uid="w", task="arms", time_s=time, sensor_embeddings=sensor)
    training = splicer.build(uid="w", task="arms", time_s=time, sensor_embeddings=sensor,
                             answer=canonical_answer("arms", "tie"))
    assert len(training.sensor_positions) == len(set(training.sensor_positions)) == 64
    assert training.sensor_positions == generation.sensor_positions
    assert training.prefix_length == generation.prefix_length
    torch.testing.assert_close(training.inputs_embeds[:training.prefix_length], generation.inputs_embeds, atol=0, rtol=0)
    assert (training.labels[:training.prefix_length] == -100).all()
    assert (training.labels[list(training.sensor_positions)] == -100).all()
    assert training.labels[-1] == 1 and (training.labels == 1).sum() == 1
    for frame, position in enumerate(training.sensor_positions):
        torch.testing.assert_close(training.inputs_embeds[position], sensor[frame])
    training.inputs_embeds.sum().backward()
    assert sensor.grad is not None and sensor.grad.norm() > 0
    assert splicer.embedding.weight.grad is None
    batch = collate_spliced([training, generation])
    assert (batch.labels[1] == -100).all()
    assert (batch.labels[~batch.attention_mask] == -100).all()
    assert (batch.position_ids[~batch.attention_mask] == 0).all()
    text_only = splicer.build(uid="t", task="arms", time_s=None, sensor_embeddings=None, text_only=True)
    assert not text_only.sensor_positions


def test_splice_marker_template_and_limits_reject_without_truncating():
    _, splicer, sensor, time = splicer_fixture()
    for text in ("no marker", MARKER + MARKER):
        with pytest.raises(ValueError, match="exactly one"):
            splicer.splice_rendered(text, uid="bad", time_s=time, sensor_embeddings=sensor)
    with pytest.raises(ValueError, match="bad.*sequence length"):
        splicer.splice_rendered("z" * 4000 + MARKER, uid="bad", time_s=time, sensor_embeddings=sensor)
    with pytest.raises(ValueError, match="hash mismatch"):
        RadarSplicer(CharacterTokenizer(), nn.Embedding(256, 32), expected_template_hash="wrong")
    with pytest.raises(ValueError, match="label"):
        canonical_answer("arms", "speeding_up")


def test_selected_ce_and_gradient_equal_full_logits_with_per_example_mean():
    bridge = FrozenQwenBridge(tiny_qwen(), expected_hidden_size=32)
    batch = collate_spliced([short_example("a", 13, 2), short_example("b", 15, 5)])
    full = bridge.full_answer_loss(batch)
    full_grad = torch.autograd.grad(full, batch.inputs_embeds, retain_graph=True)[0]
    selected = bridge.selected_answer_loss(batch)
    selected_grad = torch.autograd.grad(selected, batch.inputs_embeds)[0]
    torch.testing.assert_close(selected, full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(selected_grad, full_grad, atol=1e-5, rtol=1e-5)
    logits = torch.randn(2, 15, 128)
    labels = batch.labels
    expected = []
    for index in range(2):
        target_positions = torch.where(labels[index] != -100)[0]
        expected.append(nn.functional.cross_entropy(logits[index, target_positions - 1], labels[index, target_positions]))
    torch.testing.assert_close(answer_loss_from_logits(logits, labels), torch.stack(expected).mean())


def test_frozen_qwen_keeps_input_gradients_and_only_projector_updates():
    bridge = FrozenQwenBridge(tiny_qwen(), expected_hidden_size=32).train()
    assert not bridge.causal_lm.training
    projector = SensorProjector("M", language_dim=32)
    before_qwen, before_projector = parameter_hash(bridge), parameter_hash(projector)
    sensor = projector(torch.randn(1, 64, 256)).sensor_embeddings[0]
    example = SplicedExample(sensor, torch.tensor([-100] * 60 + [4, 5, 6, 1]), tuple(range(64)), 60, "tiny")
    loss = bridge.selected_answer_loss(collate_spliced([example]))
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
    loss.backward()
    assert all(parameter.grad is None for parameter in bridge.parameters())
    assert projector.language_mapping[0].weight.grad.norm() > 0
    optimizer.step()
    assert parameter_hash(bridge) == before_qwen
    assert parameter_hash(projector) != before_projector


def test_actual_qwen_architecture_embeddings_and_cached_eight_tokens_equal():
    bridge = FrozenQwenBridge(tiny_qwen(), expected_hidden_size=32)
    assert bridge.embedding_equivalence_error(torch.tensor([[2, 3, 4, 5, 6]])) < 1e-6
    example = short_example(length=9, train=False)
    # A sentinel outside vocabulary ensures eight steps even if token1 is emitted.
    cached = bridge.greedy_generate(example, CharacterTokenizer(), end_token_id=999, use_cache=True, max_new_tokens=8)
    uncached = bridge.greedy_generate(example, CharacterTokenizer(), end_token_id=999, use_cache=False, max_new_tokens=8)
    assert cached.token_ids == uncached.token_ids and len(cached.token_ids) == 8
    assert cached.truncated and uncached.truncated
    invalid = short_example(train=True)
    with pytest.raises(ValueError, match="teacher-forced"):
        bridge.greedy_generate(invalid, CharacterTokenizer(), end_token_id=1)


def test_eval_checkpointing_preserves_gradient_and_target_changes_it():
    bridge = FrozenQwenBridge(tiny_qwen(), expected_hidden_size=32)
    batch = collate_spliced([short_example("checkpoint", 13, 3)])
    ordinary = bridge.selected_answer_loss(batch)
    ordinary_grad = torch.autograd.grad(ordinary, batch.inputs_embeds, retain_graph=True)[0]
    bridge.enable_input_checkpointing()
    checkpointed = bridge.selected_answer_loss(batch)
    checkpointed_grad = torch.autograd.grad(checkpointed, batch.inputs_embeds, retain_graph=True)[0]
    torch.testing.assert_close(checkpointed, ordinary, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(checkpointed_grad, ordinary_grad, atol=1e-5, rtol=1e-5)
    batch.labels[batch.labels != -100] += 1
    changed = bridge.selected_answer_loss(batch)
    changed_grad = torch.autograd.grad(changed, batch.inputs_embeds)[0]
    assert not torch.allclose(ordinary_grad, changed_grad)
    assert not bridge.causal_lm.training
