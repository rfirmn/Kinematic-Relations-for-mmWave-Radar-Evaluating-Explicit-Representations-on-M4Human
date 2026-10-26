"""Frozen Qwen interface with explicit multimodal boundaries and causal loss.

No model weights or tokenizer are downloaded implicitly. Production loading
requires pinned commits, a verified local snapshot, and an accepted chat template.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MARKER = "[[RADAR_BLOCK]]"
SYSTEM_PROMPT = ('You answer kinematic questions using the supplied radar evidence. Return exactly one JSON object with the key "answer" and one allowed label. Do not add explanations or other keys. Use "unknown" when the evidence does not support a choice.')
USER_PREAMBLE = ("Radar observations contain 64 consecutive frames numbered 00 to 63. The first six frames provide derivative context. The question concerns frames 06 to 63. Left and right refer to the person's anatomical sides. Root means pelvis. Relative limb motion excludes pelvis translation.")
QUESTIONS = {
    "root": "Over frames 06 to 63, does the pelvis speed show an increasing trend, a decreasing trend, or no overall trend?",
    "arms": "Over frames 06 to 63, which arm has greater typical motion speed relative to the pelvis, using the elbows and wrists?",
    "legs": "Over frames 06 to 63, which leg has greater typical motion speed relative to the pelvis, using the knees and ankles?",
}
CHOICES = {"root": ("speeding_up", "slowing_down", "no_overall_trend", "unknown"),
           "arms": ("faster_left", "faster_right", "tie", "unknown"),
           "legs": ("faster_left", "faster_right", "tie", "unknown")}


def canonical_answer(task: str, label: str) -> str:
    if task not in CHOICES or label not in CHOICES[task]:
        raise ValueError("label is not allowed for this task")
    return json.dumps({"answer": label}, separators=(",", ":"))


def prompt_messages(task: str) -> list[dict[str, str]]:
    if task not in QUESTIONS:
        raise ValueError("unknown task")
    user = USER_PREAMBLE + "\n" + MARKER + "\n" + QUESTIONS[task] + "\nAllowed answers: " + ", ".join(CHOICES[task]) + "."
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


@dataclass
class SplicedExample:
    inputs_embeds: Tensor
    labels: Tensor
    sensor_positions: tuple[int, ...]
    prefix_length: int
    uid: str


@dataclass
class SplicedBatch:
    inputs_embeds: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    labels: Tensor
    lengths: tuple[int, ...]
    sensor_positions: tuple[tuple[int, ...], ...]
    uids: tuple[str, ...]


class RadarSplicer:
    def __init__(self, tokenizer: Any, embedding: nn.Module, *, expected_template_hash: str,
                 assistant_end_token: str | None = None, template_kwargs: dict | None = None):
        template = tokenizer.chat_template
        if not isinstance(template, str) or not template:
            raise ValueError("a pinned string chat template is required")
        self.template_hash = hashlib.sha256(template.encode()).hexdigest()
        if self.template_hash != expected_template_hash:
            raise ValueError("chat template hash mismatch")
        self.tokenizer, self.embedding = tokenizer, embedding
        self.template_kwargs = dict(template_kwargs or {})
        self.end_token = assistant_end_token if assistant_end_token is not None else tokenizer.eos_token
        end_ids = tokenizer.encode(self.end_token, add_special_tokens=False)
        if len(end_ids) != 1 or end_ids[0] not in tokenizer.all_special_ids:
            raise ValueError("assistant termination must be one registered special token")
        self.end_token_id = end_ids[0]
        if "<|im_end|>" in template and self.end_token != "<|im_end|>":
            raise ValueError("Qwen chat template requires <|im_end|> termination")

    def _ids(self, text: str) -> Tensor:
        return torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long,
                            device=self.embedding.weight.device)

    def build(self, *, uid: str, task: str, time_s: Tensor | None,
              sensor_embeddings: Tensor | None, answer: str | None = None,
              text_only: bool = False) -> SplicedExample:
        rendered = self.tokenizer.apply_chat_template(prompt_messages(task), tokenize=False,
                                                      add_generation_prompt=True, **self.template_kwargs)
        return self.splice_rendered(rendered, uid=uid, time_s=time_s,
                                    sensor_embeddings=sensor_embeddings, answer=answer,
                                    text_only=text_only)

    def splice_rendered(self, rendered: str, *, uid: str, time_s: Tensor | None,
                        sensor_embeddings: Tensor | None, answer: str | None = None,
                        text_only: bool = False) -> SplicedExample:
        if rendered.count(MARKER) != 1:
            raise ValueError(f"{uid}: rendered chat must contain exactly one radar marker")
        before, after = rendered.split(MARKER)
        pieces: list[Tensor] = []
        sensor_positions: list[int] = []
        length = 0

        def append_text(text: str) -> None:
            nonlocal length
            embedded = self.embedding(self._ids(text))
            pieces.append(embedded)
            length += embedded.shape[0]

        append_text(before)
        if text_only:
            if sensor_embeddings is not None:
                raise ValueError("T has no sensor embeddings")
            append_text("No radar evidence is provided.")
        else:
            d = self.embedding.weight.shape[1]
            if sensor_embeddings is None or sensor_embeddings.shape != (64, d):
                raise ValueError(f"{uid}: sensor embeddings must be [64,{d}]")
            if time_s is None or time_s.shape != (64,) or time_s.dtype != torch.float64:
                raise ValueError(f"{uid}: time_s must be float64 [64]")
            if not torch.isfinite(time_s).all() or not (time_s.diff() > 0).all():
                raise ValueError(f"{uid}: invalid sensor timestamps")
            elapsed = (time_s - time_s[0]).detach().cpu().tolist()
            for frame in range(64):
                append_text(f"Frame {frame:02d} at {elapsed[frame]:.4f}s: ")
                sensor_positions.append(length)
                # Keep this cast differentiable. In particular, never detach H.
                pieces.append(sensor_embeddings[frame:frame + 1].to(device=self.embedding.weight.device,
                                                                   dtype=self.embedding.weight.dtype))
                length += 1
                append_text("\n")
        append_text(after)
        prefix_length = length
        labels = torch.full((prefix_length,), -100, dtype=torch.long, device=self.embedding.weight.device)
        if answer is not None:
            answer_ids = self._ids(answer)
            if not answer_ids.numel() or self.tokenizer.decode(answer_ids.tolist(), skip_special_tokens=False) != answer:
                raise ValueError(f"{uid}: canonical answer does not round-trip through tokenizer")
            if self.end_token_id in answer_ids.tolist():
                raise ValueError(f"{uid}: answer already contains assistant termination")
            target_ids = torch.cat((answer_ids, answer_ids.new_tensor([self.end_token_id])))
            pieces.append(self.embedding(target_ids))
            labels = torch.cat((labels, target_ids))
            length += target_ids.numel()
        limit = 3040 if answer is None else 3072
        if length > limit:
            raise ValueError(f"{uid}: sequence length {length} exceeds limit {limit}; truncation is forbidden")
        return SplicedExample(torch.cat(pieces), labels, tuple(sensor_positions), prefix_length, uid)


def collate_spliced(examples: list[SplicedExample]) -> SplicedBatch:
    """Right-pad without creating labels at sensor/prefix/padding positions."""
    if not examples:
        raise ValueError("empty batch")
    width = max(example.inputs_embeds.shape[0] for example in examples)
    lengths = tuple(example.inputs_embeds.shape[0] for example in examples)
    embedded = torch.stack([F.pad(example.inputs_embeds, (0, 0, 0, width - length)) for example, length in zip(examples, lengths)])
    labels = torch.stack([F.pad(example.labels, (0, width - length), value=-100) for example, length in zip(examples, lengths)])
    attention_mask = torch.arange(width, device=embedded.device)[None, :] < torch.tensor(lengths, device=embedded.device)[:, None]
    positions = attention_mask.long().cumsum(-1) - 1
    positions.masked_fill_(~attention_mask, 0)
    return SplicedBatch(embedded, attention_mask, positions, labels, lengths,
                        tuple(example.sensor_positions for example in examples), tuple(example.uid for example in examples))


def answer_loss_from_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """One causal shift, then mean target tokens per example, then examples."""
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits/labels shape mismatch")
    if (labels[:, 0] != -100).any():
        raise ValueError("a first-position target has no causal predecessor")
    examples, targets = torch.where(labels != -100)
    counts = torch.bincount(examples, minlength=labels.shape[0])
    if (counts == 0).any():
        raise ValueError("every training example must have assistant targets")
    # Use the same 2-D CE kernel in the full-logit reference and selected-head
    # implementation; reductions over ignored prefix logits can change fp32
    # rounding before conversion of gradients to the frozen backbone's dtype.
    token_loss = F.cross_entropy(logits[examples, targets - 1].float(), labels[examples, targets], reduction="none")
    sums = token_loss.new_zeros(labels.shape[0]).scatter_add_(0, examples, token_loss)
    return (sums / counts).mean()


@dataclass
class GenerationResult:
    token_ids: list[int]
    raw_text: str
    terminated: bool
    truncated: bool
    prefix_length: int


class FrozenQwenBridge(nn.Module):
    """Autograd through frozen Qwen toward sensor embeddings stays enabled."""
    def __init__(self, causal_lm: nn.Module, expected_hidden_size: int = 1536, head_mode: str = "canonical_fp32"):
        super().__init__()
        if causal_lm.config.hidden_size != expected_hidden_size:
            raise ValueError("Qwen hidden size does not match the pinned specification")
        self.causal_lm = causal_lm.requires_grad_(False).eval()
        if head_mode not in ("native", "canonical_fp32"):
            raise ValueError("unknown verified LM-head compute mode")
        self.head_mode = head_mode
        self.checkpoint_backbone = False

    def train(self, mode: bool = True) -> "FrozenQwenBridge":
        super().train(mode)
        self.causal_lm.eval()
        return self

    @classmethod
    def from_local_snapshot(cls, path: str, *, model_commit: str, tokenizer_commit: str,
                            dtype: torch.dtype, expected_hidden_size: int = 1536,
                            head_mode: str = "canonical_fp32") -> "FrozenQwenBridge":
        if len(model_commit) != 40 or len(tokenizer_commit) != 40:
            raise ValueError("explicit model/tokenizer commit SHA required")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("main Qwen recipe requires verified 16-bit weights")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path, revision=model_commit, local_files_only=True,
                                                    torch_dtype=dtype, low_cpu_mem_usage=True)
        result = cls(model, expected_hidden_size, head_mode)
        result.model_commit, result.tokenizer_commit = model_commit, tokenizer_commit
        return result

    def enable_input_checkpointing(self) -> None:
        # HF's usual checkpoint path is conditional on model.training. Preserve
        # frozen eval/dropout behavior by checkpointing the backbone explicitly.
        self.checkpoint_backbone = True

    def _hidden_states(self, batch: SplicedBatch) -> Tensor:
        def run(embedded: Tensor) -> Tensor:
            return self.causal_lm.model(inputs_embeds=embedded, attention_mask=batch.attention_mask,
                         position_ids=batch.position_ids, use_cache=False, return_dict=True).last_hidden_state
        if self.checkpoint_backbone and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(run, batch.inputs_embeds, use_reentrant=False)
        return run(batch.inputs_embeds)

    def _head_logits_float32(self, hidden: Tensor) -> Tensor:
        """Compute the same frozen head in fp32 for both reduction paths.

        Half-precision GEMMs with different numbers of selected rows can round
        input gradients differently. Explicit fp32 computation removes that
        shape-dependent rounding; the stored Qwen parameters remain 16-bit.
        Its temporary fp32 head memory is included in native resource profiles.
        """
        head = self.causal_lm.get_output_embeddings()
        if not isinstance(head, nn.Linear):
            raise TypeError("the verified Qwen path requires a Linear LM head")
        if self.head_mode == "native":
            return head(hidden).float()
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            bias = None if head.bias is None else head.bias.float()
            weight = head.weight.float()
            shape = hidden.shape[:-1]
            rows = hidden.float().reshape(-1, hidden.shape[-1])
            # Fix the GEMM row count as well as dtype. Otherwise tiny changes in
            # head-gradient rounding can be amplified through a bf16 backbone.
            logits = torch.cat([F.linear(row, weight, bias) for row in rows.split(1)], dim=0)
            return logits.reshape(*shape, weight.shape[0])

    def selected_answer_loss(self, batch: SplicedBatch) -> Tensor:
        if (batch.labels[:, 0] != -100).any():
            raise ValueError("first-position targets have no predecessor")
        all_hidden = self._hidden_states(batch)
        example_indices, target_positions = torch.where(batch.labels != -100)
        counts = torch.bincount(example_indices, minlength=batch.labels.shape[0])
        if (counts == 0).any():
            raise ValueError("every example must have answer targets")
        hidden = all_hidden[example_indices, target_positions - 1]
        logits = self._head_logits_float32(hidden)
        loss = F.cross_entropy(logits, batch.labels[example_indices, target_positions], reduction="none")
        sums = loss.new_zeros(batch.labels.shape[0]).scatter_add_(0, example_indices, loss)
        return (sums / counts).mean()

    def full_answer_loss(self, batch: SplicedBatch) -> Tensor:
        return answer_loss_from_logits(self._head_logits_float32(self._hidden_states(batch)), batch.labels)

    @torch.no_grad()
    def greedy_generate(self, example: SplicedExample, tokenizer: Any, end_token_id: int,
                        *, use_cache: bool = True, max_new_tokens: int = 32) -> GenerationResult:
        if max_new_tokens > 32 or max_new_tokens <= 0:
            raise ValueError("generation budget must be 1..32 tokens")
        if example.inputs_embeds.shape[0] != example.prefix_length or (example.labels != -100).any():
            raise ValueError("generation must receive a prefix without teacher-forced answer")
        if example.prefix_length > 3040:
            raise ValueError(f"{example.uid}: generation prefix exceeds 3040")
        self.causal_lm.eval()
        embedded = example.inputs_embeds[None]
        next_input = embedded
        cache = None
        generated: list[int] = []
        terminated = False
        for step in range(max_new_tokens):
            full_length = example.prefix_length + step
            mask = torch.ones((1, full_length), dtype=torch.long, device=embedded.device)
            positions = torch.arange(full_length - next_input.shape[1], full_length, device=embedded.device)[None]
            outputs = self.causal_lm(inputs_embeds=next_input, attention_mask=mask, position_ids=positions,
                                     past_key_values=cache, use_cache=use_cache, return_dict=True)
            token = int(outputs.logits[0, -1].argmax())
            generated.append(token)
            if token == end_token_id:
                terminated = True
                break
            token_embedding = self.causal_lm.get_input_embeddings()(torch.tensor([[token]], device=embedded.device))
            if use_cache:
                cache = outputs.past_key_values
                if cache is None:
                    raise RuntimeError("model did not return a KV cache")
                next_input = token_embedding
            else:
                embedded = torch.cat((embedded, token_embedding), dim=1)
                next_input = embedded
        content_ids = generated[:-1] if terminated else generated
        return GenerationResult(generated, tokenizer.decode(content_ids, skip_special_tokens=False),
                                terminated, not terminated, example.prefix_length)

    @torch.no_grad()
    def embedding_equivalence_error(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> float:
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(~mask.bool(), 0)
        common = dict(attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
        by_ids = self.causal_lm(input_ids=input_ids, **common).logits
        by_embeddings = self.causal_lm(inputs_embeds=self.causal_lm.get_input_embeddings()(input_ids), **common).logits
        return float((by_ids.float() - by_embeddings.float()).abs().max())
