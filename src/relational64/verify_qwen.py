"""Run bounded, native checks on an explicitly pinned local Qwen snapshot.

The sensor tensors here are synthetic. This verifies the real pretrained
backbone and target-device software path; it never asserts M4Human accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

import torch

from .language import FrozenQwenBridge, RadarSplicer, SplicedExample, canonical_answer, collate_spliced
from .models import SensorProjector, parameter_hash


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--head-mode", choices=("native", "canonical_fp32"), default="canonical_fp32")
    args = parser.parse_args()
    report = {"schema": "relational64_qwen_check_1", "date_utc": datetime.now(timezone.utc).isoformat(),
              "evidence": "native pretrained Qwen + synthetic sensor fixtures", "actual_m4human_data": False,
              "platform": platform.platform(), "native_windows": platform.system() == "Windows",
              "snapshot": str(args.snapshot.resolve()), "model_commit": args.commit,
              "tokenizer_commit": args.commit, "checks": {}, "status": "running"}
    report["source_hashes"] = {name: file_hash(Path(__file__).with_name(name))
                               for name in ("models.py", "language.py", "verify_qwen.py")}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("native CUDA is required for this check")
        if args.snapshot.name != args.commit or len(args.commit) != 40:
            raise ValueError("snapshot directory must match the pinned commit SHA")
        import transformers
        from transformers import AutoTokenizer
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.set_num_threads(4)
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if args.dtype == "auto" else {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
        report.update(torch=torch.__version__, transformers=transformers.__version__, dtype=str(dtype), head_mode=args.head_mode,
                      cuda=torch.version.cuda, device=torch.cuda.get_device_name(0), native_cuda=True)
        report["snapshot_file_hashes"] = {item.name: file_hash(item) for item in sorted(args.snapshot.iterdir()) if item.is_file()}
        tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot), revision=args.commit, local_files_only=True)
        template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        report["chat_template_hash"] = template_hash
        bridge = FrozenQwenBridge.from_local_snapshot(str(args.snapshot), model_commit=args.commit,
                   tokenizer_commit=args.commit, dtype=dtype, head_mode=args.head_mode).to("cuda")
        splicer = RadarSplicer(tokenizer, bridge.causal_lm.get_input_embeddings(), expected_template_hash=template_hash)
        torch.cuda.reset_peak_memory_stats()
        frozen_before = parameter_hash(bridge)
        ids = tokenizer("Return a JSON object with one answer.", return_tensors="pt").input_ids.to("cuda")
        error = bridge.embedding_equivalence_error(ids)
        report["checks"]["input_ids_inputs_embeds"] = {"max_absolute_logit_error": error, "passed": error <= 1e-3}
        # Actual Qwen backbone, short synthetic embedding sequence, unequal answer
        # lengths: isolate CE math without allocating vocabulary logits for 2k tokens.
        short_examples = []
        for index, target_count in enumerate((2, 5)):
            embedding = torch.randn(17 + index, 1536, device="cuda", dtype=dtype, requires_grad=True)
            labels = torch.full((17 + index,), -100, device="cuda", dtype=torch.long)
            labels[-target_count:] = torch.tensor([11, 20, 31, 42, 53][:target_count], device="cuda")
            short_examples.append(SplicedExample(embedding, labels, (), 17 + index - target_count, f"short_{index}"))
        short_batch = collate_spliced(short_examples)
        full_loss = bridge.full_answer_loss(short_batch)
        full_gradient = torch.autograd.grad(full_loss, short_batch.inputs_embeds, retain_graph=False)[0]
        selected_loss = bridge.selected_answer_loss(short_batch)
        selected_gradient = torch.autograd.grad(selected_loss, short_batch.inputs_embeds, retain_graph=False)[0]
        loss_error = float((full_loss - selected_loss).abs())
        grad_error = float((full_gradient.float() - selected_gradient.float()).abs().max())
        grad_norm_error = float((full_gradient.float() - selected_gradient.float()).norm() / full_gradient.float().norm().clamp_min(1e-20))
        combined_tolerance_ratio = float(((full_gradient.float() - selected_gradient.float()).abs() / (1e-3 + 1e-3 * full_gradient.float().abs())).max())
        report["checks"]["selected_head_equivalence"] = {"full_loss": float(full_loss), "selected_loss": float(selected_loss),
             "absolute_loss_error": loss_error, "max_absolute_gradient_error": grad_error,
             "relative_gradient_norm_error": grad_norm_error, "full_gradient_max_abs": float(full_gradient.abs().max()),
             "combined_tolerance_ratio_max": combined_tolerance_ratio, "atol": 1e-3, "rtol": 1e-3,
             "passed": loss_error <= 1e-3 and combined_tolerance_ratio <= 1}
        del short_batch, short_examples, full_loss, selected_loss, full_gradient, selected_gradient
        torch.cuda.empty_cache()
        projector = SensorProjector("M").to("cuda")
        original_projector_hash = parameter_hash(projector)
        feature = torch.randn(1, 64, 256, device="cuda")
        sensor = projector(feature).sensor_embeddings[0]
        time_s = torch.arange(64, device="cuda", dtype=torch.float64) / 12
        training = splicer.build(uid="native_synthetic", task="arms", time_s=time_s, sensor_embeddings=sensor,
                                 answer=canonical_answer("arms", "tie"))
        generation = splicer.build(uid="native_synthetic", task="arms", time_s=time_s, sensor_embeddings=sensor)
        prefix_equal = torch.equal(training.inputs_embeds[:training.prefix_length], generation.inputs_embeds)
        loss = bridge.selected_answer_loss(collate_spliced([training]))
        optimizer = torch.optim.AdamW(projector.parameters(), lr=3e-4)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0))
        frozen_grads_none = all(parameter.grad is None for parameter in bridge.parameters())
        optimizer.step()
        report["checks"]["splicing_freeze_and_input_gradient"] = {"sensor_embeddings": len(training.sensor_positions),
             "training_tokens": training.inputs_embeds.shape[0], "generation_prefix_tokens": generation.prefix_length,
             "prefix_identical": prefix_equal, "answer_loss": float(loss), "projector_gradient_norm": gradient_norm,
             "qwen_gradients_none": frozen_grads_none, "projector_changed": parameter_hash(projector) != original_projector_hash,
             "passed": prefix_equal and len(training.sensor_positions) == 64 and math.isfinite(gradient_norm) and gradient_norm > 0 and frozen_grads_none}
        del training, loss, optimizer
        # Force eight steps for equivalence even if the sampled fixture emits its
        # ordinary termination token before step8; this sentinel is check-only.
        decode_ids = tokenizer("Continue the sequence of integers: 1, 2, 3,", return_tensors="pt").input_ids.to("cuda")
        decode_embedding = bridge.causal_lm.get_input_embeddings()(decode_ids)[0]
        decode_example = SplicedExample(decode_embedding, torch.full(decode_ids.shape[1:], -100, dtype=torch.long, device="cuda"),
                                        (), decode_ids.shape[1], "cache_equivalence")
        cached = bridge.greedy_generate(decode_example, tokenizer, -1, use_cache=True, max_new_tokens=8)
        uncached = bridge.greedy_generate(decode_example, tokenizer, -1, use_cache=False, max_new_tokens=8)
        report["checks"]["cached_vs_uncached_8_tokens"] = {"cached_ids": cached.token_ids,
             "uncached_ids": uncached.token_ids, "forced_eight_steps_for_check": True,
             "passed": cached.token_ids == uncached.token_ids and len(cached.token_ids) == 8}
        generated = bridge.greedy_generate(generation, tokenizer, splicer.end_token_id)
        report["generation_software_sample"] = {"new_token_ids": generated.token_ids, "raw_text": generated.raw_text,
             "terminated": generated.terminated, "truncated": generated.truncated,
             "interpretation": "software evidence only; no QA accuracy inference from synthetic sensors"}
        frozen_after = parameter_hash(bridge)
        report["checks"]["frozen_weight_hash"] = {"before": frozen_before, "after": frozen_after, "passed": frozen_before == frozen_after}
        report["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        report["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        report["status"] = "passed" if all(item["passed"] for item in report["checks"].values()) else "failed"
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report), "checks": report["checks"],
                      "error": report.get("error")}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
