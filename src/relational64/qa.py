"""Canonical QA, immutable label policy, and strict generation parser."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any
import numpy as np
from .kinematics import estimate_velocities

SYSTEM_PROMPT = 'You answer kinematic questions using the supplied radar evidence. Return exactly one JSON object with the key "answer" and one allowed label. Do not add explanations or other keys. Use "unknown" when the evidence does not support a choice.'
RADAR_MARKER = "[[RADAR_BLOCK]]"
TASKS = {
    "root": {"task_id": "root_speed_trend", "part": "root", "question": "Over frames 06 to 63, does the pelvis speed show an increasing trend, a decreasing trend, or no overall trend?", "classes": ("speeding_up", "slowing_down", "no_overall_trend")},
    "arms": {"task_id": "relative_limb_motion", "part": "arms", "question": "Over frames 06 to 63, which arm has greater typical motion speed relative to the pelvis, using the elbows and wrists?", "classes": ("faster_left", "faster_right", "tie")},
    "legs": {"task_id": "relative_limb_motion", "part": "legs", "question": "Over frames 06 to 63, which leg has greater typical motion speed relative to the pelvis, using the knees and ankles?", "classes": ("faster_left", "faster_right", "tie")},
}
USER_PREFIX = "Radar observations contain 64 consecutive frames numbered 00 to 63. The first six frames provide derivative context. The question concerns frames 06 to 63. Left and right refer to the person's anatomical sides. Root means pelvis. Relative limb motion excludes pelvis translation."


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class LabelPolicy:
    root_delta_mps2: float = 0.05
    root_guard_mps2: float = 0.01
    root_residual_max_mps: float = 0.15
    limb_delta_mps: float = 0.05
    limb_guard_mps: float = 0.01
    minimum_valid_endpoints: int = 44

    @property
    def hash(self):
        return stable_hash(asdict(self))


def canonical_answer(answer: str) -> str:
    return json.dumps({"answer": answer}, separators=(",", ":"))


def user_prompt(part: str) -> str:
    task = TASKS[part]
    choices = ", ".join((*task["classes"], "unknown"))
    return f"{USER_PREFIX}\n{RADAR_MARKER}\n{task['question']}\nAllowed answers: {choices}."


def classify_scalar(part, scalar, *, residual_scale=None, policy=None):
    policy = policy or LabelPolicy()
    if not np.isfinite(scalar) or (part == "root" and (residual_scale is None or not np.isfinite(residual_scale))):
        return "undefined", None, "invalid_numeric_target"
    if part == "root":
        if residual_scale > policy.root_residual_max_mps:
            return "unknown", "unknown", "nonlinear_or_unstable_speed_trend"
        delta, guard = policy.root_delta_mps2, policy.root_guard_mps2
    else:
        delta, guard = policy.limb_delta_mps, policy.limb_guard_mps
    # A numerical equality tolerance protects a mathematically exact boundary
    # from the rounding error of the float64 trajectory derivative.
    boundary = np.isclose(abs(scalar), delta+guard, rtol=0, atol=1e-12) or np.isclose(abs(scalar), delta-guard, rtol=0, atol=1e-12)
    if not boundary and scalar > delta + guard:
        return "defined", TASKS[part]["classes"][0], None
    if not boundary and scalar < -delta - guard:
        return "defined", TASKS[part]["classes"][1], None
    if not boundary and abs(scalar) < delta - guard:
        return "defined", TASKS[part]["classes"][2], None
    return "unknown", "unknown", "trend_boundary_band" if part == "root" else "limb_boundary_band"


def targets_from_velocity(timestamps, velocity, velocity_valid, *, policy=None):
    """Apply the GT label equations to velocities (also used by Rule).

    Rule callers must supply sensor-only availability; GT masks are forbidden
    in the inference path. These scalars and masks never enter a prompt.
    """
    policy = policy or LabelPolicy()
    t = np.asarray(timestamps, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    valid = np.asarray(velocity_valid, dtype=bool)
    if t.shape != (64,) or v.shape != (64, 22, 3) or valid.shape != (64, 22):
        raise ValueError("QA requires timestamps [64], velocity [64,22,3], valid [64,22]")
    valid = valid & np.isfinite(v).all(-1) & np.isfinite(t[:, None])
    result = []
    for part, joints in (("root", (0,)), ("arms", (0, 18, 19, 20, 21)), ("legs", (0, 4, 5, 7, 8))):
        support = valid[:, joints].all(axis=1)
        support[:6] = False
        n = int(support.sum())
        record = dict(task_id=TASKS[part]["task_id"], part=part, interval=[6, 63],
                      support_count=n, candidate_count=58, scalar_value=None,
                      residual_scale=None, reference_status="undefined", answer=None,
                      reason="insufficient_velocity_support")
        if n >= policy.minimum_valid_endpoints:
            if part == "root":
                times = t[support]
                speed = np.linalg.norm(v[support, 0], axis=-1)
                i, j = np.triu_indices(n, k=1)
                dt = times[j] - times[i]
                if np.any(dt <= 0):
                    record["reason"] = "invalid_numeric_target"
                    result.append(record)
                    continue
                scalar = float(np.median((speed[j] - speed[i]) / dt))
                elapsed = times - t[6]
                intercept = np.median(speed - scalar * elapsed)
                residual = speed - (intercept + scalar * elapsed)
                mad = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
                record["residual_scale"] = mad
            else:
                left, right = ((18, 20), (19, 21)) if part == "arms" else ((4, 7), (5, 8))
                rel = v[support] - v[support, 0:1]
                median_speed = np.median(np.linalg.norm(rel, axis=-1), axis=0)
                scalar = float(np.mean(median_speed[list(left)]) - np.mean(median_speed[list(right)]))
            status, answer, reason = classify_scalar(part, scalar, residual_scale=record["residual_scale"], policy=policy)
            record.update(scalar_value=scalar if np.isfinite(scalar) else None,
                          reference_status=status, answer=answer, reason=reason)
        result.append(record)
    return result


def generate_qa(window_uid, timestamps, positions, position_valid, *, protocol_id="m4human_relational64",
                policy=None, target_hash=None, window_manifest_hash=None, metadata=None):
    policy = policy or LabelPolicy()
    targets = estimate_velocities(positions, position_valid, timestamps)
    records = targets_from_velocity(timestamps, targets.velocity, targets.valid, policy=policy)
    for record in records:
        part = record["part"]
        prompt_hash = stable_hash({"system": SYSTEM_PROMPT, "user": user_prompt(part)})
        identity = f"{protocol_id}/{window_uid}/{record['task_id']}/{part}/{prompt_hash}/{policy.hash}"
        record.update(qa_uid=hashlib.sha256(identity.encode()).hexdigest(), window_uid=window_uid,
                      prompt_id=f"canonical_v1/{part}", prompt_hash=prompt_hash,
                      label_policy_hash=policy.hash, target_hash=target_hash,
                      window_manifest_hash=window_manifest_hash)
        if metadata:
            for key in ("subject_id", "recording_id", "split", "activity_id"):
                if key in metadata:
                    record[key] = metadata[key]
    return records


@dataclass(frozen=True)
class ParsedAnswer:
    status: str
    answer: str | None
    raw_text: str | None
    reason: str | None = None
    json_parse_valid: bool = False


def parse_answer(raw_text, part, *, truncated=False, runtime_error=None, eot_tokens=()):
    """Parse one JSON object without repair, substring extraction or fuzziness."""
    if part not in TASKS:
        raise ValueError(f"unknown task part {part!r}")
    if runtime_error is not None:
        return ParsedAnswer("runtime_error", None, raw_text, str(runtime_error))
    if truncated:
        return ParsedAnswer("truncated", None, raw_text, "generation_token_limit")
    if raw_text is None or (isinstance(raw_text, str) and not raw_text.strip()):
        return ParsedAnswer("missing", None, raw_text, "no_output")
    if not isinstance(raw_text, str):
        return ParsedAnswer("invalid_json", None, None, "output_not_text")
    text = raw_text.strip()
    for token in eot_tokens:
        if token and text.endswith(token):
            text = text[:-len(token)].strip()
            break
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError(f"nonfinite_constant:{value}")
    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
    except (ValueError, TypeError) as exc:
        return ParsedAnswer("invalid_json", None, raw_text, str(exc))
    if not isinstance(value, dict) or set(value) != {"answer"} or not isinstance(value["answer"], str):
        return ParsedAnswer("invalid_schema", None, raw_text, "expected_single_string_answer", True)
    if value["answer"] not in (*TASKS[part]["classes"], "unknown"):
        return ParsedAnswer("invalid_label", None, raw_text, "label_not_in_task_enum", True)
    return ParsedAnswer("valid", value["answer"], raw_text, None, True)


def fit_train_majority(records):
    if any(r.get("split") != "train" for r in records):
        raise ValueError("TrainMajority must only access train records")
    result = {}
    for part, task in TASKS.items():
        labels = [r["answer"] for r in records if r["part"] == part and r["reference_status"] == "defined"]
        if not labels:
            raise ValueError(f"no defined train target for {part}")
        result[part] = max(task["classes"], key=labels.count)
    return result


def rule_predictions(window_uid, timestamps, predicted_positions, *, policy=None):
    """No GT masks: numerical sensor state is the only Rule input."""
    mask = np.isfinite(predicted_positions).all(axis=-1)
    return generate_qa(window_uid, timestamps, predicted_positions, mask, policy=policy)


def audit_label_policy(records, *, policy=None, boundary_examples_per_task=10):
    """Train/validation diagnostics; this does not approve or lock a policy.

    Include all windows, including undefined targets. Callers must still
    inspect physical trajectories, units and coordinate/time audit evidence
    before finalization. Test labels are expressly rejected here.
    """
    from collections import Counter
    policy = policy or LabelPolicy()
    if any(r.get("split") not in ("train", "val") for r in records):
        raise ValueError("label-policy audit accepts train/val only")
    report = {"policy_hash": policy.hash, "status": "diagnostic_not_locked", "tasks": {}, "sensitivity": []}
    for part in TASKS:
        rows = [r for r in records if r["part"] == part]
        finite = [r for r in rows if r["scalar_value"] is not None and np.isfinite(r["scalar_value"])]
        values = np.asarray([r["scalar_value"] for r in finite], np.float64)
        def quantiles(array):
            return dict(zip(("min", "q05", "q25", "median", "q75", "q95", "max"), np.quantile(array, [0, .05, .25, .5, .75, .95, 1]).tolist())) if len(array) else None
        delta = policy.root_delta_mps2 if part == "root" else policy.limb_delta_mps
        guard = policy.root_guard_mps2 if part == "root" else policy.limb_guard_mps
        distance = lambda r: min(abs(abs(r["scalar_value"])-(delta-guard)), abs(abs(r["scalar_value"])-(delta+guard)))
        near = sorted(finite, key=lambda r: (distance(r), r["qa_uid"]))[:boundary_examples_per_task]
        residuals = [r["residual_scale"] for r in finite if r.get("residual_scale") is not None]
        report["tasks"][part] = {
            "qa_count": len(rows), "status_counts": dict(Counter(r["reference_status"] for r in rows)),
            "class_counts": dict(Counter(r["answer"] for r in rows if r["answer"] is not None)),
            "reason_counts": dict(Counter(r["reason"] for r in rows if r.get("reason"))),
            "scalar_quantiles": quantiles(values), "residual_quantiles": quantiles(residuals),
            "per_subject": {str(subject): {"status_counts": dict(Counter(r["reference_status"] for r in rows if r.get("subject_id") == subject)),
                                          "class_counts": dict(Counter(r["answer"] for r in rows if r.get("subject_id") == subject and r["answer"] is not None))}
                            for subject in sorted({r.get("subject_id") for r in rows}, key=str)},
            "boundary_examples": [{k: r.get(k) for k in ("qa_uid", "window_uid", "subject_id", "split", "scalar_value", "residual_scale", "support_count", "reference_status", "answer")} for r in near]}
        for delta_value in (.025, .05, .10):
            residual_values = (.075, .15, .30) if part == "root" else (policy.root_residual_max_mps,)
            for residual_max in residual_values:
                alternative = LabelPolicy(root_delta_mps2=delta_value if part == "root" else policy.root_delta_mps2,
                                          root_guard_mps2=.2*delta_value if part == "root" else policy.root_guard_mps2,
                                          root_residual_max_mps=residual_max,
                                          limb_delta_mps=delta_value if part != "root" else policy.limb_delta_mps,
                                          limb_guard_mps=.2*delta_value if part != "root" else policy.limb_guard_mps,
                                          minimum_valid_endpoints=policy.minimum_valid_endpoints)
                statuses, classes = Counter(), Counter()
                changed = 0
                for row in rows:
                    if row["reference_status"] == "undefined":
                        statuses["undefined"] += 1
                        continue
                    status, answer, _ = classify_scalar(part, row["scalar_value"], residual_scale=row.get("residual_scale"), policy=alternative)
                    statuses[status] += 1
                    classes[answer] += 1
                    changed += answer != row["answer"]
                report["sensitivity"].append({"part": part, "delta": delta_value, "guard": .2*delta_value,
                                              "residual_max": residual_max if part == "root" else None,
                                              "status_counts": dict(statuses), "class_counts": dict(classes),
                                              "changed_label_count": changed})
    return report
