"""Fixed-panel QA/physics metrics and deterministic donor controls.

All counts retain missing/invalid predictions. Cluster bootstrap reweights
subjects, never windows; donor inference reweights disjoint subject blocks.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
from itertools import product
import numpy as np
from .qa import TASKS, ParsedAnswer, parse_answer
from .kinematics import BONE_EDGES

TASK_WEIGHTS = {"root": .5, "arms": .25, "legs": .25}


def _prediction(value, part):
    if isinstance(value, ParsedAnswer):
        parsed = value
    elif isinstance(value, dict):
        parsed = ParsedAnswer(value.get("status", "missing"), value.get("answer"), value.get("raw_text"), value.get("reason"), value.get("json_parse_valid", value.get("status") == "valid"))
    else:
        parsed = parse_answer(value, part)
    if parsed.status == "valid" and parsed.answer not in (*TASKS[part]["classes"], "unknown"):
        return ParsedAnswer("invalid_label", None, parsed.raw_text, "structured_prediction_label_not_in_task_enum", parsed.json_parse_valid)
    return parsed


def _task_metric(records, predictions, part, *, secondary=False, weights=None):
    classes = (*TASKS[part]["classes"], "unknown") if secondary else TASKS[part]["classes"]
    columns = (*classes, *(("unknown",) if not secondary else ()), "invalid", "missing/truncated")
    matrix = np.zeros((len(classes), len(columns)), dtype=np.float64)
    for record in records:
        eligible = record["reference_status"] in (("defined", "unknown") if secondary else ("defined",))
        if record["part"] != part or not eligible:
            continue
        weight = 1. if weights is None else float(weights.get(record["subject_id"], 0))
        pred = _prediction(predictions.get(record["qa_uid"]), part)
        column = pred.answer if pred.status == "valid" else "missing/truncated" if pred.status in ("missing", "truncated") else "invalid"
        if column not in columns:
            column = "invalid"
        matrix[classes.index(record["answer"]), columns.index(column)] += weight
    count = float(matrix.sum())
    support = matrix.sum(axis=1)
    f1 = {}
    tp_sum = 0.
    for i, label in enumerate(classes):
        tp = matrix[i, i]
        denominator = support[i] + matrix[:, i].sum()
        f1[label] = float(2*tp/denominator) if denominator else 0.
        tp_sum += tp
    return dict(count=count, classes=list(classes), confusion_columns=list(columns),
                confusion=matrix.tolist(), class_support=dict(zip(classes, support.tolist())),
                per_class_f1=f1, macro_f1=float(np.mean(list(f1.values()))) if count else None,
                accuracy=float(tp_sum/count) if count else None,
                coverage_ceiling_macro_f1=float(np.count_nonzero(support)/len(classes)) if count else None)


def evaluate_qa(records, predictions, *, subject_weights=None):
    """Predictions map qa_uid to raw generation, ParsedAnswer, or its dict."""
    ids = [r["qa_uid"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate QA UID in shared evaluation panel")
    primary = {part: _task_metric(records, predictions, part, weights=subject_weights) for part in TASKS}
    secondary = {part: _task_metric(records, predictions, part, secondary=True, weights=subject_weights) for part in TASKS}
    def weighted(metrics, field):
        return None if any(metrics[p][field] is None for p in TASKS) else float(sum(TASK_WEIGHTS[p]*metrics[p][field] for p in TASKS))
    active = [r for r in records if r["reference_status"] in ("defined", "unknown")]
    statuses = Counter()
    json_count = valid_count = substantive_count = unknown_count = 0.
    total_weight = 0.
    for r in active:
        weight = 1. if subject_weights is None else float(subject_weights.get(r["subject_id"], 0))
        pred = _prediction(predictions.get(r["qa_uid"]), r["part"])
        statuses[pred.status] += weight
        total_weight += weight
        json_count += weight * pred.json_parse_valid
        valid_count += weight * (pred.status == "valid")
        substantive_count += weight * (pred.status == "valid" and pred.answer != "unknown")
        unknown_count += weight * (pred.status == "valid" and pred.answer == "unknown")
    rate = lambda value: float(value/total_weight) if total_weight else None
    return {"S_atomic": weighted(primary, "macro_f1"), "primary": primary,
            "secondary_4class_weighted_f1": weighted(secondary, "macro_f1"), "secondary": secondary,
            "accuracy_weighted": weighted(primary, "accuracy"),
            "accuracy_macro": None if any(primary[p]["accuracy"] is None for p in TASKS) else float(np.mean([primary[p]["accuracy"] for p in TASKS])),
            "generation": {"count": total_weight, "json_parse_rate": rate(json_count), "schema_enum_valid_rate": rate(valid_count),
                           "substantive_answer_rate": rate(substantive_count), "unknown_rate": rate(unknown_count), "status_counts": dict(statuses)},
            "reference_coverage": dict(Counter(r["reference_status"] for r in records)),
            "undefined_reasons": dict(Counter(r.get("reason") for r in records if r["reference_status"] == "undefined")),
            "support": {key: len({r[key] for r in records if r.get(key) is not None}) for key in ("subject_id", "recording_id", "window_uid", "qa_uid")}}


def paired_subject_bootstrap(records, predictions_r, predictions_s, *, draws=5000, seed=42, confidence=.95):
    """Paired subject draws, optionally multiple matching checkpoint seeds.

    Lists of prediction maps are averaged within each draw; they do not add
    clusters. Undefined draws remain null and are reported separately.
    """
    if not all("subject_id" in r for r in records):
        raise ValueError("bootstrap requires subject_id on every QA record")
    subjects = sorted({r["subject_id"] for r in records}, key=str)
    r_seeds = predictions_r if isinstance(predictions_r, list) else [predictions_r]
    s_seeds = predictions_s if isinstance(predictions_s, list) else [predictions_s]
    if len(r_seeds) != len(s_seeds) or not r_seeds:
        raise ValueError("paired seed lists must have equal nonzero length")
    point_deltas = []
    for rp, sp in zip(r_seeds, s_seeds):
        a, b = evaluate_qa(records, rp)["S_atomic"], evaluate_qa(records, sp)["S_atomic"]
        point_deltas.append(None if a is None or b is None else a-b)
    base = {"cluster_count": len(subjects), "unit": "subject", "seed": seed,
            "point_delta": None if None in point_deltas else float(np.mean(point_deltas)),
            "per_seed_deltas": point_deltas,
            "seed_sample_std": float(np.std(point_deltas, ddof=1)) if len(point_deltas)>1 and None not in point_deltas else None}
    if len(subjects) < 2:
        return dict(base, interval=None, status="insufficient_subjects", draws=0, undefined_draws=0)
    # Confusion sufficient statistics avoid reparsing every QA in every draw.
    # Repeated subjects multiply their entire confusion count, not an averaged
    # subject score (F1 is nonlinear).
    rng = np.random.default_rng(seed)
    sampled = rng.integers(len(subjects), size=(draws, len(subjects)))
    multiplicity = np.zeros((draws, len(subjects)), np.float64)
    for index in range(len(subjects)):
        multiplicity[:, index] = np.sum(sampled == index, axis=1)
    subject_records = [[r for r in records if r["subject_id"] == subject] for subject in subjects]
    def seed_draw_scores(predictions):
        # [subjects,task,true_class,prediction_column]
        counts = np.asarray([[_task_metric(rows, predictions, part)["confusion"] for part in TASKS] for rows in subject_records])
        confusion = np.einsum("ds,skij->dkij", multiplicity, counts)
        true_support = confusion.sum(-1)
        predicted_support = confusion[..., :3].sum(-2)
        diagonal = np.diagonal(confusion[..., :3], axis1=-2, axis2=-1)
        denominator = true_support + predicted_support
        f1 = np.divide(2*diagonal, denominator, out=np.zeros_like(diagonal), where=denominator>0)
        scores = (f1.mean(-1) * np.asarray(list(TASK_WEIGHTS.values()))).sum(-1)
        scores[(true_support.sum(-1) == 0).any(-1)] = np.nan
        return scores
    by_seed = np.asarray([seed_draw_scores(rp)-seed_draw_scores(sp) for rp, sp in zip(r_seeds, s_seeds)])
    mean_delta = by_seed.mean(axis=0)
    values = mean_delta[np.isfinite(mean_delta)].tolist()
    undefined = int((~np.isfinite(mean_delta)).sum())
    alpha = (1-confidence)/2
    interval = np.quantile(values, [alpha, 1-alpha]).tolist() if values else None
    status = "diagnostic_incomplete_support" if undefined > .05*draws else "diagnostic_small_subject_count" if len(subjects)<=4 else "cluster_percentile"
    return dict(base, interval=interval, status=status, draws=draws, undefined_draws=undefined, undefined_fraction=undefined/draws if draws else None, delta_draws=values)


def physical_metrics(predicted_positions, target_positions, position_valid, predicted_velocity,
                     target_velocity, velocity_valid, *, interval=None):
    """Return reducible numerator/count, with failure on nonfinite predictions.

    Aggregation is per valid window-endpoint-joint, including repeated source
    observations in overlapping windows. No Procrustes alignment is applied.
    """
    p, gt = np.asarray(predicted_positions), np.asarray(target_positions)
    v, vg = np.asarray(predicted_velocity), np.asarray(target_velocity)
    pm, vm = np.asarray(position_valid, bool), np.asarray(velocity_valid, bool)
    if p.shape != gt.shape or p.shape != v.shape or p.shape != vg.shape or p.shape[-2:] != (22, 3) or pm.shape != p.shape[:-1] or vm.shape != pm.shape:
        raise ValueError("physical metric shape mismatch")
    if np.any(pm & ~np.isfinite(gt).all(-1)) or np.any(vm & ~np.isfinite(vg).all(-1)):
        raise ValueError("valid GT mask includes nonfinite reference")
    if interval is not None:
        select = np.zeros(p.shape[-3], bool)
        select[interval[0]:interval[1]+1] = True
        pm = pm & select[:, None]
        vm = vm & select[:, None]
    output = {}
    def add(name, errors, mask, unit):
        chosen = np.asarray(errors)[mask]
        bad = int((~np.isfinite(chosen)).sum())
        finite = chosen[np.isfinite(chosen)]
        numerator = float(finite.sum())
        count = int(mask.sum())
        value = numerator/count if count and not bad else None
        output[name] = {"numerator": numerator if not bad else None, "count": count, "mean": value,
                        "unit": unit, "nonfinite_prediction_count": bad, "failure_rate": bad/count if count else None,
                        "finite_subset_mean_diagnostic": float(finite.mean()) if len(finite) else None}
        if unit == "m":
            output[name]["mean_mm"] = 1000*value if value is not None else None
    with np.errstate(invalid="ignore"):
        add("global_mpjpe", np.linalg.norm(p-gt, axis=-1), pm, "m")
        add("root_position_error", np.linalg.norm(p[..., 0, :]-gt[..., 0, :], axis=-1), pm[..., 0], "m")
        relative_p = (p-p[..., 0:1, :])-(gt-gt[..., 0:1, :])
        relative_pm = pm[..., 1:] & pm[..., 0:1]
        add("relative_mpjpe", np.linalg.norm(relative_p[..., 1:, :], axis=-1), relative_pm, "m")
        add("absolute_velocity_error", np.linalg.norm(v-vg, axis=-1), vm, "m/s")
        relative_v = (v-v[..., 0:1, :])-(vg-vg[..., 0:1, :])
        add("relative_velocity_error", np.linalg.norm(relative_v[..., 1:, :], axis=-1), vm[..., 1:] & vm[..., 0:1], "m/s")
        add("root_speed_mae", np.abs(np.linalg.norm(v[..., 0, :], axis=-1)-np.linalg.norm(vg[..., 0, :], axis=-1)), vm[..., 0], "m/s")
        parent, child = np.asarray(BONE_EDGES).T
        bone_error = np.abs(np.linalg.norm(p[..., parent, :]-p[..., child, :], axis=-1)-np.linalg.norm(gt[..., parent, :]-gt[..., child, :], axis=-1))
        add("bone_length_mae", bone_error, pm[..., parent] & pm[..., child], "m")
        add("zero_velocity_error", np.linalg.norm(vg, axis=-1), vm, "m/s")
    state_bad = int((~np.isfinite(p).all(-1)).sum())
    velocity_bad = int((~np.isfinite(v).all(-1)).sum())
    return {"status": "failed_nonfinite_prediction" if state_bad or velocity_bad else "valid",
            "aggregation": "valid_window_endpoint_joint_with_overlap", "metrics": output,
            "nonfinite_sensor_position_count": state_bad, "nonfinite_sensor_velocity_count": velocity_bad,
            "nonfinite_sensor_position_rate": state_bad/pm.size if pm.size else None,
            "nonfinite_sensor_velocity_rate": velocity_bad/vm.size if vm.size else None}


def _donor_rejection(a, b):
    for key in ("subject_id", "recording_id"):
        if a[key] == b[key]:
            return f"same_{key}"
    for key in ("split", "activity_id", "joint_map_hash", "coordinate_convention", "time_source", "label_policy_hash", "interval"):
        if key not in a or key not in b:
            return f"missing_{key}"
        if a[key] != b[key]:
            return f"mismatched_{key}"
    if a["reference_status"] != "defined" or b["reference_status"] != "defined":
        return "reference_not_defined"
    if a["answer"] == b["answer"]:
        return "same_reference_label"
    if a["answer"] not in TASKS[a["part"]]["classes"] or b["answer"] not in TASKS[b["part"]]["classes"]:
        return "invalid_reference_label"
    if min(a["support_count"], b["support_count"]) < 44:
        return "insufficient_support"
    for item in (a, b):
        mask = np.asarray(item.get("sensor_frame_mask", []), bool)
        if mask.shape != (64,) or not mask.all():
            return "incomplete_sensor_frames"
    ta, tb = np.asarray(a["timestamps"], np.float64), np.asarray(b["timestamps"], np.float64)
    if ta.shape != (64,) or tb.shape != (64,) or not np.isfinite(ta).all() or not np.isfinite(tb).all():
        return "invalid_time_grid"
    da, db = np.diff(ta), np.diff(tb)
    if np.any(da <= 0) or np.any(db <= 0) or np.any(da > .125+1e-12) or np.any(db > .125+1e-12):
        return "invalid_time_gap"
    duration_a, duration_b = ta[-1]-ta[0], tb[-1]-tb[0]
    if abs(duration_a-duration_b)/((duration_a+duration_b)/2) > .05:
        return "duration_mismatch"
    if np.max(np.abs((ta-ta[0])-(tb-tb[0]))) > .25*np.median(np.r_[da, db]):
        return "time_grid_mismatch"
    return None


def build_donor_pairs(candidates, *, seed=42):
    """Candidates are QA records joined with immutable window metadata.

    Matching labels/activity remain offline metadata. No model prediction is
    accepted by this function. Window reuse is prohibited across all tasks.
    """
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    subjects = sorted({r["subject_id"] for r in candidates}, key=lambda x: (sha(f"donor/{seed}/{x}"), str(x)))
    used = set()
    pairs = []
    rejections = Counter()
    eligible_count = 0
    candidate_pair_count = 0
    checked_pair_count = 0
    blocks = []
    by_subject = {}
    for row in candidates:
        by_subject.setdefault(row["subject_id"], []).append(row)
    for block_index in range(len(subjects)//2):
        sa, sb = subjects[2*block_index:2*block_index+2]
        block_id = sha(f"donor/{seed}/{sa}/{sb}")
        blocks.append({"block_id": block_id, "subjects": [sa, sb]})
        left, right = by_subject[sa], by_subject[sb]
        eligible = []
        task_left, task_right = {}, {}
        for row in left:
            task_left.setdefault((row["task_id"], row["part"]), []).append(row)
        for row in right:
            task_right.setdefault((row["task_id"], row["part"]), []).append(row)
        def distinct_recording_pairs(a_rows, b_rows):
            ac = Counter(row["recording_id"] for row in a_rows)
            bc = Counter(row["recording_id"] for row in b_rows)
            return len(a_rows)*len(b_rows)-sum(count*bc.get(key,0) for key,count in ac.items())
        def groups(rows, keys):
            result = {}
            for row in rows:
                result.setdefault(tuple(row[key] for key in keys), []).append(row)
            return result
        for task in sorted(task_left.keys() & task_right.keys()):
            ta, tb = task_left[task], task_right[task]
            candidate_pair_count += len(ta)*len(tb)
            # Preserve the exact rejection priority: same recording precedes
            # split, which precedes activity. Aggregate excluded cross-products
            # from contingency counts without enumerating their window pairs.
            if any("split" not in row or "activity_id" not in row for row in (*ta,*tb)):
                candidate_groups = [(ta, tb, False)]  # uncommon malformed-input diagnostic path
            else:
                distinct = distinct_recording_pairs(ta,tb)
                same_recording = len(ta)*len(tb)-distinct
                if same_recording:
                    rejections["same_recording_id"] += same_recording
                split_a, split_b = groups(ta,("split",)), groups(tb,("split",))
                compatible_splits = split_a.keys() & split_b.keys()
                same_split = sum(distinct_recording_pairs(split_a[k],split_b[k]) for k in compatible_splits)
                if distinct-same_split:
                    rejections["mismatched_split"] += distinct-same_split
                activity_a, activity_b = groups(ta,("split","activity_id")), groups(tb,("split","activity_id"))
                compatible_activity = activity_a.keys() & activity_b.keys()
                same_activity = sum(distinct_recording_pairs(activity_a[k],activity_b[k]) for k in compatible_activity)
                if same_split-same_activity:
                    rejections["mismatched_activity_id"] += same_split-same_activity
                candidate_groups = [(activity_a[k],activity_b[k],True) for k in sorted(compatible_activity,key=repr)]
            for a_rows,b_rows,recording_already_counted in candidate_groups:
                for a,b in product(a_rows,b_rows):
                    if recording_already_counted and a["recording_id"] == b["recording_id"]:
                        continue
                    checked_pair_count += 1
                    reason = _donor_rejection(a, b)
                    if reason:
                        rejections[reason] += 1
                        continue
                    key = (sha(f"{a['task_id']}/{a['part']}/{a['window_uid']}/{b['window_uid']}"), a["task_id"], a["part"], a["window_uid"], b["window_uid"])
                    eligible.append((key, a, b))
        eligible_count += len(eligible)
        for key, a, b in sorted(eligible, key=lambda x: x[0]):
            if a["window_uid"] in used or b["window_uid"] in used:
                rejections["window_already_used"] += 1
                continue
            used.update((a["window_uid"], b["window_uid"]))
            ta, tb = np.asarray(a["timestamps"]), np.asarray(b["timestamps"])
            elapsed_a, elapsed_b = ta-ta[0], tb-tb[0]
            canonical = (elapsed_a+elapsed_b)/2
            pairs.append({"pair_uid": key[0], "block_id": block_id, "task_id": a["task_id"], "part": a["part"],
                          "qa_uid_a": a["qa_uid"], "qa_uid_b": b["qa_uid"], "window_uid_a": a["window_uid"], "window_uid_b": b["window_uid"],
                          "subject_id_a": sa, "subject_id_b": sb, "activity_id": a["activity_id"], "answer_a": a["answer"], "answer_b": b["answer"],
                          "canonical_elapsed_times": canonical.tolist(), "max_grid_error_s": float(np.max(np.abs(elapsed_a-elapsed_b)))})
    return {"pairs": pairs, "blocks": blocks, "unpaired_subjects": subjects[2*(len(subjects)//2):],
            "eligible_candidate_count": eligible_count, "candidate_pair_count": candidate_pair_count,
            "checked_pair_count": checked_pair_count, "candidate_blocking": ["task_id", "part", "split", "activity_id"],
            "rejections": dict(sorted(rejections.items())), "policy_seed": seed}


def evaluate_donors(pairs, predictions, *, draws=5000, seed=42):
    """Map pair_uid to {a,b,text_token_hash_a,text_token_hash_b}."""
    rows = []
    missing_text_provenance = 0
    for pair in pairs:
        value = predictions.get(pair["pair_uid"], {})
        hash_a = value.get("text_token_hash_a", pair.get("text_token_hash_a"))
        hash_b = value.get("text_token_hash_b", pair.get("text_token_hash_b"))
        if hash_a and hash_b and hash_a != hash_b:
            raise ValueError("donor text-token hash divergent")
        if not hash_a or not hash_b:
            missing_text_provenance += 1
        pa, pb = _prediction(value.get("a"), pair["part"]), _prediction(value.get("b"), pair["part"])
        ca, cb = pa.status == "valid" and pa.answer == pair["answer_a"], pb.status == "valid" and pb.answer == pair["answer_b"]
        rows.append({"block_id": pair["block_id"], "a_correct": ca, "b_correct": cb,
                     "both_parse_valid": pa.status == pb.status == "valid", "both_correct": ca and cb,
                     "only_a": ca and not cb, "only_b": cb and not ca, "neither": not ca and not cb,
                     "true_transition": f"{pair['answer_a']}->{pair['answer_b']}", "pred_transition": f"{pa.answer if pa.status=='valid' else pa.status}->{pb.answer if pb.status=='valid' else pb.status}"})
    blocks = sorted({r["block_id"] for r in rows})
    report = {"pair_count": len(rows), "block_count": len(blocks), "status": "descriptive_only" if len(blocks)<5 else "block_cluster_percentile", "interval": None,
              "metrics": {key: float(np.mean([r[key] for r in rows])) if rows else None for key in ("a_correct", "b_correct", "both_parse_valid", "both_correct", "only_a", "only_b", "neither")},
              "transition_counts": dict(Counter(f"{r['true_transition']} | {r['pred_transition']}" for r in rows)),
              "missing_text_provenance_count": missing_text_provenance}
    if missing_text_provenance:
        report["status"] = "failed_text_provenance"
    if len(blocks)>=5 and not missing_text_provenance:
        rng = np.random.default_rng(seed)
        estimates = []
        for _ in range(draws):
            weights = Counter(rng.choice(blocks, size=len(blocks), replace=True).tolist())
            denominator = sum(weights[r["block_id"]] for r in rows)
            estimates.append(sum(weights[r["block_id"]]*r["both_correct"] for r in rows)/denominator)
        report["interval"] = np.quantile(estimates, [.025, .975]).tolist()
    return report


def condition_contrasts(scores):
    """Prespecified contrasts; null operands yield null, never silent zeros."""
    def contrast(*terms):
        if any(scores.get(condition) is None for _, condition in terms):
            return None
        return float(sum(coefficient*scores[condition] for coefficient, condition in terms))
    return {"R_minus_S": contrast((1, "R"), (-1, "S")),
            "F_minus_S": contrast((1, "F"), (-1, "S")),
            "R_minus_F": contrast((1, "R"), (-1, "F")),
            "E_minus_S": contrast((1, "E"), (-1, "S")),
            "interaction": contrast((1, "R"), (-1, "F"), (-1, "E"), (1, "S")),
            "R_minus_M": contrast((1, "R"), (-1, "M")),
            "R_minus_T": contrast((1, "R"), (-1, "T")),
            "R_minus_Maj": contrast((1, "R"), (-1, "Maj"))}


def make_task_probe(input_dim=512, hidden_dim=128):
    """Prespecified frozen-feature probe: Linear, GELU, Linear(12)."""
    import torch.nn as nn
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 12))


def probe_features(features):
    """Task-independent temporal mean/max, then detach before probe fitting."""
    import torch
    if features.ndim != 3 or features.shape[-1] != 256:
        raise ValueError("probe source must be [B,T,256]")
    source = features.detach()
    return torch.cat((source.mean(dim=1), source.amax(dim=1)), dim=-1)
