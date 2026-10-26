"""Evaluation regressions use synthetic labels, never held-out M4Human."""
import copy
import numpy as np
import pytest

from relational64.qa import TASKS, canonical_answer, parse_answer
from relational64.evaluation import evaluate_qa, paired_subject_bootstrap, physical_metrics, build_donor_pairs, evaluate_donors, probe_features, make_task_probe, condition_contrasts


def panel(subjects=("s1", "s2"), all_classes=True):
    refs = []
    for subject in subjects:
        for part, task in TASKS.items():
            for index, answer in enumerate(task["classes"] if all_classes else task["classes"][:1]):
                refs.append(dict(qa_uid=f"{subject}/{part}/{index}", window_uid=f"{subject}/w{index}", subject_id=subject, recording_id=f"{subject}/r", task_id=task["task_id"], part=part, answer=answer, reference_status="defined"))
    return refs


def correct(refs):
    return {r["qa_uid"]: canonical_answer(r["answer"]) for r in refs}


@pytest.mark.parametrize("text,status", [('{"answer":"tie"}', "valid"), ('{"answer":"unknown"}', "valid"), ('{"answer":"tie","extra":1}', "invalid_schema"), ('{"answer":"tie","answer":"tie"}', "invalid_json"), ('["tie"]', "invalid_schema"), ('{"answer":"tie"} extra', "invalid_json"), ('```json\n{"answer":"tie"}\n```', "invalid_json"), ('{"answer":"speeding_up"}', "invalid_label"), ('{"answer":"Tie"}', "invalid_label"), ('{"answer":null}', "invalid_schema"), ('{"answer":NaN}', "invalid_json"), ('{"answer":Infinity}', "invalid_json")])
def test_strict_parser(text, status):
    assert parse_answer(text, "arms").status == status


def test_parser_missing_truncated_and_explicit_eot():
    assert parse_answer(None, "arms").status == "missing"
    assert parse_answer('{"answer":"tie"}', "arms", truncated=True).status == "truncated"
    assert parse_answer(' {"answer":"tie"}<|im_end|> ', "arms", eot_tokens=("<|im_end|>",)).answer == "tie"
    assert parse_answer('{"answer":"tie"}', "arms", runtime_error="oom").status == "runtime_error"


def test_fixed_classes_and_missing_denominators():
    refs = panel()
    predictions = correct(refs)
    result = evaluate_qa(refs, predictions)
    assert result["S_atomic"] == 1
    absent = panel(all_classes=False)
    result = evaluate_qa(absent, correct(absent))
    assert result["S_atomic"] == pytest.approx(1/3)
    assert result["primary"]["arms"]["coverage_ceiling_macro_f1"] == pytest.approx(1/3)
    predictions.pop(refs[0]["qa_uid"])
    predictions[refs[1]["qa_uid"]] = canonical_answer("unknown")
    result = evaluate_qa(refs, predictions)
    assert result["primary"]["root"]["count"] == 6
    assert result["primary"]["root"]["accuracy"] == pytest.approx(4/6)
    assert result["generation"]["status_counts"]["missing"] == 1
    assert result["generation"]["unknown_rate"] == pytest.approx(1/18)
    no_legs = [r for r in refs if r["part"] != "legs"]
    assert evaluate_qa(no_legs, predictions)["S_atomic"] is None


def test_subject_bootstrap_is_paired_and_multiplicity_preserved():
    refs = panel()
    predictions = correct(refs)
    report = paired_subject_bootstrap(refs, predictions, predictions, draws=30)
    assert report["cluster_count"] == 2
    assert report["interval"] == [0, 0]
    assert all(x == 0 for x in report["delta_draws"])
    # Duplicating all windows of one subject retains the same cluster count.
    extra = [dict(r, qa_uid=r["qa_uid"]+"copy") for r in refs if r["subject_id"] == "s1"]
    report = paired_subject_bootstrap(refs+extra, correct(refs+extra), correct(refs+extra), draws=20)
    assert report["cluster_count"] == 2
    assert report["interval"] == [0, 0]
    one = panel(subjects=("s1",))
    assert paired_subject_bootstrap(one, correct(one), correct(one))["interval"] is None


def test_bootstrap_nonzero_matches_confusion_reduction_and_reports_null_draws():
    from collections import Counter
    refs = panel()
    good, weak = correct(refs), correct(refs)
    for row in refs:
        if row["subject_id"] == "s1":
            weak[row["qa_uid"]] = canonical_answer("unknown")
    report = paired_subject_bootstrap(refs, good, weak, draws=20, seed=9)
    expected = []
    for choices in np.random.default_rng(9).integers(2, size=(20, 2)):
        weights = Counter(["s1" if i == 0 else "s2" for i in choices])
        expected.append(evaluate_qa(refs, good, subject_weights=weights)["S_atomic"]-evaluate_qa(refs, weak, subject_weights=weights)["S_atomic"])
    np.testing.assert_allclose(report["delta_draws"], expected)
    partial = [r for r in refs if (r["subject_id"] == "s1") == (r["part"] == "root")]
    report = paired_subject_bootstrap(partial, correct(partial), correct(partial), draws=50)
    assert report["undefined_draws"] > 0
    assert report["status"] == "diagnostic_incomplete_support"


def test_physical_metrics_numerator_count_and_nonfinite_fail():
    gt = np.zeros((2, 64, 22, 3))
    pred = gt.copy()
    pred[..., 0] += .1
    mask = np.ones((2, 64, 22), bool)
    vm = mask.copy()
    vm[:, :6] = False
    velocity = np.zeros_like(gt)
    report = physical_metrics(pred, gt, mask, velocity, velocity, vm)
    assert report["status"] == "valid"
    metrics = report["metrics"]
    assert metrics["global_mpjpe"]["count"] == 2*64*22
    assert metrics["global_mpjpe"]["mean_mm"] == pytest.approx(100)
    assert metrics["relative_mpjpe"]["mean"] == pytest.approx(0)
    assert metrics["absolute_velocity_error"]["count"] == 2*58*22
    pred[0, 9, 2, 0] = np.nan
    report = physical_metrics(pred, gt, mask, velocity, velocity, vm)
    assert report["status"] == "failed_nonfinite_prediction"
    assert report["metrics"]["global_mpjpe"]["mean"] is None
    assert report["metrics"]["global_mpjpe"]["count"] == 2*64*22


def donor_candidates():
    output = []
    for i in range(4):
        for j, part in enumerate(TASKS):
            output.append(dict(qa_uid=f"q{i}{j}", window_uid=f"w{i}", subject_id=f"s{i}", recording_id=f"r{i}", activity_id="walk", split="test", joint_map_hash="map", coordinate_convention="radar", time_source="nominal_12hz", label_policy_hash="label", interval=[6, 63], support_count=58, reference_status="defined", answer=TASKS[part]["classes"][i % 3], task_id=TASKS[part]["task_id"], part=part, timestamps=(np.arange(64)/12).tolist(), sensor_frame_mask=[True]*64))
    return output


def test_donor_determinism_no_reuse_and_two_block_descriptive():
    candidates = donor_candidates()
    report = build_donor_pairs(candidates)
    assert report == build_donor_pairs(list(reversed(candidates)))
    pairs = report["pairs"]
    used = [p[key] for p in pairs for key in ("window_uid_a", "window_uid_b")]
    assert len(used) == len(set(used))
    assert len(report["blocks"]) == 2
    predictions = {p["pair_uid"]: {"a": canonical_answer(p["answer_a"]), "b": canonical_answer(p["answer_b"]), "text_token_hash_a": "same", "text_token_hash_b": "same"} for p in pairs}
    result = evaluate_donors(pairs, predictions)
    assert result["status"] == "descriptive_only" and result["interval"] is None
    assert result["metrics"]["both_correct"] == 1
    missing = evaluate_donors(pairs, {})
    assert missing["metrics"]["both_correct"] == 0
    assert missing["pair_count"] == len(pairs)
    assert missing["status"] == "failed_text_provenance"
    pair = pairs[0]
    predictions[pair["pair_uid"]]["b"] = None
    assert evaluate_donors(pairs, predictions)["metrics"]["both_correct"] < 1
    predictions[pair["pair_uid"]]["text_token_hash_b"] = "different"
    with pytest.raises(ValueError, match="hash"):
        evaluate_donors(pairs, predictions)


@pytest.mark.parametrize("field,value", [("recording_id", "same"), ("answer", "tie"), ("time_source", "other"), ("sensor_frame_mask", [False]*64)])
def test_donor_constraints(field, value):
    candidates = donor_candidates()
    for record in candidates:
        if field in ("recording_id", "answer", "sensor_frame_mask"):
            record[field] = value
        elif field == "time_source":
            record[field] = record["subject_id"]
        elif record["subject_id"] in ("s0", "s2"):
            record[field] = value
    result = build_donor_pairs(candidates)
    assert not result["pairs"]


def test_probe_mean_max_and_frozen_source():
    torch = pytest.importorskip("torch")
    features = torch.arange(2*64*256, dtype=torch.float32).reshape(2, 64, 256).requires_grad_()
    pooled = probe_features(features)
    assert pooled.shape == (2, 512) and not pooled.requires_grad
    assert torch.equal(pooled[:, :256], features.detach().mean(1))
    assert torch.equal(pooled[:, 256:], features.detach().amax(1))
    probe = make_task_probe()
    logits = probe(pooled)
    assert logits.shape == (2, 12)
    logits.square().mean().backward()
    assert features.grad is None


def test_prespecified_contrasts_and_interaction():
    result = condition_contrasts({"S": .3, "F": .4, "E": .35, "R": .5, "M": .2})
    assert result["R_minus_S"] == pytest.approx(.2)
    assert result["interaction"] == pytest.approx(.05)
    assert result["R_minus_T"] is None
