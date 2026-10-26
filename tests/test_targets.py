"""Synthetic software fixtures: these do not validate M4Human coordinates."""
import numpy as np
import pytest

from relational64.kinematics import derivative_operator, estimate_velocities, differentiate_torch
from relational64.qa import LabelPolicy, classify_scalar, generate_qa, targets_from_velocity, rule_predictions, fit_train_majority, audit_label_policy


def fixture_pose(t):
    return np.zeros((len(t), 22, 3), np.float64), np.ones((len(t), 22), bool)


@pytest.mark.parametrize("irregular", [False, True])
def test_constant_linear_quadratic_derivative(irregular):
    dt = np.linspace(.07, .10, 63) if irregular else np.full(63, 1/12)
    t = np.r_[0., np.cumsum(dt)]
    p, mask = fixture_pose(t)
    p[:, 0] = (2., -1., .5)
    p[:, 1] = np.column_stack((1+.3*t, 2-.2*t, .5+.1*t))
    p[:, 2, 0] = .5*t*t
    result = estimate_velocities(p, mask, t)
    assert not result.valid[:6].any()
    assert result.valid[6:].all()
    np.testing.assert_allclose(result.velocity[6:, 0], 0., atol=1e-8)
    np.testing.assert_allclose(result.velocity[6:, 1], np.tile([.3, -.2, .1], (58, 1)), atol=1e-8)
    np.testing.assert_allclose(result.velocity[6:, 2, 0], t[6:], atol=1e-8)
    np.testing.assert_allclose(result.velocity.astype(np.float32)[6:, 2, 0], t[6:], atol=1e-4)


def test_mask_is_local_and_time_gaps_have_reasons():
    t = np.arange(64, dtype=np.float64)/12
    p, mask = fixture_pose(t)
    mask[20, 2] = False
    p[20, 2] = np.nan
    result = estimate_velocities(p, mask, t)
    assert not result.valid[20:27, 2].any()
    assert result.valid[6:, 1].all()
    assert result.valid[27, 2]
    assert result.reasons[22, 2] == "invalid_position_support"
    t[32:] += .126 - 1/12
    op = derivative_operator(t)
    assert op.reasons[32] == "time_gap"
    assert not op.time_valid[32:38].any()
    assert op.time_valid[38]
    t = np.arange(64, dtype=np.float64)*.125
    assert derivative_operator(t).time_valid[6:].all()
    t[20] = t[19]
    assert derivative_operator(t).reasons[20] == "nonincreasing_timestamp"


@pytest.mark.parametrize("kind,answer", [("accelerate", "speeding_up"), ("decelerate", "slowing_down"), ("constant", "no_overall_trend")])
def test_root_theil_sen_and_mad(kind, answer):
    t = np.arange(64, dtype=np.float64)/12
    p, mask = fixture_pose(t)
    if kind == "accelerate":
        p[:, :, 0] = (.2*t+.05*t*t)[:, None]
    elif kind == "decelerate":
        p[:, :, 0] = (1.2*t-.05*t*t)[:, None]
    else:
        p[:, :, 0] = (.3*t)[:, None]
    qa = generate_qa("window", t, p, mask)
    assert qa[0]["answer"] == answer
    assert qa[0]["support_count"] == 58
    assert qa[0]["residual_scale"] < 1e-8
    assert qa[1]["answer"] == "tie"
    assert qa[2]["answer"] == "tie"


@pytest.mark.parametrize("scalar,answer", [(.05, "unknown"), (.04, "unknown"), (.06, "unknown"), (-.04, "unknown"), (-.06, "unknown"), (.039999, "no_overall_trend")])
def test_exact_root_boundary(scalar, answer):
    assert classify_scalar("root", scalar, residual_scale=0)[1] == answer
    assert classify_scalar("root", scalar, residual_scale=.151)[2] == "nonlinear_or_unstable_speed_trend"


def test_relative_limb_common_support_and_translation_invariance():
    t = np.arange(64, dtype=np.float64)/12
    p, mask = fixture_pose(t)
    p[:, [18, 20], 0] = (.2*t)[:, None]
    p[:, [19, 21], 0] = (.1*t)[:, None]
    qa = generate_qa("a", t, p, mask)
    assert qa[1]["answer"] == "faster_left"
    assert qa[1]["scalar_value"] == pytest.approx(.1)
    translated = p + np.column_stack((.4*t, .1*t*t, .05*t))[:, None, :]
    assert generate_qa("b", t, translated, mask)[1]["scalar_value"] == pytest.approx(.1)
    swapped = p.copy()
    swapped[:, [18, 20]] = p[:, [19, 21]]
    swapped[:, [19, 21]] = p[:, [18, 20]]
    assert generate_qa("c", t, swapped, mask)[1]["answer"] == "faster_right"
    velocity = np.zeros_like(p)
    velocity[:, [18, 20], 0] = .2
    velocity[:, [19, 21], 0] = .1
    valid = np.ones_like(mask)
    valid[:6] = False
    valid[6:21, 18] = False  # 43 common endpoints: all arms share the loss of support
    qa = targets_from_velocity(t, velocity, valid)
    assert qa[1]["support_count"] == 43
    assert qa[1]["reference_status"] == "undefined"
    assert qa[1]["answer"] is None and qa[1]["scalar_value"] is None
    assert qa[1]["candidate_count"] == 58
    assert qa[0]["support_count"] == qa[2]["support_count"] == 58
    valid[20, 18] = True
    qa = targets_from_velocity(t, velocity, valid)
    assert qa[1]["support_count"] == 44 and qa[1]["answer"] == "faster_left"


def test_torch_prediction_derivative_backward_without_gt_masks():
    torch = pytest.importorskip("torch")
    t = np.arange(64)/12
    op = derivative_operator(t)
    p = torch.randn(2, 64, 22, 3, requires_grad=True)
    velocity = differentiate_torch(p, op)
    assert velocity.shape == p.shape
    assert not velocity[:, :6].any()
    velocity.square().mean().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0


def test_majority_train_only_and_class_order_tiebreak():
    records = []
    for part, answers in (("root", ["speeding_up", "slowing_down"]), ("arms", ["tie", "faster_left"]), ("legs", ["faster_right", "tie"])):
        records.extend(dict(split="train", part=part, reference_status="defined", answer=answer) for answer in answers)
    assert fit_train_majority(records) == {"root": "speeding_up", "arms": "faster_left", "legs": "faster_right"}
    records[0]["split"] = "val"
    with pytest.raises(ValueError, match="only access train"):
        fit_train_majority(records)


def test_qa_hash_and_rule_oracle_diagnostic():
    t = np.arange(64)/12
    p, mask = fixture_pose(t)
    a = generate_qa("w", t, p, mask, target_hash="target", window_manifest_hash="manifest")
    b = generate_qa("w", t, p, mask, target_hash="target", window_manifest_hash="manifest")
    assert a == b
    assert len({q["qa_uid"] for q in a}) == 3
    assert [q["answer"] for q in rule_predictions("w", t, p)] == [q["answer"] for q in a]
    c = generate_qa("w", t, p, mask, policy=LabelPolicy(limb_delta_mps=.1))
    assert a[1]["qa_uid"] != c[1]["qa_uid"]


def test_label_audit_is_train_val_only_and_not_an_automatic_policy_lock():
    t = np.arange(64)/12
    p, mask = fixture_pose(t)
    records = generate_qa("w", t, p, mask, metadata={"split": "train", "subject_id": "s1"})
    report = audit_label_policy(records)
    assert report["status"] == "diagnostic_not_locked"
    assert report["tasks"]["arms"]["class_counts"] == {"tie": 1}
    assert len(report["sensitivity"]) == 15
    records[0]["split"] = "test"
    with pytest.raises(ValueError, match="train/val"):
        audit_label_policy(records)


def test_prompt_registry_matches_language_boundary_registry():
    from relational64.qa import SYSTEM_PROMPT, user_prompt
    from relational64.language import prompt_messages
    for part in ("root", "arms", "legs"):
        messages = prompt_messages(part)
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[1]["content"] == user_prompt(part)
