"""Synthetic software fixtures; these are not M4Human pairing validation."""
import torch
import pytest

from relational64.models import (BONE_EDGES, BILATERAL_EDGES, EDGES, DescriptorNormalizer,
    RadarPoseModel, SensorProjector, TaskProbe, adamw_parameter_groups,
    build_descriptors, continuous_rope, freeze_for_inference, parameter_hash)


def sensor_fixture(batch=1, points=9):
    torch.manual_seed(3)
    x = torch.randn(batch, 64, points, 4)
    mask = torch.ones(batch, 64, points, dtype=torch.bool)
    if points > 8:
        mask[:, :, 8:] = False
    time = torch.arange(64, dtype=torch.float64).expand(batch, -1) / 12
    return x, mask, time


def state_fixture(batch=2):
    torch.manual_seed(7)
    p, v = torch.randn(batch, 64, 22, 3), torch.randn(batch, 64, 22, 3)
    available = torch.ones(batch, 64, dtype=torch.bool)
    available[:, :6] = False
    return p, v, available


@pytest.mark.parametrize("batch", [1, 2])
def test_encoder_shape_padding_and_point_permutation(batch):
    model = RadarPoseModel().eval()
    x, mask, time = sensor_fixture(batch)
    with torch.no_grad():
        first = model(x, mask, time)
        changed = x.clone()
        changed[~mask] = float("nan")
        padded = model(changed, mask, time)
        permutation = torch.randperm(x.shape[2])
        permuted = model(x[:, :, permutation], mask[:, :, permutation], time)
    assert first.features.shape == (batch, 64, 256)
    assert first.positions.shape == (batch, 64, 22, 3)
    torch.testing.assert_close(first.features, padded.features, atol=0, rtol=0)
    torch.testing.assert_close(first.positions, permuted.positions, atol=1e-5, rtol=1e-5)
    assert len(model.encoder.spatial) == 3 and len(model.encoder.temporal) == 8


def test_encoder_backward_and_frozen_determinism():
    model = RadarPoseModel()
    x, mask, time = sensor_fixture(points=8)
    result = model(x, mask, time)
    result.positions.square().mean().backward()
    for part in (model.encoder.point_embedding[0], model.pose_head[-1]):
        assert part.weight.grad is not None
        assert torch.isfinite(part.weight.grad).all() and part.weight.grad.norm() > 0
    freeze_for_inference(model)
    with torch.no_grad():
        one, two = model(x, mask, time), model(x, mask, time)
    torch.testing.assert_close(one.features, two.features, atol=0, rtol=0)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_rope_nominal_actual_time_and_translation():
    q, k = torch.randn(2, 4, 64, 64), torch.randn(2, 4, 64, 64)
    time = torch.arange(64, dtype=torch.float64).expand(2, -1) / 12
    rotated_q, rotated_k = continuous_rope(q, k, time)
    phase = torch.arange(64)[:, None] / (10000 ** (torch.arange(0, 64, 2) / 64))
    expected = q.clone()
    expected[..., 0::2] = q[..., 0::2] * phase.cos() - q[..., 1::2] * phase.sin()
    expected[..., 1::2] = q[..., 0::2] * phase.sin() + q[..., 1::2] * phase.cos()
    torch.testing.assert_close(rotated_q, expected)
    shifted_q, _ = continuous_rope(q, k, time + 1000)
    torch.testing.assert_close(rotated_q, shifted_q, atol=1e-5, rtol=1e-5)
    changed = time.clone()
    changed[:, 7:] += 0.01
    actual_q, _ = continuous_rope(q, k, changed)
    assert not torch.allclose(actual_q, rotated_q)
    assert rotated_k.shape == k.shape


def test_sensor_input_validation():
    model = RadarPoseModel()
    x, mask, time = sensor_fixture()
    mask[:, 0, 0] = False
    with pytest.raises(ValueError, match="8..150"):
        model(x, mask, time)
    x, mask, time = sensor_fixture()
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        model(x, mask, time)


def test_descriptor_conditions_channels_and_geometry():
    p, v, available = state_fixture()
    data = {condition: build_descriptors(p, v, available, condition) for condition in "SFER"}
    assert len(BONE_EDGES) == 21 and len(BILATERAL_EDGES) == 8 and len(EDGES) == 29
    assert all(i != j and 0 <= i < 22 and 0 <= j < 22 for i, j in EDGES)
    for batch in data.values():
        assert batch.nodes.shape == (2, 64, 22, 7)
        assert batch.edges.shape == (2, 64, 29, 14)
    assert torch.equal(data["S"].edges, data["F"].edges)
    assert torch.equal(data["S"].nodes, data["E"].nodes)
    assert torch.equal(data["F"].nodes, data["R"].nodes)
    assert torch.equal(data["E"].edges, data["R"].edges)
    torch.testing.assert_close(data["R"].nodes[..., 0, :3], p[..., 0, :])
    torch.testing.assert_close(data["R"].nodes[..., 1:, :3], p[..., 1:, :] - p[..., :1, :])
    i, j = EDGES[0]
    torch.testing.assert_close(data["R"].edges[..., 0, :3], p[..., j, :] - p[..., i, :])
    torch.testing.assert_close(data["R"].edges[:, 6:, 0, 3:6], v[:, 6:, j] - v[:, 6:, i])
    torch.testing.assert_close(data["S"].edges[:, 6:, 0, 7:10], p[:, 6:, j])


@pytest.mark.parametrize("condition", list("SFER"))
def test_normalizer_train_only_and_warmup_zero_after_normalization(condition):
    p, v, available = state_fixture()
    descriptor = build_descriptors(p, v, available, condition)
    normalizer = DescriptorNormalizer(condition, "checkpoint_hash", "train_manifest_hash")
    with pytest.raises(ValueError, match="only access train"):
        normalizer.fit([("val", descriptor)])
    normalizer.fit([("train", descriptor)])
    transformed = normalizer(descriptor)
    assert torch.equal(transformed.nodes[:, :6, :, 3:], torch.zeros(2, 6, 22, 4))
    assert (normalizer.node_count[:, :3] == 128).all()
    assert (normalizer.node_count[:, 3:] == 116).all()
    assert torch.equal(transformed.edges[~descriptor.edge_channel_available], torch.zeros_like(transformed.edges[~descriptor.edge_channel_available]))
    assert transformed.edges[:, :6, :, :3].abs().sum() > 0
    assert normalizer.artifact()["source_checkpoint_hash"] == "checkpoint_hash"
    with pytest.raises(RuntimeError, match="frozen"):
        normalizer.fit([("train", descriptor)])


def test_explicit_alignment_zero_and_limb_translation_invariance():
    p, v, available = state_fixture(1)
    zero = build_descriptors(torch.zeros_like(p), torch.zeros_like(v), available, "R")
    assert torch.isfinite(zero.edges).all() and (zero.edges[..., 10] == 0).all()
    first = build_descriptors(p, v, available, "R")
    translation, drift = torch.randn(1, 64, 1, 3), torch.randn(1, 64, 1, 3)
    translated = build_descriptors(p + translation, v + drift, available, "R")
    torch.testing.assert_close(first.nodes[..., 1:, :], translated.nodes[..., 1:, :], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(first.edges, translated.edges, atol=1e-5, rtol=1e-5)


def test_projector_exact_counts_initialization_equality_and_shapes():
    projectors = {}
    for condition in "MSFER":
        torch.manual_seed(42)
        projectors[condition] = SensorProjector(condition)
    assert sum(p.numel() for p in projectors["M"].parameters()) == 2_761_728
    for condition in "SFER":
        assert sum(p.numel() for p in projectors[condition].parameters()) == 3_003_392
        assert parameter_hash(projectors[condition]) == parameter_hash(projectors["S"])
        for name, weight in projectors["M"].language_mapping.state_dict().items():
            assert torch.equal(weight, projectors[condition].language_mapping.state_dict()[name])
    p, v, available = state_fixture(1)
    for condition, projector in projectors.items():
        descriptor = None if condition == "M" else build_descriptors(p, v, available, condition)
        output = projector(torch.randn(1, 64, 256), descriptor)
        assert output.fused.shape == (1, 64, 256)
        assert output.sensor_embeddings.shape == (1, 64, 1536)
        output.sensor_embeddings.square().mean().backward()
        assert all(parameter.grad is not None for parameter in projector.parameters())
    groups = adamw_parameter_groups(projectors["R"])
    assert groups[0]["weight_decay"] == 0.01 and groups[1]["weight_decay"] == 0
    assert sum(len(group["params"]) for group in groups) == len(list(projectors["R"].parameters()))
    assert TaskProbe()(torch.randn(2, 64, 256)).shape == (2, 3, 4)
