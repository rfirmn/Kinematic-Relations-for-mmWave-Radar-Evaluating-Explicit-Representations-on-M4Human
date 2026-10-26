"""Pure PyTorch sensor architecture for the relational64 specification.

This is the specified mmMind-based encoder adaptation, not a claim of an exact
reproduction of every unpublished training or implementation detail of mmMind.
Only sensor points, their mask and sensor time enter the physical model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
BONE_EDGES = tuple((parent, child) for child, parent in enumerate(PARENTS) if parent >= 0)
BILATERAL_EDGES = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21))
EDGES = BONE_EDGES + BILATERAL_EDGES
STRUCTURED_CONDITIONS = frozenset("SFER")


def continuous_rope(q: Tensor, k: Tensor, time_s: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate Q/K [B,H,T,Dh] from measured seconds (never rotate values)."""
    if q.shape != k.shape or q.shape[-1] % 2:
        raise ValueError("RoPE requires equal Q/K shapes and an even head width")
    if time_s.dtype != torch.float64 or time_s.shape != (q.shape[0], q.shape[2]):
        raise ValueError("time_s must be float64 [B,T]")
    if not torch.isfinite(time_s).all() or not (time_s.diff(dim=1) > 0).all():
        raise ValueError("sensor timestamps must be finite and strictly increasing")
    # Subtract in float64, then compute phase at least in float32 under AMP.
    elapsed = (12.0 * (time_s - time_s[:, :1])).float()
    denominator = 10000.0 ** (torch.arange(0, q.shape[-1], 2, device=q.device, dtype=torch.float32) / q.shape[-1])
    phase = elapsed[:, None, :, None] / denominator
    cosine, sine = phase.cos(), phase.sin()

    def rotate(x: Tensor) -> Tensor:
        a, b = x.float()[..., 0::2], x.float()[..., 1::2]
        return torch.stack((a * cosine - b * sine, a * sine + b * cosine), dim=-1).flatten(-2).to(x.dtype)

    return rotate(q), rotate(k)


class SensorAttention(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads, self.head_dim, self.dropout = heads, dim // heads, dropout
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, query: Tensor, context: Tensor, key_mask: Tensor | None = None,
                time_s: Tensor | None = None) -> Tensor:
        def heads(x: Tensor) -> Tensor:
            return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(self.query(query)), heads(self.key(context)), heads(self.value(context))
        if time_s is not None:
            q, k = continuous_rope(q, k, time_s)
        mask = None if key_mask is None else key_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                            dropout_p=self.dropout if self.training else 0.0)
        return self.output(out.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1))


class SensorBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(256, eps=1e-5)
        self.attention = SensorAttention()
        self.norm2 = nn.LayerNorm(256, eps=1e-5)
        self.ffn = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 256))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor, mask: Tensor | None = None, time_s: Tensor | None = None) -> Tensor:
        def zero_padding(value: Tensor) -> Tensor:
            return value if mask is None else value.masked_fill(~mask[..., None], 0)
        normed = self.norm1(x)
        x = zero_padding(x + self.dropout(self.attention(normed, normed, mask, time_s)))
        return zero_padding(x + self.dropout(self.ffn(self.norm2(x))))


class RadarEncoder(nn.Module):
    """3 spatial + 8 temporal pre-LN blocks, 256 dimensions and 4 heads."""
    def __init__(self):
        super().__init__()
        self.point_embedding = nn.Sequential(nn.Linear(4, 256), nn.GELU(), nn.Linear(256, 256))
        self.position_embedding = nn.Sequential(nn.Linear(3, 256), nn.GELU(), nn.Linear(256, 256))
        self.spatial = nn.ModuleList([SensorBlock() for _ in range(3)])
        self.pool_norm = nn.LayerNorm(256, eps=1e-5)
        self.pool_query = nn.Parameter(torch.empty(1, 1, 256))
        nn.init.normal_(self.pool_query, mean=0, std=0.02)
        self.pool_attention = SensorAttention()
        self.pool_output_norm = nn.LayerNorm(256, eps=1e-5)
        self.temporal = nn.ModuleList([SensorBlock() for _ in range(8)])
        self.output_norm = nn.LayerNorm(256, eps=1e-5)

    def forward(self, points: Tensor, point_mask: Tensor, time_s: Tensor) -> Tensor:
        if points.ndim != 4 or points.shape[1] != 64 or points.shape[-1] != 4:
            raise ValueError("points must have shape [B,64,N,4]")
        if point_mask.shape != points.shape[:-1] or point_mask.dtype != torch.bool:
            raise ValueError("point_mask must be bool [B,64,N]")
        if not (8 <= points.shape[2] <= 150) or (point_mask.sum(-1) < 8).any():
            raise ValueError("each frame needs 8..150 observed points")
        if not torch.isfinite(points[point_mask]).all():
            raise ValueError("observed points contain nonfinite channels")
        if time_s.shape != points.shape[:2] or time_s.dtype != torch.float64:
            raise ValueError("time_s must be float64 [B,64]")
        if not torch.isfinite(time_s).all() or not (time_s.diff(dim=1) > 0).all():
            raise ValueError("sensor time must be finite and increasing")
        b, t, n, _ = points.shape
        # Mask before either MLP so even nonfinite padded values cannot propagate.
        clean = points.masked_fill(~point_mask[..., None], 0)
        x = self.point_embedding(clean) + self.position_embedding(clean[..., :3])
        x = x.masked_fill(~point_mask[..., None], 0).reshape(b * t, n, 256)
        spatial_mask = point_mask.reshape(b * t, n)
        for block in self.spatial:
            x = block(x, spatial_mask)
        query = self.pool_query.expand(b * t, -1, -1)
        x = self.pool_output_norm(query + self.pool_attention(query, self.pool_norm(x), spatial_mask))
        x = x.reshape(b, t, 256)
        for block in self.temporal:
            x = block(x, time_s=time_s)
        return self.output_norm(x)


@dataclass
class RadarOutput:
    features: Tensor
    positions: Tensor


class RadarPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RadarEncoder()
        self.pose_head = nn.Sequential(nn.LayerNorm(256, eps=1e-5), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 66))

    def forward(self, points: Tensor, point_mask: Tensor, time_s: Tensor) -> RadarOutput:
        features = self.encoder(points, point_mask, time_s)
        return RadarOutput(features, self.pose_head(features).reshape(*features.shape[:2], 22, 3))


def freeze_for_inference(module: nn.Module) -> nn.Module:
    """Freeze weights and disable dropout; callers must keep physical caches eval."""
    module.requires_grad_(False)
    module.eval()
    return module


@dataclass
class DescriptorBatch:
    nodes: Tensor
    edges: Tensor
    node_channel_available: Tensor
    edge_channel_available: Tensor
    velocity_available: Tensor
    condition: str


def build_descriptors(positions: Tensor, velocities: Tensor, velocity_available: Tensor,
                      condition: str) -> DescriptorBatch:
    """Continuous descriptors only: no GT status, question, threshold or answer."""
    if condition not in STRUCTURED_CONDITIONS:
        raise ValueError("descriptors require S, F, E or R")
    if positions.shape != velocities.shape or positions.shape[-2:] != (22, 3):
        raise ValueError("positions and velocities must both be [B,T,22,3]")
    if velocity_available.shape != positions.shape[:2] or velocity_available.dtype != torch.bool:
        raise ValueError("velocity_available must be sensor-derived bool [B,T]")
    # Physical state remains float32 even when a caller enables AMP.
    p, v = positions.float(), velocities.float()
    v = v.masked_fill(~velocity_available[..., None, None], 0)
    q = p - p[..., :1, :]
    w = v - v[..., :1, :]
    absolute = torch.cat((p, v, v.norm(dim=-1, keepdim=True)), dim=-1)
    local = torch.cat((q, w, w.norm(dim=-1, keepdim=True)), dim=-1)
    nodes = absolute if condition in "SE" else torch.cat((absolute[..., :1, :], local[..., 1:, :]), dim=-2)
    indices = torch.tensor(EDGES, device=p.device)
    i, j = indices[:, 0], indices[:, 1]
    if condition in "SF":
        edges = torch.cat((absolute[..., i, :], absolute[..., j, :]), dim=-1)
        velocity_channels = (3, 4, 5, 6, 10, 11, 12, 13)
    else:
        delta, relative = p[..., j, :] - p[..., i, :], v[..., j, :] - v[..., i, :]
        length, speed = delta.norm(dim=-1, keepdim=True), relative.norm(dim=-1, keepdim=True)
        denominator = length * speed
        alignment = (delta * relative).sum(-1, keepdim=True) / denominator.clamp_min(1e-12)
        alignment = torch.where((length >= 1e-6) & (speed >= 1e-6), alignment.clamp(-1, 1), 0)
        edges = torch.cat((delta, relative, length, speed, w[..., i, :].norm(dim=-1, keepdim=True),
                           w[..., j, :].norm(dim=-1, keepdim=True), alignment,
                           (q[..., i, :] + q[..., j, :]) / 2), dim=-1)
        velocity_channels = (3, 4, 5, 7, 8, 9, 10)
    node_available = torch.ones_like(nodes, dtype=torch.bool)
    edge_available = torch.ones_like(edges, dtype=torch.bool)
    node_available[..., 3:7] = velocity_available[..., None, None]
    edge_available[..., list(velocity_channels)] = velocity_available[..., None, None]
    return DescriptorBatch(nodes, edges, node_available, edge_available, velocity_available, condition)


class DescriptorNormalizer(nn.Module):
    """Role/channel Welford statistics fitted exclusively on train predictions."""
    def __init__(self, condition: str, source_checkpoint_hash: str, train_manifest_hash: str):
        super().__init__()
        if condition not in STRUCTURED_CONDITIONS or not source_checkpoint_hash or not train_manifest_hash:
            raise ValueError("condition and both provenance hashes are required")
        self.condition = condition
        self.source_checkpoint_hash, self.train_manifest_hash = source_checkpoint_hash, train_manifest_hash
        for role, shape in (("node", (22, 7)), ("edge", (29, 14))):
            self.register_buffer(role + "_mean", torch.zeros(shape, dtype=torch.float64))
            self.register_buffer(role + "_std", torch.ones(shape, dtype=torch.float64))
            self.register_buffer(role + "_count", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, batches: Iterable[tuple[str, DescriptorBatch]]) -> "DescriptorNormalizer":
        if self.fitted.item():
            raise RuntimeError("normalizer is frozen once fitted")
        accum = {role: [torch.zeros_like(getattr(self, role + "_mean")) for _ in range(3)] for role in ("node", "edge")}
        for split, batch in batches:
            if split != "train":
                raise ValueError("descriptor statistics may only access train predictions")
            if batch.condition != self.condition:
                raise ValueError("condition mismatch")
            for role, values, available in (("node", batch.nodes, batch.node_channel_available),
                                            ("edge", batch.edges, batch.edge_channel_available)):
                x, mask = values.detach().to(device="cpu", dtype=torch.float64), available.detach().cpu()
                if not torch.isfinite(x[mask]).all():
                    raise ValueError("nonfinite observed descriptor")
                n = mask.sum((0, 1)).double()
                total = torch.where(mask, x, 0).sum((0, 1))
                mean = total / n.clamp_min(1)
                m2 = torch.where(mask, (x - mean).square(), 0).sum((0, 1))
                old_n, old_mean, old_m2 = accum[role]
                combined = old_n + n
                delta = mean - old_mean
                old_mean += delta * n / combined.clamp_min(1)
                old_m2 += m2 + delta.square() * old_n * n / combined.clamp_min(1)
                old_n += n
        for role, (n, mean, m2) in accum.items():
            if (n == 0).any():
                raise ValueError("normalizer has an unobserved role/channel")
            getattr(self, role + "_count").copy_(n)
            getattr(self, role + "_mean").copy_(mean)
            getattr(self, role + "_std").copy_((m2 / n).sqrt().clamp_min(1e-6))
        self.fitted.fill_(True)
        return self

    def forward(self, batch: DescriptorBatch) -> DescriptorBatch:
        if not self.fitted.item() or batch.condition != self.condition:
            raise ValueError("unfitted normalizer or condition mismatch")
        def normalize(role: str, x: Tensor, available: Tensor) -> Tensor:
            mean, std = getattr(self, role + "_mean"), getattr(self, role + "_std")
            result = ((x.float() - mean.to(x.device).float()) / std.to(x.device).float()).clamp(-10, 10)
            return result.masked_fill(~available, 0)
        return DescriptorBatch(normalize("node", batch.nodes, batch.node_channel_available),
                               normalize("edge", batch.edges, batch.edge_channel_available),
                               batch.node_channel_available, batch.edge_channel_available,
                               batch.velocity_available, batch.condition)

    def artifact(self) -> dict:
        return {"condition": self.condition, "source_checkpoint_hash": self.source_checkpoint_hash,
                "train_manifest_hash": self.train_manifest_hash, "variance": "population",
                "availability_map": {"node_velocity": [3, 4, 5, 6],
                    "edge_velocity": [3, 4, 5, 6, 10, 11, 12, 13] if self.condition in "SF" else [3, 4, 5, 7, 8, 9, 10]},
                **{name: value.cpu().tolist() for name, value in self.state_dict().items()}}


@dataclass
class ProjectionOutput:
    fused: Tensor
    sensor_embeddings: Tensor


class SensorProjector(nn.Module):
    def __init__(self, condition: str, language_dim: int = 1536):
        super().__init__()
        if condition not in "MSFER" or len(condition) != 1:
            raise ValueError("projector condition must be M, S, F, E or R")
        self.condition = condition
        # Construct language mapping first so its initial weights match M at equal seed.
        self.language_mapping = nn.Sequential(nn.Linear(256, language_dim), nn.LayerNorm(language_dim, eps=1e-5),
                                             nn.GELU(), nn.Linear(language_dim, language_dim), nn.LayerNorm(language_dim, eps=1e-5))
        if condition != "M":
            self.node_mlp = nn.Sequential(nn.Linear(7, 128), nn.GELU(), nn.Linear(128, 128))
            self.edge_mlp = nn.Sequential(nn.Linear(14, 128), nn.GELU(), nn.Linear(128, 128))
            self.joint_id = nn.Embedding(22, 128)
            self.edge_id = nn.Embedding(29, 128)
            self.edge_type = nn.Embedding(2, 128)
            self.node_availability = nn.Embedding(2, 128)
            self.edge_availability = nn.Embedding(2, 128)
            for embedding in (self.joint_id, self.edge_id, self.edge_type, self.node_availability, self.edge_availability):
                nn.init.normal_(embedding.weight, mean=0, std=0.02)
            self.node_norm, self.edge_norm = nn.LayerNorm(128, eps=1e-5), nn.LayerNorm(128, eps=1e-5)
            self.node_query, self.edge_query = nn.Parameter(torch.empty(128)), nn.Parameter(torch.empty(128))
            nn.init.normal_(self.node_query, std=0.02)
            nn.init.normal_(self.edge_query, std=0.02)
            self.fusion = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 256))
            self.fusion_norm = nn.LayerNorm(256, eps=1e-5)
            self.register_buffer("edge_types", torch.tensor([0] * 21 + [1] * 8), persistent=False)

    def forward(self, features: Tensor, descriptors: DescriptorBatch | None = None) -> ProjectionOutput:
        if features.ndim != 3 or features.shape[1:] != (64, 256):
            raise ValueError("features must be [B,64,256]")
        if self.condition == "M":
            if descriptors is not None:
                raise ValueError("M receives Y only, without pose/velocity descriptors")
            fused = features
        else:
            if descriptors is None or descriptors.condition != self.condition:
                raise ValueError("matching normalized descriptors are required")
            available = descriptors.velocity_available.long()
            node = self.node_norm(self.node_mlp(descriptors.nodes) + self.joint_id.weight + self.node_availability(available)[..., None, :])
            edge = self.edge_norm(self.edge_mlp(descriptors.edges) + self.edge_id.weight + self.edge_type(self.edge_types)
                                  + self.edge_availability(available)[..., None, :])
            def pool(x: Tensor, query: Tensor) -> Tensor:
                weights = torch.softmax((x * query).sum(-1) / math.sqrt(128), dim=-1)
                return (x * weights[..., None]).sum(-2)
            fused = self.fusion_norm(features + self.fusion(torch.cat((features, pool(node, self.node_query), pool(edge, self.edge_query)), dim=-1)))
        return ProjectionOutput(fused, self.language_mapping(fused))


class TaskProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 12))

    def forward(self, source: Tensor) -> Tensor:
        if source.shape[1:] != (64, 256):
            raise ValueError("probe source must be [B,64,256]")
        return self.network(torch.cat((source.mean(1), source.amax(1)), -1)).reshape(-1, 3, 4)


def parameter_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        flat = tensor.detach().contiguous().reshape(-1)
        # Stream large frozen LLM tensors rather than copying a whole embedding
        # table to host RAM for each comparison.
        for chunk in flat.split(262144):
            digest.update(chunk.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def adamw_parameter_groups(module: nn.Module, weight_decay: float = 0.01) -> list[dict]:
    """Apply decay only to Linear weights; all bias/LN/embedding/query avoid it."""
    linear_ids = {id(part.weight) for part in module.modules() if isinstance(part, nn.Linear)}
    decay, no_decay = [], []
    for parameter in module.parameters():
        if parameter.requires_grad:
            (decay if id(parameter) in linear_ids else no_decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
