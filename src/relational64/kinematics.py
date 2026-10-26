"""Window-local kinematics. All reference computations use float64.

The causal derivative concerns the trajectory only; it does not make a
bidirectional radar encoder causal. No missing reference is interpolated.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
BONE_EDGES = tuple((parent, joint) for joint, parent in enumerate(PARENTS) if parent >= 0)
BILATERAL_EDGES = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21))
JOINT_NAMES = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class DerivativeOperator:
    weights: np.ndarray
    time_valid: np.ndarray
    reasons: tuple[str | None, ...]
    condition_numbers: np.ndarray

    def torch(self, *, device=None, dtype=None):
        """Return constant W for differentiable einsum('et,...tjc->...ejc', W, P)."""
        import torch
        return torch.as_tensor(self.weights, device=device, dtype=dtype or torch.float32)


@dataclass(frozen=True)
class VelocityTargets:
    velocity: np.ndarray
    valid: np.ndarray
    operator: DerivativeOperator
    reasons: np.ndarray


def derivative_operator(timestamps, *, max_gap_s: float = 0.125,
                        max_condition: float = 1e6) -> DerivativeOperator:
    """Seven-point endpoint quadratic LS derivative, scaled by support span.

    W is computed by SVD without ridge/fallback. Invalid time rows are zero;
    the first six rows always remain unavailable. A gap exactly at the limit
    is accepted (a few ulps of subtraction error are tolerated).
    """
    t = np.asarray(timestamps, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    n = len(t)
    w = np.zeros((n, n), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    cond = np.full(n, np.nan, dtype=np.float64)
    reasons: list[str | None] = ["derivative_context"] * n
    for e in range(6, n):
        support = t[e-6:e+1]
        if not np.isfinite(support).all():
            reasons[e] = "nonfinite_timestamp"
            continue
        dt = np.diff(support)
        if np.any(dt <= 0):
            reasons[e] = "nonincreasing_timestamp"
            continue
        tolerance = 8 * np.finfo(np.float64).eps * max(1., float(np.max(np.abs(support))))
        if np.any(dt > max_gap_s + tolerance):
            reasons[e] = "time_gap"
            continue
        h = support[-1] - support[0]
        x = (support - support[-1]) / h
        design = np.column_stack((np.ones(7), x, x*x))
        u, singular, vt = np.linalg.svd(design, full_matrices=False)
        rank = np.sum(singular > singular[0] * max(design.shape) * np.finfo(np.float64).eps)
        cond[e] = singular[0] / singular[-1] if singular[-1] > 0 else np.inf
        if rank != 3:
            reasons[e] = "rank_deficient_design"
            continue
        if cond[e] > max_condition:
            reasons[e] = "ill_conditioned_design"
            continue
        coefficient = ((vt.T / singular) @ u.T)[1] / h
        if not np.isfinite(coefficient).all():
            reasons[e] = "nonfinite_derivative_operator"
            continue
        w[e, e-6:e+1] = coefficient
        valid[e] = True
        reasons[e] = None
    return DerivativeOperator(w, valid, tuple(reasons), cond)


def estimate_velocities(positions, position_valid, timestamps, *, max_gap_s=0.125,
                        max_condition=1e6) -> VelocityTargets:
    """Compute GT derivatives and per-joint support masks inside one window."""
    p = np.asarray(positions, dtype=np.float64)
    mask = np.asarray(position_valid, dtype=bool)
    if p.ndim != 3 or p.shape[-1] != 3 or mask.shape != p.shape[:-1]:
        raise ValueError("expected positions [T,J,3] and position_valid [T,J]")
    if len(timestamps) != len(p):
        raise ValueError("timestamp/position length mismatch")
    op = derivative_operator(timestamps, max_gap_s=max_gap_s, max_condition=max_condition)
    clean_mask = mask & np.isfinite(p).all(axis=-1)
    clean_p = np.where(clean_mask[..., None], p, 0.)
    velocity = np.einsum("et,tjc->ejc", op.weights, clean_p)
    valid = np.zeros_like(clean_mask)
    reasons = np.full(clean_mask.shape, "derivative_context", dtype=object)
    for e in range(6, len(p)):
        if not op.time_valid[e]:
            reasons[e] = op.reasons[e]
            continue
        valid[e] = clean_mask[e-6:e+1].all(axis=0) & np.isfinite(velocity[e]).all(axis=-1)
        reasons[e] = np.where(valid[e], None, "invalid_position_support")
    velocity = np.where(valid[..., None], velocity, 0.)
    return VelocityTargets(velocity, valid, op, reasons)


def relative_velocities(velocity, valid, root_joint=0):
    velocity = np.asarray(velocity)
    valid = np.asarray(valid, dtype=bool)
    rel_valid = valid & valid[:, root_joint:root_joint+1]
    rel = velocity - velocity[:, root_joint:root_joint+1]
    return np.where(rel_valid[..., None], rel, 0.), rel_valid


def differentiate_torch(positions, operator):
    """Differentiate sensor predictions without consulting any GT mask."""
    import torch
    weights = operator.weights if isinstance(operator, DerivativeOperator) else operator
    weights = torch.as_tensor(weights, device=positions.device, dtype=positions.dtype)
    if weights.ndim == 2:
        return torch.einsum("et,...tjc->...ejc", weights, positions)
    if weights.ndim == 3 and positions.ndim == 4:
        return torch.einsum("bet,btjc->bejc", weights, positions)
    raise ValueError("W must be [T,T] or [B,T,T]")
