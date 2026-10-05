from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .geometry import planar_rotation


@dataclass(frozen=True)
class PlanarSupportPose:
    position: np.ndarray
    angle: float


@dataclass(frozen=True)
class PlanarEquivarianceProbe:
    support_pose: PlanarSupportPose
    action_points: np.ndarray
    label: str = ""


@dataclass(frozen=True)
class FiniteEquivarianceCertificate:
    """Finite black-box support-equivariance certificate for one action chunk.

    Acceptance is position-wise.  A position is authorized only when the
    observed frozen-policy action under every finite support probe agrees with
    rigid support transport and when the observed probes are mutually
    consistent under pairwise support-to-support transport.
    """

    accepted: bool
    accepted_mask: np.ndarray
    direct_max_relative_residual: np.ndarray
    pairwise_max_relative_residual: np.ndarray
    num_probes: int
    num_pairwise_checks: int
    max_relative_residual: float
    reason: str


def _pose(pose: PlanarSupportPose) -> tuple[np.ndarray, float]:
    position = np.asarray(pose.position, dtype=float)
    if position.shape != (2,):
        raise ValueError("planar support position must have shape (2,)")
    if not np.all(np.isfinite(position)) or not np.isfinite(pose.angle):
        raise ValueError("support pose must be finite")
    return position, float(pose.angle)


def transport_points_between_planar_supports(
    action_points: np.ndarray,
    *,
    source_pose: PlanarSupportPose,
    target_pose: PlanarSupportPose,
) -> np.ndarray:
    """Rigidly transport world-frame action points between support poses."""
    points = np.asarray(action_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("action_points must have shape [H, 2]")
    source_position, source_angle = _pose(source_pose)
    target_position, target_angle = _pose(target_pose)

    local = (planar_rotation(-source_angle) @ (points - source_position).T).T
    return target_position + (planar_rotation(target_angle) @ local.T).T


def _relative_point_residual(
    observed: np.ndarray,
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    error = np.linalg.norm(observed - predicted, axis=1)
    predicted_motion = np.linalg.norm(predicted - reference, axis=1)
    observed_motion = np.linalg.norm(observed - reference, axis=1)
    scale = np.maximum(np.maximum(predicted_motion, observed_motion), eps)
    return error / scale


def certify_finite_planar_equivariance(
    *,
    baseline_action_points: np.ndarray,
    baseline_support_pose: PlanarSupportPose,
    probes: list[PlanarEquivarianceProbe] | tuple[PlanarEquivarianceProbe, ...],
    max_relative_residual: float = 0.20,
    min_motion_scale: float = 1.0,
) -> FiniteEquivarianceCertificate:
    """Certify finite support-relative action equivariance from black-box probes.

    This is deliberately a *finite* test.  It does not extrapolate a local
    Jacobian.  The same-randomness frozen policy must already exhibit the
    rigid-support group action at multiple non-zero transforms.

    The threshold reuses the 0.20 rigid-geometry residual used by the existing
    CASJ anchor gate rather than tuning a new value from public outcomes.
    """
    baseline = np.asarray(baseline_action_points, dtype=float)
    if baseline.ndim != 2 or baseline.shape[1] != 2 or len(baseline) == 0:
        raise ValueError("baseline_action_points must have non-empty shape [H, 2]")
    if not np.all(np.isfinite(baseline)):
        raise ValueError("baseline action points must be finite")
    if not np.isfinite(max_relative_residual) or max_relative_residual < 0:
        raise ValueError("max_relative_residual must be finite and non-negative")
    if not np.isfinite(min_motion_scale) or min_motion_scale <= 0:
        raise ValueError("min_motion_scale must be finite and positive")
    _pose(baseline_support_pose)

    rows = tuple(probes)
    if len(rows) < 2:
        raise ValueError("at least two distinct finite probes are required")

    direct_residuals = []
    for probe in rows:
        observed = np.asarray(probe.action_points, dtype=float)
        if observed.shape != baseline.shape:
            raise ValueError("every probe action chunk must match baseline shape")
        if not np.all(np.isfinite(observed)):
            raise ValueError("probe action points must be finite")
        predicted = transport_points_between_planar_supports(
            baseline,
            source_pose=baseline_support_pose,
            target_pose=probe.support_pose,
        )
        direct_residuals.append(
            _relative_point_residual(
                observed,
                predicted,
                baseline,
                eps=min_motion_scale,
            )
        )
    direct_max = np.max(np.stack(direct_residuals, axis=0), axis=0)

    pairwise_residuals = []
    for left_index, right_index in combinations(range(len(rows)), 2):
        left = rows[left_index]
        right = rows[right_index]
        left_actions = np.asarray(left.action_points, dtype=float)
        right_actions = np.asarray(right.action_points, dtype=float)
        predicted_right = transport_points_between_planar_supports(
            left_actions,
            source_pose=left.support_pose,
            target_pose=right.support_pose,
        )
        pairwise_residuals.append(
            _relative_point_residual(
                right_actions,
                predicted_right,
                left_actions,
                eps=min_motion_scale,
            )
        )

    pairwise_max = np.max(np.stack(pairwise_residuals, axis=0), axis=0)
    accepted_mask = np.logical_and(
        direct_max <= max_relative_residual,
        pairwise_max <= max_relative_residual,
    )
    accepted = bool(np.all(accepted_mask))
    if accepted:
        reason = (
            "all action positions satisfy finite support-equivariance and "
            "pairwise closure across the frozen probe set"
        )
    else:
        reason = (
            "one or more action positions violate finite support-equivariance "
            "or cross-probe closure"
        )
    return FiniteEquivarianceCertificate(
        accepted=accepted,
        accepted_mask=accepted_mask,
        direct_max_relative_residual=direct_max,
        pairwise_max_relative_residual=pairwise_max,
        num_probes=len(rows),
        num_pairwise_checks=len(rows) * (len(rows) - 1) // 2,
        max_relative_residual=float(max_relative_residual),
        reason=reason,
    )
