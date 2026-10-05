from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanarRigidAnchorFingerprint:
    """Rigid point-action fingerprint for a planar support pose.

    The support intervention chart is (x, y, theta): translations move the
    support center in world coordinates and theta rotates the support about its
    own center. For an action target rigidly attached at local offset r,

        a = p + R(theta) r,

    the ideal action-support Jacobian is

        [ 1  0  -v_y ]
        [ 0  1   v_x ],

    where v = a - p is the world-frame lever arm.
    """

    accepted: bool
    anchor_world: np.ndarray
    anchor_local: np.ndarray
    predicted_jacobian: np.ndarray
    translation_residual: float
    anchor_consistency_residual: float
    jacobian_residual: float
    reason: str


def planar_rotation(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=float)


def planar_point_jacobian(anchor_world: np.ndarray) -> np.ndarray:
    """Ideal d(action_xy) / d(support_x, support_y, support_theta)."""
    v = np.asarray(anchor_world, dtype=float)
    if v.shape != (2,):
        raise ValueError("anchor_world must have shape (2,)")
    return np.array(
        [
            [1.0, 0.0, -v[1]],
            [0.0, 1.0, v[0]],
        ],
        dtype=float,
    )


def infer_planar_rigid_anchor(
    casj_block: np.ndarray,
    *,
    support_position: np.ndarray,
    support_angle: float,
    action_point: np.ndarray,
    max_translation_residual: float = 0.20,
    max_anchor_consistency_residual: float = 0.20,
    max_jacobian_residual: float = 0.20,
) -> PlanarRigidAnchorFingerprint:
    """Recover and verify a latent support-relative point anchor from CASJ.

    casj_block must map the physical intervention coordinates
    (delta x, delta y, delta theta) to the decoded 2D action target.
    """
    block = np.asarray(casj_block, dtype=float)
    support = np.asarray(support_position, dtype=float)
    action = np.asarray(action_point, dtype=float)
    if block.shape != (2, 3):
        raise ValueError("casj_block must have shape (2, 3)")
    if support.shape != (2,) or action.shape != (2,):
        raise ValueError("support_position and action_point must have shape (2,)")

    translation_residual = float(np.linalg.norm(block[:, :2] - np.eye(2)) / np.sqrt(2.0))

    rotational_column = block[:, 2]
    anchor_world = np.array(
        [rotational_column[1], -rotational_column[0]],
        dtype=float,
    )
    action_anchor = action - support
    anchor_scale = max(float(np.linalg.norm(action_anchor)), 1.0)
    anchor_consistency = float(
        np.linalg.norm(anchor_world - action_anchor) / anchor_scale
    )

    anchor_local = planar_rotation(-support_angle) @ anchor_world
    predicted = planar_point_jacobian(anchor_world)
    jacobian_residual = float(
        np.linalg.norm(block - predicted)
        / max(float(np.linalg.norm(predicted)), 1e-12)
    )

    failures = [
        (
            translation_residual > max_translation_residual,
            "translation columns are not rigid-follow compatible",
        ),
        (
            anchor_consistency > max_anchor_consistency_residual,
            "rotation response does not match the current action/support lever arm",
        ),
        (
            jacobian_residual > max_jacobian_residual,
            "CASJ block is not explained by one rigid point anchor",
        ),
    ]
    for failed, reason in failures:
        if failed:
            return PlanarRigidAnchorFingerprint(
                False,
                anchor_world,
                anchor_local,
                predicted,
                translation_residual,
                anchor_consistency,
                jacobian_residual,
                reason,
            )

    return PlanarRigidAnchorFingerprint(
        True,
        anchor_world,
        anchor_local,
        predicted,
        translation_residual,
        anchor_consistency,
        jacobian_residual,
        "CASJ is consistent with a rigid support-relative point anchor",
    )


def transport_planar_anchor(
    *,
    anchor_local: np.ndarray,
    new_support_position: np.ndarray,
    new_support_angle: float,
) -> np.ndarray:
    """Move a recovered local anchor with a new planar support pose."""
    anchor = np.asarray(anchor_local, dtype=float)
    position = np.asarray(new_support_position, dtype=float)
    if anchor.shape != (2,) or position.shape != (2,):
        raise ValueError("anchor_local and new_support_position must have shape (2,)")
    return position + planar_rotation(new_support_angle) @ anchor



@dataclass(frozen=True)
class SpatialRigidPointFingerprint:
    """Rigid 3D point-action fingerprint for a six-DoF support pose.

    Intervention chart:
      - the first three coordinates translate the support center in world axes;
      - the last three coordinates apply an infinitesimal world-axis rotation
        about the support center.

    For a point action a = p + R r, with v = a - p,

        delta a = delta p + omega x v
                = delta p - [v]_x omega,

    so the ideal 3x6 CASJ block is [I_3, -[v]_x].
    """

    accepted: bool
    anchor_world: np.ndarray
    anchor_local: np.ndarray
    predicted_jacobian: np.ndarray
    translation_residual: float
    rotational_structure_residual: float
    anchor_consistency_residual: float
    jacobian_residual: float
    reason: str


def skew(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    if v.shape != (3,):
        raise ValueError("vector must have shape (3,)")
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def vee(skew_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(skew_matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("skew_matrix must have shape (3, 3)")
    return np.array(
        [matrix[2, 1], matrix[0, 2], matrix[1, 0]],
        dtype=float,
    )


def axis_angle_rotation(rotation_vector: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map for a 3D rotation vector."""
    vector = np.asarray(rotation_vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError("rotation_vector must have shape (3,)")
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3) + skew(vector)
    axis = vector / theta
    k = skew(axis)
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def spatial_point_jacobian(anchor_world: np.ndarray) -> np.ndarray:
    """Ideal 3x6 point-action CASJ for [world translation, world rotation]."""
    v = np.asarray(anchor_world, dtype=float)
    if v.shape != (3,):
        raise ValueError("anchor_world must have shape (3,)")
    return np.concatenate([np.eye(3), -skew(v)], axis=1)


def infer_spatial_rigid_point_anchor(
    casj_block: np.ndarray,
    *,
    support_position: np.ndarray,
    support_rotation: np.ndarray,
    action_point: np.ndarray,
    max_translation_residual: float = 0.20,
    max_rotational_structure_residual: float = 0.20,
    max_anchor_consistency_residual: float = 0.20,
    max_jacobian_residual: float = 0.20,
) -> SpatialRigidPointFingerprint:
    """Recover a latent 3D object-local action anchor from a 3x6 CASJ block."""
    block = np.asarray(casj_block, dtype=float)
    position = np.asarray(support_position, dtype=float)
    rotation = np.asarray(support_rotation, dtype=float)
    action = np.asarray(action_point, dtype=float)
    if block.shape != (3, 6):
        raise ValueError("casj_block must have shape (3, 6)")
    if position.shape != (3,) or action.shape != (3,):
        raise ValueError("support_position and action_point must have shape (3,)")
    if rotation.shape != (3, 3):
        raise ValueError("support_rotation must have shape (3, 3)")

    translation = block[:, :3]
    rotational = block[:, 3:]
    translation_residual = float(
        np.linalg.norm(translation - np.eye(3)) / np.sqrt(3.0)
    )

    # Project the noisy rotational block to the nearest skew-symmetric matrix.
    rotational_skew = 0.5 * (rotational - rotational.T)
    symmetric_part = 0.5 * (rotational + rotational.T)
    rotational_scale = max(float(np.linalg.norm(rotational)), 1.0)
    rotational_structure_residual = float(
        np.linalg.norm(symmetric_part) / rotational_scale
    )

    # Ideal B = -[v]_x, hence [v]_x = -skew(B).
    anchor_world = vee(-rotational_skew)
    action_anchor = action - position
    anchor_scale = max(float(np.linalg.norm(action_anchor)), 1.0)
    anchor_consistency = float(
        np.linalg.norm(anchor_world - action_anchor) / anchor_scale
    )
    anchor_local = rotation.T @ anchor_world

    predicted = spatial_point_jacobian(anchor_world)
    jacobian_residual = float(
        np.linalg.norm(block - predicted)
        / max(float(np.linalg.norm(predicted)), 1e-12)
    )

    failures = [
        (
            translation_residual > max_translation_residual,
            "translation block is not rigid-follow compatible",
        ),
        (
            rotational_structure_residual > max_rotational_structure_residual,
            "rotation block is not skew-compatible with a rigid point anchor",
        ),
        (
            anchor_consistency > max_anchor_consistency_residual,
            "rotation response does not match the current 3D action/support lever arm",
        ),
        (
            jacobian_residual > max_jacobian_residual,
            "CASJ block is not explained by one 3D rigid point anchor",
        ),
    ]
    for failed, reason in failures:
        if failed:
            return SpatialRigidPointFingerprint(
                False,
                anchor_world,
                anchor_local,
                predicted,
                translation_residual,
                rotational_structure_residual,
                anchor_consistency,
                jacobian_residual,
                reason,
            )

    return SpatialRigidPointFingerprint(
        True,
        anchor_world,
        anchor_local,
        predicted,
        translation_residual,
        rotational_structure_residual,
        anchor_consistency,
        jacobian_residual,
        "CASJ is consistent with one rigid 3D support-relative point anchor",
    )


def transport_spatial_point_anchor(
    *,
    anchor_local: np.ndarray,
    new_support_position: np.ndarray,
    new_support_rotation: np.ndarray,
) -> np.ndarray:
    """Transport a recovered 3D local point anchor with a new support pose."""
    anchor = np.asarray(anchor_local, dtype=float)
    position = np.asarray(new_support_position, dtype=float)
    rotation = np.asarray(new_support_rotation, dtype=float)
    if anchor.shape != (3,) or position.shape != (3,):
        raise ValueError("anchor_local and new_support_position must have shape (3,)")
    if rotation.shape != (3, 3):
        raise ValueError("new_support_rotation must have shape (3, 3)")
    return position + rotation @ anchor