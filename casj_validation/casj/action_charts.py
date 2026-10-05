from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .transport import decode_body, decode_spatial, invert_pose


def _pose_chunk(chunk: np.ndarray) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=float)
    if chunk.ndim != 3 or chunk.shape[1:] != (4, 4):
        raise ValueError("SE3 action chunk must have shape [H, 4, 4]")
    return chunk


def _vee(skew_matrix: np.ndarray) -> np.ndarray:
    return np.array(
        [
            skew_matrix[2, 1],
            skew_matrix[0, 2],
            skew_matrix[1, 0],
        ],
        dtype=float,
    )


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Log map SO(3) -> R^3 using the principal branch."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    cosine = float(
        np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    )
    theta = float(np.arccos(cosine))
    if theta < 1e-8:
        return 0.5 * _vee(rotation - rotation.T)
    if np.pi - theta < 1e-5:
        # Principal-axis extraction near pi, where sin(theta) is ill-conditioned.
        diagonal = np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diagonal)
        index = int(np.argmax(axis))
        if axis[index] < 1e-8:
            raise ValueError("rotation log is numerically singular near pi")
        for j in range(3):
            if j == index:
                continue
            axis[j] = (
                rotation[index, j] + rotation[j, index]
            ) / (4.0 * axis[index])
        axis /= np.linalg.norm(axis)
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * _vee(
        rotation - rotation.T
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def se3_log(transform: np.ndarray) -> np.ndarray:
    """Principal SE(3) log coordinates [rho, omega].

    The returned vector is the twist whose matrix exponential reconstructs the
    relative transform, using the standard SO(3) left-Jacobian inverse for the
    translational component.
    """
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("invalid homogeneous transform")

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    omega = so3_log(rotation)
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)

    if theta < 1e-8:
        jacobian_inverse = (
            np.eye(3)
            - 0.5 * omega_hat
            + (1.0 / 12.0) * (omega_hat @ omega_hat)
        )
    else:
        coefficient = (
            1.0 / (theta * theta)
            - (1.0 + np.cos(theta))
            / (2.0 * theta * np.sin(theta))
        )
        jacobian_inverse = (
            np.eye(3)
            - 0.5 * omega_hat
            + coefficient * (omega_hat @ omega_hat)
        )

    rho = jacobian_inverse @ translation
    return np.concatenate([rho, omega])


def so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """Exponential map R^3 -> SO(3)."""
    omega = np.asarray(rotation_vector, dtype=float)
    if omega.shape != (3,):
        raise ValueError("rotation_vector must have shape (3,)")
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    if theta < 1e-8:
        return (
            np.eye(3)
            + omega_hat
            + 0.5 * (omega_hat @ omega_hat)
        )
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * omega_hat + b * (omega_hat @ omega_hat)


def se3_exp(twist: np.ndarray) -> np.ndarray:
    """Exponential map [rho, omega] -> SE(3)."""
    twist = np.asarray(twist, dtype=float)
    if twist.shape != (6,):
        raise ValueError("twist must have shape (6,)")
    rho = twist[:3]
    omega = twist[3:]
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    rotation = so3_exp(omega)

    if theta < 1e-8:
        left_jacobian = (
            np.eye(3)
            + 0.5 * omega_hat
            + (1.0 / 6.0) * (omega_hat @ omega_hat)
        )
    else:
        a = (1.0 - np.cos(theta)) / (theta * theta)
        b = (theta - np.sin(theta)) / (theta * theta * theta)
        left_jacobian = (
            np.eye(3)
            + a * omega_hat
            + b * (omega_hat @ omega_hat)
        )

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = left_jacobian @ rho
    return transform


class EuclideanActionChart:
    """Action chart for vectors whose local response is ordinary subtraction."""

    def decode(self, raw_chunk: np.ndarray, *, observation: Any) -> np.ndarray:
        del observation
        chunk = np.asarray(raw_chunk, dtype=float)
        if chunk.ndim != 2:
            raise ValueError("Euclidean action chunk must have shape [H, D]")
        return chunk

    def encode(self, decoded_chunk: np.ndarray, *, observation: Any) -> np.ndarray:
        del observation
        return self.decode(decoded_chunk, observation=None)

    def retract(
        self,
        baseline_decoded: np.ndarray,
        local_delta: np.ndarray,
    ) -> np.ndarray:
        baseline = np.asarray(baseline_decoded, dtype=float)
        delta = np.asarray(local_delta, dtype=float)
        if baseline.shape != delta.shape:
            raise ValueError("Euclidean baseline/local_delta shapes must match")
        return baseline + delta

    def local_response(
        self,
        changed: np.ndarray,
        baseline: np.ndarray,
    ) -> np.ndarray:
        changed = np.asarray(changed, dtype=float)
        baseline = np.asarray(baseline, dtype=float)
        if changed.shape != baseline.shape or changed.ndim != 2:
            raise ValueError("Euclidean decoded chunks must share shape [H, D]")
        return changed - baseline

    def pairing_error(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        if first.shape != second.shape:
            return float("inf")
        return float(
            np.linalg.norm(second - first)
            / max(
                float(np.linalg.norm(first)),
                float(np.linalg.norm(second)),
                1e-12,
            )
        )


class SE3MatrixActionChart:
    """Left-trivialized action response for absolute SE(3) pose chunks."""

    def decode(self, raw_chunk: np.ndarray, *, observation: Any) -> np.ndarray:
        del observation
        return _pose_chunk(raw_chunk)

    def encode(self, decoded_chunk: np.ndarray, *, observation: Any) -> np.ndarray:
        del observation
        return _pose_chunk(decoded_chunk).copy()

    def retract(
        self,
        baseline_decoded: np.ndarray,
        local_delta: np.ndarray,
    ) -> np.ndarray:
        baseline = _pose_chunk(baseline_decoded)
        delta = np.asarray(local_delta, dtype=float)
        if delta.shape != (len(baseline), 6):
            raise ValueError(
                f"SE3 local_delta must have shape {(len(baseline), 6)}"
            )
        return np.stack(
            [se3_exp(delta[i]) @ baseline[i] for i in range(len(baseline))],
            axis=0,
        )

    def local_response(
        self,
        changed: np.ndarray,
        baseline: np.ndarray,
    ) -> np.ndarray:
        # changed and baseline are already decoded task-space poses. Do not
        # dispatch through subclass decode() again: body/spatial/joint charts
        # would otherwise reinterpret absolute poses as native actions.
        changed = _pose_chunk(changed)
        baseline = _pose_chunk(baseline)
        if changed.shape != baseline.shape:
            raise ValueError("SE3 decoded chunks must have the same horizon")
        return np.stack(
            [
                se3_log(changed[i] @ invert_pose(baseline[i]))
                for i in range(len(baseline))
            ],
            axis=0,
        )

    def pairing_error(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        response = self.local_response(second, first)
        return float(np.linalg.norm(response))


@dataclass(frozen=True)
class ReferenceRelativeSE3ActionChart(SE3MatrixActionChart):
    """Chunk poses expressed independently relative to one reference pose.

    Native action i is:
        R_i = inv(T_ref) @ T_i

    Decoding therefore uses:
        T_i = T_ref @ R_i

    This differs from body/incremental deltas, where each delta composes with
    the previous decoded pose.
    """

    anchor_pose: np.ndarray
    runtime_anchor_pose: np.ndarray | None = None

    def decode(self, raw_chunk: np.ndarray, *, observation: Any) -> np.ndarray:
        del observation
        relative = _pose_chunk(raw_chunk)
        anchor = np.asarray(self.anchor_pose, dtype=float)
        return np.stack([anchor @ item for item in relative], axis=0)

    def encode(self, decoded_chunk: np.ndarray, *, observation: Any) -> np.ndarray:
        del observation
        poses = _pose_chunk(decoded_chunk)
        anchor = (
            self.anchor_pose
            if self.runtime_anchor_pose is None
            else np.asarray(self.runtime_anchor_pose, dtype=float)
        )
        anchor_inv = invert_pose(anchor)
        return np.stack([anchor_inv @ pose for pose in poses], axis=0)


@dataclass(frozen=True)
class BodyRelativeSE3ActionChart(SE3MatrixActionChart):
    """Decode body deltas and re-encode from the actual runtime anchor."""

    anchor_pose: np.ndarray
    runtime_anchor_pose: np.ndarray | None = None

    def decode(self, raw_chunk: np.ndarray, *, observation: Any) -> np.ndarray:
        del observation
        return decode_body(self.anchor_pose, np.asarray(raw_chunk, dtype=float))

    def encode(self, decoded_chunk: np.ndarray, *, observation: Any) -> np.ndarray:
        del observation
        anchor = (
            self.anchor_pose
            if self.runtime_anchor_pose is None
            else self.runtime_anchor_pose
        )
        from .transport import encode_body
        return encode_body(anchor, _pose_chunk(decoded_chunk))


@dataclass(frozen=True)
class SpatialRelativeSE3ActionChart(SE3MatrixActionChart):
    """Decode spatial deltas and re-encode from the actual runtime anchor."""

    anchor_pose: np.ndarray
    runtime_anchor_pose: np.ndarray | None = None

    def decode(self, raw_chunk: np.ndarray, *, observation: Any) -> np.ndarray:
        del observation
        return decode_spatial(self.anchor_pose, np.asarray(raw_chunk, dtype=float))

    def encode(self, decoded_chunk: np.ndarray, *, observation: Any) -> np.ndarray:
        del observation
        anchor = (
            self.anchor_pose
            if self.runtime_anchor_pose is None
            else self.runtime_anchor_pose
        )
        from .transport import encode_spatial
        return encode_spatial(anchor, _pose_chunk(decoded_chunk))


@dataclass(frozen=True)
class JointPositionSE3ActionChart(SE3MatrixActionChart):
    """Decode joint chunks by FK and re-encode task poses by sequential IK."""

    fk: Callable[[np.ndarray], np.ndarray]
    ik: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    runtime_seed: np.ndarray | None = None
    max_translation_error: float = 1e-3
    max_rotation_error_rad: float = np.deg2rad(1.0)
    max_joint_step: float | None = None

    def decode(self, raw_chunk: np.ndarray, *, observation: Any) -> np.ndarray:
        del observation
        joints = np.asarray(raw_chunk, dtype=float)
        if joints.ndim != 2:
            raise ValueError("joint-position action chunk must have shape [H, dof]")
        return np.stack(
            [np.asarray(self.fk(q), dtype=float) for q in joints],
            axis=0,
        )

    def encode(self, decoded_chunk: np.ndarray, *, observation: Any) -> np.ndarray:
        del observation
        targets = _pose_chunk(decoded_chunk)
        if self.ik is None or self.runtime_seed is None:
            raise ValueError(
                "joint-position re-encoding requires ik and runtime_seed"
            )

        seed = np.asarray(self.runtime_seed, dtype=float).copy()
        solved = []
        for target in targets:
            q = np.asarray(self.ik(target, seed), dtype=float)
            if q.shape != seed.shape:
                raise ValueError(
                    f"IK returned shape {q.shape}, expected {seed.shape}"
                )
            reconstructed = np.asarray(self.fk(q), dtype=float)
            error = invert_pose(target) @ reconstructed
            translation_error = float(np.linalg.norm(error[:3, 3]))
            rotation_error = float(np.linalg.norm(so3_log(error[:3, :3])))
            if translation_error > self.max_translation_error:
                raise ValueError(
                    "IK task-space translation residual exceeds CASJ chart limit"
                )
            if rotation_error > self.max_rotation_error_rad:
                raise ValueError(
                    "IK task-space rotation residual exceeds CASJ chart limit"
                )
            if self.max_joint_step is not None:
                if float(np.linalg.norm(q - seed)) > self.max_joint_step:
                    raise ValueError(
                        "IK joint continuity step exceeds CASJ chart limit"
                    )
            solved.append(q)
            seed = q
        return np.stack(solved, axis=0)