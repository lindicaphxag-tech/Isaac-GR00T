from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class TransportRejected(RuntimeError):
    pass


def _pose(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape != (4, 4):
        raise ValueError("expected a 4x4 homogeneous transform")
    if not np.allclose(x[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("invalid homogeneous bottom row")
    return x


def invert_pose(pose: np.ndarray) -> np.ndarray:
    pose = _pose(pose)
    out = np.eye(4)
    out[:3, :3] = pose[:3, :3].T
    out[:3, 3] = -(pose[:3, :3].T @ pose[:3, 3])
    return out


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def transport_pose_chunk(
    chunk: np.ndarray,
    *,
    old_support: np.ndarray,
    new_support: np.ndarray,
) -> np.ndarray:
    """Exact finite support transport: P'_i = F_new inv(F_old) P_i."""
    chunk = np.asarray(chunk, dtype=float)
    if chunk.ndim != 3 or chunk.shape[1:] != (4, 4):
        raise ValueError("chunk must have shape [H, 4, 4]")
    correction = _pose(new_support) @ invert_pose(old_support)
    return np.stack([correction @ _pose(p) for p in chunk])


def decode_body(anchor: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    pose = _pose(anchor)
    out = []
    for delta in np.asarray(deltas, dtype=float):
        pose = pose @ _pose(delta)
        out.append(pose.copy())
    return np.stack(out)


def encode_body(anchor: np.ndarray, poses: np.ndarray) -> np.ndarray:
    previous = _pose(anchor)
    out = []
    for pose in np.asarray(poses, dtype=float):
        pose = _pose(pose)
        out.append(invert_pose(previous) @ pose)
        previous = pose
    return np.stack(out)


def decode_spatial(anchor: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    pose = _pose(anchor)
    out = []
    for delta in np.asarray(deltas, dtype=float):
        pose = _pose(delta) @ pose
        out.append(pose.copy())
    return np.stack(out)


def encode_spatial(anchor: np.ndarray, poses: np.ndarray) -> np.ndarray:
    previous = _pose(anchor)
    out = []
    for pose in np.asarray(poses, dtype=float):
        pose = _pose(pose)
        out.append(pose @ invert_pose(previous))
        previous = pose
    return np.stack(out)


def transport_body_chunk(
    deltas: np.ndarray,
    *,
    nominal_anchor: np.ndarray,
    actual_anchor: np.ndarray,
    old_support: np.ndarray,
    new_support: np.ndarray,
) -> np.ndarray:
    """Decode -> exact task-space transport -> encode from actual runtime anchor."""
    nominal = decode_body(nominal_anchor, deltas)
    moved = transport_pose_chunk(nominal, old_support=old_support, new_support=new_support)
    return encode_body(actual_anchor, moved)


def transport_spatial_chunk(
    deltas: np.ndarray,
    *,
    nominal_anchor: np.ndarray,
    actual_anchor: np.ndarray,
    old_support: np.ndarray,
    new_support: np.ndarray,
) -> np.ndarray:
    nominal = decode_spatial(nominal_anchor, deltas)
    moved = transport_pose_chunk(nominal, old_support=old_support, new_support=new_support)
    return encode_spatial(actual_anchor, moved)


@dataclass(frozen=True)
class JointProjection:
    positions: np.ndarray
    translation_error: np.ndarray
    rotation_error_rad: np.ndarray


def transport_joint_chunk(
    joints: np.ndarray,
    *,
    fk: Callable[[np.ndarray], np.ndarray],
    ik: Callable[[np.ndarray, np.ndarray], np.ndarray],
    old_support: np.ndarray,
    new_support: np.ndarray,
    seed: np.ndarray,
    max_translation_error: float = 1e-3,
    max_rotation_error_rad: float = np.deg2rad(1.0),
    max_joint_step: float | None = None,
) -> JointProjection:
    """Joint actions have no universal SE(3) group action: project FK -> transport -> IK."""
    joints = np.asarray(joints, dtype=float)
    targets = transport_pose_chunk(
        np.stack([fk(q) for q in joints]),
        old_support=old_support,
        new_support=new_support,
    )

    q_prev = np.asarray(seed, dtype=float).copy()
    solved, terrs, rerrs = [], [], []
    for target in targets:
        q = np.asarray(ik(target, q_prev), dtype=float)
        reconstructed = _pose(fk(q))
        error = invert_pose(target) @ reconstructed
        terr = float(np.linalg.norm(error[:3, 3]))
        rerr = rotation_angle(error[:3, :3])
        if terr > max_translation_error or rerr > max_rotation_error_rad:
            raise TransportRejected("IK projection does not preserve transported task-space intent")
        if max_joint_step is not None and np.linalg.norm(q - q_prev) > max_joint_step:
            raise TransportRejected("joint continuity bound exceeded")
        solved.append(q)
        terrs.append(terr)
        rerrs.append(rerr)
        q_prev = q

    return JointProjection(
        positions=np.stack(solved),
        translation_error=np.asarray(terrs),
        rotation_error_rad=np.asarray(rerrs),
    )