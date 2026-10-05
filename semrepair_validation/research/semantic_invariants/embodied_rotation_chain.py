"""Dependency-light witness for composed rotation-representation failures.

This module models a semantic chain rather than a specific simulator object:

    target orientation
      -> trajectory-converter representation
      -> normalized controller semantics
      -> realized orientation

It is motivated by retrospective ManiSkill issues #1138 (axis-angle converter
feeding an XYZ-Euler controller) and #1469/#1472 (rotation sign inversion during
controller scaling). These are seed cases only; this module is not prospective
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, pi, sin, sqrt


Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Vector3 = tuple[float, float, float]


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _matmul(a: Matrix3, b: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def _transpose(a: Matrix3) -> Matrix3:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def euler_xyz_matrix(euler: Vector3) -> Matrix3:
    """Match the XYZ convention used by ManiSkill's controller implementation."""
    x, y, z = euler
    rx: Matrix3 = (
        (1.0, 0.0, 0.0),
        (0.0, cos(x), -sin(x)),
        (0.0, sin(x), cos(x)),
    )
    ry: Matrix3 = (
        (cos(y), 0.0, sin(y)),
        (0.0, 1.0, 0.0),
        (-sin(y), 0.0, cos(y)),
    )
    rz: Matrix3 = (
        (cos(z), -sin(z), 0.0),
        (sin(z), cos(z), 0.0),
        (0.0, 0.0, 1.0),
    )
    return _matmul(_matmul(rx, ry), rz)


def compact_axis_angle_from_matrix(matrix: Matrix3) -> Vector3:
    """Return the principal compact axis-angle vector for a rotation matrix."""
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    angle = acos(_clip((trace - 1.0) / 2.0))
    if abs(angle) < 1.0e-14:
        return (0.0, 0.0, 0.0)

    denom = 2.0 * sin(angle)
    axis = (
        (matrix[2][1] - matrix[1][2]) / denom,
        (matrix[0][2] - matrix[2][0]) / denom,
        (matrix[1][0] - matrix[0][1]) / denom,
    )
    return tuple(component * angle for component in axis)  # type: ignore[return-value]


def angular_distance_radians(a: Matrix3, b: Matrix3) -> float:
    """Geodesic SO(3) distance between two rotation matrices."""
    relative = _matmul(_transpose(a), b)
    trace = relative[0][0] + relative[1][1] + relative[2][2]
    return acos(_clip((trace - 1.0) / 2.0))


@dataclass(frozen=True)
class RotationChainWitness:
    target_euler: Vector3
    converter_output: Vector3
    controller_input: Vector3
    error_radians: float

    @property
    def error_degrees(self) -> float:
        return self.error_radians * 180.0 / pi


def evaluate_rotation_chain(
    target_euler: Vector3,
    *,
    converter_representation: str,
    controller_sign: int = 1,
) -> RotationChainWitness:
    """Evaluate converter/controller composition against target orientation.

    Args:
        target_euler: Canonical intended XYZ-Euler delta.
        converter_representation: "euler" for the repaired converter or
            "axis-angle" for the retrospective #1138 behavior.
        controller_sign: 1 for sign-preserving scaling or -1 for the
            retrospective #1469 behavior under symmetric bounds.
    """
    if converter_representation not in {"euler", "axis-angle"}:
        raise ValueError("converter_representation must be 'euler' or 'axis-angle'")
    if controller_sign not in {-1, 1}:
        raise ValueError("controller_sign must be -1 or 1")

    target_matrix = euler_xyz_matrix(target_euler)
    if converter_representation == "euler":
        converter_output = target_euler
    else:
        converter_output = compact_axis_angle_from_matrix(target_matrix)

    controller_input = tuple(controller_sign * value for value in converter_output)
    realized = euler_xyz_matrix(controller_input)
    return RotationChainWitness(
        target_euler=target_euler,
        converter_output=converter_output,
        controller_input=controller_input,
        error_radians=angular_distance_radians(target_matrix, realized),
    )


def vector_norm(value: Vector3) -> float:
    return sqrt(sum(component * component for component in value))
