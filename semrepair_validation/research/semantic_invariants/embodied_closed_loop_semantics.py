"""Closed-loop orientation assay for composed embodied semantic failures.

This is a deterministic mathematical assay, not a simulator benchmark.  It
models the ManiSkill-style composition seed:

canonical XYZ-Euler delta
-> converter emits compact axis-angle
-> controller sign convention is inverted
-> controller interprets the 3-vector as XYZ Euler.

The assay measures both goal convergence and deviation from the canonical
fully-repaired trajectory.  It demonstrates why task success alone can hide a
remaining semantic mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, degrees, sin
from typing import Callable

from .embodied_rotation_chain import (
    Matrix3,
    Vector3,
    angular_distance_radians,
    compact_axis_angle_from_matrix,
    euler_xyz_matrix,
)
from .embodied_repair_runtime import SemanticRepairMediator
from .embodied_repair_verification import verify_repair_against_heldout
from .embodied_repair_synthesis import (
    RepairExample,
    RepairProgram,
    bank_counterexample_finder,
    build_vector_repair_catalog,
    cegis_repair,
)
from .embodied_semantic_compiler import compile_semantic_boundary
from .embodied_semantic_types import SemanticTensorType


def _matmul(a: Matrix3, b: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def _transpose(a: Matrix3) -> Matrix3:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _matrix_to_euler_xyz(matrix: Matrix3) -> Vector3:
    y = asin(max(-1.0, min(1.0, float(matrix[0][2]))))
    cy = cos(y)
    if abs(cy) > 1.0e-10:
        x = atan2(-matrix[1][2], matrix[2][2])
        z = atan2(-matrix[0][1], matrix[0][0])
    else:
        z = 0.0
        x = atan2(matrix[2][1], matrix[1][1])
    return (float(x), float(y), float(z))


def _clip_components(value: Vector3, limit: float) -> Vector3:
    return tuple(max(-limit, min(limit, item)) for item in value)  # type: ignore[return-value]


def hidden_composed_boundary(canonical_euler_delta: Vector3) -> Vector3:
    """Retrospective #1138 + #1469 failure shape at the controller boundary."""
    rotation = euler_xyz_matrix(canonical_euler_delta)
    axis_angle = compact_axis_angle_from_matrix(rotation)
    return tuple(-item for item in axis_angle)  # type: ignore[return-value]


def no_repair(observed: Vector3) -> Vector3:
    return observed


def representation_only_repair(observed: Vector3) -> Vector3:
    # Correct representation, but preserve the already-inverted sign.
    from .embodied_repair_synthesis import _axis_angle_to_matrix, _matrix_to_euler_xyz as convert

    return convert(_axis_angle_to_matrix(observed))  # type: ignore[return-value]


def sign_only_repair(observed: Vector3) -> Vector3:
    # Restore sign, but the controller still interprets compact axis-angle as Euler.
    return tuple(-item for item in observed)  # type: ignore[return-value]


def full_composed_repair(observed: Vector3) -> Vector3:
    from .embodied_repair_synthesis import _axis_angle_to_matrix, _matrix_to_euler_xyz as convert

    sign_restored = tuple(-item for item in observed)
    return convert(_axis_angle_to_matrix(sign_restored))  # type: ignore[return-value]




def _composed_repair_example(euler: Vector3, *, label: str) -> RepairExample:
    return RepairExample(
        observed=hidden_composed_boundary(euler),
        expected=euler,
        label=label,
    )


def synthesize_verified_composed_repair() -> RepairProgram:
    """Synthesize and independently bank-verify the composed ManiSkill-style repair."""

    source = SemanticTensorType(
        role="action",
        entity="ee_rotation_delta",
        frame="root_aligned_body",
        representation="axis_angle",
        convention="positive_action_negative_rotation",
        unit="rad",
        provenance="requested",
    )
    target = source.updated(
        representation="euler_xyz",
        convention="positive_action_positive_rotation",
    )

    seed_angles = (
        (0.04, 0.04, 0.04),
        (0.03, -0.04, 0.05),
    )
    heldout_angles = (
        (-0.04, 0.03, 0.02),
        (0.05, -0.02, -0.03),
        (-0.02, -0.04, 0.05),
        (0.02, 0.05, -0.03),
        (0.04, 0.03, -0.05),
        (-0.05, 0.02, 0.04),
    )
    seeds = tuple(
        _composed_repair_example(value, label=f"seed-{index}")
        for index, value in enumerate(seed_angles)
    )

    compilation = compile_semantic_boundary(
        source,
        target,
        repair_examples=seeds,
        max_repair_depth=2,
        repair_atol=1.0e-9,
    )
    if compilation.repair_candidate is None:
        raise RuntimeError("semantic compiler did not produce a repair candidate")

    heldout = tuple(
        _composed_repair_example(value, label=f"heldout-{index}")
        for index, value in enumerate(heldout_angles)
    )
    verification = cegis_repair(
        seeds,
        build_vector_repair_catalog(3),
        counterexample_finder=bank_counterexample_finder(
            heldout,
            atol=1.0e-9,
        ),
        max_depth=2,
        allowed_families=set(compilation.repair_families),
        atol=1.0e-9,
    )
    if verification.status != "verified" or verification.program is None:
        raise RuntimeError(
            f"composed repair did not verify: {verification.status!r}"
        )
    if verification.program.name != compilation.repair_candidate.name:
        raise RuntimeError(
            "compiler candidate and independent CEGIS verification disagree: "
            f"{compilation.repair_candidate.name!r} != {verification.program.name!r}"
        )
    return verification.program


def compiler_mediated_repair() -> Callable[[Vector3], Vector3]:
    """Return a runtime repair callable backed by a verified synthesized program."""

    program = synthesize_verified_composed_repair()
    contract_id = "embodied/representation/controller-roundtrip@0.2"
    heldout = (
        _composed_repair_example((-0.04, 0.03, 0.02), label="runtime-cert-0"),
        _composed_repair_example((0.05, -0.02, -0.03), label="runtime-cert-1"),
        _composed_repair_example((-0.02, -0.04, 0.05), label="runtime-cert-2"),
    )
    certificate = verify_repair_against_heldout(
        contract_id=contract_id,
        program=program,
        heldout=heldout,
        verifier_id="closed-loop-independent-bank-v1",
        atol=1.0e-9,
    )
    mediator = SemanticRepairMediator.from_certificate(
        contract_id=contract_id,
        program=program,
        certificate=certificate,
    )

    def repair(observed: Vector3) -> Vector3:
        receipt = mediator.mediate(
            observed,
            metadata={"assay": "composed-rotation-closed-loop"},
        )
        return receipt.executed  # type: ignore[return-value]

    return repair


@dataclass(frozen=True)
class ClosedLoopAssayResult:
    mode: str
    errors_degrees: tuple[float, ...]
    path_length_degrees: float
    steps_to_tolerance: int | None
    final_error_degrees: float
    max_deviation_from_reference_degrees: float


def rollout_orientation_servo(
    *,
    goal_euler: Vector3 = (0.35, -0.25, 0.30),
    repair: Callable[[Vector3], Vector3] = full_composed_repair,
    horizon: int = 12,
    max_component_step: float = 0.04,
    tolerance_degrees: float = 0.01,
) -> tuple[Matrix3, ...]:
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if max_component_step <= 0:
        raise ValueError("max_component_step must be positive")

    identity: Matrix3 = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    goal = euler_xyz_matrix(goal_euler)
    current = identity
    trajectory = [current]

    for _ in range(horizon):
        error_deg = degrees(angular_distance_radians(current, goal))
        if error_deg <= tolerance_degrees:
            trajectory.append(current)
            continue

        relative = _matmul(_transpose(current), goal)
        canonical_delta = _clip_components(
            _matrix_to_euler_xyz(relative),
            max_component_step,
        )
        observed = hidden_composed_boundary(canonical_delta)
        controller_delta = repair(observed)
        current = _matmul(current, euler_xyz_matrix(controller_delta))
        trajectory.append(current)

    return tuple(trajectory)


def summarize_rollout(
    trajectory: tuple[Matrix3, ...],
    *,
    mode: str,
    goal_euler: Vector3 = (0.35, -0.25, 0.30),
    reference: tuple[Matrix3, ...] | None = None,
    tolerance_degrees: float = 0.01,
) -> ClosedLoopAssayResult:
    goal = euler_xyz_matrix(goal_euler)
    errors = tuple(
        degrees(angular_distance_radians(state, goal))
        for state in trajectory
    )
    path_length = sum(
        degrees(angular_distance_radians(left, right))
        for left, right in zip(trajectory, trajectory[1:])
    )
    steps_to_tolerance = next(
        (index for index, error in enumerate(errors) if error <= tolerance_degrees),
        None,
    )

    max_deviation = 0.0
    if reference is not None:
        if len(reference) != len(trajectory):
            raise ValueError("reference and trajectory lengths differ")
        max_deviation = max(
            degrees(angular_distance_radians(actual, expected))
            for actual, expected in zip(trajectory, reference)
        )

    return ClosedLoopAssayResult(
        mode=mode,
        errors_degrees=errors,
        path_length_degrees=path_length,
        steps_to_tolerance=steps_to_tolerance,
        final_error_degrees=errors[-1],
        max_deviation_from_reference_degrees=max_deviation,
    )


def run_composed_rotation_assay() -> dict[str, ClosedLoopAssayResult]:
    repairs = {
        "none": no_repair,
        "representation_only": representation_only_repair,
        "sign_only": sign_only_repair,
        "full_composed": full_composed_repair,
        "compiler_mediated": compiler_mediated_repair(),
    }
    trajectories = {
        name: rollout_orientation_servo(repair=repair)
        for name, repair in repairs.items()
    }
    reference = trajectories["full_composed"]
    return {
        name: summarize_rollout(
            trajectory,
            mode=name,
            reference=reference,
        )
        for name, trajectory in trajectories.items()
    }