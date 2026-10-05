"""Native ManiSkill #1138 representation-boundary execution.

This script must run against a separately checked-out frozen ManiSkill source.
It invokes the production trajectory converter and production controller decoder;
it does not copy their numerical implementations.

Claim boundary:
- verifies the converter -> controller representation boundary only;
- deliberately bypasses PDEEPoseController._clip_and_scale_action because
  ManiSkill #1469/#1472 owns a separate rotation-sign defect there;
- does not claim a simulator rollout or task-success improvement.
"""

from __future__ import annotations

import argparse
import json
from math import acos, degrees

import numpy as np
import sapien
import torch

from mani_skill.agents.controllers import PDEEPoseController
from mani_skill.trajectory.utils.actions.conversion import delta_pose_to_pd_ee_delta
from mani_skill.utils import gym_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.pose import Pose


def _orientation_error_degrees(q_expected: torch.Tensor, q_actual: torch.Tensor) -> float:
    dot = torch.abs(torch.sum(q_expected * q_actual)).clamp(0.0, 1.0)
    return degrees(2.0 * acos(float(dot)))


def run_native_boundary(target_euler=(0.5, 0.5, 0.5)) -> dict[str, object]:
    dtype = torch.float64
    target = torch.tensor([target_euler], dtype=dtype)
    target_matrix = rotation_conversions.euler_angles_to_matrix(target, "XYZ")
    expected_q = rotation_conversions.matrix_to_quaternion(target_matrix)[0]

    # Instantiate only the production controller surface required by the
    # converter and decoder. No simulator scene or IK is required for this
    # representation-boundary contract.
    controller = object.__new__(PDEEPoseController)
    controller.config = type(
        "_Config",
        (),
        {
            "use_delta": True,
            "normalize_action": True,
            "frame": "root_translation:root_aligned_body_rotation",
        },
    )()
    controller.action_space_low = torch.tensor([-1.0] * 6, dtype=dtype)
    controller.action_space_high = torch.tensor([1.0] * 6, dtype=dtype)

    delta_pose = sapien.Pose(
        [0.0, 0.0, 0.0],
        expected_q.detach().cpu().numpy(),
    )

    # Production trajectory converter on frozen source: quaternion -> compact
    # axis-angle -> normalized action.
    converted_normalized = delta_pose_to_pd_ee_delta(controller, delta_pose)
    converted_physical = gym_utils.clip_and_scale_action(
        torch.as_tensor(converted_normalized, dtype=dtype),
        controller.action_space_low,
        controller.action_space_high,
    ).unsqueeze(0)

    identity = Pose.create_from_pq(
        torch.zeros((1, 3), dtype=dtype),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
    )

    # Production controller decoder interprets the same three rotation
    # dimensions as XYZ Euler.
    broken_target = PDEEPoseController.compute_target_pose(
        controller,
        identity,
        converted_physical,
    )
    broken_error = _orientation_error_degrees(expected_q, broken_target.q[0])

    # SemRepair's repair boundary for #1138 is representation-only: provide the
    # canonical XYZ-Euler representation expected by the unchanged controller.
    repaired_physical = torch.tensor(
        [[0.0, 0.0, 0.0, *target_euler]],
        dtype=dtype,
    )
    repaired_target = PDEEPoseController.compute_target_pose(
        controller,
        identity,
        repaired_physical,
    )
    repaired_error = _orientation_error_degrees(expected_q, repaired_target.q[0])

    return {
        "schema": "semrepair-maniskill-native-boundary/v0.1",
        "target_euler_xyz_rad": list(target_euler),
        "production_converter_rotation": [float(x) for x in converted_physical[0, 3:]],
        "broken_so3_error_degrees": broken_error,
        "repaired_so3_error_degrees": repaired_error,
        "production_functions": [
            "mani_skill.trajectory.utils.actions.conversion.delta_pose_to_pd_ee_delta",
            "mani_skill.agents.controllers.PDEEPoseController.compute_target_pose",
        ],
        "claim_boundary": (
            "native production converter/controller representation boundary; "
            "separate normalized-rotation sign path intentionally excluded"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_native_boundary()

    # Emit measurements before enforcing the contract so a failing CI run still
    # preserves the exact native evidence in logs/artifacts.
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    else:
        print(report, flush=True)

    assert report["broken_so3_error_degrees"] > 5.0
    assert report["repaired_so3_error_degrees"] < 1.0e-5


if __name__ == "__main__":
    main()