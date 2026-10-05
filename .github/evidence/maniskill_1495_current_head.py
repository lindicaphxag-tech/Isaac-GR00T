import math
import sys
from types import SimpleNamespace

import numpy as np
import sapien
import torch

from mani_skill.agents.controllers import PDEEPoseController
from mani_skill.trajectory.utils.actions.conversion import delta_pose_to_pd_ee_delta
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.pose import Pose


mode = sys.argv[1]
if mode not in {"before-sign-fix", "after-sign-fix"}:
    raise SystemExit("usage: maniskill_1495_current_head.py {before-sign-fix|after-sign-fix}")

dtype = torch.float64
expected_euler = torch.tensor([[0.31, -0.27, 0.42]], dtype=dtype)
expected_quat = rotation_conversions.matrix_to_quaternion(
    rotation_conversions.euler_angles_to_matrix(expected_euler, "XYZ")
)
inverse_delta = rotation_conversions.quaternion_invert(expected_quat)

controller = object.__new__(PDEEPoseController)
controller.config = SimpleNamespace(
    use_delta=True,
    normalize_action=True,
    frame="root_translation:root_aligned_body_rotation",
    rot_lower=-2 * math.pi,
    rot_upper=2 * math.pi,
)
controller.action_space_low = torch.tensor(
    [-0.1, -0.1, -0.1, -2 * math.pi, -2 * math.pi, -2 * math.pi],
    dtype=dtype,
)
controller.action_space_high = -controller.action_space_low

# The trajectory-conversion path supplies current * target.inv(), i.e. inverse
# desired relative rotation. PR #1495 owns exactly this representation boundary.
delta_pose = sapien.Pose([0.01, -0.02, 0.03], inverse_delta[0].numpy())
normalized = delta_pose_to_pd_ee_delta(controller, delta_pose)
normalized_t = torch.as_tensor(normalized, dtype=dtype).unsqueeze(0)

# Exercise the production controller scaling + decoder, not a duplicate decoder.
physical = controller._clip_and_scale_action(normalized_t)
identity = Pose.create_from_pq(
    torch.zeros((1, 3), dtype=dtype),
    torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
)
target = controller.compute_target_pose(identity, physical)

dot = torch.abs(torch.sum(expected_quat[0] * target.q[0]))
dot = float(torch.clamp(dot, 0.0, 1.0))
error_deg = math.degrees(2.0 * math.acos(dot))
position_error = float(
    torch.linalg.vector_norm(target.p[0] - torch.tensor([0.01, -0.02, 0.03], dtype=dtype))
)

print(
    f"mode={mode} orientation_error_deg={error_deg:.9f} "
    f"position_error={position_error:.12g}"
)

if mode == "before-sign-fix":
    # Current controller main still applies rot_lower, so the representation-only
    # PR is intentionally not end-to-end green until dependency #1472 lands.
    if not error_deg > 5.0:
        raise AssertionError(f"expected sign dependency to remain visible, got {error_deg} deg")
else:
    if not error_deg < 1e-4:
        raise AssertionError(f"expected composed #1495 + #1472 round-trip <1e-4 deg, got {error_deg}")
    if not position_error < 1e-7:
        raise AssertionError(f"unexpected position drift: {position_error}")
