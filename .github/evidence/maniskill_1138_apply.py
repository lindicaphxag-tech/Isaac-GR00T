from pathlib import Path
import sys


root = Path(sys.argv[1])

conversion = root / "mani_skill/trajectory/utils/actions/conversion.py"
text = conversion.read_text()

replacements = [
    (
        "from transforms3d.quaternions import quat2axangle\n",
        "",
    ),
    (
        '''def compact_axis_angle_from_quaternion(quat: np.ndarray) -> np.ndarray:
    theta, omega = quat2axangle(quat)
    # - 2 * np.pi to make the angle symmetrical around 0
    if omega > np.pi:
        omega = omega - 2 * np.pi
    return omega * theta
''',
        '''def xyz_euler_from_quaternion(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion to the XYZ-Euler representation used by PDEEPoseController."""
    quat_tensor = torch.as_tensor(quat)
    return (
        rotation_conversions.matrix_to_euler_angles(
            rotation_conversions.quaternion_to_matrix(quat_tensor),
            "XYZ",
        )
        .cpu()
        .numpy()
    )
''',
    ),
    (
        "        compact_axis_angle_from_quaternion(delta_pose.q),\n",
        "        xyz_euler_from_quaternion(delta_pose.q),\n",
    ),
]

for old, new in replacements:
    if text.count(old) != 1:
        raise RuntimeError(f"conversion anchor count={text.count(old)} for {old!r}")
    text = text.replace(old, new)
conversion.write_text(text)

docs = root / "docs/source/user_guide/concepts/controllers.md"
text = docs.read_text()
old = '''- arm_pd_ee_delta_pose (6-dim): both position ( $p$ ) and rotation ( $R$ ) are controlled. Rotation is represented as axis-angle in the end-effector frame.

  $\\bar{p}(t)=p(t)+a_{p}$, $\\bar{R}(t)=R(t) \\cdot e^{[a_{R}]_{\\times}}$, $\\bar{q}(t)=IK(\\bar{p}(t), \\bar{R}(t))$
'''
new = '''- arm_pd_ee_delta_pose (6-dim): both position ( $p$ ) and rotation ( $R$ ) are controlled. The three rotation action dimensions are XYZ Euler angles, matching the controller's Rotation section and implementation.

  $\\bar{p}(t)=p(t)+a_{p}$. The rotation increment is constructed from the XYZ Euler action $a_R$ and composed according to the configured rotation frame; then $\\bar{q}(t)=IK(\\bar{p}(t), \\bar{R}(t))$.
'''
if text.count(old) != 1:
    raise RuntimeError(f"docs anchor count={text.count(old)}")
docs.write_text(text.replace(old, new))

test = root / "tests/test_action_conversion.py"
test.write_text('''from types import SimpleNamespace

import pytest
import sapien
import torch

from mani_skill.agents.controllers import PDEEPoseController
from mani_skill.trajectory.utils.actions.conversion import delta_pose_to_pd_ee_delta
from mani_skill.utils import gym_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.pose import Pose


@pytest.mark.parametrize(
    "euler",
    [
        [0.3, 0.2, -0.4],
        [0.5, 0.5, 0.5],
        [0.8, -0.4, 0.6],
    ],
)
def test_pd_ee_delta_pose_converter_matches_controller_xyz_euler_semantics(euler):
    """Converted rotations must reconstruct the target under the controller decoder."""
    dtype = torch.float64
    expected_euler = torch.tensor([euler], dtype=dtype)
    expected_quat = rotation_conversions.matrix_to_quaternion(
        rotation_conversions.euler_angles_to_matrix(expected_euler, "XYZ")
    )

    controller = object.__new__(PDEEPoseController)
    controller.config = SimpleNamespace(
        use_delta=True,
        normalize_action=True,
        frame="root_translation:root_aligned_body_rotation",
    )
    controller.action_space_low = torch.full((6,), -1.0, dtype=dtype)
    controller.action_space_high = torch.full((6,), 1.0, dtype=dtype)

    delta_pose = sapien.Pose([0.0, 0.0, 0.0], expected_quat[0].numpy())
    normalized = delta_pose_to_pd_ee_delta(controller, delta_pose)

    physical_action = gym_utils.clip_and_scale_action(
        torch.as_tensor(normalized, dtype=dtype),
        controller.action_space_low,
        controller.action_space_high,
    ).unsqueeze(0)

    identity = Pose.create_from_pq(
        torch.zeros((1, 3), dtype=dtype),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype),
    )
    reconstructed = PDEEPoseController.compute_target_pose(
        controller, identity, physical_action
    )

    torch.testing.assert_close(
        torch.abs(torch.sum(expected_quat[0] * reconstructed.q[0])),
        torch.tensor(1.0, dtype=dtype),
        atol=1e-7,
        rtol=1e-7,
    )
''')
