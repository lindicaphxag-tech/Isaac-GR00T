import numpy as np

from casj.finite_equivariance import (
    PlanarEquivarianceProbe,
    PlanarSupportPose,
    certify_finite_planar_equivariance,
    transport_points_between_planar_supports,
)


def _pose(x, y, theta):
    return PlanarSupportPose(np.array([x, y], dtype=float), theta)


def test_finite_equivariance_accepts_true_rigid_support_bound_chunk():
    base_pose = _pose(10.0, 20.0, 0.2)
    baseline = np.array([[12.0, 21.0], [9.0, 24.0], [13.0, 18.0]])
    poses = [
        _pose(12.0, 19.0, 0.35),
        _pose(7.0, 23.0, -0.1),
        _pose(11.0, 25.0, 0.6),
    ]
    probes = [
        PlanarEquivarianceProbe(
            pose,
            transport_points_between_planar_supports(
                baseline, source_pose=base_pose, target_pose=pose
            ),
        )
        for pose in poses
    ]
    cert = certify_finite_planar_equivariance(
        baseline_action_points=baseline,
        baseline_support_pose=base_pose,
        probes=probes,
    )
    assert cert.accepted
    assert cert.accepted_mask.tolist() == [True, True, True]
    np.testing.assert_allclose(cert.direct_max_relative_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(cert.pairwise_max_relative_residual, 0.0, atol=1e-12)


def test_world_fixed_actions_are_rejected():
    base_pose = _pose(0.0, 0.0, 0.0)
    baseline = np.array([[3.0, 1.0], [2.0, -2.0]])
    probes = [
        PlanarEquivarianceProbe(_pose(4.0, 0.0, 0.2), baseline.copy()),
        PlanarEquivarianceProbe(_pose(-3.0, 2.0, -0.3), baseline.copy()),
    ]
    cert = certify_finite_planar_equivariance(
        baseline_action_points=baseline,
        baseline_support_pose=base_pose,
        probes=probes,
    )
    assert not cert.accepted
    assert not np.any(cert.accepted_mask)


def test_certificate_is_position_selective():
    base_pose = _pose(0.0, 0.0, 0.0)
    baseline = np.array([[1.0, 0.0], [8.0, 8.0]])
    poses = [_pose(2.0, 1.0, 0.2), _pose(-1.0, 3.0, -0.4)]
    probes = []
    for pose in poses:
        moved = transport_points_between_planar_supports(
            baseline, source_pose=base_pose, target_pose=pose
        )
        moved[1] = baseline[1]  # second action is actually world-fixed
        probes.append(PlanarEquivarianceProbe(pose, moved))
    cert = certify_finite_planar_equivariance(
        baseline_action_points=baseline,
        baseline_support_pose=base_pose,
        probes=probes,
    )
    assert not cert.accepted
    assert cert.accepted_mask.tolist() == [True, False]


def test_runtime_transport_needs_no_policy_query_after_certificate():
    base_pose = _pose(2.0, -1.0, 0.1)
    runtime_pose = _pose(20.0, 5.0, 1.0)
    baseline = np.array([[4.0, 3.0], [0.0, -2.0]])
    transported = transport_points_between_planar_supports(
        baseline,
        source_pose=base_pose,
        target_pose=runtime_pose,
    )
    assert transported.shape == baseline.shape
    assert np.all(np.isfinite(transported))
