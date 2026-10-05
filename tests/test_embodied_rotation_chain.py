from research.semantic_invariants.embodied_rotation_chain import (
    angular_distance_radians,
    euler_xyz_matrix,
    evaluate_rotation_chain,
)


TARGET = (0.08, 0.08, 0.08)


def test_axis_angle_converter_reinterpreted_as_xyz_euler_is_not_semantically_exact():
    result = evaluate_rotation_chain(
        TARGET,
        converter_representation="axis-angle",
        controller_sign=1,
    )

    assert result.error_degrees > 0.3


def test_fixing_converter_alone_does_not_repair_independent_controller_sign_bug():
    result = evaluate_rotation_chain(
        TARGET,
        converter_representation="euler",
        controller_sign=-1,
    )

    assert result.error_degrees > 15.0


def test_fixing_sign_alone_leaves_axis_angle_vs_euler_mismatch():
    result = evaluate_rotation_chain(
        TARGET,
        converter_representation="axis-angle",
        controller_sign=1,
    )

    assert result.error_degrees > 0.3


def test_composed_historical_failures_are_worse_than_representation_mismatch_alone():
    representation_only = evaluate_rotation_chain(
        TARGET,
        converter_representation="axis-angle",
        controller_sign=1,
    )
    composed = evaluate_rotation_chain(
        TARGET,
        converter_representation="axis-angle",
        controller_sign=-1,
    )

    assert composed.error_degrees > 15.0
    assert composed.error_degrees > 20 * representation_only.error_degrees


def test_repaired_converter_and_sign_preserve_target_orientation():
    result = evaluate_rotation_chain(
        TARGET,
        converter_representation="euler",
        controller_sign=1,
    )

    assert result.error_degrees < 1.0e-5


def test_angular_distance_self_comparison_has_zero_floor():
    matrix = euler_xyz_matrix((0.35, -0.25, 0.30))
    assert angular_distance_radians(matrix, matrix) == 0.0


def test_angular_distance_keeps_real_small_rotation_above_floor():
    baseline = euler_xyz_matrix((0.0, 0.0, 0.0))
    perturbed = euler_xyz_matrix((1.0e-6, 0.0, 0.0))
    distance = angular_distance_radians(baseline, perturbed)

    assert distance > 0.0
    assert abs(distance - 1.0e-6) < 1.0e-8