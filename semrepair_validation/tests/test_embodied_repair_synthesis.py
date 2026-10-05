from math import acos, cos, sin, sqrt

from research.semantic_invariants.embodied_repair_synthesis import (
    RepairExample,
    bank_counterexample_finder,
    build_vector_repair_catalog,
    cegis_repair,
    synthesize_minimal_repair,
)
from research.semantic_invariants.embodied_rotation_chain import euler_xyz_matrix


def _matrix_to_axis_angle(matrix):
    angle = acos(
        max(
            -1.0,
            min(
                1.0,
                (matrix[0][0] + matrix[1][1] + matrix[2][2] - 1.0) / 2.0,
            ),
        )
    )
    if angle < 1.0e-14:
        return (0.0, 0.0, 0.0)
    denominator = 2.0 * sin(angle)
    axis = (
        (matrix[2][1] - matrix[1][2]) / denominator,
        (matrix[0][2] - matrix[2][0]) / denominator,
        (matrix[1][0] - matrix[0][1]) / denominator,
    )
    return tuple(component * angle for component in axis)


def test_synthesizes_joint_order_repair():
    catalog = build_vector_repair_catalog(3)
    examples = [
        RepairExample((2.0, 3.0, 1.0), (1.0, 2.0, 3.0)),
        RepairExample((20.0, 30.0, 10.0), (10.0, 20.0, 30.0)),
    ]

    result = synthesize_minimal_repair(
        examples,
        catalog,
        max_depth=1,
        allowed_families={"ordering"},
    )

    assert result.status == "unique"
    assert result.program is not None
    assert result.program.name == "permute(2, 0, 1)"


def test_synthesizes_composed_order_and_sign_repair():
    catalog = build_vector_repair_catalog(3)
    examples = [
        RepairExample((2.0, -3.0, 1.0), (1.0, 2.0, 3.0)),
        RepairExample((5.0, -7.0, 4.0), (4.0, 5.0, 7.0)),
    ]

    result = synthesize_minimal_repair(
        examples,
        catalog,
        max_depth=2,
        allowed_families={"ordering", "sign"},
    )

    assert result.program is not None
    assert result.program.cost == 3
    assert result.program.apply(examples[0].observed) == examples[0].expected
    assert result.program.apply(examples[1].observed) == examples[1].expected


def test_synthesizes_rotation_representation_repair():
    catalog = build_vector_repair_catalog(3)
    targets = [(0.08, 0.08, 0.08), (0.04, -0.07, 0.06)]
    examples = [
        RepairExample(_matrix_to_axis_angle(euler_xyz_matrix(target)), target)
        for target in targets
    ]

    result = synthesize_minimal_repair(
        examples,
        catalog,
        max_depth=1,
        allowed_families={"rotation-representation"},
        atol=1.0e-9,
    )

    assert result.status == "unique"
    assert result.program is not None
    assert result.program.name == "axis-angle->euler-xyz"


def test_synthesizes_rigid_body_velocity_frame_repair():
    catalog = build_vector_repair_catalog(3)
    context_a = {"angular": (0.0, 0.0, 2.0), "com_offset": (0.1, 0.0, 0.0)}
    context_b = {"angular": (0.0, 3.0, 0.0), "com_offset": (0.0, 0.0, 0.2)}
    examples = [
        RepairExample((0.0, 0.2, 0.0), (0.0, 0.0, 0.0), context_a),
        RepairExample((0.6, 0.0, 0.0), (0.0, 0.0, 0.0), context_b),
    ]

    result = synthesize_minimal_repair(
        examples,
        catalog,
        max_depth=1,
        allowed_families={"rigid-body-frame"},
    )

    assert result.status == "unique"
    assert result.program is not None
    assert result.program.name == "com-velocity->link-velocity"


def test_cegis_uses_counterexample_to_escape_ambiguous_zero_example():
    catalog = build_vector_repair_catalog(3)
    seed = [RepairExample((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), label="uninformative")]
    held_out = [
        RepairExample((-1.0, 2.0, -3.0), (1.0, 2.0, 3.0), label="sign witness"),
        RepairExample((-4.0, 5.0, -6.0), (4.0, 5.0, 6.0), label="second witness"),
    ]

    result = cegis_repair(
        seed,
        catalog,
        counterexample_finder=bank_counterexample_finder(held_out),
        max_depth=1,
        allowed_families={"sign"},
    )

    assert result.status == "verified"
    assert result.rounds == 2
    assert result.program is not None
    assert result.program.name == "sign(-1, 1, -1)"


def test_unknown_repair_fails_closed():
    catalog = build_vector_repair_catalog(3)
    result = synthesize_minimal_repair(
        [RepairExample((1.0, 2.0, 3.0), (1.0, 4.0, 9.0))],
        catalog,
        max_depth=1,
        allowed_families={"ordering", "sign", "unit"},
    )

    assert result.status == "no_solution"
    assert result.program is None
