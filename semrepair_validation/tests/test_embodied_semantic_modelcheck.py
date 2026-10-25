from research.semantic_invariants.embodied_semantic_modelcheck import (
    check_repair_optimality,
)
from research.semantic_invariants.embodied_semantic_types import (
    SemanticAdapter,
    SemanticTensorType,
)


def _rotation(rep):
    return SemanticTensorType(
        role="action",
        entity="ee_rotation_delta",
        representation=rep,
        provenance="requested",
    )


def test_finite_model_check_confirms_repair_optimality():
    axis = _rotation("axis_angle")
    quat = _rotation("quaternion")
    euler = _rotation("euler_xyz")
    adapters = [
        SemanticAdapter(
            "axis-to-quat",
            requires={"representation": "axis_angle"},
            produces={"representation": "quaternion"},
            cost=1.0,
        ),
        SemanticAdapter(
            "quat-to-euler",
            requires={"representation": "quaternion"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
        SemanticAdapter(
            "axis-to-euler-expensive",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=3.0,
        ),
    ]

    report = check_repair_optimality([axis, quat, euler], adapters, max_steps=3)

    assert report.passed
    assert report.checked_pairs == 9
    assert report.optimality_violations == 0
    assert report.nonforgeable_violations == 0
