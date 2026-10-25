import pytest

from research.semantic_invariants.embodied_semantic_source_bridge import (
    ConflictingSourceSemantics,
    compile_source_boundary,
    infer_type_from_source,
)
from research.semantic_invariants.embodied_semantic_types import (
    SemanticAdapter,
    SemanticTensorType,
)
from research.semantic_invariants.embodied_source_semantics import (
    DEFAULT_ROTATION_RULES,
    SourceSemanticRule,
)


def _rotation_base():
    return SemanticTensorType(
        role="action",
        entity="ee_rotation_delta",
        frame="root_aligned_body",
        representation=None,
        unit="rad",
        provenance="requested",
        embodiment="maniskill-robot",
    )


def test_real_maniskill_style_source_infers_conflict_and_compiles_repair():
    producer_source = """
delta_pose = np.r_[
    delta_pose.p,
    compact_axis_angle_from_quaternion(delta_pose.q),
]
"""
    consumer_source = """
delta_quat = matrix_to_quaternion(
    rotation_conversions.euler_angles_to_matrix(delta_rot, "XYZ")
)
"""
    adapters = (
        SemanticAdapter(
            "axis-angle-to-quaternion",
            requires={"representation": "axis_angle"},
            produces={"representation": "quaternion"},
            effects=("reencode",),
        ),
        SemanticAdapter(
            "quaternion-to-euler-xyz",
            requires={"representation": "quaternion"},
            produces={"representation": "euler_xyz"},
            effects=("reencode",),
        ),
    )

    result = compile_source_boundary(
        producer_source=producer_source,
        consumer_source=consumer_source,
        producer_base=_rotation_base(),
        consumer_base=_rotation_base(),
        rules=DEFAULT_ROTATION_RULES,
        adapters=adapters,
    )

    assert result.producer.semantic_type.representation == "axis_angle"
    assert result.consumer.semantic_type.representation == "euler_xyz"
    assert result.compilation.status == "repaired"
    assert [
        item.name for item in result.compilation.adapter_plan.adapters
    ] == [
        "axis-angle-to-quaternion",
        "quaternion-to-euler-xyz",
    ]
    assert result.producer.evidence[0].rule_id == "rotation/axis-angle-output"


def test_explicit_type_cannot_be_silently_overwritten_by_source():
    base = _rotation_base().updated(representation="euler_xyz")
    source = "encoded = compact_axis_angle_from_quaternion(delta_pose.q)"

    with pytest.raises(ConflictingSourceSemantics, match="conflicts with source"):
        infer_type_from_source(base, source, DEFAULT_ROTATION_RULES)


def test_conflicting_source_facts_fail_closed():
    rules = (
        SourceSemanticRule(
            rule_id="mode/a",
            callee_suffix="emit_a",
            facts={"mode": "a"},
        ),
        SourceSemanticRule(
            rule_id="mode/b",
            callee_suffix="emit_b",
            facts={"mode": "b"},
        ),
    )
    base = SemanticTensorType(role="action", entity="command", mode=None)

    with pytest.raises(ConflictingSourceSemantics, match="conflicting semantics"):
        infer_type_from_source(
            base,
            "x = emit_a(x)\ny = emit_b(y)",
            rules,
        )


def test_source_bridge_keeps_unknown_fields_unknown():
    base = SemanticTensorType(
        role="state",
        entity="joint_position",
        representation="vector",
        ordering=None,
    )
    result = infer_type_from_source(base, "x = x + 1", DEFAULT_ROTATION_RULES)

    assert result.semantic_type.ordering is None
    assert result.evidence == ()
