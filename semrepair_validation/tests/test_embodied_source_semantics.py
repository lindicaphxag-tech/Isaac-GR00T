from research.semantic_invariants.embodied_source_semantics import (
    DEFAULT_ROTATION_RULES,
    SourceSemanticRule,
    compare_boundary_semantics,
    infer_source_semantic_facts,
)


def test_extracts_literal_xyz_euler_semantics():
    source = """
delta_quat = matrix_to_quaternion(
    rotation_conversions.euler_angles_to_matrix(delta_rot, "XYZ")
)
"""
    facts = infer_source_semantic_facts(source, DEFAULT_ROTATION_RULES)
    assert any(
        fact.field == "representation" and fact.value == "euler_xyz"
        for fact in facts
    )


def test_dynamic_rotation_convention_does_not_become_false_evidence():
    source = """
delta_quat = rotation_conversions.euler_angles_to_matrix(delta_rot, convention)
"""
    facts = infer_source_semantic_facts(source, DEFAULT_ROTATION_RULES)
    assert facts == ()


def test_detects_axis_angle_producer_to_euler_consumer_conflict():
    producer = "encoded = compact_axis_angle_from_quaternion(delta_pose.q)"
    consumer = 'matrix = euler_angles_to_matrix(delta_rot, "XYZ")'
    producer_facts = infer_source_semantic_facts(producer, DEFAULT_ROTATION_RULES)
    consumer_facts = infer_source_semantic_facts(consumer, DEFAULT_ROTATION_RULES)
    conflicts = compare_boundary_semantics(producer_facts, consumer_facts)
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.field == "representation"
    assert conflict.producer_value == "axis_angle"
    assert conflict.consumer_value == "euler_xyz"


def test_rule_registry_supports_units_without_core_code_changes():
    rules = [
        SourceSemanticRule(
            rule_id="units/cm-to-m-output",
            callee_suffix="centimeters_to_meters",
            facts={"unit": "m"},
        ),
        SourceSemanticRule(
            rule_id="units/consumer-centimeters",
            callee_suffix="consume_centimeters",
            facts={"unit": "cm"},
        ),
    ]
    producer = infer_source_semantic_facts("x = centimeters_to_meters(x)", rules)
    consumer = infer_source_semantic_facts("consume_centimeters(x)", rules)
    conflicts = compare_boundary_semantics(producer, consumer)
    assert len(conflicts) == 1
    assert conflicts[0].field == "unit"