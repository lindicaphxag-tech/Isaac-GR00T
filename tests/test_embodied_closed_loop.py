import pytest

from research.semantic_invariants.embodied_closed_loop import run_joint_order_case
from research.semantic_invariants.embodied_runtime_adapters import (
    MissingRuntimeAdapter,
    execute_adapter_plan,
)
from research.semantic_invariants.embodied_semantic_types import (
    SemanticAdapter,
    SemanticTensorType,
    synthesize_unique_adapter_plan,
)


def test_semantic_compilation_changes_closed_loop_outcome():
    broken = run_joint_order_case(repaired=False, steps=3)
    repaired = run_joint_order_case(repaired=True, steps=3)

    assert broken.final_error > 1.0
    assert repaired.final_error == pytest.approx(0.0)
    assert repaired.adapter_applied == ("policy-to-controller-order",)


def test_runtime_execution_requires_certified_implementation():
    source = SemanticTensorType(
        role="action",
        entity="joint_delta",
        ordering="a",
        provenance="requested",
    )
    target = source.updated(ordering="b")
    adapter = SemanticAdapter(
        "a-to-b",
        requires={"ordering": "a"},
        produces={"ordering": "b"},
    )
    plan = synthesize_unique_adapter_plan(source, target, [adapter])

    with pytest.raises(MissingRuntimeAdapter):
        execute_adapter_plan((1.0, 2.0), plan, {})
