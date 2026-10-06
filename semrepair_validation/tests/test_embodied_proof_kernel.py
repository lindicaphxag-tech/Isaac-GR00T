from research.semantic_invariants.embodied_compilation_certificate import (
    issue_compilation_certificate,
)
from research.semantic_invariants.embodied_proof_kernel import (
    verify_semantic_compilation,
)
from research.semantic_invariants.embodied_refinement import (
    SemanticEvidence,
    refine_executed_action,
)
from research.semantic_invariants.embodied_semantic_compiler import (
    compile_semantic_boundary,
)
from research.semantic_invariants.embodied_semantic_types import (
    SemanticAdapter,
    SemanticTensorType,
)


def test_full_proof_kernel_accepts_refinement_plus_unique_adapter_chain():
    source = SemanticTensorType(
        role="action",
        entity="ee_rotation_delta",
        representation="axis_angle",
        unit="rad",
        provenance="requested",
    )
    target = source.updated(
        representation="euler_xyz",
        provenance="executed",
    )
    receipt = SemanticEvidence(
        kind="execution_receipt",
        subject_id="action-7",
        claims={"executed": True, "controller_id": "arm"},
        issuer="controller",
    )
    refinement = refine_executed_action(source, receipt)
    adapter = SemanticAdapter(
        "axis-to-euler",
        requires={"representation": "axis_angle"},
        produces={"representation": "euler_xyz"},
        cost=1.0,
    )

    result = compile_semantic_boundary(
        source,
        target,
        refinements=(refinement,),
        adapters=(adapter,),
    )
    cert = issue_compilation_certificate(result, adapters=(adapter,))

    checked = verify_semantic_compilation(
        cert,
        adapters=(adapter,),
        refinement_evidence_by_digest={receipt.digest: receipt},
    )

    assert checked.valid
    assert checked.certificate_integrity
    assert checked.refinement.verified_refinements == 1
    assert checked.adapter.valid


def test_full_proof_kernel_fails_when_owner_receipt_is_missing():
    source = SemanticTensorType(
        role="action",
        entity="joint_command",
        representation="joint_position",
        provenance="requested",
    )
    target = source.updated(provenance="executed")
    receipt = SemanticEvidence(
        kind="execution_receipt",
        subject_id="action-7",
        claims={"executed": True, "controller_id": "arm"},
        issuer="controller",
    )
    refinement = refine_executed_action(source, receipt)

    result = compile_semantic_boundary(
        source,
        target,
        refinements=(refinement,),
    )
    cert = issue_compilation_certificate(result, adapters=())

    checked = verify_semantic_compilation(
        cert,
        adapters=(),
        refinement_evidence_by_digest={},
    )

    assert not checked.valid
    assert any("missing raw evidence" in reason for reason in checked.reasons)


def test_full_proof_kernel_fails_after_adapter_registry_drift():
    source = SemanticTensorType(
        role="action",
        entity="ee_rotation_delta",
        representation="axis_angle",
        unit="rad",
        provenance="requested",
    )
    target = source.updated(representation="euler_xyz")
    original = (
        SemanticAdapter(
            "axis-to-euler-a",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
    )
    result = compile_semantic_boundary(source, target, adapters=original)
    cert = issue_compilation_certificate(result, adapters=original)

    expanded = original + (
        SemanticAdapter(
            "axis-to-euler-b",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
    )
    checked = verify_semantic_compilation(cert, adapters=expanded)

    assert not checked.valid
    assert any("registry digest has drifted" in reason for reason in checked.reasons)
