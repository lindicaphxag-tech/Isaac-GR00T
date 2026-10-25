from dataclasses import replace

from research.semantic_invariants.embodied_compilation_certificate import (
    issue_compilation_certificate,
)
from research.semantic_invariants.embodied_refinement import (
    SemanticEvidence,
    refine_executed_action,
    refine_sensor_freshness,
)
from research.semantic_invariants.embodied_refinement_verifier import (
    verify_refinement_records,
)
from research.semantic_invariants.embodied_semantic_compiler import (
    CompilationResult,
    compile_semantic_boundary,
)
from research.semantic_invariants.embodied_semantic_types import SemanticTensorType


def test_refinement_checker_recomputes_execution_receipt_semantics():
    source = SemanticTensorType(
        role="action",
        entity="joint_command",
        representation="joint_position",
        provenance="requested",
    )
    target = source.updated(provenance="executed")
    evidence = SemanticEvidence(
        kind="execution_receipt",
        subject_id="a-1",
        claims={"executed": True, "controller_id": "arm"},
        issuer="controller",
    )
    refinement = refine_executed_action(source, evidence)
    result = compile_semantic_boundary(
        source,
        target,
        refinements=(refinement,),
    )
    cert = issue_compilation_certificate(result, adapters=())

    check = verify_refinement_records(
        cert,
        evidence_by_digest={evidence.digest: evidence},
    )

    assert check.valid
    assert check.verified_refinements == 1


def test_refinement_checker_rejects_freshness_claim_with_stale_raw_sample():
    source = SemanticTensorType(
        role="observation",
        entity="camera_frame",
        freshness="stale",
        provenance="observed",
    )
    target = source.updated(freshness="fresh")
    fresh = SemanticEvidence(
        kind="sensor_sample",
        subject_id="frame-2",
        claims={"sample_time": 10.1},
        issuer="camera",
    )
    refinement = refine_sensor_freshness(source, fresh, due_time=10.0)
    result = compile_semantic_boundary(
        source,
        target,
        refinements=(refinement,),
    )
    cert = issue_compilation_certificate(result, adapters=())

    stale = SemanticEvidence(
        kind="sensor_sample",
        subject_id="frame-2",
        claims={"sample_time": 9.9},
        issuer="camera",
    )
    forged_map = {fresh.digest: stale}

    check = verify_refinement_records(cert, evidence_by_digest=forged_map)

    assert not check.valid
    assert any("digest mismatch" in reason for reason in check.reasons)


def test_refinement_checker_rejects_forged_after_type_even_with_rehashed_certificate_record():
    source = SemanticTensorType(
        role="action",
        entity="joint_command",
        representation="joint_position",
        provenance="requested",
    )
    target = source.updated(provenance="executed")
    evidence = SemanticEvidence(
        kind="execution_receipt",
        subject_id="a-1",
        claims={"executed": True, "controller_id": "arm"},
        issuer="controller",
    )
    refinement = refine_executed_action(source, evidence)
    result = compile_semantic_boundary(
        source,
        target,
        refinements=(refinement,),
    )
    cert = issue_compilation_certificate(result, adapters=())

    record = dict(cert.refinement_records[0])
    forged_after = dict(record["after"])
    forged_after["provenance"] = "requested"
    record["after"] = forged_after
    forged = replace(
        cert,
        refinement_records=(record,),
        refined_source=forged_after,
    )

    check = verify_refinement_records(
        forged,
        evidence_by_digest={evidence.digest: evidence},
    )

    assert not check.valid
    assert any("after-type mismatch" in reason for reason in check.reasons)


def test_no_refinement_records_require_inferred_and_refined_types_to_match():
    source = SemanticTensorType(
        role="state",
        entity="joint_position",
        representation="vector",
    )
    result = compile_semantic_boundary(source, source)
    cert = issue_compilation_certificate(result, adapters=())

    check = verify_refinement_records(cert, evidence_by_digest={})

    assert check.valid
    assert check.verified_refinements == 0
