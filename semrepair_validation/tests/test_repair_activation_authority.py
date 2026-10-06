from dataclasses import replace

from research.semantic_invariants.embodied_repair_synthesis import (
    RepairExample,
    build_vector_repair_catalog,
    synthesize_minimal_repair,
)
from research.semantic_invariants.embodied_repair_verification import (
    verify_repair_against_heldout,
)
from research.semantic_invariants.repair_activation_authority import (
    ActivationGateEvidence,
    ActivationGatedSemanticRepairMediator,
    RepairActivationRejected,
    issue_activation_gate_evidence,
    issue_repair_activation_certificate,
    verify_activation_gate_evidence,
    verify_repair_activation_certificate,
)


CONTRACT_ID = "embodied/repair-activation-test@0.4"


def _program():
    result = synthesize_minimal_repair(
        [
            RepairExample((-1.0, 2.0, -3.0), (1.0, 2.0, 3.0)),
            RepairExample((-4.0, 5.0, -6.0), (4.0, 5.0, 6.0)),
        ],
        build_vector_repair_catalog(3),
        max_depth=1,
        allowed_families={"sign"},
    )
    assert result.program is not None
    return result.program


def _local(program):
    return verify_repair_against_heldout(
        contract_id=CONTRACT_ID,
        program=program,
        heldout=[
            RepairExample((-0.25, 0.5, -0.75), (0.25, 0.5, 0.75)),
            RepairExample((-0.8, -0.2, -0.1), (0.8, -0.2, 0.1)),
        ],
        verifier_id="independent-heldout-v1",
    )


def _gate(program, name, *, status="pass", independent=False):
    kinds = {
        "anchor": "external-semantic-anchor",
        "value": "paired-semantic-fidelity",
        "protocol": "protocol-trace",
        "interaction": "factorial-repair-interaction",
        "execution": "paired-execution-replay",
    }
    return issue_activation_gate_evidence(
        gate=name,
        contract_id=CONTRACT_ID,
        program=program,
        status=status,
        evidence_digest=(name[0] * 64),
        evidence_scope_digest=(name[-1] * 64),
        evaluator_id=f"{name}-evaluator-v1",
        evidence_kind=kinds[name],
        independent_of_repair_path=independent,
    )


def _passing_gates(program):
    return (
        _gate(program, "anchor", independent=True),
        _gate(program, "value"),
        _gate(program, "protocol"),
        _gate(program, "interaction"),
        _gate(program, "execution", independent=True),
    )


def test_local_verification_plus_all_activation_gates_authorizes():
    program = _program()
    local = _local(program)
    activation = issue_repair_activation_certificate(
        contract_id=CONTRACT_ID,
        program=program,
        local_verification=local,
        gates=_passing_gates(program),
    )

    assert verify_repair_activation_certificate(
        activation,
        contract_id=CONTRACT_ID,
        program=program,
        local_verification=local,
    )


def test_missing_gate_fails_closed():
    program = _program()
    local = _local(program)

    try:
        issue_repair_activation_certificate(
            contract_id=CONTRACT_ID,
            program=program,
            local_verification=local,
            gates=_passing_gates(program)[:-1],
        )
    except RepairActivationRejected:
        pass
    else:
        raise AssertionError("missing execution gate must reject activation")


def test_unknown_or_failed_gate_fails_closed():
    program = _program()
    local = _local(program)
    gates = list(_passing_gates(program))
    gates[3] = _gate(program, "interaction", status="unknown")

    try:
        issue_repair_activation_certificate(
            contract_id=CONTRACT_ID,
            program=program,
            local_verification=local,
            gates=gates,
        )
    except RepairActivationRejected:
        pass
    else:
        raise AssertionError("unknown interaction evidence must block activation")


def test_circular_anchor_evidence_cannot_authorize():
    program = _program()
    local = _local(program)
    gates = list(_passing_gates(program))
    gates[0] = _gate(program, "anchor", independent=False)

    try:
        issue_repair_activation_certificate(
            contract_id=CONTRACT_ID,
            program=program,
            local_verification=local,
            gates=gates,
        )
    except RepairActivationRejected:
        pass
    else:
        raise AssertionError("self-consistency cannot substitute for an external anchor")


def test_execution_gate_must_be_independent_of_local_semantic_path():
    program = _program()
    local = _local(program)
    gates = list(_passing_gates(program))
    gates[-1] = _gate(program, "execution", independent=False)

    try:
        issue_repair_activation_certificate(
            contract_id=CONTRACT_ID,
            program=program,
            local_verification=local,
            gates=gates,
        )
    except RepairActivationRejected:
        pass
    else:
        raise AssertionError("local semantic evidence cannot self-authorize execution")


def test_tampered_activation_digest_is_rejected():
    program = _program()
    local = _local(program)
    activation = issue_repair_activation_certificate(
        contract_id=CONTRACT_ID,
        program=program,
        local_verification=local,
        gates=_passing_gates(program),
    )
    tampered = replace(activation, decision_digest="0" * 64)

    assert not verify_repair_activation_certificate(
        tampered,
        contract_id=CONTRACT_ID,
        program=program,
        local_verification=local,
    )


def test_local_certificate_cannot_be_rebound_to_different_program():
    original = _program()
    local = _local(original)

    other_result = synthesize_minimal_repair(
        [
            RepairExample((1.0, -2.0, 3.0), (1.0, 2.0, 3.0)),
            RepairExample((4.0, -5.0, 6.0), (4.0, 5.0, 6.0)),
        ],
        build_vector_repair_catalog(3),
        max_depth=1,
        allowed_families={"sign"},
    )
    assert other_result.program is not None

    try:
        issue_repair_activation_certificate(
            contract_id=CONTRACT_ID,
            program=other_result.program,
            local_verification=local,
            gates=_passing_gates(other_result.program),
        )
    except RepairActivationRejected:
        pass
    else:
        raise AssertionError("activation must remain bound to verified executable identity")


def test_tampered_gate_evidence_cannot_be_reused_as_authority():
    program = _program()
    local = _local(program)
    gate = _passing_gates(program)[0]
    tampered = replace(gate, evidence_digest="f" * 64)

    assert not verify_activation_gate_evidence(
        tampered,
        contract_id=CONTRACT_ID,
        program=program,
    )

    gates = list(_passing_gates(program))
    gates[0] = tampered
    try:
        issue_repair_activation_certificate(
            contract_id=CONTRACT_ID,
            program=program,
            local_verification=local,
            gates=gates,
        )
    except RepairActivationRejected:
        pass
    else:
        raise AssertionError("tampered gate evidence must not authorize activation")


def test_gate_evidence_is_bound_to_program_identity():
    program = _program()
    gate = _passing_gates(program)[0]

    other_result = synthesize_minimal_repair(
        [
            RepairExample((1.0, -2.0, 3.0), (1.0, 2.0, 3.0)),
            RepairExample((4.0, -5.0, 6.0), (4.0, 5.0, 6.0)),
        ],
        build_vector_repair_catalog(3),
        max_depth=1,
        allowed_families={"sign"},
    )
    assert other_result.program is not None

    assert not verify_activation_gate_evidence(
        gate,
        contract_id=CONTRACT_ID,
        program=other_result.program,
    )


def test_activation_gated_mediator_carries_authority_digest():
    program = _program()
    local = _local(program)
    activation = issue_repair_activation_certificate(
        contract_id=CONTRACT_ID,
        program=program,
        local_verification=local,
        gates=_passing_gates(program),
    )
    mediator = ActivationGatedSemanticRepairMediator(
        contract_id=CONTRACT_ID,
        program=program,
        local_verification=local,
        activation_certificate=activation,
    )

    receipt = mediator.mediate((-0.4, 0.2, -0.1))

    assert receipt.executed == (0.4, 0.2, 0.1)
    assert receipt.metadata["activation_certificate_digest"] == activation.decision_digest
    assert receipt.metadata["activation_gate_status"] == {
        "anchor": "pass",
        "value": "pass",
        "protocol": "pass",
        "interaction": "pass",
        "execution": "pass",
    }
