from research.semantic_invariants.embodied_repair_runtime import (
    RepairGuardRejectedError,
    RepairNotVerifiedError,
    SemanticRepairMediator,
)
from research.semantic_invariants.embodied_repair_verification import (
    verify_repair_against_heldout,
)
from research.semantic_invariants.embodied_repair_synthesis import (
    RepairExample,
    build_vector_repair_catalog,
    synthesize_minimal_repair,
)


def _sign_repair():
    catalog = build_vector_repair_catalog(3)
    result = synthesize_minimal_repair(
        [
            RepairExample((-1.0, 2.0, -3.0), (1.0, 2.0, 3.0)),
            RepairExample((-4.0, 5.0, -6.0), (4.0, 5.0, 6.0)),
        ],
        catalog,
        max_depth=1,
        allowed_families={"sign"},
    )
    assert result.program is not None
    return result.program


def _certified_mediator(program=None, *, guard=None):
    program = program or _sign_repair()
    contract_id = "embodied/representation/controller-roundtrip@0.2"
    certificate = verify_repair_against_heldout(
        contract_id=contract_id,
        program=program,
        heldout=[
            RepairExample((-0.25, 0.5, -0.75), (0.25, 0.5, 0.75)),
            RepairExample((-0.8, -0.2, -0.1), (0.8, -0.2, 0.1)),
        ],
        verifier_id="independent-heldout-v1",
    )
    return SemanticRepairMediator.from_certificate(
        contract_id=contract_id,
        program=program,
        certificate=certificate,
        guard=guard,
    )


def test_verified_repair_mediates_and_records_requested_vs_executed_semantics():
    mediator = _certified_mediator()

    receipt = mediator.mediate(
        (-2.0, 1.0, -0.5),
        metadata={"source": "unit-test"},
    )

    assert receipt.requested == (-2.0, 1.0, -0.5)
    assert receipt.executed == (2.0, 1.0, 0.5)
    assert receipt.requested != receipt.executed
    assert receipt.program == "sign(-1, 1, -1)"
    assert len(receipt.provenance_digest) == 64
    assert mediator.replay_matches(receipt)


def test_runtime_installation_fails_closed_for_unverified_candidate():
    try:
        SemanticRepairMediator(
            contract_id="embodied/test@0.1",
            program=_sign_repair(),
            certificate=None,  # type: ignore[arg-type]
        )
    except RepairNotVerifiedError:
        pass
    else:
        raise AssertionError("unverified repairs must never be installed")


def test_existing_safety_guard_observes_repaired_action():
    seen = []

    def guard(executed, context):
        seen.append(executed)
        return max(abs(item) for item in executed) <= 1.0

    mediator = _certified_mediator(guard=guard)

    receipt = mediator.mediate((-0.4, 0.2, -0.1))
    assert seen == [receipt.executed]


def test_guard_rejection_prevents_execution_receipt():
    mediator = _certified_mediator(guard=lambda executed, context: False)

    try:
        mediator.mediate((-0.4, 0.2, -0.1))
    except RepairGuardRejectedError:
        pass
    else:
        raise AssertionError("a rejected repaired action must fail closed")


def test_provenance_digest_changes_when_semantic_execution_changes():
    mediator = _certified_mediator()

    first = mediator.mediate((-0.4, 0.2, -0.1))
    second = mediator.mediate((-0.5, 0.2, -0.1))

    assert first.provenance_digest != second.provenance_digest


def test_runtime_installs_verified_repair_from_program_bound_certificate():
    program = _sign_repair()
    contract_id = "embodied/representation/controller-roundtrip@0.2"
    certificate = verify_repair_against_heldout(
        contract_id=contract_id,
        program=program,
        heldout=[
            RepairExample((-0.25, 0.5, -0.75), (0.25, 0.5, 0.75)),
            RepairExample((-0.8, -0.2, -0.1), (0.8, -0.2, 0.1)),
        ],
        verifier_id="independent-heldout-v1",
    )

    mediator = SemanticRepairMediator.from_certificate(
        contract_id=contract_id,
        program=program,
        certificate=certificate,
    )
    receipt = mediator.mediate((-0.3, 0.4, -0.2))

    assert receipt.executed == (0.3, 0.4, 0.2)


def test_runtime_rejects_certificate_reuse_for_other_contract():
    program = _sign_repair()
    certificate = verify_repair_against_heldout(
        contract_id="embodied/a@0.1",
        program=program,
        heldout=[RepairExample((-1.0, 2.0, -3.0), (1.0, 2.0, 3.0))],
        verifier_id="independent-heldout-v1",
    )

    try:
        SemanticRepairMediator.from_certificate(
            contract_id="embodied/b@0.1",
            program=program,
            certificate=certificate,
        )
    except RepairNotVerifiedError:
        pass
    else:
        raise AssertionError("certificate must be bound to its contract")