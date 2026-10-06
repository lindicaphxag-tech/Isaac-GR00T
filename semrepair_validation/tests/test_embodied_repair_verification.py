import pytest

from research.semantic_invariants.embodied_repair_synthesis import (
    RepairExample,
    RepairPrimitive,
    RepairProgram,
    build_vector_repair_catalog,
    synthesize_minimal_repair,
)
from research.semantic_invariants.embodied_repair_verification import (
    RepairVerificationFailed,
    certificate_matches_program,
    verify_repair_against_heldout,
)


def _sign_program():
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


def test_heldout_bank_issues_program_bound_certificate():
    program = _sign_program()
    cert = verify_repair_against_heldout(
        contract_id="embodied/test@0.1",
        program=program,
        heldout=[
            RepairExample((-0.5, 0.25, -0.75), (0.5, 0.25, 0.75)),
            RepairExample((-2.0, -1.0, -3.0), (2.0, -1.0, 3.0)),
        ],
        verifier_id="heldout-bank-v1",
    )

    assert cert.status == "verified"
    assert cert.cases == 2
    assert certificate_matches_program(
        cert,
        contract_id="embodied/test@0.1",
        program=program,
    )


def test_failed_heldout_case_does_not_issue_certificate():
    program = _sign_program()

    with pytest.raises(RepairVerificationFailed):
        verify_repair_against_heldout(
            contract_id="embodied/test@0.1",
            program=program,
            heldout=[
                RepairExample((-0.5, 0.25, -0.75), (99.0, 0.25, 0.75)),
            ],
            verifier_id="heldout-bank-v1",
        )


def test_certificate_is_not_transferable_to_other_contract():
    program = _sign_program()
    cert = verify_repair_against_heldout(
        contract_id="embodied/a@0.1",
        program=program,
        heldout=[RepairExample((-1.0, 2.0, -3.0), (1.0, 2.0, 3.0))],
        verifier_id="heldout-bank-v1",
    )

    assert not certificate_matches_program(
        cert,
        contract_id="embodied/b@0.1",
        program=program,
    )


def test_certificate_rejects_same_named_primitive_with_different_implementation_identity():
    program = _sign_program()
    cert = verify_repair_against_heldout(
        contract_id="embodied/test@0.1",
        program=program,
        heldout=[RepairExample((-1.0, 2.0, -3.0), (1.0, 2.0, 3.0))],
        verifier_id="heldout-bank-v1",
    )

    original = program.operations[0]
    substituted = RepairProgram(
        (
            RepairPrimitive(
                name=original.name,
                family=original.family,
                cost=original.cost,
                apply_fn=original.apply_fn,
                implementation_id="attacker-v1:same-name-different-code",
            ),
        )
    )

    assert not certificate_matches_program(
        cert,
        contract_id="embodied/test@0.1",
        program=substituted,
    )


def test_unversioned_primitive_cannot_receive_installable_certificate():
    program = RepairProgram(
        (
            RepairPrimitive(
                name="anonymous-sign",
                family="sign",
                cost=1,
                apply_fn=lambda value, context: tuple(-x for x in value),
            ),
        )
    )

    with pytest.raises(RepairVerificationFailed, match="implementation identity"):
        verify_repair_against_heldout(
            contract_id="embodied/test@0.1",
            program=program,
            heldout=[RepairExample((-1.0, -2.0), (1.0, 2.0))],
            verifier_id="heldout-bank-v1",
        )

def test_same_declared_identity_with_changed_apply_fn_invalidates_old_certificate():
    original = RepairProgram(
        (
            RepairPrimitive(
                name="semantic-cast",
                family="representation",
                cost=1,
                apply_fn=lambda value, context: tuple(value),
                implementation_id="semantic-cast@v1",
            ),
        )
    )
    cert = verify_repair_against_heldout(
        contract_id="embodied/implementation-binding@0.1",
        program=original,
        heldout=[RepairExample((1.0, 2.0), (1.0, 2.0))],
        verifier_id="implementation-binding-bank-v1",
    )

    drifted = RepairProgram(
        (
            RepairPrimitive(
                name="semantic-cast",
                family="representation",
                cost=1,
                apply_fn=lambda value, context: tuple(reversed(value)),
                implementation_id="semantic-cast@v1",
            ),
        )
    )

    assert not certificate_matches_program(
        cert,
        contract_id="embodied/implementation-binding@0.1",
        program=drifted,
    )
