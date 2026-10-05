import pytest

from research.semantic_invariants.embodied_repair_synthesis import (
    RepairExample,
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
