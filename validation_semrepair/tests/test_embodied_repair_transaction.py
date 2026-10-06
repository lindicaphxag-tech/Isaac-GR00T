import pytest

from validation_semrepair.embodied_repair_interactions import (
    RepairOutcome,
    analyze_repair_lattice,
)
from validation_semrepair.embodied_repair_transaction import (
    RepairTransactionEvidenceMismatch,
    RepairTransactionRejected,
    ShadowProbeObservation,
    authorize_semantic_repair_commit,
    prepare_semantic_repair_transaction,
)
from validation_semrepair.embodied_semantic_observability import (
    SemanticDiagnosisHypothesis,
    synthesize_minimal_diagnostic_taps,
)
from validation_semrepair.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
)


def _factor(name, transport, evidence):
    return SemanticTransportFactor(name, transport, evidence)


def _hypothesis(name, factors):
    return SemanticDiagnosisHypothesis(name=name, factors=tuple(factors))


def _interaction():
    return analyze_repair_lattice(
        subject="masked joint-order pipeline",
        metric="tracking_error",
        objective="minimize",
        repairs=("producer-order", "dispatch-order"),
        outcomes=(
            RepairOutcome(frozenset(), 1.0e-7, "both-faults-cancel"),
            RepairOutcome(
                frozenset(("producer-order",)),
                3.6,
                "producer-only",
            ),
            RepairOutcome(
                frozenset(("dispatch-order",)),
                3.6,
                "dispatch-only",
            ),
            RepairOutcome(
                frozenset(("producer-order", "dispatch-order")),
                1.0e-7,
                "fully-correct",
            ),
        ),
        tolerance=1.0e-6,
    )


def _diagnosis(*, fault_evidence="fault/p"):
    identity = MonomialSemanticTransport.identity(2)
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    correct = _hypothesis(
        "correct",
        (
            _factor("producer", identity, "correct/p"),
            _factor("dispatch", identity, "correct/d"),
        ),
    )
    masked = _hypothesis(
        "double-swap",
        (
            _factor("producer", swap, fault_evidence),
            _factor("dispatch", swap, "fault/d"),
        ),
    )
    return synthesize_minimal_diagnostic_taps((correct, masked))


def _certificate():
    return prepare_semantic_repair_transaction(
        interaction=_interaction(),
        diagnosis=_diagnosis(),
        requested_repairs=("producer-order",),
        implementations={
            "producer-order": "producer-fix@abc",
            "dispatch-order": "dispatch-fix@def",
        },
        repair_hypotheses={
            "producer-order": "double-swap",
            "dispatch-order": "double-swap",
        },
        reference_hypothesis="correct",
    )


def _passing_observation(certificate):
    requirement = certificate.probe_requirements[0]
    return ShadowProbeObservation(
        fault_hypothesis=requirement.fault_hypothesis,
        tap_after_factor=requirement.tap_after_factor,
        basis_index=requirement.basis_index,
        observed=requirement.expected_reference,
        evidence_id="shadow-run/1",
    )


def test_prepare_expands_single_hotfix_to_atomic_closure():
    certificate = _certificate()

    assert certificate.requested_repairs == ("producer-order",)
    assert certificate.atomic_closure == (
        "dispatch-order",
        "producer-order",
    )
    assert len(certificate.probe_requirements) == 1
    assert certificate.probe_requirements[0].tap_after_factor == "producer"


def test_shadow_verified_atomic_bundle_is_authorized():
    interaction = _interaction()
    diagnosis = _diagnosis()
    certificate = prepare_semantic_repair_transaction(
        interaction=interaction,
        diagnosis=diagnosis,
        requested_repairs=("producer-order",),
        implementations={
            "producer-order": "producer-fix@abc",
            "dispatch-order": "dispatch-fix@def",
        },
        repair_hypotheses={
            "producer-order": "double-swap",
            "dispatch-order": "double-swap",
        },
        reference_hypothesis="correct",
    )

    authorization = authorize_semantic_repair_commit(
        interaction=interaction,
        diagnosis=diagnosis,
        certificate=certificate,
        requested_implementations={
            "producer-order": "producer-fix@abc",
            "dispatch-order": "dispatch-fix@def",
        },
        observations=(_passing_observation(certificate),),
    )

    assert authorization.repairs == (
        "dispatch-order",
        "producer-order",
    )
    assert authorization.transaction_digest == certificate.digest
    assert authorization.observation_digest


def test_partial_hotfix_cannot_commit():
    interaction = _interaction()
    diagnosis = _diagnosis()
    certificate = prepare_semantic_repair_transaction(
        interaction=interaction,
        diagnosis=diagnosis,
        requested_repairs=("producer-order",),
        implementations={
            "producer-order": "producer-fix@abc",
            "dispatch-order": "dispatch-fix@def",
        },
        repair_hypotheses={
            "producer-order": "double-swap",
            "dispatch-order": "double-swap",
        },
        reference_hypothesis="correct",
    )

    with pytest.raises(RepairTransactionRejected, match="atomic closure"):
        authorize_semantic_repair_commit(
            interaction=interaction,
            diagnosis=diagnosis,
            certificate=certificate,
            requested_implementations={
                "producer-order": "producer-fix@abc",
            },
            observations=(_passing_observation(certificate),),
        )


def test_wrong_shadow_semantics_reject_commit():
    interaction = _interaction()
    diagnosis = _diagnosis()
    certificate = prepare_semantic_repair_transaction(
        interaction=interaction,
        diagnosis=diagnosis,
        requested_repairs=("producer-order",),
        implementations={
            "producer-order": "producer-fix@abc",
            "dispatch-order": "dispatch-fix@def",
        },
        repair_hypotheses={
            "producer-order": "double-swap",
            "dispatch-order": "double-swap",
        },
        reference_hypothesis="correct",
    )
    requirement = certificate.probe_requirements[0]
    wrong = ShadowProbeObservation(
        fault_hypothesis=requirement.fault_hypothesis,
        tap_after_factor=requirement.tap_after_factor,
        basis_index=requirement.basis_index,
        observed=(0.0, 1.0),
        evidence_id="shadow-run/wrong",
    )

    with pytest.raises(RepairTransactionRejected, match="reference semantics"):
        authorize_semantic_repair_commit(
            interaction=interaction,
            diagnosis=diagnosis,
            certificate=certificate,
            requested_implementations={
                "producer-order": "producer-fix@abc",
                "dispatch-order": "dispatch-fix@def",
            },
            observations=(wrong,),
        )


def test_diagnosis_evidence_drift_invalidates_prepared_transaction():
    interaction = _interaction()
    original = _diagnosis(fault_evidence="fault/source-v1")
    certificate = prepare_semantic_repair_transaction(
        interaction=interaction,
        diagnosis=original,
        requested_repairs=("producer-order",),
        implementations={
            "producer-order": "producer-fix@abc",
            "dispatch-order": "dispatch-fix@def",
        },
        repair_hypotheses={
            "producer-order": "double-swap",
            "dispatch-order": "double-swap",
        },
        reference_hypothesis="correct",
    )
    changed = _diagnosis(fault_evidence="fault/source-v2")

    with pytest.raises(
        RepairTransactionEvidenceMismatch,
        match="diagnosis evidence changed",
    ):
        authorize_semantic_repair_commit(
            interaction=interaction,
            diagnosis=changed,
            certificate=certificate,
            requested_implementations={
                "producer-order": "producer-fix@abc",
                "dispatch-order": "dispatch-fix@def",
            },
            observations=(_passing_observation(certificate),),
        )


def test_implementation_drift_invalidates_prepared_transaction():
    interaction = _interaction()
    diagnosis = _diagnosis()
    certificate = prepare_semantic_repair_transaction(
        interaction=interaction,
        diagnosis=diagnosis,
        requested_repairs=("producer-order",),
        implementations={
            "producer-order": "producer-fix@abc",
            "dispatch-order": "dispatch-fix@def",
        },
        repair_hypotheses={
            "producer-order": "double-swap",
            "dispatch-order": "double-swap",
        },
        reference_hypothesis="correct",
    )

    with pytest.raises(
        RepairTransactionEvidenceMismatch,
        match="implementation drift",
    ):
        authorize_semantic_repair_commit(
            interaction=interaction,
            diagnosis=diagnosis,
            certificate=certificate,
            requested_implementations={
                "producer-order": "producer-fix@changed",
                "dispatch-order": "dispatch-fix@def",
            },
            observations=(_passing_observation(certificate),),
        )
