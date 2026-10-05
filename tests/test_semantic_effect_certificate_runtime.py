from dataclasses import replace

from research.semantic_invariants.semantic_effect_certificate import (
    EffectCommitPolicy,
    issue_effect_certificate,
    verify_effect_certificate,
)
from research.semantic_invariants.semantic_effect_commit import (
    EffectClass,
    EffectStatus,
    RuntimeDecision,
    SemanticEffectIntent,
)
from research.semantic_invariants.semantic_effect_evidence import EffectObservation
from research.semantic_invariants.semantic_effect_runtime import CertifiedSemanticEffectRuntime


CONTRACT = "embodied/effect/handoff@0.1"


def _intent(effect_class=EffectClass.IRREVERSIBLE):
    return SemanticEffectIntent(
        effect_id="handoff:episode-5:instrument-2",
        action_name="handoff_instrument",
        effect_class=effect_class,
        dependency_version="plan-v7",
        precondition=lambda state: state["holder"] == "robot",
        postcondition=lambda state: state["holder"] == "human",
    )


def test_certificate_binds_effect_identity_class_dependency_and_policy():
    intent = _intent()
    cert = issue_effect_certificate(
        intent,
        policy=EffectCommitPolicy(min_commit_planes=2),
        semantic_contract_id=CONTRACT,
    )

    assert verify_effect_certificate(cert)
    assert cert.effect_class == "irreversible"
    assert cert.policy["min_commit_planes"] == 2


def test_tampered_policy_invalidates_certificate():
    cert = issue_effect_certificate(
        _intent(),
        policy=EffectCommitPolicy(min_commit_planes=2),
        semantic_contract_id=CONTRACT,
    )
    tampered = replace(cert, policy={**cert.policy, "min_commit_planes": 1})

    assert not verify_effect_certificate(tampered)


def test_certified_runtime_requires_two_independent_planes_before_commit():
    intent = _intent()
    cert = issue_effect_certificate(
        intent,
        policy=EffectCommitPolicy(min_commit_planes=2),
        semantic_contract_id=CONTRACT,
    )
    runtime = CertifiedSemanticEffectRuntime()

    prepared = runtime.prepare(
        intent,
        cert,
        semantic_contract_id=CONTRACT,
        current_state={"holder": "robot"},
        current_dependency_version="plan-v7",
    )
    assert prepared.decision == RuntimeDecision.DISPATCH
    runtime.mark_dispatched(intent, cert, semantic_contract_id=CONTRACT)

    one_plane = runtime.resolve(
        intent,
        cert,
        (EffectObservation("camera", "camera", {"holder": "human"}),),
        semantic_contract_id=CONTRACT,
    )
    assert one_plane.outcome.status == EffectStatus.AMBIGUOUS
    assert one_plane.outcome.decision == RuntimeDecision.BLOCK


def test_certified_runtime_commits_when_policy_quorum_is_met():
    intent = _intent()
    cert = issue_effect_certificate(
        intent,
        policy=EffectCommitPolicy(min_commit_planes=2),
        semantic_contract_id=CONTRACT,
    )
    runtime = CertifiedSemanticEffectRuntime()
    runtime.prepare(
        intent,
        cert,
        semantic_contract_id=CONTRACT,
        current_state={"holder": "robot"},
        current_dependency_version="plan-v7",
    )
    runtime.mark_dispatched(intent, cert, semantic_contract_id=CONTRACT)

    result = runtime.resolve(
        intent,
        cert,
        (
            EffectObservation("camera", "camera", {"holder": "human"}),
            EffectObservation("force", "proprioception", {"holder": "human"}),
        ),
        semantic_contract_id=CONTRACT,
    )

    assert result.outcome.status == EffectStatus.COMMITTED
    assert result.outcome.decision == RuntimeDecision.ADVANCE
    assert result.certificate_digest == cert.decision_digest


def test_certificate_for_different_effect_class_is_rejected():
    intent = _intent()
    cert = issue_effect_certificate(
        _intent(effect_class=EffectClass.REVERSIBLE),
        policy=EffectCommitPolicy(),
        semantic_contract_id=CONTRACT,
    )
    runtime = CertifiedSemanticEffectRuntime()

    try:
        runtime.prepare(
            intent,
            cert,
            semantic_contract_id=CONTRACT,
            current_state={"holder": "robot"},
            current_dependency_version="plan-v7",
        )
    except ValueError as error:
        assert "does not match intent" in str(error)
    else:
        raise AssertionError("mismatched effect certificate must fail closed")


def test_certificate_for_different_semantic_contract_is_rejected():
    intent = _intent()
    cert = issue_effect_certificate(
        intent,
        policy=EffectCommitPolicy(),
        semantic_contract_id=CONTRACT,
    )
    runtime = CertifiedSemanticEffectRuntime()

    try:
        runtime.prepare(
            intent,
            cert,
            semantic_contract_id="embodied/effect/other@0.1",
            current_state={"holder": "robot"},
            current_dependency_version="plan-v7",
        )
    except ValueError as error:
        assert "semantic contract" in str(error)
    else:
        raise AssertionError("certificate must bind semantic contract identity")