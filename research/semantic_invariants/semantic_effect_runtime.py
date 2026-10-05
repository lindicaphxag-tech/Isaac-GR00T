"""Certified runtime wrapper for Semantic Effect Commit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

from .semantic_effect_certificate import (
    EffectCommitPolicy,
    SemanticEffectCertificate,
    assert_certificate_authorizes,
)
from .semantic_effect_commit import CommitOutcome, SemanticEffectIntent, SemanticEffectRuntime
from .semantic_effect_evidence import EffectEvidenceSummary, EffectObservation, resolve_with_triangulated_evidence


State = TypeVar("State")


@dataclass(frozen=True)
class CertifiedResolution:
    outcome: CommitOutcome
    evidence: EffectEvidenceSummary
    certificate_digest: str
    semantic_contract_id: str


class CertifiedSemanticEffectRuntime(Generic[State]):
    """Fail closed unless the effect intent and evidence policy are certified."""

    def __init__(self, runtime: SemanticEffectRuntime[State] | None = None) -> None:
        self.runtime = runtime or SemanticEffectRuntime()

    def prepare(
        self,
        intent: SemanticEffectIntent[State],
        certificate: SemanticEffectCertificate,
        *,
        semantic_contract_id: str,
        current_state: State,
        current_dependency_version: str,
    ) -> CommitOutcome:
        assert_certificate_authorizes(
            certificate,
            intent,
            semantic_contract_id=semantic_contract_id,
        )
        return self.runtime.prepare(
            intent,
            current_state=current_state,
            current_dependency_version=current_dependency_version,
        )

    def mark_dispatched(
        self,
        intent: SemanticEffectIntent[State],
        certificate: SemanticEffectCertificate,
        *,
        semantic_contract_id: str,
    ):
        assert_certificate_authorizes(
            certificate,
            intent,
            semantic_contract_id=semantic_contract_id,
        )
        return self.runtime.mark_dispatched(intent.effect_id)

    def resolve(
        self,
        intent: SemanticEffectIntent[State],
        certificate: SemanticEffectCertificate,
        observations: Sequence[EffectObservation[State]],
        *,
        semantic_contract_id: str,
        executor_acknowledged_success: bool = False,
        executor_acknowledged_abort: bool = False,
    ) -> CertifiedResolution:
        policy: EffectCommitPolicy = assert_certificate_authorizes(
            certificate,
            intent,
            semantic_contract_id=semantic_contract_id,
        )
        outcome, evidence = resolve_with_triangulated_evidence(
            self.runtime,
            intent,
            observations,
            min_commit_planes=policy.min_commit_planes,
            min_abort_planes=policy.min_abort_planes,
            executor_acknowledged_success=executor_acknowledged_success,
            executor_acknowledged_abort=executor_acknowledged_abort,
        )
        return CertifiedResolution(
            outcome=outcome,
            evidence=evidence,
            certificate_digest=certificate.decision_digest,
            semantic_contract_id=semantic_contract_id,
        )