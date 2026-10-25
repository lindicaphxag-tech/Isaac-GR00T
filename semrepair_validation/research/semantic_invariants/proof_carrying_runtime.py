"""Proof-carrying runtime for reliable embodied execution.

This is the single authority entry point that connects:

semantic compilation proof
-> effect certificate
-> physical-effect intent
-> ambiguity-safe execution
-> independent observation-plane commit

Search, synthesis, planning, and executor acknowledgements remain untrusted.
Every physical transition is re-authorized against the current semantic
environment before it can proceed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, TypeVar, Generic

from .embodied_compilation_certificate import SemanticCompilationCertificate
from .embodied_refinement import SemanticEvidence
from .embodied_semantic_types import SemanticAdapter
from .semantic_effect_certificate import SemanticEffectCertificate
from .semantic_effect_commit import CommitOutcome, SemanticEffectIntent
from .semantic_effect_evidence import EffectObservation
from .semantic_effect_proof_kernel import (
    EmbodiedExecutionProofResult,
    verify_embodied_execution_bundle,
)
from .semantic_effect_runtime import (
    CertifiedResolution,
    CertifiedSemanticEffectRuntime,
)


State = TypeVar("State")


class ExecutionProofRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class AuthorizedExecution:
    compilation_digest: str
    effect_certificate_digest: str
    effect_id: str
    semantic_contract_id: str
    proof: EmbodiedExecutionProofResult


class ProofCarryingSemanticRuntime(Generic[State]):
    """Fail-closed authority wrapper for physical embodied effects."""

    def __init__(self, runtime: CertifiedSemanticEffectRuntime[State] | None = None) -> None:
        self.runtime = runtime or CertifiedSemanticEffectRuntime()

    def authorize(
        self,
        compilation_certificate: SemanticCompilationCertificate,
        effect_certificate: SemanticEffectCertificate,
        intent: SemanticEffectIntent[State],
        *,
        semantic_contract_id: str,
        adapters: Sequence[SemanticAdapter],
        adapter_evidence_identity: Mapping[str, str] | None = None,
        refinement_evidence_by_digest: Mapping[str, SemanticEvidence] | None = None,
        model_depth: int = 5,
    ) -> AuthorizedExecution:
        proof = verify_embodied_execution_bundle(
            compilation_certificate,
            effect_certificate,
            intent,
            adapters=adapters,
            adapter_evidence_identity=adapter_evidence_identity,
            refinement_evidence_by_digest=refinement_evidence_by_digest,
            model_depth=model_depth,
        )
        if not proof.valid:
            raise ExecutionProofRejected("; ".join(proof.reasons) or "execution proof rejected")
        if effect_certificate.semantic_contract_id != semantic_contract_id:
            raise ExecutionProofRejected("effect certificate semantic contract mismatch")
        return AuthorizedExecution(
            compilation_digest=compilation_certificate.decision_digest,
            effect_certificate_digest=effect_certificate.decision_digest,
            effect_id=intent.effect_id,
            semantic_contract_id=semantic_contract_id,
            proof=proof,
        )

    def prepare(
        self,
        compilation_certificate: SemanticCompilationCertificate,
        effect_certificate: SemanticEffectCertificate,
        intent: SemanticEffectIntent[State],
        *,
        semantic_contract_id: str,
        adapters: Sequence[SemanticAdapter],
        current_state: State,
        current_dependency_version: str,
        adapter_evidence_identity: Mapping[str, str] | None = None,
        refinement_evidence_by_digest: Mapping[str, SemanticEvidence] | None = None,
        model_depth: int = 5,
    ) -> tuple[AuthorizedExecution, CommitOutcome]:
        auth = self.authorize(
            compilation_certificate,
            effect_certificate,
            intent,
            semantic_contract_id=semantic_contract_id,
            adapters=adapters,
            adapter_evidence_identity=adapter_evidence_identity,
            refinement_evidence_by_digest=refinement_evidence_by_digest,
            model_depth=model_depth,
        )
        if current_dependency_version != auth.compilation_digest:
            raise ExecutionProofRejected("runtime dependency is stale relative to compilation proof")
        outcome = self.runtime.prepare(
            intent,
            effect_certificate,
            semantic_contract_id=semantic_contract_id,
            current_state=current_state,
            current_dependency_version=current_dependency_version,
        )
        return auth, outcome

    def mark_dispatched(
        self,
        authorization: AuthorizedExecution,
        compilation_certificate: SemanticCompilationCertificate,
        effect_certificate: SemanticEffectCertificate,
        intent: SemanticEffectIntent[State],
        *,
        adapters: Sequence[SemanticAdapter],
        adapter_evidence_identity: Mapping[str, str] | None = None,
        refinement_evidence_by_digest: Mapping[str, SemanticEvidence] | None = None,
        model_depth: int = 5,
    ):
        refreshed = self.authorize(
            compilation_certificate,
            effect_certificate,
            intent,
            semantic_contract_id=authorization.semantic_contract_id,
            adapters=adapters,
            adapter_evidence_identity=adapter_evidence_identity,
            refinement_evidence_by_digest=refinement_evidence_by_digest,
            model_depth=model_depth,
        )
        if refreshed.compilation_digest != authorization.compilation_digest:
            raise ExecutionProofRejected("semantic compilation changed after preparation")
        if refreshed.effect_certificate_digest != authorization.effect_certificate_digest:
            raise ExecutionProofRejected("effect certificate changed after preparation")
        return self.runtime.mark_dispatched(
            intent,
            effect_certificate,
            semantic_contract_id=authorization.semantic_contract_id,
        )

    def resolve(
        self,
        authorization: AuthorizedExecution,
        compilation_certificate: SemanticCompilationCertificate,
        effect_certificate: SemanticEffectCertificate,
        intent: SemanticEffectIntent[State],
        observations: Sequence[EffectObservation[State]],
        *,
        adapters: Sequence[SemanticAdapter],
        adapter_evidence_identity: Mapping[str, str] | None = None,
        refinement_evidence_by_digest: Mapping[str, SemanticEvidence] | None = None,
        model_depth: int = 5,
        executor_acknowledged_success: bool = False,
        executor_acknowledged_abort: bool = False,
    ) -> CertifiedResolution:
        refreshed = self.authorize(
            compilation_certificate,
            effect_certificate,
            intent,
            semantic_contract_id=authorization.semantic_contract_id,
            adapters=adapters,
            adapter_evidence_identity=adapter_evidence_identity,
            refinement_evidence_by_digest=refinement_evidence_by_digest,
            model_depth=model_depth,
        )
        if refreshed.compilation_digest != authorization.compilation_digest:
            raise ExecutionProofRejected("semantic compilation changed before commit")
        if refreshed.effect_certificate_digest != authorization.effect_certificate_digest:
            raise ExecutionProofRejected("effect certificate changed before commit")
        return self.runtime.resolve(
            intent,
            effect_certificate,
            observations,
            semantic_contract_id=authorization.semantic_contract_id,
            executor_acknowledged_success=executor_acknowledged_success,
            executor_acknowledged_abort=executor_acknowledged_abort,
        )