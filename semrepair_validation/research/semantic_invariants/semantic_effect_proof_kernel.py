"""Trusted kernel for proof-carrying embodied execution.

This bridges two previously separate guarantees:

1. semantic compilation is independently verified against a frozen adapter /
   evidence environment;
2. physical-effect execution is authorized by an effect certificate whose
   dependency version is bound to that exact compilation decision digest.

A stale or recompiled semantic adapter therefore invalidates the prepared
physical action instead of silently executing under changed meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .embodied_compilation_certificate import SemanticCompilationCertificate
from .embodied_proof_kernel import SemanticProofKernelResult, verify_semantic_compilation
from .embodied_refinement import SemanticEvidence
from .embodied_semantic_types import SemanticAdapter
from .semantic_effect_certificate import (
    SemanticEffectCertificate,
    certificate_matches_intent,
    verify_effect_certificate,
)
from .semantic_effect_commit import SemanticEffectIntent
from .semantic_effect_model_check import bounded_model_check


@dataclass(frozen=True)
class EmbodiedExecutionProofResult:
    valid: bool
    semantic_compilation: SemanticProofKernelResult
    effect_certificate_valid: bool
    effect_matches_intent: bool
    dependency_bound_to_compilation: bool
    runtime_model_safe: bool
    model_depth: int
    reasons: tuple[str, ...]


def verify_embodied_execution_bundle(
    compilation_certificate: SemanticCompilationCertificate,
    effect_certificate: SemanticEffectCertificate,
    intent: SemanticEffectIntent,
    *,
    adapters: Sequence[SemanticAdapter],
    adapter_evidence_identity: Mapping[str, str] | None = None,
    refinement_evidence_by_digest: Mapping[str, SemanticEvidence] | None = None,
    model_depth: int = 5,
) -> EmbodiedExecutionProofResult:
    """Verify compiler-to-physical-effect provenance without trusting search."""

    semantic = verify_semantic_compilation(
        compilation_certificate,
        adapters=adapters,
        adapter_evidence_identity=adapter_evidence_identity,
        refinement_evidence_by_digest=refinement_evidence_by_digest,
    )

    effect_integrity = verify_effect_certificate(effect_certificate)
    effect_matches = certificate_matches_intent(effect_certificate, intent)
    dependency_bound = (
        intent.dependency_version == compilation_certificate.decision_digest
        and effect_certificate.dependency_version == compilation_certificate.decision_digest
    )

    model_report = bounded_model_check(max_depth=model_depth)
    runtime_safe = bool(model_report["safe"])

    reasons: list[str] = []
    if not semantic.valid:
        reasons.append("semantic compilation proof failed")
    if not effect_integrity:
        reasons.append("effect certificate integrity failed")
    if not effect_matches:
        reasons.append("effect certificate does not match intent")
    if not dependency_bound:
        reasons.append("physical effect is not bound to semantic compilation decision")
    if not runtime_safe:
        reasons.append("bounded effect-runtime model checker found a safety violation")

    return EmbodiedExecutionProofResult(
        valid=(
            semantic.valid
            and effect_integrity
            and effect_matches
            and dependency_bound
            and runtime_safe
        ),
        semantic_compilation=semantic,
        effect_certificate_valid=effect_integrity,
        effect_matches_intent=effect_matches,
        dependency_bound_to_compilation=dependency_bound,
        runtime_model_safe=runtime_safe,
        model_depth=model_depth,
        reasons=tuple(reasons),
    )