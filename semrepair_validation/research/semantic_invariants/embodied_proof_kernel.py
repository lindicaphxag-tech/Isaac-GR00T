"""Small trusted kernel for proof-carrying embodied semantic compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .embodied_compilation_certificate import (
    SemanticCompilationCertificate,
    verify_compilation_certificate,
)
from .embodied_compilation_verifier import (
    CompilationProofCheck,
    verify_pure_adapter_certificate,
)
from .embodied_refinement import SemanticEvidence
from .embodied_refinement_verifier import (
    RefinementProofCheck,
    verify_refinement_records,
)
from .embodied_semantic_types import SemanticAdapter


@dataclass(frozen=True)
class SemanticProofKernelResult:
    valid: bool
    certificate_integrity: bool
    refinement: RefinementProofCheck
    adapter: CompilationProofCheck
    reasons: tuple[str, ...]


def verify_semantic_compilation(
    certificate: SemanticCompilationCertificate,
    *,
    adapters: Sequence[SemanticAdapter],
    adapter_evidence_identity: Mapping[str, str] | None = None,
    refinement_evidence_by_digest: Mapping[str, SemanticEvidence] | None = None,
    max_adapter_steps: int = 6,
) -> SemanticProofKernelResult:
    """Verify a compilation bundle without trusting the compiler search path."""

    integrity = verify_compilation_certificate(certificate)
    refinement = verify_refinement_records(
        certificate,
        evidence_by_digest=dict(refinement_evidence_by_digest or {}),
    )
    adapter = verify_pure_adapter_certificate(
        certificate,
        adapters=adapters,
        evidence_identity=dict(adapter_evidence_identity or {}),
        max_steps=max_adapter_steps,
    )

    reasons: list[str] = []
    if not integrity:
        reasons.append("certificate integrity check failed")
    reasons.extend(f"refinement: {item}" for item in refinement.reasons)
    reasons.extend(f"adapter: {item}" for item in adapter.reasons)

    return SemanticProofKernelResult(
        valid=integrity and refinement.valid and adapter.valid,
        certificate_integrity=integrity,
        refinement=refinement,
        adapter=adapter,
        reasons=tuple(reasons),
    )
