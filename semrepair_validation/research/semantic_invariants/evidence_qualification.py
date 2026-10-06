"""Program-bound qualification for evidence used in repair authorization.

A repair gate must not become authoritative merely because an evaluator returns
"pass". The evidence channel itself needs qualification:

- source identity is bound to the measured entities;
- the measurement is semantically identifying rather than circular;
- the result is repeatable/deterministic under its frozen protocol.

This module binds those claims to the same contract, executable repair program,
evidence digest, and evidence scope used by the activation gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

from .embodied_repair_synthesis import RepairProgram
from .embodied_repair_verification import repair_program_fingerprint


EVIDENCE_QUALIFICATION_SCHEMA = "semrepair-evidence-qualification/v0.1"
VALID_QUALIFICATION_STATUS = {"pass", "fail", "unknown"}


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=repr,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EvidenceQualificationCertificate:
    """Integrity-bound qualification of one measurement/evidence channel."""

    schema: str
    contract_id: str
    program_fingerprint: str
    evidence_digest: str
    evidence_scope_digest: str
    source_identity_status: str
    identifiability_status: str
    repeatability_status: str
    qualifier_id: str
    metadata: dict[str, Any] | None
    decision_digest: str

    @property
    def qualified(self) -> bool:
        return (
            self.source_identity_status == "pass"
            and self.identifiability_status == "pass"
            and self.repeatability_status == "pass"
        )


def _unsigned_payload(
    *,
    contract_id: str,
    program_fingerprint: str,
    evidence_digest: str,
    evidence_scope_digest: str,
    source_identity_status: str,
    identifiability_status: str,
    repeatability_status: str,
    qualifier_id: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": EVIDENCE_QUALIFICATION_SCHEMA,
        "contract_id": contract_id,
        "program_fingerprint": program_fingerprint,
        "evidence_digest": evidence_digest,
        "evidence_scope_digest": evidence_scope_digest,
        "source_identity_status": source_identity_status,
        "identifiability_status": identifiability_status,
        "repeatability_status": repeatability_status,
        "qualifier_id": qualifier_id,
        "metadata": dict(metadata or {}),
    }


def issue_evidence_qualification_certificate(
    *,
    contract_id: str,
    program: RepairProgram,
    evidence_digest: str,
    evidence_scope_digest: str,
    source_identity_status: str,
    identifiability_status: str,
    repeatability_status: str,
    qualifier_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> EvidenceQualificationCertificate:
    statuses = (
        source_identity_status,
        identifiability_status,
        repeatability_status,
    )
    if any(status not in VALID_QUALIFICATION_STATUS for status in statuses):
        raise ValueError(
            "qualification statuses must be one of pass/fail/unknown"
        )
    if not contract_id:
        raise ValueError("contract_id must be non-empty")
    if not evidence_digest:
        raise ValueError("evidence_digest must be non-empty")
    if not evidence_scope_digest:
        raise ValueError("evidence_scope_digest must be non-empty")
    if not qualifier_id:
        raise ValueError("qualifier_id must be non-empty")

    payload = _unsigned_payload(
        contract_id=contract_id,
        program_fingerprint=repair_program_fingerprint(program),
        evidence_digest=evidence_digest,
        evidence_scope_digest=evidence_scope_digest,
        source_identity_status=source_identity_status,
        identifiability_status=identifiability_status,
        repeatability_status=repeatability_status,
        qualifier_id=qualifier_id,
        metadata=metadata,
    )
    return EvidenceQualificationCertificate(
        **payload,
        decision_digest=_digest(payload),
    )


def verify_evidence_qualification_certificate(
    certificate: EvidenceQualificationCertificate | None,
    *,
    contract_id: str,
    program: RepairProgram,
    evidence_digest: str,
    evidence_scope_digest: str,
) -> bool:
    if certificate is None:
        return False
    if certificate.schema != EVIDENCE_QUALIFICATION_SCHEMA:
        return False
    if certificate.contract_id != contract_id:
        return False
    try:
        program_fingerprint = repair_program_fingerprint(program)
    except Exception:
        return False
    if certificate.program_fingerprint != program_fingerprint:
        return False
    if certificate.evidence_digest != evidence_digest:
        return False
    if certificate.evidence_scope_digest != evidence_scope_digest:
        return False
    if not certificate.qualifier_id:
        return False
    statuses = (
        certificate.source_identity_status,
        certificate.identifiability_status,
        certificate.repeatability_status,
    )
    if any(status not in VALID_QUALIFICATION_STATUS for status in statuses):
        return False

    payload = _unsigned_payload(
        contract_id=certificate.contract_id,
        program_fingerprint=certificate.program_fingerprint,
        evidence_digest=certificate.evidence_digest,
        evidence_scope_digest=certificate.evidence_scope_digest,
        source_identity_status=certificate.source_identity_status,
        identifiability_status=certificate.identifiability_status,
        repeatability_status=certificate.repeatability_status,
        qualifier_id=certificate.qualifier_id,
        metadata=certificate.metadata,
    )
    return certificate.decision_digest == _digest(payload)


def assert_evidence_qualified(
    certificate: EvidenceQualificationCertificate | None,
    *,
    contract_id: str,
    program: RepairProgram,
    evidence_digest: str,
    evidence_scope_digest: str,
) -> EvidenceQualificationCertificate:
    if not verify_evidence_qualification_certificate(
        certificate,
        contract_id=contract_id,
        program=program,
        evidence_digest=evidence_digest,
        evidence_scope_digest=evidence_scope_digest,
    ):
        raise ValueError(
            "evidence qualification certificate is missing, invalid, stale, "
            "or bound to different evidence"
        )
    assert certificate is not None
    if not certificate.qualified:
        raise ValueError(
            "evidence channel is not qualified: "
            f"source_identity={certificate.source_identity_status!r}, "
            f"identifiability={certificate.identifiability_status!r}, "
            f"repeatability={certificate.repeatability_status!r}"
        )
    return certificate


def evidence_qualification_fingerprint(
    certificate: EvidenceQualificationCertificate,
) -> str:
    return _digest(asdict(certificate))
