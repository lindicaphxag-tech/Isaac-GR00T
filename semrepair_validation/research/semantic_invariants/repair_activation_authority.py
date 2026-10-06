"""Fail-closed activation authority for verified semantic repairs.

Local held-out verification proves that one repair program satisfies one frozen
semantic witness bank. It does not by itself prove that the repair is
identifiable, interaction-safe, protocol-compatible, or execution-safe.

This module adds a second, program-bound certificate whose only purpose is
runtime activation authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from .embodied_repair_runtime import RepairReceipt, SemanticRepairMediator
from .embodied_repair_synthesis import RepairProgram, Vector
from .embodied_repair_verification import (
    RepairVerificationCertificate,
    certificate_matches_program,
    repair_program_fingerprint,
)


ACTIVATION_CERTIFICATE_SCHEMA = "semrepair-activation-authority/v0.1"
REQUIRED_ACTIVATION_GATES = (
    "anchor",
    "value",
    "protocol",
    "interaction",
    "execution",
)
VALID_GATE_STATUS = {"pass", "fail", "unknown"}


class RepairActivationRejected(RuntimeError):
    pass


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=repr,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def verification_certificate_fingerprint(
    certificate: RepairVerificationCertificate,
) -> str:
    return _digest(asdict(certificate))


@dataclass(frozen=True)
class ActivationGateEvidence:
    """One independently inspectable activation obligation.

    independent_of_repair_path does not assert social independence. It means
    the evidence is not algebraically produced only by the same transformation
    path whose semantics it is supposed to identify.
    """

    gate: str
    status: str
    evidence_digest: str
    evaluator_id: str
    evidence_kind: str
    independent_of_repair_path: bool
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.gate not in REQUIRED_ACTIVATION_GATES:
            raise ValueError(f"unknown activation gate: {self.gate!r}")
        if self.status not in VALID_GATE_STATUS:
            raise ValueError(f"invalid gate status: {self.status!r}")
        if not self.evidence_digest:
            raise ValueError("evidence_digest must be non-empty")
        if not self.evaluator_id:
            raise ValueError("evaluator_id must be non-empty")
        if not self.evidence_kind:
            raise ValueError("evidence_kind must be non-empty")


@dataclass(frozen=True)
class RepairActivationCertificate:
    schema: str
    contract_id: str
    program_fingerprint: str
    local_verification_fingerprint: str
    gates: tuple[ActivationGateEvidence, ...]
    issuer_version: str
    decision_digest: str

    @property
    def gate_map(self) -> dict[str, ActivationGateEvidence]:
        return {gate.gate: gate for gate in self.gates}


def _gate_payload(gate: ActivationGateEvidence) -> dict[str, Any]:
    return {
        "gate": gate.gate,
        "status": gate.status,
        "evidence_digest": gate.evidence_digest,
        "evaluator_id": gate.evaluator_id,
        "evidence_kind": gate.evidence_kind,
        "independent_of_repair_path": gate.independent_of_repair_path,
        "metadata": gate.metadata,
    }


def _validate_gate_set(
    gates: Sequence[ActivationGateEvidence],
) -> tuple[ActivationGateEvidence, ...]:
    gates = tuple(gates)
    names = [gate.gate for gate in gates]
    if len(set(names)) != len(names):
        raise RepairActivationRejected(
            "activation certificate contains duplicate gates"
        )

    missing = sorted(set(REQUIRED_ACTIVATION_GATES) - set(names))
    extra = sorted(set(names) - set(REQUIRED_ACTIVATION_GATES))
    if missing or extra:
        raise RepairActivationRejected(
            f"activation gate set mismatch: missing={missing!r} extra={extra!r}"
        )

    not_passed = [gate.gate for gate in gates if gate.status != "pass"]
    if not_passed:
        raise RepairActivationRejected(
            f"activation remains fail-closed; non-passing gates={not_passed!r}"
        )

    gate_map = {gate.gate: gate for gate in gates}
    if not gate_map["anchor"].independent_of_repair_path:
        raise RepairActivationRejected(
            "anchor evidence is circular with the repair path"
        )

    if not gate_map["execution"].independent_of_repair_path:
        raise RepairActivationRejected(
            "execution gate must use evidence independent of local repair verification"
        )

    return tuple(
        sorted(
            gates,
            key=lambda gate: REQUIRED_ACTIVATION_GATES.index(gate.gate),
        )
    )


def issue_repair_activation_certificate(
    *,
    contract_id: str,
    program: RepairProgram,
    local_verification: RepairVerificationCertificate,
    gates: Sequence[ActivationGateEvidence],
    issuer_version: str = "0.4-dev",
) -> RepairActivationCertificate:
    """Issue activation authority only after every required obligation passes."""
    if not contract_id:
        raise ValueError("contract_id must be non-empty")
    if not certificate_matches_program(
        local_verification,
        contract_id=contract_id,
        program=program,
    ):
        raise RepairActivationRejected(
            "activation requires a matching program-bound local verification certificate"
        )

    ordered = _validate_gate_set(gates)
    payload = {
        "schema": ACTIVATION_CERTIFICATE_SCHEMA,
        "contract_id": contract_id,
        "program_fingerprint": repair_program_fingerprint(program),
        "local_verification_fingerprint": verification_certificate_fingerprint(
            local_verification
        ),
        "gates": [_gate_payload(gate) for gate in ordered],
        "issuer_version": issuer_version,
    }
    return RepairActivationCertificate(
        schema=ACTIVATION_CERTIFICATE_SCHEMA,
        contract_id=contract_id,
        program_fingerprint=payload["program_fingerprint"],
        local_verification_fingerprint=payload[
            "local_verification_fingerprint"
        ],
        gates=ordered,
        issuer_version=issuer_version,
        decision_digest=_digest(payload),
    )


def verify_repair_activation_certificate(
    certificate: RepairActivationCertificate | None,
    *,
    contract_id: str,
    program: RepairProgram,
    local_verification: RepairVerificationCertificate,
) -> bool:
    if certificate is None:
        return False
    if certificate.schema != ACTIVATION_CERTIFICATE_SCHEMA:
        return False
    if certificate.contract_id != contract_id:
        return False
    if not certificate_matches_program(
        local_verification,
        contract_id=contract_id,
        program=program,
    ):
        return False

    try:
        ordered = _validate_gate_set(certificate.gates)
        program_fingerprint = repair_program_fingerprint(program)
    except (RepairActivationRejected, ValueError):
        return False

    local_fingerprint = verification_certificate_fingerprint(local_verification)
    if certificate.program_fingerprint != program_fingerprint:
        return False
    if certificate.local_verification_fingerprint != local_fingerprint:
        return False

    payload = {
        "schema": certificate.schema,
        "contract_id": certificate.contract_id,
        "program_fingerprint": certificate.program_fingerprint,
        "local_verification_fingerprint": certificate.local_verification_fingerprint,
        "gates": [_gate_payload(gate) for gate in ordered],
        "issuer_version": certificate.issuer_version,
    }
    return certificate.decision_digest == _digest(payload)


def assert_repair_activation_authorized(
    certificate: RepairActivationCertificate | None,
    *,
    contract_id: str,
    program: RepairProgram,
    local_verification: RepairVerificationCertificate,
) -> RepairActivationCertificate:
    if not verify_repair_activation_certificate(
        certificate,
        contract_id=contract_id,
        program=program,
        local_verification=local_verification,
    ):
        raise RepairActivationRejected(
            "repair activation certificate is missing, invalid, stale, or incomplete"
        )
    assert certificate is not None
    return certificate


class ActivationGatedSemanticRepairMediator(SemanticRepairMediator):
    """Runtime mediator requiring local correctness plus activation authority."""

    def __init__(
        self,
        *,
        contract_id: str,
        program: RepairProgram,
        local_verification: RepairVerificationCertificate,
        activation_certificate: RepairActivationCertificate,
        guard=None,
    ) -> None:
        activation = assert_repair_activation_authorized(
            activation_certificate,
            contract_id=contract_id,
            program=program,
            local_verification=local_verification,
        )
        super().__init__(
            contract_id=contract_id,
            program=program,
            certificate=local_verification,
            guard=guard,
        )
        self.activation_certificate = activation

    def mediate(
        self,
        requested: Vector,
        *,
        context=None,
        metadata=None,
    ) -> RepairReceipt:
        receipt = super().mediate(
            requested,
            context=context,
            metadata=metadata,
        )
        enriched = dict(receipt.metadata)
        enriched["activation_certificate_digest"] = (
            self.activation_certificate.decision_digest
        )
        enriched["activation_gate_status"] = {
            gate.gate: gate.status for gate in self.activation_certificate.gates
        }
        return replace(receipt, metadata=enriched)
