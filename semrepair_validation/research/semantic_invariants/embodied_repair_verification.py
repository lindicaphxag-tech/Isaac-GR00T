"""Independent verification certificates for synthesized semantic repairs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Sequence

from .embodied_repair_synthesis import RepairExample, RepairProgram, program_fits


class RepairVerificationFailed(RuntimeError):
    pass


def repair_program_fingerprint(program: RepairProgram) -> str:
    """Fingerprint the declared repair DSL program, not arbitrary Python code."""
    payload = [
        {
            "name": operation.name,
            "family": operation.family,
            "cost": operation.cost,
        }
        for operation in program.operations
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def evidence_bank_fingerprint(examples: Sequence[RepairExample]) -> str:
    payload = [
        {
            "observed": tuple(float(x) for x in example.observed),
            "expected": tuple(float(x) for x in example.expected),
            "context": dict(example.context),
            "label": example.label,
        }
        for example in examples
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RepairVerificationCertificate:
    contract_id: str
    program_fingerprint: str
    evidence_fingerprint: str
    verifier_id: str
    cases: int
    atol: float
    status: str = "verified"

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise ValueError("contract_id must be non-empty")
        if not self.verifier_id:
            raise ValueError("verifier_id must be non-empty")
        if self.cases <= 0:
            raise ValueError("verification requires at least one held-out case")
        if self.status != "verified":
            raise ValueError("certificate status must be verified")


def verify_repair_against_heldout(
    *,
    contract_id: str,
    program: RepairProgram,
    heldout: Sequence[RepairExample],
    verifier_id: str,
    atol: float = 1.0e-8,
) -> RepairVerificationCertificate:
    """Issue a certificate only if every frozen held-out case passes."""
    if not heldout:
        raise RepairVerificationFailed("held-out verification bank is empty")
    if not program_fits(program, heldout, atol=atol):
        raise RepairVerificationFailed(
            "repair program failed at least one held-out semantic witness"
        )

    return RepairVerificationCertificate(
        contract_id=contract_id,
        program_fingerprint=repair_program_fingerprint(program),
        evidence_fingerprint=evidence_bank_fingerprint(heldout),
        verifier_id=verifier_id,
        cases=len(heldout),
        atol=float(atol),
    )


def certificate_matches_program(
    certificate: RepairVerificationCertificate,
    *,
    contract_id: str,
    program: RepairProgram,
) -> bool:
    return (
        certificate.status == "verified"
        and certificate.contract_id == contract_id
        and certificate.program_fingerprint == repair_program_fingerprint(program)
    )
