"""Independent verification certificates for synthesized semantic repairs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
from pathlib import Path
import textwrap
from typing import Sequence

from .embodied_repair_synthesis import RepairExample, RepairProgram, program_fits


class RepairVerificationFailed(RuntimeError):
    pass


def _callable_implementation_digest(program_operation) -> str:
    """Bind a certified primitive to the Python source that will execute."""
    fn = program_operation.apply_fn
    if not program_operation.implementation_id:
        raise RepairVerificationFailed(
            f"repair primitive {program_operation.name!r} has no stable implementation identity"
        )

    try:
        source = textwrap.dedent(inspect.getsource(fn)).strip()
        source_file = inspect.getsourcefile(fn)
    except (OSError, TypeError) as exc:
        raise RepairVerificationFailed(
            f"repair primitive {program_operation.name!r} has no inspectable implementation source"
        ) from exc

    if not source or not source_file:
        raise RepairVerificationFailed(
            f"repair primitive {program_operation.name!r} has no inspectable implementation source"
        )

    source_path = Path(source_file)
    if not source_path.is_file():
        raise RepairVerificationFailed(
            f"repair primitive {program_operation.name!r} source file is unavailable"
        )

    closure_values = []
    for cell in fn.__closure__ or ():
        try:
            closure_values.append(repr(cell.cell_contents))
        except ValueError:
            closure_values.append("<empty>")

    payload = {
        "implementation_id": program_operation.implementation_id,
        "module": getattr(fn, "__module__", None),
        "qualname": getattr(fn, "__qualname__", None),
        "callable_source_sha256": sha256(source.encode("utf-8")).hexdigest(),
        "source_file_sha256": sha256(source_path.read_bytes()).hexdigest(),
        "defaults": repr(getattr(fn, "__defaults__", None)),
        "kwdefaults": repr(getattr(fn, "__kwdefaults__", None)),
        "closure": closure_values,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def repair_program_fingerprint(program: RepairProgram) -> str:
    """Fingerprint semantic identity and the exact executable implementation."""
    payload = []
    for operation in program.operations:
        if not operation.implementation_id:
            raise RepairVerificationFailed(
                f"repair primitive {operation.name!r} has no stable implementation identity"
            )
        payload.append(
            {
                "name": operation.name,
                "family": operation.family,
                "cost": operation.cost,
                "implementation_id": operation.implementation_id,
                "implementation_digest": _callable_implementation_digest(operation),
            }
        )
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
    certificate: RepairVerificationCertificate | None,
    *,
    contract_id: str,
    program: RepairProgram,
) -> bool:
    if certificate is None:
        return False
    if certificate.status != "verified" or certificate.contract_id != contract_id:
        return False
    try:
        fingerprint = repair_program_fingerprint(program)
    except RepairVerificationFailed:
        return False
    return certificate.program_fingerprint == fingerprint