"""Fail-closed runtime mediation for verified embodied semantic repairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Callable, Mapping

from .embodied_repair_synthesis import RepairProgram, Vector
from .embodied_repair_verification import (
    RepairVerificationCertificate,
    certificate_matches_program,
)


Context = Mapping[str, Any]


class RepairNotVerifiedError(RuntimeError):
    pass


class RepairGuardRejectedError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepairReceipt:
    contract_id: str
    program: str
    verification_status: str
    requested: Vector
    executed: Vector
    provenance_digest: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _digest_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class SemanticRepairMediator:
    """Apply only independently verified repair programs at runtime.

    The mediator records requested and actually executed semantics separately.
    A caller may provide a post-repair guard representing an existing safety
    layer. The guard always observes the repaired action that would be executed.
    """

    def __init__(
        self,
        *,
        contract_id: str,
        program: RepairProgram,
        certificate: RepairVerificationCertificate,
        guard: Callable[[Vector, Context], bool] | None = None,
    ) -> None:
        if not contract_id:
            raise ValueError("contract_id must be non-empty")
        if not certificate_matches_program(
            certificate,
            contract_id=contract_id,
            program=program,
        ):
            raise RepairNotVerifiedError(
                "runtime installation requires a valid program-bound verification certificate"
            )
        self.contract_id = contract_id
        self.program = program
        self.verification_status = certificate.status
        self.verification_certificate = certificate
        self.guard = guard

    @classmethod
    def from_certificate(
        cls,
        *,
        contract_id: str,
        program: RepairProgram,
        certificate: RepairVerificationCertificate,
        guard: Callable[[Vector, Context], bool] | None = None,
    ) -> "SemanticRepairMediator":
        """Install a repair only when an independent certificate matches it."""
        if not certificate_matches_program(
            certificate,
            contract_id=contract_id,
            program=program,
        ):
            raise RepairNotVerifiedError(
                "verification certificate does not match contract/program"
            )
        return cls(
            contract_id=contract_id,
            program=program,
            certificate=certificate,
            guard=guard,
        )

    def mediate(
        self,
        requested: Vector,
        *,
        context: Context | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RepairReceipt:
        context = context or {}
        executed = self.program.apply(requested, context)

        if self.guard is not None and not self.guard(executed, context):
            raise RepairGuardRejectedError(
                f"post-repair guard rejected action for {self.contract_id!r}"
            )

        payload = {
            "contract_id": self.contract_id,
            "program": self.program.name,
            "verification_status": self.verification_status,
            "requested": requested,
            "executed": executed,
            "context": context,
            "metadata": dict(metadata or {}),
        }
        return RepairReceipt(
            contract_id=self.contract_id,
            program=self.program.name,
            verification_status=self.verification_status,
            requested=tuple(float(item) for item in requested),
            executed=tuple(float(item) for item in executed),
            provenance_digest=_digest_payload(payload),
            metadata=dict(metadata or {}),
        )

    def replay_matches(
        self,
        receipt: RepairReceipt,
        *,
        context: Context | None = None,
    ) -> bool:
        """Re-run the repair and ensure the executed semantic action is stable."""
        replayed = self.program.apply(receipt.requested, context or {})
        return replayed == receipt.executed