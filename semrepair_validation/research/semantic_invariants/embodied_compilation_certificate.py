"""Proof-carrying receipts for embodied semantic compilation.

A repair that was uniquely minimal under one adapter registry and evidence set
may stop being unique after the registry or its contextual evidence changes.
This module binds an installable compilation decision to the exact semantic
environment under which it was justified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Mapping, Sequence

from .embodied_semantic_compiler import CompilationResult
from .embodied_semantic_types import SemanticAdapter


COMPILER_CERTIFICATE_SCHEMA = "embodied-semantic-compilation-certificate/v0.1"


def _digest_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def adapter_registry_digest(adapters: Sequence[SemanticAdapter]) -> str:
    """Hash the semantic adapter registry, independent of insertion order."""
    records = []
    for adapter in adapters:
        records.append(
            {
                "name": adapter.name,
                "requires": dict(sorted(adapter.requires.items())),
                "produces": dict(sorted(adapter.produces.items())),
                "cost": float(adapter.cost),
                "effects": list(adapter.effects),
                "required_evidence": list(adapter.required_evidence),
            }
        )
    records.sort(key=lambda item: (item["name"], _digest_json(item)))
    return _digest_json(records)


def evidence_identity_digest(evidence_identity: Mapping[str, str]) -> str:
    """Hash stable identities of contextual witnesses."""
    return _digest_json(dict(sorted(evidence_identity.items())))


@dataclass(frozen=True)
class SemanticCompilationCertificate:
    schema: str
    compiler_version: str
    registry_digest: str
    evidence_digest: str
    source: dict[str, object]
    target: dict[str, object]
    inferred_source: dict[str, object]
    refined_source: dict[str, object]
    selected_adapter_path: tuple[str, ...]
    selected_cost: float | None
    repair_candidate: str | None
    repair_families: tuple[str, ...]
    refinement_records: tuple[dict[str, object], ...]
    effects: tuple[str, ...]
    decision_digest: str

    @property
    def installable(self) -> bool:
        return self.repair_candidate is None


def _decision_payload(
    result: CompilationResult,
    *,
    registry_digest: str,
    evidence_digest: str,
    compiler_version: str,
) -> dict[str, object]:
    plan = result.adapter_plan
    return {
        "schema": COMPILER_CERTIFICATE_SCHEMA,
        "compiler_version": compiler_version,
        "registry_digest": registry_digest,
        "evidence_digest": evidence_digest,
        "source": asdict(result.source),
        "target": asdict(result.target),
        "inferred_source": asdict(result.inferred_source),
        "refined_source": asdict(result.refined_source),
        "selected_adapter_path": (
            [adapter.name for adapter in plan.adapters] if plan is not None else []
        ),
        "selected_cost": plan.total_cost if plan is not None else None,
        "repair_candidate": (
            result.repair_candidate.name if result.repair_candidate is not None else None
        ),
        "repair_families": list(result.repair_families),
        "refinement_records": [
            {
                "before": asdict(item.before),
                "after": asdict(item.after),
                "evidence_digest": item.evidence_digest,
                "effects": list(item.effects),
                "proof_context": [
                    [str(key), value] for key, value in item.proof_context
                ],
            }
            for item in result.refinements
        ],
        "effects": list(result.effects),
    }


def issue_compilation_certificate(
    result: CompilationResult,
    *,
    adapters: Sequence[SemanticAdapter],
    evidence_identity: Mapping[str, str] | None = None,
    compiler_version: str = "0.1",
) -> SemanticCompilationCertificate:
    registry_hash = adapter_registry_digest(adapters)
    evidence_identity = dict(evidence_identity or {})

    if result.adapter_plan is not None:
        missing = set(result.adapter_plan.evidence_used) - set(evidence_identity)
        if missing:
            raise ValueError(
                "certificate requires stable identities for consumed evidence: "
                f"{tuple(sorted(missing))!r}"
            )

    evidence_hash = evidence_identity_digest(evidence_identity)
    payload = _decision_payload(
        result,
        registry_digest=registry_hash,
        evidence_digest=evidence_hash,
        compiler_version=compiler_version,
    )
    digest = _digest_json(payload)

    return SemanticCompilationCertificate(
        schema=COMPILER_CERTIFICATE_SCHEMA,
        compiler_version=compiler_version,
        registry_digest=registry_hash,
        evidence_digest=evidence_hash,
        source=payload["source"],
        target=payload["target"],
        inferred_source=payload["inferred_source"],
        refined_source=payload["refined_source"],
        selected_adapter_path=tuple(payload["selected_adapter_path"]),
        selected_cost=payload["selected_cost"],
        repair_candidate=payload["repair_candidate"],
        repair_families=tuple(payload["repair_families"]),
        refinement_records=tuple(payload["refinement_records"]),
        effects=tuple(payload["effects"]),
        decision_digest=digest,
    )


def certificate_matches_environment(
    certificate: SemanticCompilationCertificate,
    *,
    adapters: Sequence[SemanticAdapter],
    evidence_identity: Mapping[str, str] | None = None,
) -> bool:
    return (
        certificate.registry_digest == adapter_registry_digest(adapters)
        and certificate.evidence_digest
        == evidence_identity_digest(dict(evidence_identity or {}))
    )


def verify_compilation_certificate(
    certificate: SemanticCompilationCertificate,
) -> bool:
    payload = {
        "schema": certificate.schema,
        "compiler_version": certificate.compiler_version,
        "registry_digest": certificate.registry_digest,
        "evidence_digest": certificate.evidence_digest,
        "source": certificate.source,
        "target": certificate.target,
        "inferred_source": certificate.inferred_source,
        "refined_source": certificate.refined_source,
        "selected_adapter_path": list(certificate.selected_adapter_path),
        "selected_cost": certificate.selected_cost,
        "repair_candidate": certificate.repair_candidate,
        "repair_families": list(certificate.repair_families),
        "refinement_records": list(certificate.refinement_records),
        "effects": list(certificate.effects),
    }
    return _digest_json(payload) == certificate.decision_digest