"""Proof-carrying policy certificates for Semantic Effect Commit.

The certificate binds a logical physical effect to the assumptions under which
it may be executed and committed. Runtime code must not silently change effect
class, dependency version, or evidence thresholds after certification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

from .semantic_effect_commit import EffectClass, SemanticEffectIntent


EFFECT_CERTIFICATE_SCHEMA = "semantic-effect-commit-certificate/v0.1"


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EffectCommitPolicy:
    """Frozen runtime policy for one logical physical-effect family."""

    min_commit_planes: int = 2
    min_abort_planes: int = 1
    require_fresh_evidence: bool = True
    allow_idempotent_retry: bool = True

    def __post_init__(self) -> None:
        if self.min_commit_planes < 1:
            raise ValueError("min_commit_planes must be positive")
        if self.min_abort_planes < 1:
            raise ValueError("min_abort_planes must be positive")


@dataclass(frozen=True)
class SemanticEffectCertificate:
    schema: str
    effect_id: str
    action_name: str
    effect_class: str
    dependency_version: str
    policy: dict[str, object]
    semantic_contract_id: str
    issuer_version: str
    decision_digest: str


def issue_effect_certificate(
    intent: SemanticEffectIntent,
    *,
    policy: EffectCommitPolicy,
    semantic_contract_id: str,
    issuer_version: str = "0.1",
) -> SemanticEffectCertificate:
    if not semantic_contract_id:
        raise ValueError("semantic_contract_id must be non-empty")

    payload = {
        "schema": EFFECT_CERTIFICATE_SCHEMA,
        "effect_id": intent.effect_id,
        "action_name": intent.action_name,
        "effect_class": intent.effect_class.value,
        "dependency_version": intent.dependency_version,
        "policy": asdict(policy),
        "semantic_contract_id": semantic_contract_id,
        "issuer_version": issuer_version,
    }
    return SemanticEffectCertificate(
        **payload,
        decision_digest=_digest(payload),
    )


def verify_effect_certificate(certificate: SemanticEffectCertificate) -> bool:
    payload = {
        "schema": certificate.schema,
        "effect_id": certificate.effect_id,
        "action_name": certificate.action_name,
        "effect_class": certificate.effect_class,
        "dependency_version": certificate.dependency_version,
        "policy": certificate.policy,
        "semantic_contract_id": certificate.semantic_contract_id,
        "issuer_version": certificate.issuer_version,
    }
    return (
        certificate.schema == EFFECT_CERTIFICATE_SCHEMA
        and _digest(payload) == certificate.decision_digest
    )


def certificate_matches_intent(
    certificate: SemanticEffectCertificate,
    intent: SemanticEffectIntent,
) -> bool:
    return (
        verify_effect_certificate(certificate)
        and certificate.effect_id == intent.effect_id
        and certificate.action_name == intent.action_name
        and certificate.effect_class == intent.effect_class.value
        and certificate.dependency_version == intent.dependency_version
    )


def policy_from_certificate(certificate: SemanticEffectCertificate) -> EffectCommitPolicy:
    if not verify_effect_certificate(certificate):
        raise ValueError("effect certificate integrity check failed")
    return EffectCommitPolicy(**certificate.policy)


def assert_certificate_authorizes(
    certificate: SemanticEffectCertificate,
    intent: SemanticEffectIntent,
    *,
    semantic_contract_id: str,
) -> EffectCommitPolicy:
    if not certificate_matches_intent(certificate, intent):
        raise ValueError("effect certificate does not match intent")
    if certificate.semantic_contract_id != semantic_contract_id:
        raise ValueError("effect certificate does not match semantic contract")
    return policy_from_certificate(certificate)