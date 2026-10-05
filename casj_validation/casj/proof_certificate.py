from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping

from .trust_region import DirectionalTrustRegion


CASJ_RUNTIME_CERTIFICATE_SCHEMA = "casj-runtime-trust-certificate/v0.1"


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def mapping_digest(payload: Mapping[str, object]) -> str:
    """Stable identity for a semantic/action/intervention contract mapping."""
    return _canonical_digest(dict(payload))


@dataclass(frozen=True)
class CASJRuntimeTrustCertificate:
    schema: str
    policy_digest: str
    action_contract_digest: str
    baseline_context_digest: str
    support_id: str
    intervention_chart_digest: str
    repair_implementation_digest: str
    evidence_digest: str
    direction_digest: str
    repair_mode: str
    certified_motion_multiplier: float
    mathematical_trust_radius: float | str
    operational_radius_cap: float
    threshold_profile: dict[str, float]
    certificate_digest: str

    def payload(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("certificate_digest")
        return data


@dataclass(frozen=True)
class CASJAuthorizationDecision:
    authorized: bool
    reason: str


def issue_runtime_trust_certificate(
    trust: DirectionalTrustRegion,
    *,
    policy_digest: str,
    action_contract_digest: str,
    baseline_context_digest: str,
    support_id: str,
    intervention_chart_digest: str,
    repair_implementation_digest: str,
    evidence_digest: str,
    direction_digest: str,
    repair_mode: str,
    operational_radius_cap: float,
    threshold_profile: Mapping[str, float] | None = None,
) -> CASJRuntimeTrustCertificate:
    """Bind a mathematical trust region to finite runtime authority."""
    identities = {
        "policy_digest": policy_digest,
        "action_contract_digest": action_contract_digest,
        "baseline_context_digest": baseline_context_digest,
        "support_id": support_id,
        "intervention_chart_digest": intervention_chart_digest,
        "repair_implementation_digest": repair_implementation_digest,
        "evidence_digest": evidence_digest,
        "direction_digest": direction_digest,
        "repair_mode": repair_mode,
    }
    missing = [name for name, value in identities.items() if not value]
    if missing:
        raise ValueError(f"certificate identities must be non-empty: {missing!r}")
    if not trust.accepted or trust.radius <= 0:
        raise ValueError("cannot issue runtime authority from a rejected trust region")
    if not isfinite(operational_radius_cap) or operational_radius_cap <= 0:
        raise ValueError("operational_radius_cap must be finite and positive")

    mathematical_radius = trust.radius
    certified = min(float(mathematical_radius), operational_radius_cap)
    radius_repr: float | str = (
        float(mathematical_radius) if isfinite(mathematical_radius) else "inf"
    )
    payload: dict[str, object] = {
        "schema": CASJ_RUNTIME_CERTIFICATE_SCHEMA,
        **identities,
        "certified_motion_multiplier": float(certified),
        "mathematical_trust_radius": radius_repr,
        "operational_radius_cap": float(operational_radius_cap),
        "threshold_profile": {
            str(key): float(value)
            for key, value in sorted(dict(threshold_profile or {}).items())
        },
    }
    digest = _canonical_digest(payload)
    return CASJRuntimeTrustCertificate(
        **payload,
        certificate_digest=digest,
    )


def verify_runtime_trust_certificate(
    certificate: CASJRuntimeTrustCertificate,
) -> bool:
    if certificate.schema != CASJ_RUNTIME_CERTIFICATE_SCHEMA:
        return False
    if certificate.certified_motion_multiplier <= 0:
        return False
    if not isfinite(certificate.operational_radius_cap):
        return False
    return _canonical_digest(certificate.payload()) == certificate.certificate_digest


def authorize_runtime_repair(
    certificate: CASJRuntimeTrustCertificate,
    *,
    requested_motion_multiplier: float,
    policy_digest: str,
    action_contract_digest: str,
    baseline_context_digest: str,
    support_id: str,
    intervention_chart_digest: str,
    repair_implementation_digest: str,
    evidence_digest: str,
    direction_digest: str,
    repair_mode: str,
) -> CASJAuthorizationDecision:
    """Fail closed if identity or physical authority has drifted."""
    if requested_motion_multiplier < 0:
        return CASJAuthorizationDecision(False, "requested motion multiplier is negative")
    if not verify_runtime_trust_certificate(certificate):
        return CASJAuthorizationDecision(False, "certificate integrity check failed")

    expected = {
        "policy_digest": policy_digest,
        "action_contract_digest": action_contract_digest,
        "baseline_context_digest": baseline_context_digest,
        "support_id": support_id,
        "intervention_chart_digest": intervention_chart_digest,
        "repair_implementation_digest": repair_implementation_digest,
        "evidence_digest": evidence_digest,
        "direction_digest": direction_digest,
        "repair_mode": repair_mode,
    }
    for field, value in expected.items():
        if getattr(certificate, field) != value:
            return CASJAuthorizationDecision(False, f"{field} changed since certification")
    if requested_motion_multiplier > certificate.certified_motion_multiplier:
        return CASJAuthorizationDecision(
            False, "requested physical motion exceeds certified trust radius"
        )
    return CASJAuthorizationDecision(True, "proof-carrying CASJ repair authorized")
