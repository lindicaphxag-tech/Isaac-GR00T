"""Independent SemRepair verifier for CASJ runtime trust certificates.

The verifier intentionally does not import the CASJ package. CASJ may issue a
certificate, but runtime authority is checked here from the serialized schema,
stable identities, and finite physical authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping, Any


CASJ_RUNTIME_CERTIFICATE_SCHEMA = "casj-runtime-trust-certificate/v0.1"


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CASJProofCheck:
    valid: bool
    integrity: bool
    identity_bound: bool
    within_authority: bool
    reasons: tuple[str, ...]


def verify_casj_runtime_certificate(
    certificate: Mapping[str, Any],
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
    expected_threshold_profile: Mapping[str, float] | None = None,
) -> CASJProofCheck:
    """Verify serialized CASJ authority without trusting its issuer runtime."""
    cert = dict(certificate)
    reasons: list[str] = []

    if cert.get("schema") != CASJ_RUNTIME_CERTIFICATE_SCHEMA:
        reasons.append("unsupported CASJ runtime certificate schema")

    supplied_digest = cert.pop("certificate_digest", None)
    integrity = isinstance(supplied_digest, str) and (
        _canonical_digest(cert) == supplied_digest
    )
    if not integrity:
        reasons.append("CASJ certificate integrity check failed")

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
    identity_bound = True
    for field, value in expected.items():
        if cert.get(field) != value:
            identity_bound = False
            reasons.append(f"CASJ identity mismatch: {field}")

    if expected_threshold_profile is not None:
        expected_thresholds = {
            str(key): float(value)
            for key, value in sorted(dict(expected_threshold_profile).items())
        }
        if cert.get("threshold_profile") != expected_thresholds:
            identity_bound = False
            reasons.append("CASJ threshold profile changed since certification")

    within_authority = True
    try:
        requested = float(requested_motion_multiplier)
        certified = float(cert.get("certified_motion_multiplier"))
        cap = float(cert.get("operational_radius_cap"))
    except (TypeError, ValueError):
        requested = certified = cap = float("nan")

    if not isfinite(requested) or requested < 0:
        within_authority = False
        reasons.append("requested motion multiplier is invalid")
    if not isfinite(certified) or certified <= 0:
        within_authority = False
        reasons.append("certified motion multiplier is invalid")
    if not isfinite(cap) or cap <= 0:
        within_authority = False
        reasons.append("operational radius cap is invalid")
    if isfinite(certified) and isfinite(cap) and certified > cap + 1e-12:
        within_authority = False
        reasons.append("certified motion exceeds operational radius cap")

    mathematical = cert.get("mathematical_trust_radius")
    if mathematical == "inf":
        math_radius = float("inf")
    else:
        try:
            math_radius = float(mathematical)
        except (TypeError, ValueError):
            math_radius = float("nan")
    if math_radius != float("inf") and (not isfinite(math_radius) or math_radius <= 0):
        within_authority = False
        reasons.append("mathematical trust radius is invalid")
    if (
        isfinite(certified)
        and math_radius != float("inf")
        and isfinite(math_radius)
        and certified > math_radius + 1e-12
    ):
        within_authority = False
        reasons.append("certified motion exceeds mathematical trust radius")
    if isfinite(requested) and isfinite(certified) and requested > certified + 1e-12:
        within_authority = False
        reasons.append("requested physical motion exceeds certified authority")

    valid = (
        cert.get("schema") == CASJ_RUNTIME_CERTIFICATE_SCHEMA
        and integrity
        and identity_bound
        and within_authority
    )
    return CASJProofCheck(
        valid=valid,
        integrity=integrity,
        identity_bound=identity_bound,
        within_authority=within_authority,
        reasons=tuple(reasons),
    )
