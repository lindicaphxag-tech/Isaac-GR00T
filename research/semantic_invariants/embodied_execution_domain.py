"""Execution-domain projection certificates for semantic repairs.

A semantically correct adapter can still emit an action outside the consumer's
admissible execution domain.  Clipping such an action changes its meaning and
must therefore be treated as a second semantic transformation, not an
implementation detail.

This module provides a small proof-carrying boundary:
- the consumer domain is explicit and identity-bound;
- a feasible action is proposed;
- the consumer forward semantics is applied;
- the semantic residual to the intended target is measured;
- independent verification recomputes feasibility and residual;
- residual above policy tolerance fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Callable, Sequence


Vector = tuple[float, ...]
ForwardSemantics = Callable[[Vector], Vector]
SemanticDistance = Callable[[Vector, Vector], float]

PROJECTION_CERTIFICATE_SCHEMA = "semrepair-execution-projection/v0.1"


def _digest_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class L2BallExecutionDomain:
    domain_id: str
    dimension: int
    radius: float = 1.0
    center: Vector = ()

    def __post_init__(self) -> None:
        if not self.domain_id:
            raise ValueError("domain_id must be non-empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be finite and positive")
        if self.center and len(self.center) != self.dimension:
            raise ValueError("center dimensionality mismatch")

    @property
    def effective_center(self) -> Vector:
        return self.center or (0.0,) * self.dimension

    @property
    def digest(self) -> str:
        return _digest_json(
            {
                "kind": "l2-ball",
                "domain_id": self.domain_id,
                "dimension": self.dimension,
                "radius": self.radius,
                "center": list(self.effective_center),
            }
        )

    def contains(self, value: Sequence[float], *, atol: float = 1.0e-9) -> bool:
        if len(value) != self.dimension:
            return False
        squared = sum(
            (float(item) - center) ** 2
            for item, center in zip(value, self.effective_center, strict=True)
        )
        return squared <= (self.radius + atol) ** 2


@dataclass(frozen=True)
class ProjectionCertificate:
    schema: str
    contract_id: str
    domain_digest: str
    forward_model_id: str
    target_digest: str
    target: Vector
    action: Vector
    realized: Vector
    residual: float
    max_residual: float
    feasible: bool
    accepted: bool
    decision_digest: str


@dataclass(frozen=True)
class ProjectionVerification:
    valid: bool
    accepted: bool
    reasons: tuple[str, ...]
    recomputed_residual: float | None


def _decision_payload(
    *,
    contract_id: str,
    domain_digest: str,
    forward_model_id: str,
    target: Vector,
    action: Vector,
    realized: Vector,
    residual: float,
    max_residual: float,
    feasible: bool,
    accepted: bool,
) -> dict[str, object]:
    return {
        "schema": PROJECTION_CERTIFICATE_SCHEMA,
        "contract_id": contract_id,
        "domain_digest": domain_digest,
        "forward_model_id": forward_model_id,
        "target_digest": _digest_json(list(target)),
        "target": list(target),
        "action": list(action),
        "realized": list(realized),
        "residual": residual,
        "max_residual": max_residual,
        "feasible": feasible,
        "accepted": accepted,
    }


def issue_projection_certificate(
    *,
    contract_id: str,
    domain: L2BallExecutionDomain,
    target: Sequence[float],
    action: Sequence[float],
    forward: ForwardSemantics,
    forward_model_id: str,
    distance: SemanticDistance,
    max_residual: float,
) -> ProjectionCertificate:
    if not contract_id:
        raise ValueError("contract_id must be non-empty")
    if not forward_model_id:
        raise ValueError("forward_model_id must be non-empty")
    if not math.isfinite(max_residual) or max_residual < 0:
        raise ValueError("max_residual must be finite and non-negative")

    target_tuple = tuple(float(x) for x in target)
    action_tuple = tuple(float(x) for x in action)
    if len(action_tuple) != domain.dimension:
        raise ValueError("action dimensionality does not match execution domain")

    realized_tuple = tuple(float(x) for x in forward(action_tuple))
    residual = float(distance(target_tuple, realized_tuple))
    if not math.isfinite(residual) or residual < 0:
        raise ValueError("semantic distance must be finite and non-negative")

    feasible = domain.contains(action_tuple)
    accepted = feasible and residual <= max_residual
    payload = _decision_payload(
        contract_id=contract_id,
        domain_digest=domain.digest,
        forward_model_id=forward_model_id,
        target=target_tuple,
        action=action_tuple,
        realized=realized_tuple,
        residual=residual,
        max_residual=max_residual,
        feasible=feasible,
        accepted=accepted,
    )
    digest = _digest_json(payload)
    return ProjectionCertificate(
        schema=PROJECTION_CERTIFICATE_SCHEMA,
        contract_id=contract_id,
        domain_digest=domain.digest,
        forward_model_id=forward_model_id,
        target_digest=payload["target_digest"],
        target=target_tuple,
        action=action_tuple,
        realized=realized_tuple,
        residual=residual,
        max_residual=max_residual,
        feasible=feasible,
        accepted=accepted,
        decision_digest=digest,
    )


def verify_projection_certificate(
    certificate: ProjectionCertificate,
    *,
    contract_id: str,
    domain: L2BallExecutionDomain,
    target: Sequence[float],
    forward: ForwardSemantics,
    forward_model_id: str,
    distance: SemanticDistance,
    atol: float = 1.0e-8,
) -> ProjectionVerification:
    reasons: list[str] = []
    target_tuple = tuple(float(x) for x in target)

    if certificate.schema != PROJECTION_CERTIFICATE_SCHEMA:
        reasons.append("projection certificate schema mismatch")
    if certificate.contract_id != contract_id:
        reasons.append("projection contract identity mismatch")
    if certificate.domain_digest != domain.digest:
        reasons.append("execution-domain identity has drifted")
    if certificate.forward_model_id != forward_model_id:
        reasons.append("consumer forward-semantics identity has drifted")
    if certificate.target_digest != _digest_json(list(target_tuple)):
        reasons.append("projection target identity has drifted")
    if certificate.target != target_tuple:
        reasons.append("projection target payload mismatch")

    feasible = domain.contains(certificate.action)
    if feasible != certificate.feasible:
        reasons.append("recorded feasibility disagrees with execution domain")

    recomputed_residual: float | None = None
    try:
        realized = tuple(float(x) for x in forward(certificate.action))
        recomputed_residual = float(distance(target_tuple, realized))
        if len(realized) != len(certificate.realized):
            reasons.append("realized semantic dimensionality mismatch")
        elif any(
            abs(a - b) > atol
            for a, b in zip(realized, certificate.realized, strict=True)
        ):
            reasons.append("consumer forward semantics no longer reproduce certificate")
        if abs(recomputed_residual - certificate.residual) > atol:
            reasons.append("projection residual no longer reproduces certificate")
    except Exception as exc:
        reasons.append(f"consumer forward semantics failed during verification: {exc}")

    expected_accepted = bool(
        feasible
        and recomputed_residual is not None
        and recomputed_residual <= certificate.max_residual + atol
    )
    if expected_accepted != certificate.accepted:
        reasons.append("recorded projection decision is inconsistent")

    payload = _decision_payload(
        contract_id=certificate.contract_id,
        domain_digest=certificate.domain_digest,
        forward_model_id=certificate.forward_model_id,
        target=certificate.target,
        action=certificate.action,
        realized=certificate.realized,
        residual=certificate.residual,
        max_residual=certificate.max_residual,
        feasible=certificate.feasible,
        accepted=certificate.accepted,
    )
    if _digest_json(payload) != certificate.decision_digest:
        reasons.append("projection certificate decision digest is invalid")

    return ProjectionVerification(
        valid=not reasons,
        accepted=not reasons and expected_accepted,
        reasons=tuple(reasons),
        recomputed_residual=recomputed_residual,
    )


class ProjectionAuthorityRequired(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionDomainAuthorization:
    authorized_action: Vector
    projection_used: bool
    projection_certificate_digest: str | None
    residual: float
    reasons: tuple[str, ...]


def authorize_execution_domain(
    *,
    contract_id: str,
    raw_action: Sequence[float],
    semantic_target: Sequence[float],
    domain: L2BallExecutionDomain,
    forward: ForwardSemantics,
    forward_model_id: str,
    distance: SemanticDistance,
    projection_certificate: ProjectionCertificate | None = None,
) -> ExecutionDomainAuthorization:
    """Authorize the concrete value that may cross the consumer boundary.

    If the raw repair output is already admissible, no projection authority is
    required and the exact raw action is retained.

    If it is outside the consumer execution domain, clipping/projection is a
    *second semantic transformation*.  A caller cannot silently substitute a
    bounded value: an independent ProjectionCertificate must verify against the
    current target, execution-domain identity and consumer forward semantics.
    """

    raw = tuple(float(x) for x in raw_action)
    target = tuple(float(x) for x in semantic_target)

    if domain.contains(raw):
        realized = tuple(float(x) for x in forward(raw))
        residual = float(distance(target, realized))
        return ExecutionDomainAuthorization(
            authorized_action=raw,
            projection_used=False,
            projection_certificate_digest=None,
            residual=residual,
            reasons=(),
        )

    if projection_certificate is None:
        raise ProjectionAuthorityRequired(
            "repair output is outside the consumer execution domain; "
            "silent clipping has no semantic authority"
        )

    verification = verify_projection_certificate(
        projection_certificate,
        contract_id=contract_id,
        domain=domain,
        target=target,
        forward=forward,
        forward_model_id=forward_model_id,
        distance=distance,
    )
    if not verification.valid:
        raise ProjectionAuthorityRequired(
            "projection certificate failed independent verification: "
            + "; ".join(verification.reasons)
        )
    if not verification.accepted:
        raise ProjectionAuthorityRequired(
            "projection is validly described but exceeds the authorized "
            "semantic residual"
        )

    return ExecutionDomainAuthorization(
        authorized_action=projection_certificate.action,
        projection_used=True,
        projection_certificate_digest=projection_certificate.decision_digest,
        residual=float(verification.recomputed_residual),
        reasons=(),
    )
