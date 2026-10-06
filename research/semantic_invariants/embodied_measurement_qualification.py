"""Qualification gate for evidence used to authorize semantic repairs.

Repeatability is not enough.  A measurement can be perfectly self-consistent
while remaining invariant to a latent semantic error (for example, a wrong
forward mapping cancelled by a matching wrong inverse).  Evidence may influence
repair authority only when it is both repeatable and semantically identifying.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Sequence


@dataclass(frozen=True)
class MeasurementWorld:
    """One semantic world used to test identifiability of a measurement."""

    evidence_id: str
    measurement_digest: str
    semantic_target_digest: str


@dataclass(frozen=True)
class MeasurementQualificationCertificate:
    measurement_id: str
    semantic_anchor_id: str | None
    repeat_outcome_digests: tuple[str, ...]
    worlds: tuple[MeasurementWorld, ...]
    minimum_repeats: int
    repeatable: bool
    identifiable: bool
    qualified: bool
    decision: str
    reasons: tuple[str, ...]
    digest: str


class MeasurementNotQualified(RuntimeError):
    pass


def _canonical_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _semantic_collisions(
    worlds: Sequence[MeasurementWorld],
) -> dict[str, tuple[str, ...]]:
    targets_by_measurement: dict[str, set[str]] = {}
    for world in worlds:
        targets_by_measurement.setdefault(world.measurement_digest, set()).add(
            world.semantic_target_digest
        )
    return {
        measurement: tuple(sorted(targets))
        for measurement, targets in targets_by_measurement.items()
        if len(targets) > 1
    }


def qualify_measurement(
    *,
    measurement_id: str,
    semantic_anchor_id: str | None,
    repeat_outcome_digests: Sequence[str],
    worlds: Sequence[MeasurementWorld] = (),
    minimum_repeats: int = 2,
) -> MeasurementQualificationCertificate:
    if not measurement_id:
        raise ValueError("measurement_id must be non-empty")
    if minimum_repeats < 2:
        raise ValueError("minimum_repeats must be at least 2")

    repeats = tuple(str(item) for item in repeat_outcome_digests)
    world_tuple = tuple(worlds)

    repeatable = (
        len(repeats) >= minimum_repeats
        and len(set(repeats)) == 1
    )

    collisions = _semantic_collisions(world_tuple)
    has_external_anchor = bool(semantic_anchor_id and semantic_anchor_id.strip())
    identifiable = has_external_anchor and not collisions

    reasons: list[str] = []
    if len(repeats) < minimum_repeats:
        reasons.append(
            f"only {len(repeats)} repeat outcomes; need at least {minimum_repeats}"
        )
    elif len(set(repeats)) != 1:
        reasons.append("repeat outcomes are not identical under the frozen protocol")

    if not has_external_anchor:
        reasons.append("measurement has no external semantic anchor")
    if collisions:
        for measurement, targets in sorted(collisions.items()):
            reasons.append(
                "measurement is semantically non-identifying: "
                f"{measurement!r} maps to distinct semantic targets {targets!r}"
            )

    qualified = repeatable and identifiable
    if qualified:
        decision = "measurement_qualified"
    elif not identifiable:
        decision = "non_identifying_measurement"
    else:
        decision = "non_repeatable_measurement"

    payload = {
        "measurement_id": measurement_id,
        "semantic_anchor_id": semantic_anchor_id,
        "repeat_outcome_digests": repeats,
        "worlds": [asdict(item) for item in world_tuple],
        "minimum_repeats": minimum_repeats,
        "repeatable": repeatable,
        "identifiable": identifiable,
        "qualified": qualified,
        "decision": decision,
        "reasons": reasons,
    }

    return MeasurementQualificationCertificate(
        measurement_id=measurement_id,
        semantic_anchor_id=semantic_anchor_id,
        repeat_outcome_digests=repeats,
        worlds=world_tuple,
        minimum_repeats=minimum_repeats,
        repeatable=repeatable,
        identifiable=identifiable,
        qualified=qualified,
        decision=decision,
        reasons=tuple(reasons),
        digest=_canonical_digest(payload),
    )


def require_qualified_measurement(
    certificate: MeasurementQualificationCertificate,
) -> None:
    if not certificate.qualified:
        detail = "; ".join(certificate.reasons) or certificate.decision
        raise MeasurementNotQualified(
            f"measurement {certificate.measurement_id!r} is not qualified: {detail}"
        )
