"""Factorial analysis of interacting semantic repairs in embodied pipelines.

A physically wrong pipeline can look deceptively healthy when two semantic
faults partially cancel.  In that setting, repairing either boundary alone can
make end-to-end error worse even though the repair is locally correct.

This module treats candidate repairs as factors in a complete repair lattice.
It detects non-monotone repair edges, strict compensating bundles, and
higher-order interaction terms.  The output is evidence-bound by a canonical
digest so the classification cannot silently outlive the measurements that
produced it.

The analysis is intentionally metric-agnostic.  It only assumes the supplied
metric has a declared optimization direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from math import isfinite
from typing import Literal, Sequence

from .embodied_measurement_qualification import (
    MeasurementQualificationCertificate,
    MeasurementNotQualified,
    require_qualified_measurement,
)


Objective = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class RepairOutcome:
    repairs: frozenset[str]
    value: float
    evidence_id: str

    def __post_init__(self) -> None:
        if not isfinite(self.value):
            raise ValueError("repair outcome must be finite")
        if not self.evidence_id:
            raise ValueError("repair outcome requires an evidence_id")


@dataclass(frozen=True)
class RepairEdgeViolation:
    before: tuple[str, ...]
    added_repair: str
    after: tuple[str, ...]
    before_loss: float
    after_loss: float

    @property
    def regression(self) -> float:
        return self.after_loss - self.before_loss


@dataclass(frozen=True)
class CompensatingRepairBundle:
    repairs: tuple[str, ...]
    baseline_loss: float
    repaired_loss: float
    proper_subset_losses: tuple[tuple[tuple[str, ...], float], ...]
    masking_mode: Literal["partial", "complete"]

    @property
    def closure_gain(self) -> float:
        return self.baseline_loss - self.repaired_loss

    @property
    def behaviorally_masked(self) -> bool:
        return self.masking_mode == "complete"


@dataclass(frozen=True)
class RepairInteractionCertificate:
    subject: str
    metric: str
    objective: Objective
    repairs: tuple[str, ...]
    outcomes: tuple[RepairOutcome, ...]
    edge_violations: tuple[RepairEdgeViolation, ...]
    compensating_bundles: tuple[CompensatingRepairBundle, ...]
    mobius_terms: tuple[tuple[tuple[str, ...], float], ...]
    digest: str
    # Historical digest-only certificates are non-authoritative until they
    # have been reissued with a tolerance-bound certificate digest.
    analysis_tolerance: float = 0.0

    @property
    def has_repair_paradox(self) -> bool:
        return bool(self.compensating_bundles)

    def interaction_for(self, repairs: Sequence[str]) -> float:
        key = tuple(sorted(repairs))
        for subset, value in self.mobius_terms:
            if subset == key:
                return value
        raise KeyError(key)


def _powerset(items: tuple[str, ...]) -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset(combo)
        for size in range(len(items) + 1)
        for combo in combinations(items, size)
    )


def _canonical_loss(value: float, objective: Objective) -> float:
    if objective == "minimize":
        return value
    if objective == "maximize":
        return -value
    raise ValueError(f"unknown objective: {objective!r}")


def _canonical_payload(
    *,
    subject: str,
    metric: str,
    objective: Objective,
    repairs: tuple[str, ...],
    outcomes: tuple[RepairOutcome, ...],
    tolerance: float,
) -> bytes:
    payload = {
        "subject": subject,
        "metric": metric,
        "objective": objective,
        "repairs": list(repairs),
        "analysis_tolerance": tolerance,
        "outcomes": [
            {
                "repairs": sorted(item.repairs),
                "value": item.value,
                "evidence_id": item.evidence_id,
            }
            for item in outcomes
        ],
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def analyze_repair_lattice(
    *,
    subject: str,
    metric: str,
    objective: Objective,
    repairs: Sequence[str],
    outcomes: Sequence[RepairOutcome],
    tolerance: float = 0.0,
) -> RepairInteractionCertificate:
    """Analyze a complete factorial repair lattice.

    A *strict compensating bundle* is a set of two or more repairs for which:

    1. applying the whole set is no worse than the baseline metric; and
    2. every non-empty proper subset is worse than baseline.

    Two modes are distinguished:

    - partial masking: the fully repaired set improves on a still-imperfect
      faulty baseline;
    - complete masking: the faulty baseline and fully repaired system are
      behaviorally equivalent within tolerance, while every partial repair is
      worse.

    The second case is especially dangerous: task success can make a
    semantically wrong pipeline indistinguishable from a correct one until one
    boundary is fixed, replaced, or reused in a different stack.

    The complete power set is required. Missing cells fail closed rather than
    being imputed.
    """

    if not subject:
        raise ValueError("subject must be non-empty")
    if not metric:
        raise ValueError("metric must be non-empty")
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    repair_names = tuple(sorted(dict.fromkeys(repairs)))
    if not repair_names:
        raise ValueError("at least one repair is required")
    if len(repair_names) != len(tuple(repairs)):
        raise ValueError("repair names must be unique")

    expected = set(_powerset(repair_names))
    by_subset: dict[frozenset[str], RepairOutcome] = {}
    for item in outcomes:
        unknown = item.repairs - set(repair_names)
        if unknown:
            raise ValueError(f"outcome references unknown repairs: {sorted(unknown)}")
        if item.repairs in by_subset:
            raise ValueError(f"duplicate outcome for subset: {sorted(item.repairs)}")
        by_subset[item.repairs] = item

    missing = expected - set(by_subset)
    extra = set(by_subset) - expected
    if missing or extra:
        raise ValueError(
            "repair lattice must be complete; "
            f"missing={sorted(map(sorted, missing))}, "
            f"extra={sorted(map(sorted, extra))}"
        )

    ordered_outcomes = tuple(
        by_subset[subset]
        for subset in sorted(expected, key=lambda s: (len(s), tuple(sorted(s))))
    )
    loss = {
        subset: _canonical_loss(item.value, objective)
        for subset, item in by_subset.items()
    }
    baseline = loss[frozenset()]

    edge_violations: list[RepairEdgeViolation] = []
    for subset in expected:
        for repair in repair_names:
            if repair in subset:
                continue
            after = subset | {repair}
            if loss[after] > loss[subset] + tolerance:
                edge_violations.append(
                    RepairEdgeViolation(
                        before=tuple(sorted(subset)),
                        added_repair=repair,
                        after=tuple(sorted(after)),
                        before_loss=loss[subset],
                        after_loss=loss[after],
                    )
                )

    bundles: list[CompensatingRepairBundle] = []
    for subset in expected:
        if len(subset) < 2:
            continue
        if loss[subset] > baseline + tolerance:
            continue

        proper_nonempty = [
            other
            for other in expected
            if other and other < subset
        ]
        if proper_nonempty and all(
            loss[other] > baseline + tolerance
            for other in proper_nonempty
        ):
            bundles.append(
                CompensatingRepairBundle(
                    repairs=tuple(sorted(subset)),
                    baseline_loss=baseline,
                    repaired_loss=loss[subset],
                    proper_subset_losses=tuple(
                        (tuple(sorted(other)), loss[other])
                        for other in sorted(
                            proper_nonempty,
                            key=lambda s: (len(s), tuple(sorted(s))),
                        )
                    ),
                    masking_mode=(
                        "partial"
                        if loss[subset] < baseline - tolerance
                        else "complete"
                    ),
                )
            )

    # Möbius transform of the canonical-loss set function.  Order-1 terms are
    # main effects; order >=2 terms quantify non-additive interaction.
    mobius: list[tuple[tuple[str, ...], float]] = []
    for subset in sorted(expected, key=lambda s: (len(s), tuple(sorted(s)))):
        if not subset:
            continue
        total = 0.0
        subset_items = tuple(sorted(subset))
        for size in range(len(subset_items) + 1):
            for combo in combinations(subset_items, size):
                inner = frozenset(combo)
                sign = -1.0 if (len(subset) - len(inner)) % 2 else 1.0
                total += sign * loss[inner]
        mobius.append((subset_items, total))

    digest = sha256(
        _canonical_payload(
            subject=subject,
            metric=metric,
            objective=objective,
            repairs=repair_names,
            outcomes=ordered_outcomes,
            tolerance=tolerance,
        )
    ).hexdigest()

    return RepairInteractionCertificate(
        subject=subject,
        metric=metric,
        objective=objective,
        repairs=repair_names,
        outcomes=ordered_outcomes,
        edge_violations=tuple(
            sorted(
                edge_violations,
                key=lambda item: (item.before, item.added_repair),
            )
        ),
        compensating_bundles=tuple(
            sorted(bundles, key=lambda item: (len(item.repairs), item.repairs))
        ),
        mobius_terms=tuple(mobius),
        digest=digest,
        analysis_tolerance=tolerance,
    )


@dataclass(frozen=True)
class RepairInteractionAuthorization:
    authorized: bool
    requested_repairs: tuple[str, ...]
    reasons: tuple[str, ...]
    certificate_digest: str


def verify_repair_interaction_certificate(
    certificate: RepairInteractionCertificate,
) -> None:
    """Independently re-derive all claims before using an interaction certificate.

    A caller must not obtain execution authority by supplying arbitrary
    "compensating_bundles", evidence IDs, or a self-asserted digest. Analysis
    tolerance changes the classification and must be part of the digest.
    This verifies internal integrity, NOT source/protocol identity or external
    evidence independence; those require separate runtime trust checks.
    """
    if not isfinite(certificate.analysis_tolerance) or certificate.analysis_tolerance < 0:
        raise ValueError("interaction certificate has invalid analysis tolerance")
    expected = analyze_repair_lattice(
        subject=certificate.subject,
        metric=certificate.metric,
        objective=certificate.objective,
        repairs=certificate.repairs,
        outcomes=certificate.outcomes,
        tolerance=certificate.analysis_tolerance,
    )
    if certificate != expected:
        raise ValueError("interaction certificate integrity verification failed")


def authorize_repair_subset(
    certificate: RepairInteractionCertificate,
    requested_repairs: Sequence[str],
    *,
    reject_measured_regression: bool = True,
    tolerance: float = 0.0,
) -> RepairInteractionAuthorization:
    """Fail closed on partial repairs inside a measured compensating bundle.

    Local repair verification is insufficient when end-to-end evidence shows
    that multiple semantic defects interact. If a requested set is a non-empty
    proper subset of any strict compensating bundle, it is denied even when each
    local repair is independently valid.

    The gate can also reject a requested subset whose measured canonical loss is
    worse than the all-fault baseline.
    """

    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    verify_repair_interaction_certificate(certificate)

    requested = frozenset(requested_repairs)
    known = set(certificate.repairs)
    unknown = requested - known
    if unknown:
        raise ValueError(f"requested unknown repairs: {sorted(unknown)}")

    reasons: list[str] = []

    for bundle in certificate.compensating_bundles:
        bundle_set = frozenset(bundle.repairs)
        if requested and requested < bundle_set:
            reasons.append(
                "requested repair set is an incomplete subset of compensating "
                f"bundle {bundle.repairs!r}"
            )

    by_subset = {item.repairs: item for item in certificate.outcomes}
    baseline = by_subset[frozenset()].value
    selected = by_subset.get(requested)
    if selected is None:
        reasons.append("requested repair set has no measured factorial outcome")
    elif reject_measured_regression:
        before = _canonical_loss(baseline, certificate.objective)
        after = _canonical_loss(selected.value, certificate.objective)
        if after > before + tolerance:
            reasons.append(
                "requested repair set regresses the measured end-to-end metric "
                f"from {baseline!r} to {selected.value!r}"
            )

    return RepairInteractionAuthorization(
        authorized=not reasons,
        requested_repairs=tuple(sorted(requested)),
        reasons=tuple(reasons),
        certificate_digest=certificate.digest,
    )


@dataclass(frozen=True)
class ReplicatedRepairAuthorization:
    authorized: bool
    requested_repairs: tuple[str, ...]
    replicate_authorizations: tuple[RepairInteractionAuthorization, ...]
    reasons: tuple[str, ...]

    @property
    def unanimous(self) -> bool:
        decisions = {item.authorized for item in self.replicate_authorizations}
        return len(decisions) <= 1


def authorize_repair_subset_across_replicates(
    certificates: Sequence[RepairInteractionCertificate],
    requested_repairs: Sequence[str],
    *,
    reject_measured_regression: bool = True,
    tolerance: float = 0.0,
) -> ReplicatedRepairAuthorization:
    """Authorize only when every frozen interaction replicate permits activation.

    Embodied replay/task metrics can be nondeterministic. A favorable replicate
    must not erase an unfavorable one. All certificates must describe the same
    subject/metric/objective/repair set; then the requested repair set is
    authorized iff every replicate-level authorization is true.

    This is deliberately conservative. Conflicting replicate outcomes remain
    evidence of ambiguity until a more direct semantic metric or additional
    evidence resolves the disagreement.
    """
    if not certificates:
        raise ValueError("at least one interaction certificate is required")

    first = certificates[0]
    signature = (
        first.subject,
        first.metric,
        first.objective,
        first.repairs,
    )
    for certificate in certificates[1:]:
        other = (
            certificate.subject,
            certificate.metric,
            certificate.objective,
            certificate.repairs,
        )
        if other != signature:
            raise ValueError(
                "replicated interaction certificates must share subject, metric, "
                "objective, and repair set"
            )

    decisions = tuple(
        authorize_repair_subset(
            certificate,
            requested_repairs,
            reject_measured_regression=reject_measured_regression,
            tolerance=tolerance,
        )
        for certificate in certificates
    )

    reasons: list[str] = []
    for index, decision in enumerate(decisions):
        if decision.authorized:
            continue
        if decision.reasons:
            reasons.extend(
                f"replicate[{index}] {reason}" for reason in decision.reasons
            )
        else:
            reasons.append(f"replicate[{index}] denied without a reason")

    return ReplicatedRepairAuthorization(
        authorized=all(item.authorized for item in decisions),
        requested_repairs=tuple(sorted(set(requested_repairs))),
        replicate_authorizations=decisions,
        reasons=tuple(reasons),
    )



@dataclass(frozen=True)
class RepairAuthorizationPlane:
    plane_id: str
    certificate: RepairInteractionCertificate

    def __post_init__(self) -> None:
        if not self.plane_id:
            raise ValueError("plane_id must be non-empty")


@dataclass(frozen=True)
class MultiPlaneRepairAuthorization:
    authorized: bool
    requested_repairs: tuple[str, ...]
    plane_decisions: tuple[tuple[str, RepairInteractionAuthorization], ...]
    reasons: tuple[str, ...]


def authorize_repair_subset_across_planes(
    planes: Sequence[RepairAuthorizationPlane],
    requested_repairs: Sequence[str],
    *,
    reject_measured_regression: bool = True,
    tolerance: float = 0.0,
) -> MultiPlaneRepairAuthorization:
    """Require independent authorization on every declared evidence plane.

    A repair can be semantically correct yet still regress an execution-domain
    metric.  Semantic fidelity, replay/task non-regression, safety and
    post-effect evidence therefore remain distinct authority planes.

    All planes must describe the same candidate repair set.  Their metric,
    objective and evidence may differ.  Missing planes are never inferred.
    """

    if not planes:
        raise ValueError("at least one authorization plane is required")

    repair_signature = planes[0].certificate.repairs
    seen_plane_ids: set[str] = set()
    decisions: list[tuple[str, RepairInteractionAuthorization]] = []
    reasons: list[str] = []

    for plane in planes:
        if plane.plane_id in seen_plane_ids:
            raise ValueError(f"duplicate authorization plane: {plane.plane_id!r}")
        seen_plane_ids.add(plane.plane_id)

        if plane.certificate.repairs != repair_signature:
            raise ValueError(
                "all authorization planes must describe the same repair set"
            )

        decision = authorize_repair_subset(
            plane.certificate,
            requested_repairs,
            reject_measured_regression=reject_measured_regression,
            tolerance=tolerance,
        )
        decisions.append((plane.plane_id, decision))
        if not decision.authorized:
            if decision.reasons:
                reasons.extend(
                    f"{plane.plane_id}: {reason}" for reason in decision.reasons
                )
            else:
                reasons.append(f"{plane.plane_id}: denied without a reason")

    return MultiPlaneRepairAuthorization(
        authorized=all(decision.authorized for _, decision in decisions),
        requested_repairs=tuple(sorted(set(requested_repairs))),
        plane_decisions=tuple(decisions),
        reasons=tuple(reasons),
    )



@dataclass(frozen=True)
class QualifiedRepairAuthorizationPlane:
    plane_id: str
    certificate: RepairInteractionCertificate
    measurement: MeasurementQualificationCertificate

    def __post_init__(self) -> None:
        if not self.plane_id:
            raise ValueError("plane_id must be non-empty")


@dataclass(frozen=True)
class QualifiedMultiPlaneRepairAuthorization:
    authorized: bool
    requested_repairs: tuple[str, ...]
    plane_decisions: tuple[tuple[str, RepairInteractionAuthorization], ...]
    measurement_digests: tuple[tuple[str, str], ...]
    reasons: tuple[str, ...]


def authorize_repair_subset_across_qualified_planes(
    planes: Sequence[QualifiedRepairAuthorizationPlane],
    requested_repairs: Sequence[str],
    *,
    reject_measured_regression: bool = True,
    tolerance: float = 0.0,
) -> QualifiedMultiPlaneRepairAuthorization:
    """Require qualified measurements before any evidence plane can authorize.

    Repeatable but semantically non-identifying measurements are rejected before
    their interaction metric is consulted.  This prevents exact self-consistency
    signals from acquiring repair authority when the same latent semantic error
    can produce the same observation.
    """
    if not planes:
        raise ValueError("at least one qualified authorization plane is required")

    repair_signature = planes[0].certificate.repairs
    seen: set[str] = set()
    decisions: list[tuple[str, RepairInteractionAuthorization]] = []
    measurement_digests: list[tuple[str, str]] = []
    reasons: list[str] = []

    for plane in planes:
        if plane.plane_id in seen:
            raise ValueError(f"duplicate authorization plane: {plane.plane_id!r}")
        seen.add(plane.plane_id)

        if plane.certificate.repairs != repair_signature:
            raise ValueError(
                "all qualified authorization planes must describe the same repair set"
            )

        try:
            require_qualified_measurement(plane.measurement)
        except MeasurementNotQualified as exc:
            reasons.append(f"{plane.plane_id}: {exc}")
            continue

        decision = authorize_repair_subset(
            plane.certificate,
            requested_repairs,
            reject_measured_regression=reject_measured_regression,
            tolerance=tolerance,
        )
        decisions.append((plane.plane_id, decision))
        measurement_digests.append((plane.plane_id, plane.measurement.digest))

        if not decision.authorized:
            if decision.reasons:
                reasons.extend(
                    f"{plane.plane_id}: {reason}" for reason in decision.reasons
                )
            else:
                reasons.append(f"{plane.plane_id}: denied without a reason")

    return QualifiedMultiPlaneRepairAuthorization(
        authorized=(len(decisions) == len(planes)) and all(
            decision.authorized for _, decision in decisions
        ),
        requested_repairs=tuple(sorted(set(requested_repairs))),
        plane_decisions=tuple(decisions),
        measurement_digests=tuple(measurement_digests),
        reasons=tuple(reasons),
    )
