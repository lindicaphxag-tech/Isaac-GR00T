"""Composed pre-dispatch authority kernel for semantic repairs.

The kernel closes an architectural bypass: local repair verification,
interaction-aware atomic deployment, and execution-domain projection are not
independent optional helpers at dispatch time. A caller submits the complete
proof context and receives one implementation-bound authority receipt.

Effect/post-dispatch evidence remains a later phase and is deliberately not
collapsed into this pre-dispatch judgment.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Sequence

from .embodied_atomic_repair import (
    AtomicRepairBundleCertificate,
    AtomicRepairRequired,
    authorize_repair_deployment,
    required_atomic_closure,
)
from .embodied_execution_domain import (
    ExecutionDomainAuthorization,
    L2BallExecutionDomain,
    ProjectionCertificate,
    SemanticDistance,
    authorize_execution_domain,
)
from .embodied_measurement_qualification import (
    MeasurementQualificationCertificate,
    MeasurementNotQualified,
    require_qualified_measurement,
)
from .embodied_repair_interactions import (
    QualifiedRepairAuthorizationPlane,
    RepairInteractionCertificate,
    authorize_repair_subset_across_qualified_planes,
)
from .embodied_repair_synthesis import RepairProgram, Vector
from .embodied_repair_verification import (
    RepairVerificationCertificate,
    certificate_matches_program,
)


class PreDispatchAuthorityRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalRepairProof:
    repair_name: str
    contract_id: str
    program: RepairProgram
    certificate: RepairVerificationCertificate
    measurement: MeasurementQualificationCertificate | None = None


@dataclass(frozen=True)
class PreDispatchAuthority:
    repair_names: tuple[str, ...]
    local_program_fingerprints: tuple[tuple[str, str], ...]
    measurement_digests: tuple[tuple[str, str], ...]
    interaction_digest: str | None
    atomic_certificate_digest: str | None
    projection_certificate_digest: str | None
    qualified_plane_digests: tuple[tuple[str, str, str], ...]
    authorized_action: Vector
    projection_used: bool
    semantic_residual: float
    authority_digest: str


def _digest(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def authorize_pre_dispatch(
    *,
    repairs: Sequence[LocalRepairProof],
    raw_action: Sequence[float],
    semantic_target: Sequence[float],
    domain: L2BallExecutionDomain,
    forward: Callable[[Vector], Vector],
    forward_model_id: str,
    distance: SemanticDistance,
    interaction: RepairInteractionCertificate | None = None,
    atomic_certificate: AtomicRepairBundleCertificate | None = None,
    projection_certificate: ProjectionCertificate | None = None,
    qualified_planes: Sequence[QualifiedRepairAuthorizationPlane] = (),
) -> PreDispatchAuthority:
    """Compose local, interaction and execution-domain proof obligations."""

    if not repairs:
        raise PreDispatchAuthorityRejected("at least one local repair proof is required")

    by_name: dict[str, LocalRepairProof] = {}
    for proof in repairs:
        if not proof.repair_name:
            raise PreDispatchAuthorityRejected("repair_name must be non-empty")
        if proof.repair_name in by_name:
            raise PreDispatchAuthorityRejected(
                f"duplicate local repair proof: {proof.repair_name!r}"
            )
        if not certificate_matches_program(
            proof.certificate,
            contract_id=proof.contract_id,
            program=proof.program,
        ):
            raise PreDispatchAuthorityRejected(
                f"local repair certificate does not match {proof.repair_name!r}"
            )
        if proof.measurement is None:
            raise PreDispatchAuthorityRejected(
                f"local repair proof {proof.repair_name!r} has no measurement qualification"
            )
        try:
            require_qualified_measurement(proof.measurement)
        except MeasurementNotQualified as exc:
            raise PreDispatchAuthorityRejected(
                f"measurement for {proof.repair_name!r} is not qualified: {exc}"
            ) from exc
        by_name[proof.repair_name] = proof

    names = tuple(sorted(by_name))
    fingerprints = tuple(
        (name, by_name[name].certificate.program_fingerprint) for name in names
    )
    measurement_digests = tuple(
        (name, by_name[name].measurement.digest) for name in names
    )

    interaction_digest: str | None = None
    atomic_digest: str | None = None

    if interaction is not None:
        interaction_digest = interaction.digest
        known = set(interaction.repairs)
        unknown = set(names) - known
        if unknown:
            raise PreDispatchAuthorityRejected(
                f"interaction certificate does not cover repairs: {sorted(unknown)}"
            )

        closure = required_atomic_closure(
            interaction=interaction,
            requested_repairs=names,
        )
        if closure != names:
            raise AtomicRepairRequired(
                "requested repair set is not closed under compensating bundles; "
                f"required={closure!r}, requested={names!r}"
            )

        touches_compensating_bundle = any(
            set(bundle.repairs) & set(names)
            for bundle in interaction.compensating_bundles
        )
        if touches_compensating_bundle:
            if atomic_certificate is None:
                raise AtomicRepairRequired(
                    "interaction evidence requires an implementation-bound "
                    "atomic repair certificate"
                )
            atomic = authorize_repair_deployment(
                interaction=interaction,
                certificate=atomic_certificate,
                requested_implementations=dict(fingerprints),
            )
            atomic_digest = atomic.certificate_digest
        elif atomic_certificate is not None:
            raise PreDispatchAuthorityRejected(
                "atomic certificate supplied but no compensating bundle is touched"
            )
    elif atomic_certificate is not None:
        raise PreDispatchAuthorityRejected(
            "atomic certificate cannot be verified without interaction evidence"
        )

    qualified_plane_digests: tuple[tuple[str, str, str], ...] = ()
    if qualified_planes:
        if interaction is not None and any(
            plane.certificate.subject != interaction.subject
            for plane in qualified_planes
        ):
            raise PreDispatchAuthorityRejected(
                "qualified-plane subject/context does not match interaction evidence"
            )
        plane_result = authorize_repair_subset_across_qualified_planes(
            qualified_planes,
            names,
        )
        if not plane_result.authorized:
            detail = "; ".join(plane_result.reasons) or "qualified evidence plane rejected"
            raise PreDispatchAuthorityRejected(
                f"qualified evidence planes rejected repair activation: {detail}"
            )
        qualified_plane_digests = tuple(
            (
                plane.plane_id,
                plane.certificate.digest,
                plane.measurement.digest,
            )
            for plane in qualified_planes
        )

    execution: ExecutionDomainAuthorization = authorize_execution_domain(
        contract_id="+".join(by_name[name].contract_id for name in names),
        raw_action=raw_action,
        semantic_target=semantic_target,
        domain=domain,
        forward=forward,
        forward_model_id=forward_model_id,
        distance=distance,
        projection_certificate=projection_certificate,
    )

    payload = {
        "repair_names": list(names),
        "local_program_fingerprints": [list(item) for item in fingerprints],
        "measurement_digests": [list(item) for item in measurement_digests],
        "interaction_digest": interaction_digest,
        "atomic_certificate_digest": atomic_digest,
        "projection_certificate_digest": execution.projection_certificate_digest,
        "qualified_plane_digests": [list(item) for item in qualified_plane_digests],
        "authorized_action": list(execution.authorized_action),
        "projection_used": execution.projection_used,
        "semantic_residual": execution.residual,
        "domain_digest": domain.digest,
        "forward_model_id": forward_model_id,
        "semantic_target": [float(x) for x in semantic_target],
    }

    return PreDispatchAuthority(
        repair_names=names,
        local_program_fingerprints=fingerprints,
        measurement_digests=measurement_digests,
        interaction_digest=interaction_digest,
        atomic_certificate_digest=atomic_digest,
        projection_certificate_digest=execution.projection_certificate_digest,
        qualified_plane_digests=qualified_plane_digests,
        authorized_action=execution.authorized_action,
        projection_used=execution.projection_used,
        semantic_residual=execution.residual,
        authority_digest=_digest(payload),
    )


def authorize_evidence_qualified_pre_dispatch(
    *,
    repairs: Sequence[LocalRepairProof],
    raw_action: Sequence[float],
    semantic_target: Sequence[float],
    domain: L2BallExecutionDomain,
    forward: Callable[[Vector], Vector],
    forward_model_id: str,
    distance: SemanticDistance,
    qualified_planes: Sequence[QualifiedRepairAuthorizationPlane],
    interaction: RepairInteractionCertificate | None = None,
    atomic_certificate: AtomicRepairBundleCertificate | None = None,
    projection_certificate: ProjectionCertificate | None = None,
    required_plane_ids: Sequence[str] = (
        "semantic-fidelity",
        "execution-effect",
    ),
) -> PreDispatchAuthority:
    """High-level deployment gate that cannot bypass qualified evidence planes.

    The low-level authorize_pre_dispatch function remains a compositional kernel.
    This function is the deployment entry point: all required evidence planes
    must be present, measurement-qualified, and independently authorize the
    requested repair set before physical dispatch can be granted.
    """

    required = tuple(dict.fromkeys(str(item) for item in required_plane_ids))
    if not required or any(not item for item in required):
        raise PreDispatchAuthorityRejected(
            "required evidence plane ids must be unique and non-empty"
        )

    by_id = {plane.plane_id: plane for plane in qualified_planes}
    if len(by_id) != len(tuple(qualified_planes)):
        raise PreDispatchAuthorityRejected("qualified evidence plane ids must be unique")

    missing = tuple(sorted(set(required) - set(by_id)))
    if missing:
        raise PreDispatchAuthorityRejected(
            f"missing required qualified evidence planes: {missing!r}"
        )

    return authorize_pre_dispatch(
        repairs=repairs,
        raw_action=raw_action,
        semantic_target=semantic_target,
        domain=domain,
        forward=forward,
        forward_model_id=forward_model_id,
        distance=distance,
        interaction=interaction,
        atomic_certificate=atomic_certificate,
        projection_certificate=projection_certificate,
        qualified_planes=qualified_planes,
    )
