"""Version-bound authorization for semantic repair interaction evidence.

Interaction results are only interpretable for the executable stack that
produced them. This module binds a factorial result to source/config/protocol
bytes and requires a separately pinned trust-root digest before authorizing.

A digest is not a signature. The operator must supply `trusted_seal_digest`
from a trusted release or deployment control plane; self-supplying a fresh
digest gives NO independent provenance guarantee. This module cannot prevent
a TOCTOU swap after verification: recheck immediately before every dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping, Sequence

from .embodied_repair_interactions import (
    RepairInteractionAuthorization,
    RepairInteractionCertificate,
    authorize_repair_subset,
    verify_repair_interaction_certificate,
)


class ExecutionContextMismatch(RuntimeError):
    """A repair's live executable or evidence context differs from its seal."""


@dataclass(frozen=True)
class VersionBoundRepairContext:
    subject: str
    interaction_digest: str
    source_digests: tuple[tuple[str, str], ...]
    configuration_digest: str
    protocol_digest: str
    seal_digest: str


@dataclass(frozen=True)
class VersionBoundRepairAuthorization:
    context_seal: str
    interaction_digest: str
    decision: RepairInteractionAuthorization


def _sha(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("context inputs must be raw bytes")
    return sha256(data).hexdigest()


def _sources(source_files: Mapping[str, bytes]) -> tuple[tuple[str, str], ...]:
    if not source_files:
        raise ValueError("at least one executable source blob is required")
    if any(not isinstance(name, str) or not name.strip() for name in source_files):
        raise ValueError("source labels must be non-empty")
    return tuple(sorted((name, _sha(blob)) for name, blob in source_files.items()))


def _seal(
    subject: str,
    interaction_digest: str,
    sources: tuple[tuple[str, str], ...],
    configuration_digest: str,
    protocol_digest: str,
) -> str:
    payload = {
        "schema": "semrepair-version-bound-repair-v1",
        "subject": subject,
        "interaction_digest": interaction_digest,
        "source_digests": [list(item) for item in sources],
        "configuration_digest": configuration_digest,
        "protocol_digest": protocol_digest,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def freeze_version_bound_context(
    *,
    interaction: RepairInteractionCertificate,
    source_files: Mapping[str, bytes],
    configuration: bytes,
    protocol: bytes,
) -> VersionBoundRepairContext:
    """Snapshot the exact code+configuration+protocol used for a result.

    This returns a self-authored digest. Issuance alone grants zero authority.
    """
    verify_repair_interaction_certificate(interaction)
    sources = _sources(source_files)
    config = _sha(configuration)
    protocol_hash = _sha(protocol)
    seal = _seal(interaction.subject, interaction.digest, sources, config, protocol_hash)
    return VersionBoundRepairContext(
        subject=interaction.subject,
        interaction_digest=interaction.digest,
        source_digests=sources,
        configuration_digest=config,
        protocol_digest=protocol_hash,
        seal_digest=seal,
    )


def verify_live_version_bound_context(
    *,
    frozen: VersionBoundRepairContext,
    interaction: RepairInteractionCertificate,
    trusted_seal_digest: str,
    live_source_files: Mapping[str, bytes],
    live_configuration: bytes,
    live_protocol: bytes,
) -> None:
    """Fail closed unless every live byte and the independently pinned root match."""
    verify_repair_interaction_certificate(interaction)
    if not isinstance(trusted_seal_digest, str) or len(trusted_seal_digest) != 64:
        raise ExecutionContextMismatch("missing or invalid trusted context seal")
    if not isinstance(frozen.seal_digest, str) or frozen.seal_digest != trusted_seal_digest:
        raise ExecutionContextMismatch("frozen context seal is not trusted")

    try:
        expected = freeze_version_bound_context(
            interaction=interaction,
            source_files=live_source_files,
            configuration=live_configuration,
            protocol=live_protocol,
        )
    except (ValueError, TypeError, AttributeError) as exc:
        raise ExecutionContextMismatch("invalid live execution-context inputs") from exc

    if frozen != expected:
        raise ExecutionContextMismatch(
            "live source, configuration, protocol, subject or interaction evidence drift"
        )


def authorize_version_bound_repair_subset(
    *,
    frozen: VersionBoundRepairContext,
    interaction: RepairInteractionCertificate,
    trusted_seal_digest: str,
    live_source_files: Mapping[str, bytes],
    live_configuration: bytes,
    live_protocol: bytes,
    requested_repairs: Sequence[str],
) -> VersionBoundRepairAuthorization:
    """Version-bind the primary interaction decision; call again before dispatch.

    The resulting decision is not a physical safety proof. Other SemRepair
    measurement, local-program, execution-domain and post-effect gates remain
    mandatory for full deployment.
    """
    verify_live_version_bound_context(
        frozen=frozen,
        interaction=interaction,
        trusted_seal_digest=trusted_seal_digest,
        live_source_files=live_source_files,
        live_configuration=live_configuration,
        live_protocol=live_protocol,
    )
    decision = authorize_repair_subset(interaction, requested_repairs)
    return VersionBoundRepairAuthorization(
        context_seal=frozen.seal_digest,
        interaction_digest=interaction.digest,
        decision=decision,
    )


def authorize_version_bound_pre_dispatch(
    *,
    frozen: VersionBoundRepairContext,
    trusted_seal_digest: str,
    live_source_files: Mapping[str, bytes],
    live_configuration: bytes,
    live_protocol: bytes,
    interaction: RepairInteractionCertificate,
    repairs,
    raw_action,
    semantic_target,
    domain,
    forward,
    forward_model_id: str,
    distance,
    qualified_planes,
    atomic_certificate=None,
    projection_certificate=None,
):
    """Mandatory context check wrapped around SemRepair's *full* dispatch gate.

    This intentionally has no fallback to an unqualified/local-only decision.
    The external dispatch adapter must call `verify_live_version_bound_context`
    again immediately before each physical/simulator effect, since a context
    can drift between prepare and dispatch (TOCTOU).
    """
    from .embodied_authority_kernel import authorize_evidence_qualified_pre_dispatch

    verify_live_version_bound_context(
        frozen=frozen,
        interaction=interaction,
        trusted_seal_digest=trusted_seal_digest,
        live_source_files=live_source_files,
        live_configuration=live_configuration,
        live_protocol=live_protocol,
    )
    return authorize_evidence_qualified_pre_dispatch(
        repairs=repairs,
        raw_action=raw_action,
        semantic_target=semantic_target,
        domain=domain,
        forward=forward,
        forward_model_id=forward_model_id,
        distance=distance,
        qualified_planes=qualified_planes,
        interaction=interaction,
        atomic_certificate=atomic_certificate,
        projection_certificate=projection_certificate,
    )
