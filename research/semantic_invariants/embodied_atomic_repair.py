"""Atomic deployment gates for interacting semantic repairs.

Factorial repair analysis can reveal a dangerous non-monotone regime: two
locally correct repairs form a compensating bundle, so deploying only one
repair makes the end-to-end system worse than the known baseline.

This module turns that diagnosis into a deployment constraint.  A bundle
certificate binds:

- the exact repair-interaction evidence digest;
- the required repair set;
- the concrete implementation identity of every repair; and
- the masking mode that justified atomic deployment.

The gate does not apply repairs itself.  It authorizes a deployment plan only
when the complete certified bundle is requested with the exact implementations
that were reviewed.  Partial hotfixes and implementation drift fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping, Sequence

from .embodied_repair_interactions import RepairInteractionCertificate


class AtomicRepairRequired(RuntimeError):
    pass


class AtomicRepairEvidenceMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class AtomicRepairBundleCertificate:
    subject: str
    interaction_digest: str
    required_repairs: tuple[str, ...]
    implementations: tuple[tuple[str, str], ...]
    masking_mode: str
    digest: str

    def implementation_for(self, repair: str) -> str:
        for name, identity in self.implementations:
            if name == repair:
                return identity
        raise KeyError(repair)


@dataclass(frozen=True)
class AtomicRepairAuthorization:
    subject: str
    repairs: tuple[str, ...]
    implementations: tuple[tuple[str, str], ...]
    interaction_digest: str
    certificate_digest: str


def _certificate_digest(
    *,
    subject: str,
    interaction_digest: str,
    required_repairs: Sequence[str],
    implementations: Sequence[tuple[str, str]],
    masking_mode: str,
) -> str:
    payload = {
        "subject": subject,
        "interaction_digest": interaction_digest,
        "required_repairs": list(required_repairs),
        "implementations": [list(item) for item in implementations],
        "masking_mode": masking_mode,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()



def required_atomic_closure(
    *,
    interaction: RepairInteractionCertificate,
    requested_repairs: Sequence[str],
) -> tuple[str, ...]:
    """Return the least repair set closed under all compensating bundles.

    If a requested repair touches one member of a compensating bundle, every
    member is required. Overlapping bundles are resolved to a fixed point, so a
    local hotfix can expose transitive deployment dependencies.
    """

    closure = set(requested_repairs)
    known = set(interaction.repairs)
    unknown = closure - known
    if unknown:
        raise ValueError(f"unknown repairs: {sorted(unknown)}")

    changed = True
    while changed:
        changed = False
        for bundle in interaction.compensating_bundles:
            members = set(bundle.repairs)
            if closure & members and not members <= closure:
                closure.update(members)
                changed = True

    return tuple(sorted(closure))

def certify_atomic_repair_bundle(
    *,
    interaction: RepairInteractionCertificate,
    repairs: Sequence[str],
    implementations: Mapping[str, str],
) -> AtomicRepairBundleCertificate:
    """Bind one detected compensating bundle to concrete repair identities."""

    required = tuple(sorted(dict.fromkeys(repairs)))
    if len(required) < 2:
        raise ValueError("atomic repair bundle requires at least two repairs")

    bundle = next(
        (
            item
            for item in interaction.compensating_bundles
            if item.repairs == required
        ),
        None,
    )
    if bundle is None:
        raise AtomicRepairEvidenceMismatch(
            "requested repair set is not a compensating bundle in the "
            "interaction certificate"
        )

    missing = [name for name in required if not implementations.get(name)]
    if missing:
        raise ValueError(
            f"missing concrete implementation identities for repairs: {missing}"
        )

    bound = tuple((name, implementations[name]) for name in required)
    digest = _certificate_digest(
        subject=interaction.subject,
        interaction_digest=interaction.digest,
        required_repairs=required,
        implementations=bound,
        masking_mode=bundle.masking_mode,
    )
    return AtomicRepairBundleCertificate(
        subject=interaction.subject,
        interaction_digest=interaction.digest,
        required_repairs=required,
        implementations=bound,
        masking_mode=bundle.masking_mode,
        digest=digest,
    )


def authorize_repair_deployment(
    *,
    interaction: RepairInteractionCertificate,
    certificate: AtomicRepairBundleCertificate,
    requested_implementations: Mapping[str, str],
) -> AtomicRepairAuthorization:
    """Authorize only the complete implementation-bound repair bundle.

    Calling this gate means a deployment intends to touch at least one member
    of a known compensating bundle.  The entire certified bundle must therefore
    be present.  "Fix one now, fix the other later" is rejected.
    """

    if certificate.interaction_digest != interaction.digest:
        raise AtomicRepairEvidenceMismatch(
            "repair-interaction evidence digest changed after certification"
        )

    required = set(certificate.required_repairs)
    requested = set(requested_implementations)

    touched = required & requested
    if not touched:
        raise AtomicRepairRequired(
            "deployment does not include any member of the certified bundle"
        )

    missing = required - requested
    if missing:
        raise AtomicRepairRequired(
            "compensating semantic repairs must deploy atomically; "
            f"missing={sorted(missing)}"
        )

    extra = requested - required
    if extra:
        raise AtomicRepairEvidenceMismatch(
            "atomic bundle certificate does not authorize extra repairs; "
            f"extra={sorted(extra)}"
        )

    for repair in certificate.required_repairs:
        expected = certificate.implementation_for(repair)
        actual = requested_implementations[repair]
        if actual != expected:
            raise AtomicRepairEvidenceMismatch(
                f"implementation drift for {repair!r}: "
                f"{actual!r} != {expected!r}"
            )

    bundle = next(
        (
            item
            for item in interaction.compensating_bundles
            if item.repairs == certificate.required_repairs
        ),
        None,
    )
    if bundle is None or bundle.masking_mode != certificate.masking_mode:
        raise AtomicRepairEvidenceMismatch(
            "certified masking classification no longer matches interaction evidence"
        )

    expected_digest = _certificate_digest(
        subject=certificate.subject,
        interaction_digest=certificate.interaction_digest,
        required_repairs=certificate.required_repairs,
        implementations=certificate.implementations,
        masking_mode=certificate.masking_mode,
    )
    if expected_digest != certificate.digest:
        raise AtomicRepairEvidenceMismatch("atomic bundle certificate was modified")

    return AtomicRepairAuthorization(
        subject=certificate.subject,
        repairs=certificate.required_repairs,
        implementations=certificate.implementations,
        interaction_digest=certificate.interaction_digest,
        certificate_digest=certificate.digest,
    )
