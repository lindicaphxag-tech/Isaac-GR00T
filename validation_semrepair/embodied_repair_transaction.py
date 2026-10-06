"""Diagnosis-bound atomic transactions for interacting semantic repairs.

A locally correct hotfix can make a pipeline worse when multiple semantic
faults compensate.  This module connects three existing pieces:

1. factorial repair-interaction evidence;
2. minimum-cost semantic diagnosis taps; and
3. implementation-bound atomic deployment.

The transaction is prepared against frozen interaction and diagnosis digests.
It stages the least atomic repair closure, requires concrete implementation
identities, and emits shadow-probe obligations that must match a declared
reference semantic hypothesis before the bundle may commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping, Sequence

from .embodied_atomic_repair import required_atomic_closure
from .embodied_repair_interactions import RepairInteractionCertificate
from .embodied_semantic_observability import (
    PairwiseDiagnosticWitness,
    SemanticDiagnosisPlan,
)


class RepairTransactionRejected(RuntimeError):
    pass


class RepairTransactionEvidenceMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class ShadowProbeRequirement:
    fault_hypothesis: str
    tap_after_factor: str | None
    basis_index: int
    expected_reference: tuple[float, ...]


@dataclass(frozen=True)
class ShadowProbeObservation:
    fault_hypothesis: str
    tap_after_factor: str | None
    basis_index: int
    observed: tuple[float, ...]
    evidence_id: str

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("shadow observation requires evidence_id")


@dataclass(frozen=True)
class SemanticRepairTransactionCertificate:
    subject: str
    reference_hypothesis: str
    requested_repairs: tuple[str, ...]
    atomic_closure: tuple[str, ...]
    implementations: tuple[tuple[str, str], ...]
    repair_hypotheses: tuple[tuple[str, str], ...]
    interaction_digest: str
    diagnosis_digest: str
    probe_requirements: tuple[ShadowProbeRequirement, ...]
    digest: str

    def implementation_for(self, repair: str) -> str:
        for name, identity in self.implementations:
            if name == repair:
                return identity
        raise KeyError(repair)


@dataclass(frozen=True)
class SemanticRepairTransactionAuthorization:
    subject: str
    repairs: tuple[str, ...]
    implementations: tuple[tuple[str, str], ...]
    interaction_digest: str
    diagnosis_digest: str
    transaction_digest: str
    observation_digest: str


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def _witness_for_pair(
    diagnosis: SemanticDiagnosisPlan,
    left: str,
    right: str,
) -> PairwiseDiagnosticWitness | None:
    target = _canonical_pair(left, right)
    for witness in diagnosis.pair_witnesses:
        if _canonical_pair(witness.left, witness.right) == target:
            return witness
    return None


def _expected_for_reference(
    witness: PairwiseDiagnosticWitness,
    reference_hypothesis: str,
) -> tuple[float, ...]:
    if witness.left == reference_hypothesis:
        return witness.left_observed
    if witness.right == reference_hypothesis:
        return witness.right_observed
    raise ValueError("diagnostic witness does not include reference hypothesis")


def _transaction_digest(
    *,
    subject: str,
    reference_hypothesis: str,
    requested_repairs: Sequence[str],
    atomic_closure: Sequence[str],
    implementations: Sequence[tuple[str, str]],
    repair_hypotheses: Sequence[tuple[str, str]],
    interaction_digest: str,
    diagnosis_digest: str,
    requirements: Sequence[ShadowProbeRequirement],
) -> str:
    payload = {
        "subject": subject,
        "reference_hypothesis": reference_hypothesis,
        "requested_repairs": list(requested_repairs),
        "atomic_closure": list(atomic_closure),
        "implementations": [list(item) for item in implementations],
        "repair_hypotheses": [list(item) for item in repair_hypotheses],
        "interaction_digest": interaction_digest,
        "diagnosis_digest": diagnosis_digest,
        "probe_requirements": [
            {
                "fault_hypothesis": item.fault_hypothesis,
                "tap_after_factor": item.tap_after_factor,
                "basis_index": item.basis_index,
                "expected_reference": list(item.expected_reference),
            }
            for item in requirements
        ],
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def prepare_semantic_repair_transaction(
    *,
    interaction: RepairInteractionCertificate,
    diagnosis: SemanticDiagnosisPlan,
    requested_repairs: Sequence[str],
    implementations: Mapping[str, str],
    repair_hypotheses: Mapping[str, str],
    reference_hypothesis: str,
) -> SemanticRepairTransactionCertificate:
    """Prepare a diagnosis-bound atomic repair transaction.

    Every repair in the transitive atomic closure must have a concrete
    implementation identity and a frozen fault hypothesis that is distinguishable
    from the reference hypothesis under the supplied diagnosis plan.
    """

    if not diagnosis.complete:
        raise RepairTransactionRejected(
            "diagnosis plan contains intrinsically unidentifiable hypotheses"
        )
    if reference_hypothesis not in diagnosis.hypothesis_names:
        raise ValueError("reference hypothesis is absent from diagnosis plan")

    requested = tuple(sorted(dict.fromkeys(requested_repairs)))
    if not requested:
        raise ValueError("at least one requested repair is required")
    closure = required_atomic_closure(
        interaction=interaction,
        requested_repairs=requested,
    )

    missing_impl = [name for name in closure if not implementations.get(name)]
    if missing_impl:
        raise RepairTransactionRejected(
            f"atomic closure lacks implementation identities: {missing_impl}"
        )
    missing_hyp = [name for name in closure if not repair_hypotheses.get(name)]
    if missing_hyp:
        raise RepairTransactionRejected(
            f"atomic closure lacks fault hypotheses: {missing_hyp}"
        )

    extra_impl = set(implementations) - set(closure)
    if extra_impl:
        raise RepairTransactionRejected(
            "transaction contains implementations outside atomic closure: "
            f"{sorted(extra_impl)}"
        )

    bound_impl = tuple((name, implementations[name]) for name in closure)
    bound_hyp = tuple((name, repair_hypotheses[name]) for name in closure)

    requirements: dict[
        tuple[str, str | None, int], ShadowProbeRequirement
    ] = {}
    for _, fault_hypothesis in bound_hyp:
        if fault_hypothesis == reference_hypothesis:
            raise RepairTransactionRejected(
                "fault hypothesis cannot equal the reference hypothesis"
            )
        if fault_hypothesis not in diagnosis.hypothesis_names:
            raise RepairTransactionRejected(
                f"fault hypothesis {fault_hypothesis!r} is absent from diagnosis"
            )
        witness = _witness_for_pair(
            diagnosis,
            reference_hypothesis,
            fault_hypothesis,
        )
        if witness is None:
            raise RepairTransactionRejected(
                "no frozen diagnostic witness separates reference from "
                f"{fault_hypothesis!r}"
            )
        requirement = ShadowProbeRequirement(
            fault_hypothesis=fault_hypothesis,
            tap_after_factor=witness.tap_after_factor,
            basis_index=witness.basis_index,
            expected_reference=_expected_for_reference(
                witness,
                reference_hypothesis,
            ),
        )
        key = (
            requirement.fault_hypothesis,
            requirement.tap_after_factor,
            requirement.basis_index,
        )
        requirements[key] = requirement

    ordered_requirements = tuple(
        requirements[key]
        for key in sorted(
            requirements,
            key=lambda item: (
                item[0],
                item[1] or "~output",
                item[2],
            ),
        )
    )
    digest = _transaction_digest(
        subject=interaction.subject,
        reference_hypothesis=reference_hypothesis,
        requested_repairs=requested,
        atomic_closure=closure,
        implementations=bound_impl,
        repair_hypotheses=bound_hyp,
        interaction_digest=interaction.digest,
        diagnosis_digest=diagnosis.digest,
        requirements=ordered_requirements,
    )

    return SemanticRepairTransactionCertificate(
        subject=interaction.subject,
        reference_hypothesis=reference_hypothesis,
        requested_repairs=requested,
        atomic_closure=closure,
        implementations=bound_impl,
        repair_hypotheses=bound_hyp,
        interaction_digest=interaction.digest,
        diagnosis_digest=diagnosis.digest,
        probe_requirements=ordered_requirements,
        digest=digest,
    )


def _observation_digest(
    observations: Sequence[ShadowProbeObservation],
) -> str:
    payload = [
        {
            "fault_hypothesis": item.fault_hypothesis,
            "tap_after_factor": item.tap_after_factor,
            "basis_index": item.basis_index,
            "observed": list(item.observed),
            "evidence_id": item.evidence_id,
        }
        for item in observations
    ]
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def authorize_semantic_repair_commit(
    *,
    interaction: RepairInteractionCertificate,
    diagnosis: SemanticDiagnosisPlan,
    certificate: SemanticRepairTransactionCertificate,
    requested_implementations: Mapping[str, str],
    observations: Sequence[ShadowProbeObservation],
    atol: float = 0.0,
) -> SemanticRepairTransactionAuthorization:
    """Authorize commit only after exact closure and shadow probes agree."""

    if atol < 0 or not isfinite(atol):
        raise ValueError("atol must be finite and non-negative")
    if interaction.digest != certificate.interaction_digest:
        raise RepairTransactionEvidenceMismatch(
            "repair-interaction evidence changed after prepare"
        )
    if diagnosis.digest != certificate.diagnosis_digest:
        raise RepairTransactionEvidenceMismatch(
            "diagnosis evidence changed after prepare"
        )

    requested = set(requested_implementations)
    required = set(certificate.atomic_closure)
    if requested != required:
        missing = sorted(required - requested)
        extra = sorted(requested - required)
        raise RepairTransactionRejected(
            f"commit must match atomic closure; missing={missing}, extra={extra}"
        )

    for repair in certificate.atomic_closure:
        if requested_implementations[repair] != certificate.implementation_for(
            repair
        ):
            raise RepairTransactionEvidenceMismatch(
                f"implementation drift for repair {repair!r}"
            )

    recomputed = _transaction_digest(
        subject=certificate.subject,
        reference_hypothesis=certificate.reference_hypothesis,
        requested_repairs=certificate.requested_repairs,
        atomic_closure=certificate.atomic_closure,
        implementations=certificate.implementations,
        repair_hypotheses=certificate.repair_hypotheses,
        interaction_digest=certificate.interaction_digest,
        diagnosis_digest=certificate.diagnosis_digest,
        requirements=certificate.probe_requirements,
    )
    if recomputed != certificate.digest:
        raise RepairTransactionEvidenceMismatch(
            "transaction certificate was modified"
        )

    by_key: dict[
        tuple[str, str | None, int], ShadowProbeObservation
    ] = {}
    for item in observations:
        key = (
            item.fault_hypothesis,
            item.tap_after_factor,
            item.basis_index,
        )
        if key in by_key:
            raise RepairTransactionRejected(
                f"duplicate shadow observation for {key!r}"
            )
        by_key[key] = item

    required_keys = {
        (
            item.fault_hypothesis,
            item.tap_after_factor,
            item.basis_index,
        )
        for item in certificate.probe_requirements
    }
    if set(by_key) != required_keys:
        missing = sorted(required_keys - set(by_key), key=repr)
        extra = sorted(set(by_key) - required_keys, key=repr)
        raise RepairTransactionRejected(
            f"shadow evidence set mismatch; missing={missing}, extra={extra}"
        )

    for requirement in certificate.probe_requirements:
        key = (
            requirement.fault_hypothesis,
            requirement.tap_after_factor,
            requirement.basis_index,
        )
        observed = by_key[key].observed
        if len(observed) != len(requirement.expected_reference):
            raise RepairTransactionRejected(
                f"shadow observation dimension mismatch for {key!r}"
            )
        residual = max(
            abs(float(actual) - float(expected))
            for actual, expected in zip(
                observed,
                requirement.expected_reference,
                strict=True,
            )
        )
        if residual > atol:
            raise RepairTransactionRejected(
                "shadow semantic probe does not match reference semantics for "
                f"{requirement.fault_hypothesis!r}: residual={residual}"
            )

    ordered_observations = tuple(
        by_key[key]
        for key in sorted(
            by_key,
            key=lambda item: (
                item[0],
                item[1] or "~output",
                item[2],
            ),
        )
    )
    return SemanticRepairTransactionAuthorization(
        subject=certificate.subject,
        repairs=certificate.atomic_closure,
        implementations=certificate.implementations,
        interaction_digest=certificate.interaction_digest,
        diagnosis_digest=certificate.diagnosis_digest,
        transaction_digest=certificate.digest,
        observation_digest=_observation_digest(ordered_observations),
    )
