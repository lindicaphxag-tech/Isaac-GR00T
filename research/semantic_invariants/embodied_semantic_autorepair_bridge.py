"""Bridge active semantic diagnosis to bounded repair synthesis.

This module is deliberately conservative.  A semantic diagnosis does not grant
execution authority.  It converts a *singleton* hidden-semantic diagnosis into
per-boundary repair obligations that can be passed to the existing independent
verification/certificate pipeline.

Two rules are enforced:

1. a non-singleton diagnosis never triggers repair synthesis;
2. more than one non-identity boundary always requires a factorial interaction
   assay before any repair set can become deployment-eligible, because
   compensating semantic faults can make locally correct hotfixes regress the
   composed system.

The synthesis backend is the existing bounded repair DSL.  Unsupported or
ambiguous repairs remain explicit failure modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Sequence

from .embodied_repair_synthesis import (
    RepairExample,
    RepairProgram,
    build_vector_repair_catalog,
    synthesize_minimal_repair,
)
from .embodied_semantic_observability import SemanticDiagnosisHypothesis
from .embodied_semantic_transport import MonomialSemanticTransport


@dataclass(frozen=True)
class BoundaryRepairObligation:
    boundary: str
    evidence_id: str
    semantic_transport_digest: str
    status: str
    program_name: str | None
    program_implementation_ids: tuple[str, ...]
    example_count: int
    alternatives: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosisRepairPlan:
    diagnosis_status: str
    hypothesis_name: str | None
    obligations: tuple[BoundaryRepairObligation, ...]
    nonidentity_boundaries: tuple[str, ...]
    requires_interaction_assay: bool
    eligible_for_independent_verification: bool
    reasons: tuple[str, ...]
    digest: str


def _transport_digest(transport: MonomialSemanticTransport) -> str:
    payload = {
        "source_for_output": list(transport.source_for_output),
        "scale": list(transport.scale),
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _basis_examples(
    transport: MonomialSemanticTransport,
    *,
    boundary: str,
) -> tuple[RepairExample, ...]:
    examples: list[RepairExample] = []
    for index in range(transport.dimension):
        expected = tuple(
            1.0 if i == index else 0.0
            for i in range(transport.dimension)
        )
        observed = transport.apply(expected)
        examples.append(
            RepairExample(
                observed=observed,
                expected=expected,
                label=f"{boundary}/e{index}",
            )
        )
    return tuple(examples)


def _program_ids(program: RepairProgram | None) -> tuple[str, ...]:
    if program is None:
        return ()
    return tuple(operation.implementation_id for operation in program.operations)


def _plan_digest(
    *,
    diagnosis_status: str,
    hypothesis_name: str | None,
    obligations: Sequence[BoundaryRepairObligation],
    nonidentity_boundaries: Sequence[str],
    requires_interaction_assay: bool,
    eligible_for_independent_verification: bool,
    reasons: Sequence[str],
) -> str:
    payload = {
        "diagnosis_status": diagnosis_status,
        "hypothesis_name": hypothesis_name,
        "obligations": [
            {
                "boundary": item.boundary,
                "evidence_id": item.evidence_id,
                "semantic_transport_digest": item.semantic_transport_digest,
                "status": item.status,
                "program_name": item.program_name,
                "program_implementation_ids": list(
                    item.program_implementation_ids
                ),
                "example_count": item.example_count,
                "alternatives": list(item.alternatives),
            }
            for item in obligations
        ],
        "nonidentity_boundaries": list(nonidentity_boundaries),
        "requires_interaction_assay": requires_interaction_assay,
        "eligible_for_independent_verification": (
            eligible_for_independent_verification
        ),
        "reasons": list(reasons),
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def compile_diagnosis_to_repair_plan(
    candidate_hypotheses: Sequence[SemanticDiagnosisHypothesis],
    *,
    max_depth: int = 2,
    atol: float = 1.0e-8,
) -> DiagnosisRepairPlan:
    """Compile a singleton diagnosis into conservative repair obligations.

    The result is never an execution certificate.  A plan is merely eligible
    for independent verification when every non-identity boundary has a unique
    bounded repair and no multi-boundary interaction assay is still required.
    """

    candidates = tuple(candidate_hypotheses)
    if len(candidates) != 1:
        names = tuple(sorted(item.name for item in candidates))
        reasons = (
            "repair synthesis requires a singleton semantic diagnosis; "
            f"remaining={names!r}",
        )
        digest = _plan_digest(
            diagnosis_status="ambiguous",
            hypothesis_name=None,
            obligations=(),
            nonidentity_boundaries=(),
            requires_interaction_assay=False,
            eligible_for_independent_verification=False,
            reasons=reasons,
        )
        return DiagnosisRepairPlan(
            diagnosis_status="ambiguous",
            hypothesis_name=None,
            obligations=(),
            nonidentity_boundaries=(),
            requires_interaction_assay=False,
            eligible_for_independent_verification=False,
            reasons=reasons,
            digest=digest,
        )

    hypothesis = candidates[0]
    dimension = hypothesis.factors[0].transport.dimension
    catalog = build_vector_repair_catalog(dimension)

    obligations: list[BoundaryRepairObligation] = []
    nonidentity: list[str] = []
    reasons: list[str] = []

    for factor in hypothesis.factors:
        transport = factor.transport
        transport_id = _transport_digest(transport)
        if transport.is_identity(atol=atol):
            obligations.append(
                BoundaryRepairObligation(
                    boundary=factor.name,
                    evidence_id=factor.evidence_id,
                    semantic_transport_digest=transport_id,
                    status="identity",
                    program_name="identity",
                    program_implementation_ids=(),
                    example_count=0,
                    alternatives=("identity",),
                )
            )
            continue

        nonidentity.append(factor.name)
        examples = _basis_examples(transport, boundary=factor.name)
        result = synthesize_minimal_repair(
            examples,
            catalog,
            max_depth=max_depth,
            atol=atol,
        )
        alternatives = tuple(item.name for item in result.alternatives)
        if result.status == "unique" and result.program is not None:
            status = "unique"
            program_name = result.program.name
            implementation_ids = _program_ids(result.program)
        elif result.status == "ambiguous":
            status = "ambiguous"
            program_name = None
            implementation_ids = ()
            reasons.append(
                f"boundary {factor.name!r} has multiple minimum-cost repairs: "
                f"{alternatives!r}"
            )
        else:
            status = result.status
            program_name = None
            implementation_ids = ()
            reasons.append(
                f"boundary {factor.name!r} has no verified bounded repair "
                f"under max_depth={max_depth}"
            )

        obligations.append(
            BoundaryRepairObligation(
                boundary=factor.name,
                evidence_id=factor.evidence_id,
                semantic_transport_digest=transport_id,
                status=status,
                program_name=program_name,
                program_implementation_ids=implementation_ids,
                example_count=len(examples),
                alternatives=alternatives,
            )
        )

    requires_interaction = len(nonidentity) > 1
    if requires_interaction:
        reasons.append(
            "multiple non-identity semantic boundaries require a factorial "
            "interaction assay before any repair subset can be authorized"
        )

    unique_nonidentity = all(
        item.status == "unique"
        for item in obligations
        if item.boundary in nonidentity
    )
    eligible = bool(nonidentity) and unique_nonidentity and not requires_interaction

    if not nonidentity:
        reasons.append("diagnosed chain is already semantically identity")

    digest = _plan_digest(
        diagnosis_status="singleton",
        hypothesis_name=hypothesis.name,
        obligations=obligations,
        nonidentity_boundaries=nonidentity,
        requires_interaction_assay=requires_interaction,
        eligible_for_independent_verification=eligible,
        reasons=reasons,
    )
    return DiagnosisRepairPlan(
        diagnosis_status="singleton",
        hypothesis_name=hypothesis.name,
        obligations=tuple(obligations),
        nonidentity_boundaries=tuple(nonidentity),
        requires_interaction_assay=requires_interaction,
        eligible_for_independent_verification=eligible,
        reasons=tuple(reasons),
        digest=digest,
    )
