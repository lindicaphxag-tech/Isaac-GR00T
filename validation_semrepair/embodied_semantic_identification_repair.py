"""Bind adaptive semantic identification to factor-wise repair authority.

A net end-to-end transport is not sufficient for repair authorization: two
wrong semantic transports may cancel exactly.  Therefore a diagnosis leaf is
bound to the *factorized* hidden-semantic hypothesis that was identified by the
experiment-design policy.

Only a uniquely identified hypothesis may authorize a repair bundle.  Every
non-identity factor receives its own inverse transport and the bundle digest
binds the diagnosis policy digest, factor order, factor evidence identities,
and concrete inverse implementations.

Unexpected observations, ambiguous diagnosis, or missing hypotheses fail
closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Mapping, Sequence

from .embodied_semantic_experiment_design import (
    DiagnosisDecision,
    DiagnosisLeaf,
    Observation,
    OptimalDiagnosisResult,
    SemanticExperiment,
    _observation_compatible,
)
from .embodied_semantic_observability import SemanticDiagnosisHypothesis
from .embodied_semantic_transport import MonomialSemanticTransport


@dataclass(frozen=True)
class DiagnosisExecutionStep:
    experiment: str
    observed: Observation
    reachable_hypotheses: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosisExecution:
    status: str
    identified_hypothesis: str | None
    policy_digest: str
    steps: tuple[DiagnosisExecutionStep, ...]
    reason: str


@dataclass(frozen=True)
class FactorRepair:
    factor_name: str
    source_evidence_id: str
    repair: MonomialSemanticTransport


@dataclass(frozen=True)
class SemanticRepairBundleAuthorization:
    status: str
    identified_hypothesis: str
    diagnosis_digest: str
    repairs: tuple[FactorRepair, ...]
    digest: str

    @property
    def repair_count(self) -> int:
        return len(self.repairs)


def execute_diagnosis_policy(
    result: OptimalDiagnosisResult,
    experiments: Sequence[SemanticExperiment],
    *,
    observe: Callable[[SemanticExperiment], Observation],
) -> DiagnosisExecution:
    """Execute a frozen diagnostic policy against one unknown module chain."""

    if not result.complete or result.policy is None:
        return DiagnosisExecution(
            status="refused",
            identified_hypothesis=None,
            policy_digest=result.digest,
            steps=(),
            reason="diagnostic policy is not complete",
        )

    experiment_by_name = {item.name: item for item in experiments}
    policy = result.policy
    steps: list[DiagnosisExecutionStep] = []

    while isinstance(policy, DiagnosisDecision):
        if policy.experiment not in experiment_by_name:
            return DiagnosisExecution(
                status="refused",
                identified_hypothesis=None,
                policy_digest=result.digest,
                steps=tuple(steps),
                reason="policy references missing experiment",
            )
        experiment = experiment_by_name[policy.experiment]
        observed = observe(experiment)
        compatible_branches = []
        for item in policy.branches:
            child_hypotheses = item.child.hypotheses
            if any(
                _observation_compatible(
                    experiment.outcome_for(hypothesis),
                    observed,
                    atol=experiment.observation_atol,
                )
                for hypothesis in child_hypotheses
            ):
                compatible_branches.append(item)

        if len(compatible_branches) != 1:
            return DiagnosisExecution(
                status="refused",
                identified_hypothesis=None,
                policy_digest=result.digest,
                steps=tuple(steps),
                reason=(
                    "observation is outside the declared error bound"
                    if not compatible_branches
                    else "observation is ambiguous under the declared error bound"
                ),
            )
        branch = compatible_branches[0]
        reachable = branch.child.hypotheses
        steps.append(
            DiagnosisExecutionStep(
                experiment=experiment.name,
                observed=observed,
                reachable_hypotheses=reachable,
            )
        )
        policy = branch.child

    if not isinstance(policy, DiagnosisLeaf) or len(policy.hypotheses) != 1:
        return DiagnosisExecution(
            status="refused",
            identified_hypothesis=None,
            policy_digest=result.digest,
            steps=tuple(steps),
            reason="policy terminated without unique identification",
        )

    return DiagnosisExecution(
        status="identified",
        identified_hypothesis=policy.hypotheses[0],
        policy_digest=result.digest,
        steps=tuple(steps),
        reason="unique semantic hypothesis identified",
    )


def _bundle_digest(
    *,
    hypothesis: SemanticDiagnosisHypothesis,
    diagnosis_digest: str,
    repairs: Sequence[FactorRepair],
) -> str:
    payload = {
        "hypothesis": hypothesis.name,
        "diagnosis_digest": diagnosis_digest,
        "repairs": [
            {
                "factor_name": item.factor_name,
                "source_evidence_id": item.source_evidence_id,
                "source_for_output": list(
                    item.repair.source_for_output
                ),
                "scale": list(item.repair.scale),
            }
            for item in repairs
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


def authorize_factorwise_inverse_bundle(
    execution: DiagnosisExecution,
    hypotheses: Sequence[SemanticDiagnosisHypothesis],
) -> SemanticRepairBundleAuthorization | None:
    """Authorize only the factor-wise inverse bundle for a unique diagnosis."""

    if (
        execution.status != "identified"
        or execution.identified_hypothesis is None
    ):
        return None

    by_name = {item.name: item for item in hypotheses}
    if len(by_name) != len(tuple(hypotheses)):
        raise ValueError("semantic hypothesis names must be unique")
    if execution.identified_hypothesis not in by_name:
        return None

    hypothesis = by_name[execution.identified_hypothesis]
    repairs: list[FactorRepair] = []
    for factor in hypothesis.factors:
        if factor.transport.is_identity():
            continue
        inverse = factor.transport.inverse()
        if not factor.transport.then(inverse).is_identity(atol=1.0e-12):
            raise ValueError(
                f"factor {factor.name!r} inverse failed identity check"
            )
        repairs.append(
            FactorRepair(
                factor_name=factor.name,
                source_evidence_id=factor.evidence_id,
                repair=inverse,
            )
        )

    repairs_tuple = tuple(repairs)
    digest = _bundle_digest(
        hypothesis=hypothesis,
        diagnosis_digest=execution.policy_digest,
        repairs=repairs_tuple,
    )
    return SemanticRepairBundleAuthorization(
        status=(
            "authorized"
            if repairs_tuple
            else "no_repair_needed"
        ),
        identified_hypothesis=hypothesis.name,
        diagnosis_digest=execution.policy_digest,
        repairs=repairs_tuple,
        digest=digest,
    )


def verify_factorwise_bundle(
    authorization: SemanticRepairBundleAuthorization,
    hypothesis: SemanticDiagnosisHypothesis,
) -> bool:
    """Check that the bundle exactly repairs every non-identity local factor."""

    if authorization.identified_hypothesis != hypothesis.name:
        return False
    expected = {
        factor.name: factor
        for factor in hypothesis.factors
        if not factor.transport.is_identity()
    }
    actual = {item.factor_name: item for item in authorization.repairs}
    if set(expected) != set(actual):
        return False

    for factor_name, factor in expected.items():
        item = actual[factor_name]
        if item.source_evidence_id != factor.evidence_id:
            return False
        if not factor.transport.then(item.repair).is_identity(atol=1.0e-12):
            return False

    expected_digest = _bundle_digest(
        hypothesis=hypothesis,
        diagnosis_digest=authorization.diagnosis_digest,
        repairs=authorization.repairs,
    )
    return expected_digest == authorization.digest
