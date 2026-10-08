"""Fail-closed execution of a frozen active semantic diagnostic decision tree.

An offline optimal experiment *plan* is not authority to dispatch probes.
The consumer pins both the canonical problem digest and the entire tree digest,
validates each branch and its cumulative risk, and only then exposes the next
admissible probe. The observation trace must be independently authenticated
by the host execution/evidence plane; source IDs alone are not authentication.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Sequence

from .embodied_semantic_experiment_design import (
    ExperimentDecisionNode,
    SemanticExperiment,
    SemanticExperimentPlan,
)


class DiagnosticExecutionRejected(RuntimeError):
    """The plan, observed path or risk authority cannot be trusted."""


@dataclass(frozen=True)
class DiagnosticObservedStep:
    experiment_name: str
    observation_signature: tuple[int | float, ...]
    evidence_id: str


@dataclass(frozen=True)
class DiagnosticProbeAuthority:
    experiment: SemanticExperiment
    plan_digest: str
    tree_commitment: str
    step_index: int
    risk_consumed: float
    risk_after_probe: float
    remaining_risk_after_probe: float
    hypotheses: tuple[str, ...]
    # Assigned only by the trusted durable reservation store; never by the
    # offline planner. The physical adapter must use this as its idempotency key.
    reservation_token: str | None = None


@dataclass(frozen=True)
class DiagnosticResolution:
    hypotheses: tuple[str, ...]
    identified: bool
    risk_consumed: float
    observations_used: int


def _canonical_node(node: ExperimentDecisionNode) -> dict[str, object]:
    exp = node.experiment
    return {
        "hypotheses": list(node.hypotheses),
        "experiment": None if exp is None else {
            "name": exp.name,
            "probe": list(exp.probe),
            "tap_after_factor": exp.tap_after_factor,
            "cost": exp.cost,
            "risk": exp.risk,
        },
        "observation_classes": [
            {"signature": list(c.signature), "hypotheses": list(c.hypotheses)}
            for c in node.observation_classes
        ],
        "children": [_canonical_node(c) for c in node.children],
        "complete": node.complete,
        "expected_remaining_cost": node.expected_remaining_cost,
        "worst_case_remaining_cost": node.worst_case_remaining_cost,
    }


def diagnostic_tree_commitment(plan: SemanticExperimentPlan) -> str:
    """Digest the *whole plan*, not only the inputs used by its optimizer.

    This is an unkeyed commitment; deployers must pin the expected digest via
    a separate, trusted release/authorization channel before use.
    """
    payload = {
        "schema": "semrepair-diagnostic-execution-v1",
        "problem_digest": plan.digest,
        "hypothesis_names": list(plan.hypothesis_names),
        "admissible_experiments": list(plan.admissible_experiments),
        "objective": plan.objective,
        "max_risk": "unbounded" if plan.max_risk == float("inf") else plan.max_risk,
        "risk_weight": plan.risk_weight,
        "total_risk_budget": (
            "unbounded" if plan.total_risk_budget == float("inf")
            else plan.total_risk_budget
        ),
        "complete": plan.complete,
        "root": _canonical_node(plan.root),
    }
    return sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_tree(node: ExperimentDecisionNode, remaining: float) -> float:
    if not isinstance(node.hypotheses, tuple) or (
        not node.hypotheses or len(node.hypotheses) != len(set(node.hypotheses))
    ):
        raise DiagnosticExecutionRejected("non-canonical hypothesis set")
    if any(
        not isfinite(v) or v < 0 for v in (
            node.expected_remaining_cost, node.worst_case_remaining_cost
        )
    ):
        raise DiagnosticExecutionRejected("invalid node cost")
    if node.experiment is None:
        if node.children or node.observation_classes:
            raise DiagnosticExecutionRejected("terminal node has child branches")
        if node.complete != (len(node.hypotheses) == 1):
            raise DiagnosticExecutionRejected("terminal identification flag forged")
        return 0.0

    exp = node.experiment
    if (
        not isfinite(exp.risk) or exp.risk < 0 or exp.risk > remaining + 1e-12
        or not isfinite(exp.cost) or exp.cost < 0
    ):
        raise DiagnosticExecutionRejected("probe exceeds remaining risk authority")
    if len(node.children) < 2 or len(node.children) != len(node.observation_classes):
        raise DiagnosticExecutionRejected("probe does not partition the hypotheses")
    if not node.complete:
        raise DiagnosticExecutionRejected("partial diagnostic tree cannot dispatch")

    seen: set[str] = set()
    signatures: set[tuple[int | float, ...]] = set()
    maximum: float = 0.0
    for cl, child in zip(node.observation_classes, node.children, strict=True):
        signature = tuple(cl.signature)
        if (
            not signature
            or any(not isfinite(float(x)) for x in signature)
            or signature in signatures
        ):
            raise DiagnosticExecutionRejected("invalid or duplicate observation signature")
        signatures.add(signature)
        if (
            tuple(sorted(cl.hypotheses)) != cl.hypotheses
            or cl.hypotheses != child.hypotheses
            or seen.intersection(cl.hypotheses)
        ):
            raise DiagnosticExecutionRejected("observation branch does not match child")
        seen.update(cl.hypotheses)
        maximum = max(
            maximum, _validate_tree(child, max(0.0, remaining - exp.risk))
        )
    if seen != set(node.hypotheses):
        raise DiagnosticExecutionRejected("diagnosis tree omits possible hypotheses")
    return exp.risk + maximum


def authorize_next_diagnostic_probe(
    *,
    plan: SemanticExperimentPlan,
    trusted_problem_digest: str,
    trusted_tree_commitment: str,
    trace: Sequence[DiagnosticObservedStep] = (),
) -> DiagnosticProbeAuthority | DiagnosticResolution:
    """Replay the entire observed path; refuse any stale or impossible step.

    The caller must obtain the expected digests and authenticated observations
    from outside the candidate repair generator. Replaying a trace does not
    prevent a rollback or double-physical-dispatch by itself; the host must
    persist a monotonically advancing episode cursor across physical effects.
    """
    if (
        not trusted_problem_digest
        or trusted_problem_digest != plan.digest
        or not trusted_tree_commitment
        or trusted_tree_commitment != diagnostic_tree_commitment(plan)
    ):
        raise DiagnosticExecutionRejected("untrusted or changed diagnostic plan")
    if not plan.complete or not isfinite(plan.total_risk_budget):
        raise DiagnosticExecutionRejected(
            "physical diagnosis requires a complete finite-risk-budget plan"
        )
    if (
        plan.total_risk_budget < 0
        or tuple(sorted(plan.hypothesis_names)) != plan.root.hypotheses
    ):
        raise DiagnosticExecutionRejected("diagnostic root or budget was modified")
    maximum = _validate_tree(plan.root, plan.total_risk_budget)
    if maximum > plan.total_risk_budget + 1e-9:
        raise DiagnosticExecutionRejected("diagnostic tree is not globally risk-bounded")
    if abs(maximum - plan.max_path_risk) > 1e-9:
        raise DiagnosticExecutionRejected("maximum path risk does not match the plan")

    node = plan.root
    consumed = 0.0
    used_evidence: set[str] = set()
    for step in trace:
        if node.experiment is None:
            raise DiagnosticExecutionRejected("observation after diagnosis completed")
        if step.experiment_name != node.experiment.name:
            raise DiagnosticExecutionRejected("observation belongs to wrong probe")
        if (
            not step.evidence_id or step.evidence_id in used_evidence
            or not isinstance(step.observation_signature, tuple)
            or any(not isfinite(float(x)) for x in step.observation_signature)
        ):
            raise DiagnosticExecutionRejected("missing, reused or invalid observation evidence")
        used_evidence.add(step.evidence_id)
        matches = [
            idx for idx, cl in enumerate(node.observation_classes)
            if tuple(cl.signature) == step.observation_signature
        ]
        if len(matches) != 1:
            raise DiagnosticExecutionRejected("unexpected semantic observation")
        consumed += node.experiment.risk
        if consumed > plan.total_risk_budget + 1e-9:
            raise DiagnosticExecutionRejected("diagnostic risk budget exhausted")
        node = node.children[matches[0]]

    if node.experiment is None:
        return DiagnosticResolution(
            hypotheses=node.hypotheses, identified=len(node.hypotheses) == 1,
            risk_consumed=consumed, observations_used=len(trace),
        )
    after = consumed + node.experiment.risk
    if after > plan.total_risk_budget + 1e-9:
        raise DiagnosticExecutionRejected("next probe exceeds episode risk budget")
    return DiagnosticProbeAuthority(
        experiment=node.experiment,
        plan_digest=plan.digest,
        tree_commitment=trusted_tree_commitment,
        step_index=len(trace),
        risk_consumed=consumed,
        risk_after_probe=after,
        remaining_risk_after_probe=max(0.0, plan.total_risk_budget - after),
        hypotheses=node.hypotheses,
    )
