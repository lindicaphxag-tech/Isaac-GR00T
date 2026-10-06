"""Exact safe experiment planning for hidden embodied interface semantics.

This module solves a finite, deliberately scoped problem:

* a frozen set of semantic hypotheses is given;
* each admissible experiment jointly chooses an input probe and an observation
  boundary (tap);
* experiments have explicit review cost and physical-risk budget;
* the planner either constructs an adaptive diagnosis tree with minimum
  worst-case cost under the cumulative risk budget, or returns a conservative
  impossibility/budget certificate.

The core distinction is between intrinsic observational equivalence, where no
provided experiment can separate two hypotheses even without a risk budget,
and budget-limited ambiguity, where the hypothesis set is separable in
principle but not under the allowed cumulative physical-risk budget.

The solver is exact over the frozen finite hypothesis/experiment set. It is
not a claim of optimal experiment design for arbitrary nonlinear dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping, Sequence


def _fraction(value: float | int) -> Fraction:
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError("cost/risk values must be finite")
    return Fraction(str(numeric))


@dataclass(frozen=True)
class SemanticHypothesis:
    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("hypothesis name must be non-empty")


@dataclass(frozen=True)
class JointSemanticExperiment:
    """One joint intervention/observation design.

    probe names the externally applied intervention or diagnostic input.
    tap names where the semantic observation is read. The same probe at two
    different taps is therefore represented by two different experiments.

    outcomes is frozen model evidence: each hypothesis must predict one
    symbolic observation for this experiment.
    """

    name: str
    probe: str
    tap: str
    outcomes: Mapping[str, str]
    probe_cost: float = 1.0
    tap_cost: float = 0.0
    risk: float = 0.0
    evidence_for: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.probe or not self.tap:
            raise ValueError("experiment name/probe/tap must be non-empty")
        if self.probe_cost < 0 or self.tap_cost < 0:
            raise ValueError("experiment costs must be non-negative")
        if self.risk < 0:
            raise ValueError("experiment risk must be non-negative")
        for value in (self.probe_cost, self.tap_cost, self.risk):
            if not isfinite(float(value)):
                raise ValueError("experiment costs/risks must be finite")
        if len(set(self.evidence_for)) != len(self.evidence_for):
            raise ValueError("evidence_for entries must be unique")
        if any(not claim for claim in self.evidence_for):
            raise ValueError("evidence_for entries must be non-empty")

    @property
    def cost(self) -> float:
        return float(self.probe_cost + self.tap_cost)


@dataclass(frozen=True)
class DiagnosisLeaf:
    hypotheses: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class DiagnosisBranch:
    outcome: str
    child: "DiagnosisNode"


@dataclass(frozen=True)
class DiagnosisDecision:
    experiment: str
    probe: str
    tap: str
    branches: tuple[DiagnosisBranch, ...]


DiagnosisNode = DiagnosisLeaf | DiagnosisDecision


@dataclass(frozen=True)
class PlanMetrics:
    worst_case_cost: float
    worst_case_risk: float
    worst_case_depth: int


@dataclass(frozen=True)
class SafeSemanticExperimentPlan:
    status: str
    hypotheses: tuple[str, ...]
    max_total_risk: float
    required_evidence: str | None
    authorized_experiments: tuple[str, ...]
    root: DiagnosisNode
    metrics: PlanMetrics
    intrinsic_equivalence_classes: tuple[tuple[str, ...], ...]
    authorized_equivalence_classes: tuple[tuple[str, ...], ...]
    digest: str

    @property
    def complete(self) -> bool:
        return self.status == "identified"


def _validate_inputs(
    hypotheses: Sequence[SemanticHypothesis],
    experiments: Sequence[JointSemanticExperiment],
) -> tuple[tuple[str, ...], tuple[JointSemanticExperiment, ...]]:
    names = tuple(item.name for item in hypotheses)
    if not names:
        raise ValueError("at least one hypothesis is required")
    if len(set(names)) != len(names):
        raise ValueError("hypothesis names must be unique")

    items = tuple(experiments)
    if len({item.name for item in items}) != len(items):
        raise ValueError("experiment names must be unique")

    expected = set(names)
    for experiment in items:
        observed = set(experiment.outcomes)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"experiment {experiment.name!r} outcome domain mismatch: "
                f"missing={missing}, extra={extra}"
            )
    return names, items


def observational_equivalence_classes(
    hypotheses: Sequence[SemanticHypothesis],
    experiments: Sequence[JointSemanticExperiment],
) -> tuple[tuple[str, ...], ...]:
    """Return exact observational-equivalence classes over all experiments."""

    names, items = _validate_inputs(hypotheses, experiments)
    signatures: dict[tuple[str, ...], list[str]] = {}
    for name in names:
        signature = tuple(item.outcomes[name] for item in items)
        signatures.setdefault(signature, []).append(name)
    return tuple(
        sorted(
            (tuple(sorted(group)) for group in signatures.values()),
            key=lambda group: group,
        )
    )


def _partition(
    subset: frozenset[str],
    experiment: JointSemanticExperiment,
) -> tuple[tuple[str, frozenset[str]], ...]:
    groups: dict[str, set[str]] = {}
    for name in subset:
        groups.setdefault(experiment.outcomes[name], set()).add(name)
    return tuple(
        sorted(
            ((outcome, frozenset(group)) for outcome, group in groups.items()),
            key=lambda item: item[0],
        )
    )


def _node_payload(node: DiagnosisNode) -> dict:
    if isinstance(node, DiagnosisLeaf):
        return {
            "kind": "leaf",
            "hypotheses": list(node.hypotheses),
            "reason": node.reason,
        }
    return {
        "kind": "decision",
        "experiment": node.experiment,
        "probe": node.probe,
        "tap": node.tap,
        "branches": [
            {"outcome": branch.outcome, "child": _node_payload(branch.child)}
            for branch in node.branches
        ],
    }


def _plan_digest(
    *,
    names: tuple[str, ...],
    experiments: tuple[JointSemanticExperiment, ...],
    max_total_risk: Fraction,
    required_evidence: str | None,
    authorized_experiments: tuple[str, ...],
    root: DiagnosisNode,
    intrinsic: tuple[tuple[str, ...], ...],
    authorized_equivalence: tuple[tuple[str, ...], ...],
) -> str:
    payload = {
        "hypotheses": list(names),
        "max_total_risk": str(max_total_risk),
        "required_evidence": required_evidence,
        "authorized_experiments": list(authorized_experiments),
        "experiments": [
            {
                "name": item.name,
                "probe": item.probe,
                "tap": item.tap,
                "probe_cost": item.probe_cost,
                "tap_cost": item.tap_cost,
                "risk": item.risk,
                "evidence_for": sorted(item.evidence_for),
                "outcomes": dict(sorted(item.outcomes.items())),
            }
            for item in sorted(experiments, key=lambda x: x.name)
        ],
        "intrinsic_equivalence_classes": [list(group) for group in intrinsic],
        "authorized_equivalence_classes": [
            list(group) for group in authorized_equivalence
        ],
        "tree": _node_payload(root),
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class _Solved:
    status: str
    node: DiagnosisNode
    cost: Fraction
    risk: Fraction
    depth: int


def synthesize_safe_semantic_experiment_plan(
    hypotheses: Sequence[SemanticHypothesis],
    experiments: Sequence[JointSemanticExperiment],
    *,
    max_total_risk: float = 0.0,
    required_evidence: str | None = None,
) -> SafeSemanticExperimentPlan:
    """Synthesize an exact adaptive diagnosis plan under cumulative risk.

    Objective, lexicographically:

    1. require complete singleton identification when feasible;
    2. minimize worst-case cumulative review/instrumentation cost;
    3. minimize worst-case cumulative physical risk;
    4. minimize worst-case number of experiments;
    5. deterministic experiment-name tie-break.

    When required_evidence is set, only experiments explicitly authorized for
    that claim may appear in the diagnosis tree. An informative but
    claim-inadmissible metric therefore cannot authorize a semantic conclusion.

    If complete identification is impossible, the returned root is a
    conservative ambiguity leaf. The status distinguishes intrinsic
    observational equivalence, claim-evidence-scope insufficiency, and
    ambiguity caused only by the risk budget.
    """

    names, all_items = _validate_inputs(hypotheses, experiments)
    if required_evidence is not None and not required_evidence:
        raise ValueError("required_evidence must be non-empty when provided")

    items = tuple(
        item
        for item in all_items
        if required_evidence is None or required_evidence in item.evidence_for
    )
    authorized_names = tuple(sorted(item.name for item in items))
    risk_budget = _fraction(max_total_risk)
    if risk_budget < 0:
        raise ValueError("max_total_risk must be non-negative")

    # Intrinsic equivalence is computed against the entire supplied experiment
    # language. Claim-relative equivalence is computed only over experiments
    # authorized to support the requested evidence claim.
    intrinsic = observational_equivalence_classes(hypotheses, all_items)
    authorized_equivalence = (
        observational_equivalence_classes(hypotheses, items)
        if items
        else (tuple(sorted(names)),)
    )
    intrinsically_ambiguous = {
        name
        for group in intrinsic
        if len(group) > 1
        for name in group
    }
    claim_ambiguous = {
        name
        for group in authorized_equivalence
        if len(group) > 1
        for name in group
    }

    experiment_cost = {
        item.name: _fraction(item.probe_cost) + _fraction(item.tap_cost)
        for item in items
    }
    experiment_risk = {item.name: _fraction(item.risk) for item in items}

    @lru_cache(maxsize=None)
    def solve(subset_tuple: tuple[str, ...], remaining: Fraction) -> _Solved:
        subset = frozenset(subset_tuple)
        if len(subset) == 1:
            only = next(iter(subset))
            return _Solved(
                status="identified",
                node=DiagnosisLeaf((only,), "singleton"),
                cost=Fraction(0),
                risk=Fraction(0),
                depth=0,
            )

        candidates: list[tuple[tuple[Fraction, Fraction, int, str], _Solved]] = []
        for experiment in items:
            erisk = experiment_risk[experiment.name]
            if erisk > remaining:
                continue
            partition = _partition(subset, experiment)
            if len(partition) <= 1:
                continue

            children: list[tuple[str, _Solved]] = []
            feasible = True
            for outcome, child_subset in partition:
                child = solve(tuple(sorted(child_subset)), remaining - erisk)
                if child.status != "identified":
                    feasible = False
                    break
                children.append((outcome, child))

            if not feasible:
                continue

            worst_child_cost = max(child.cost for _, child in children)
            worst_child_risk = max(child.risk for _, child in children)
            worst_child_depth = max(child.depth for _, child in children)
            total_cost = experiment_cost[experiment.name] + worst_child_cost
            total_risk = erisk + worst_child_risk
            total_depth = 1 + worst_child_depth

            node = DiagnosisDecision(
                experiment=experiment.name,
                probe=experiment.probe,
                tap=experiment.tap,
                branches=tuple(
                    DiagnosisBranch(outcome=outcome, child=child.node)
                    for outcome, child in sorted(children, key=lambda x: x[0])
                ),
            )
            solved = _Solved(
                status="identified",
                node=node,
                cost=total_cost,
                risk=total_risk,
                depth=total_depth,
            )
            key = (total_cost, total_risk, total_depth, experiment.name)
            candidates.append((key, solved))

        if candidates:
            return min(candidates, key=lambda pair: pair[0])[1]

        has_intrinsic_pair = any(
            len(set(group) & subset) > 1 for group in intrinsic
        )
        has_claim_scope_pair = any(
            len(set(group) & subset) > 1 for group in authorized_equivalence
        )
        if subset <= intrinsically_ambiguous or has_intrinsic_pair:
            reason = "intrinsic_observational_equivalence"
            status = "unidentifiable"
        elif required_evidence is not None and (
            subset <= claim_ambiguous or has_claim_scope_pair
        ):
            reason = "claim_evidence_scope_insufficient"
            status = "evidence_scope_limited"
        else:
            reason = "risk_budget_insufficient"
            status = "budget_limited"

        return _Solved(
            status=status,
            node=DiagnosisLeaf(tuple(sorted(subset)), reason),
            cost=Fraction(0),
            risk=Fraction(0),
            depth=0,
        )

    solved = solve(tuple(sorted(names)), risk_budget)
    digest = _plan_digest(
        names=names,
        experiments=all_items,
        max_total_risk=risk_budget,
        required_evidence=required_evidence,
        authorized_experiments=authorized_names,
        root=solved.node,
        intrinsic=intrinsic,
        authorized_equivalence=authorized_equivalence,
    )
    return SafeSemanticExperimentPlan(
        status=solved.status,
        hypotheses=names,
        max_total_risk=float(risk_budget),
        required_evidence=required_evidence,
        authorized_experiments=authorized_names,
        root=solved.node,
        metrics=PlanMetrics(
            worst_case_cost=float(solved.cost),
            worst_case_risk=float(solved.risk),
            worst_case_depth=solved.depth,
        ),
        intrinsic_equivalence_classes=intrinsic,
        authorized_equivalence_classes=authorized_equivalence,
        digest=digest,
    )


def collect_plan_experiments(node: DiagnosisNode) -> tuple[str, ...]:
    """Return all experiment names appearing anywhere in a plan tree."""

    if isinstance(node, DiagnosisLeaf):
        return ()
    names = {node.experiment}
    for branch in node.branches:
        names.update(collect_plan_experiments(branch.child))
    return tuple(sorted(names))


def render_plan(node: DiagnosisNode, *, indent: str = "") -> str:
    """Human-readable deterministic rendering for review artifacts."""

    if isinstance(node, DiagnosisLeaf):
        return (
            f"{indent}LEAF[{node.reason}]: "
            + ", ".join(node.hypotheses)
        )

    lines = [
        f"{indent}EXPERIMENT {node.experiment}: "
        f"probe={node.probe} tap={node.tap}"
    ]
    for branch in node.branches:
        lines.append(f"{indent}  if {branch.outcome!r}:")
        lines.append(render_plan(branch.child, indent=indent + "    "))
    return "\n".join(lines)
