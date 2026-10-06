"""Exact safe experiment planning for hidden embodied interface semantics.

The finite solver treats a diagnostic experiment as a joint choice of:

* physical / software probe;
* semantic observation boundary (tap);
* claim-authorized evidence scope;
* review / instrumentation cost;
* physical-risk cost; and
* a frozen hypothesis-to-observation model.

Numeric predicted observations may carry an explicit L-infinity error bound.
The planner then reasons over the exact finite arrangement of the resulting
closed error boxes: every realizable compatibility set is an adversarial branch
that the adaptive diagnosis tree must handle. This means a production tolerance
changes identifiability itself rather than being hidden in a test-only allclose.

The solver is exact over the supplied finite hypothesis/experiment language and
bounded numeric observation model. It is not a claim about arbitrary nonlinear
system identification.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from math import isfinite
from typing import Mapping, Sequence


ScalarObservation = str | int | float | bool | None
Observation = ScalarObservation | tuple[float, ...]
MAX_ROBUST_REGION_POINTS = 50_000


def _fraction(value: float | int) -> Fraction:
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError("cost/risk values must be finite")
    return Fraction(str(numeric))


def _numeric_vector(value: Observation) -> tuple[float, ...] | None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError("numeric observations must be finite")
        return (numeric,)
    if isinstance(value, tuple):
        if not value:
            raise ValueError("numeric observation tuples must be non-empty")
        vector: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return None
            numeric = float(item)
            if not isfinite(numeric):
                raise ValueError("numeric observations must be finite")
            vector.append(numeric)
        return tuple(vector)
    return None


def _canonical_observation(value: Observation) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


@dataclass(frozen=True)
class SemanticHypothesis:
    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("hypothesis name must be non-empty")


@dataclass(frozen=True)
class JointSemanticExperiment:
    """One joint intervention/observation design."""

    name: str
    probe: str
    tap: str
    outcomes: Mapping[str, Observation]
    probe_cost: float = 1.0
    tap_cost: float = 0.0
    risk: float = 0.0
    evidence_for: tuple[str, ...] = ()
    observation_atol: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not self.probe or not self.tap:
            raise ValueError("experiment name/probe/tap must be non-empty")
        if self.probe_cost < 0 or self.tap_cost < 0:
            raise ValueError("experiment costs must be non-negative")
        if self.risk < 0:
            raise ValueError("experiment risk must be non-negative")
        for value in (
            self.probe_cost,
            self.tap_cost,
            self.risk,
            self.observation_atol,
        ):
            if not isfinite(float(value)):
                raise ValueError("experiment costs/risks/tolerance must be finite")
        if self.observation_atol < 0:
            raise ValueError("observation_atol must be non-negative")
        if len(set(self.evidence_for)) != len(self.evidence_for):
            raise ValueError("evidence_for entries must be unique")
        if any(not claim for claim in self.evidence_for):
            raise ValueError("evidence_for entries must be non-empty")

        vectors = [_numeric_vector(value) for value in self.outcomes.values()]
        numeric_count = sum(vector is not None for vector in vectors)
        if 0 < numeric_count < len(vectors):
            raise ValueError(
                "an experiment cannot mix numeric and symbolic outcomes"
            )
        if numeric_count:
            dimensions = {len(vector) for vector in vectors if vector is not None}
            if len(dimensions) != 1:
                raise ValueError(
                    "numeric outcomes in one experiment must share dimension"
                )
        elif self.observation_atol != 0:
            raise ValueError(
                "observation_atol is only valid for numeric outcomes"
            )

    @property
    def cost(self) -> float:
        return float(self.probe_cost + self.tap_cost)

    @property
    def numeric(self) -> bool:
        return bool(self.outcomes) and all(
            _numeric_vector(value) is not None
            for value in self.outcomes.values()
        )


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


def _outcomes_overlap(
    left: Observation,
    right: Observation,
    *,
    atol: float,
) -> bool:
    left_vector = _numeric_vector(left)
    right_vector = _numeric_vector(right)
    if left_vector is None or right_vector is None:
        return left == right
    if len(left_vector) != len(right_vector):
        return False
    return max(
        abs(a - b) for a, b in zip(left_vector, right_vector, strict=True)
    ) <= (2.0 * atol)


def compatible_hypotheses_for_observation(
    experiment: JointSemanticExperiment,
    observation: Observation,
    *,
    hypotheses: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return every hypothesis compatible with one runtime observation.

    A caller must not silently choose one element when this tuple has size 0 or
    greater than 1. Zero means the observation lies outside the declared model;
    multiple hypotheses means the observation is ambiguous under the declared
    tolerance.
    """

    names = tuple(hypotheses) if hypotheses is not None else tuple(
        experiment.outcomes
    )
    if experiment.numeric:
        observed = _numeric_vector(observation)
        if observed is None:
            return ()
        compatible: list[str] = []
        for name in names:
            expected = _numeric_vector(experiment.outcomes[name])
            assert expected is not None
            if len(expected) != len(observed):
                continue
            error = max(
                abs(a - b)
                for a, b in zip(expected, observed, strict=True)
            )
            if error <= experiment.observation_atol:
                compatible.append(name)
        return tuple(sorted(compatible))

    return tuple(
        sorted(
            name
            for name in names
            if experiment.outcomes[name] == observation
        )
    )


def _numeric_region_points(
    subset: frozenset[str],
    experiment: JointSemanticExperiment,
) -> tuple[tuple[float, ...], ...]:
    vectors = {
        name: _numeric_vector(experiment.outcomes[name])
        for name in subset
    }
    if any(vector is None for vector in vectors.values()):
        raise ValueError("numeric region requested for symbolic experiment")
    concrete = {
        name: vector for name, vector in vectors.items() if vector is not None
    }
    dimension = len(next(iter(concrete.values())))
    per_axis: list[tuple[float, ...]] = []
    for axis in range(dimension):
        boundaries = sorted(
            {
                value
                for vector in concrete.values()
                for value in (
                    vector[axis] - experiment.observation_atol,
                    vector[axis] + experiment.observation_atol,
                )
            }
        )
        representatives = set(boundaries)
        representatives.update(
            (left + right) / 2.0
            for left, right in zip(boundaries, boundaries[1:], strict=False)
            if left < right
        )
        per_axis.append(tuple(sorted(representatives)))

    count = 1
    for axis_points in per_axis:
        count *= len(axis_points)
    if count > MAX_ROBUST_REGION_POINTS:
        raise ValueError(
            "robust observation arrangement exceeds finite region cap: "
            f"{count} > {MAX_ROBUST_REGION_POINTS}"
        )
    return tuple(product(*per_axis))


def _possible_observation_branches(
    subset: frozenset[str],
    experiment: JointSemanticExperiment,
) -> tuple[tuple[str, frozenset[str]], ...]:
    """Enumerate every realizable posterior hypothesis subset.

    For symbolic outcomes this is ordinary exact grouping.

    For numeric outcomes with bounded L-infinity error, the compatibility set can
    change only at box boundaries. Endpoints plus one representative point from
    every open interval on every axis therefore induce every possible
    compatibility subset of the finite closed-box arrangement.
    """

    if not experiment.numeric:
        groups: dict[str, set[str]] = {}
        for name in subset:
            key = _canonical_observation(experiment.outcomes[name])
            groups.setdefault(key, set()).add(name)
        return tuple(
            sorted(
                (
                    (label, frozenset(group))
                    for label, group in groups.items()
                ),
                key=lambda item: item[0],
            )
        )

    unique: dict[tuple[str, ...], frozenset[str]] = {}
    for point in _numeric_region_points(subset, experiment):
        compatible = compatible_hypotheses_for_observation(
            experiment,
            point,
            hypotheses=tuple(sorted(subset)),
        )
        if not compatible:
            continue
        child = frozenset(compatible)
        unique[tuple(sorted(child))] = child

    return tuple(
        (
            "compatible[" + ",".join(key) + "]",
            unique[key],
        )
        for key in sorted(unique)
    )


def observational_equivalence_classes(
    hypotheses: Sequence[SemanticHypothesis],
    experiments: Sequence[JointSemanticExperiment],
) -> tuple[tuple[str, ...], ...]:
    """Return conservative pairwise-indistinguishability components.

    Exact symbolic observations recover ordinary equivalence classes. Under
    bounded numeric uncertainty, pairwise overlap need not be transitive; this
    function therefore returns connected components only as a human-readable
    conservative summary. The actual planner does not use this summary to
    decide identifiability. It reasons over the exact finite observation
    arrangement.
    """

    names, items = _validate_inputs(hypotheses, experiments)
    adjacency = {name: {name} for name in names}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            indistinguishable = all(
                _outcomes_overlap(
                    item.outcomes[left],
                    item.outcomes[right],
                    atol=item.observation_atol,
                )
                for item in items
            )
            if indistinguishable:
                adjacency[left].add(right)
                adjacency[right].add(left)

    unseen = set(names)
    groups: list[tuple[str, ...]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        unseen -= component
        groups.append(tuple(sorted(component)))
    return tuple(sorted(groups))


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
                "observation_atol": item.observation_atol,
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


def _make_identifiability_checker(
    experiments: tuple[JointSemanticExperiment, ...],
):
    @lru_cache(maxsize=None)
    def identifiable(subset_tuple: tuple[str, ...]) -> bool:
        subset = frozenset(subset_tuple)
        if len(subset) <= 1:
            return True

        for experiment in experiments:
            branches = _possible_observation_branches(subset, experiment)
            if len(branches) <= 1:
                continue
            child_sets = tuple(child for _, child in branches)
            if any(child == subset for child in child_sets):
                continue
            if all(
                identifiable(tuple(sorted(child)))
                for child in child_sets
            ):
                return True
        return False

    return identifiable


def synthesize_safe_semantic_experiment_plan(
    hypotheses: Sequence[SemanticHypothesis],
    experiments: Sequence[JointSemanticExperiment],
    *,
    max_total_risk: float = 0.0,
    required_evidence: str | None = None,
) -> SafeSemanticExperimentPlan:
    """Synthesize an exact adaptive diagnosis plan under cumulative risk."""

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

    intrinsic = observational_equivalence_classes(hypotheses, all_items)
    authorized_equivalence = (
        observational_equivalence_classes(hypotheses, items)
        if items
        else (tuple(sorted(names)),)
    )
    can_identify_intrinsically = _make_identifiability_checker(all_items)
    can_identify_with_claim = _make_identifiability_checker(items)

    experiment_cost = {
        item.name: _fraction(item.probe_cost) + _fraction(item.tap_cost)
        for item in items
    }
    experiment_risk = {
        item.name: _fraction(item.risk) for item in items
    }

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

        candidates: list[
            tuple[tuple[Fraction, Fraction, int, str], _Solved]
        ] = []
        for experiment in items:
            erisk = experiment_risk[experiment.name]
            if erisk > remaining:
                continue
            branches = _possible_observation_branches(subset, experiment)
            if len(branches) <= 1:
                continue
            child_sets = tuple(child for _, child in branches)
            if any(child == subset for child in child_sets):
                continue

            children: list[tuple[str, _Solved]] = []
            feasible = True
            for label, child_subset in branches:
                child = solve(
                    tuple(sorted(child_subset)),
                    remaining - erisk,
                )
                if child.status != "identified":
                    feasible = False
                    break
                children.append((label, child))

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
                    DiagnosisBranch(outcome=label, child=child.node)
                    for label, child in sorted(children, key=lambda x: x[0])
                ),
            )
            solved = _Solved(
                status="identified",
                node=node,
                cost=total_cost,
                risk=total_risk,
                depth=total_depth,
            )
            key = (
                total_cost,
                total_risk,
                total_depth,
                experiment.name,
            )
            candidates.append((key, solved))

        if candidates:
            return min(candidates, key=lambda pair: pair[0])[1]

        sorted_subset = tuple(sorted(subset))
        if not can_identify_intrinsically(sorted_subset):
            reason = "intrinsic_observational_equivalence"
            status = "unidentifiable"
        elif required_evidence is not None and not can_identify_with_claim(
            sorted_subset
        ):
            reason = "claim_evidence_scope_insufficient"
            status = "evidence_scope_limited"
        else:
            reason = "risk_budget_insufficient"
            status = "budget_limited"

        return _Solved(
            status=status,
            node=DiagnosisLeaf(sorted_subset, reason),
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
        return f"{indent}LEAF[{node.reason}]: " + ", ".join(node.hypotheses)

    lines = [
        f"{indent}EXPERIMENT {node.experiment}: "
        f"probe={node.probe} tap={node.tap}"
    ]
    for branch in node.branches:
        lines.append(f"{indent}  if {branch.outcome!r}:")
        lines.append(render_plan(branch.child, indent=indent + "    "))
    return "\n".join(lines)
