"""Exact adaptive experiment design for hidden embodied semantics.

This module sits above the existing semantic-inference and observability layers.

The existing active inference routine greedily chooses the next probe by
information gain. The observability layer can choose a minimum-cost static set
of internal taps for a frozen hypothesis set. Those solve different subproblems.

Here we solve the finite deterministic joint problem exactly: which probe should
be applied, which boundary should be observed, and what should be tested next
after each possible observation?

The result is an adaptive diagnostic decision tree with an executable
optimality certificate over the frozen candidate experiment table. The solver
fails closed when surviving semantic hypotheses are observationally equivalent
under every admissible experiment.

This does not claim novelty for generic optimal decision trees or active fault
diagnosis. The intended use is embodied semantic ABI identification.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
import json
from typing import Mapping, Sequence


ScalarObservation = str | int | float | bool | None
Observation = ScalarObservation | tuple["Observation", ...]


def _canonical_observation(value: Observation) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


@dataclass(frozen=True)
class SemanticExperiment:
    """One frozen diagnostic experiment over a finite hypothesis set."""

    name: str
    outcomes: tuple[tuple[str, Observation], ...]
    cost: float = 1.0
    risk: float = 0.0
    probe: str | None = None
    tap_after_factor: str | None = None
    observation_atol: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name must be non-empty")
        if (not isfinite(self.cost)) or self.cost <= 0:
            raise ValueError("experiment cost must be finite and positive")
        if (not isfinite(self.risk)) or self.risk < 0:
            raise ValueError("experiment risk must be finite and non-negative")
        if (not isfinite(self.observation_atol)) or self.observation_atol < 0:
            raise ValueError(
                "observation_atol must be finite and non-negative"
            )
        names = tuple(name for name, _ in self.outcomes)
        if len(names) != len(set(names)):
            raise ValueError("experiment outcomes contain duplicate hypotheses")

    def outcome_for(self, hypothesis: str) -> Observation:
        for name, value in self.outcomes:
            if name == hypothesis:
                return value
        raise KeyError(
            f"experiment {self.name!r} has no outcome for hypothesis "
            f"{hypothesis!r}"
        )

    @property
    def outcome_map(self) -> dict[str, Observation]:
        return dict(self.outcomes)


@dataclass(frozen=True)
class DiagnosisLeaf:
    hypotheses: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosisBranch:
    outcome: Observation
    child: "DiagnosisPolicy"


@dataclass(frozen=True)
class DiagnosisDecision:
    hypotheses: tuple[str, ...]
    experiment: str
    branches: tuple[DiagnosisBranch, ...]


DiagnosisPolicy = DiagnosisLeaf | DiagnosisDecision


@dataclass(frozen=True)
class OptimalDiagnosisResult:
    status: str
    objective: str
    optimal_cost: float | None
    hypothesis_names: tuple[str, ...]
    experiment_names: tuple[str, ...]
    policy: DiagnosisPolicy | None
    unresolved_groups: tuple[tuple[str, ...], ...]
    bellman_states: int
    max_risk: float
    risk_weight: float
    digest: str

    @property
    def complete(self) -> bool:
        return self.status == "optimal" and self.policy is not None


@dataclass(frozen=True)
class DiagnosisVerification:
    valid: bool
    policy_cost: float | None
    independently_optimal_cost: float | None
    digest_matches: bool
    message: str


def _effective_cost(experiment: SemanticExperiment, risk_weight: float) -> float:
    return experiment.cost + risk_weight * experiment.risk


def _numeric_observation(value: Observation) -> tuple[float, ...] | None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return None
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, tuple):
        flattened: list[float] = []
        for item in value:
            numeric = _numeric_observation(item)
            if numeric is None:
                return None
            flattened.extend(numeric)
        return tuple(flattened)
    return None


def _observation_distance(left: Observation, right: Observation) -> float:
    lhs = _numeric_observation(left)
    rhs = _numeric_observation(right)
    if lhs is None or rhs is None or len(lhs) != len(rhs):
        return (
            0.0
            if _canonical_observation(left) == _canonical_observation(right)
            else float("inf")
        )
    return max(abs(a - b) for a, b in zip(lhs, rhs, strict=True))


def _observation_compatible(
    expected: Observation,
    observed: Observation,
    *,
    atol: float,
) -> bool:
    return _observation_distance(expected, observed) <= atol


def _partition(
    hypotheses: frozenset[str],
    experiment: SemanticExperiment,
) -> tuple[tuple[Observation, frozenset[str]], ...]:
    """Conservative robust partition under bounded observation error.

    Each predicted observation denotes an L-infinity ball of radius
    observation_atol. Hypotheses whose balls overlap cannot be assigned to
    different branches without risking an ambiguous runtime observation.
    Connected overlap components are therefore merged conservatively.
    """

    ordered = tuple(sorted(hypotheses))
    if experiment.observation_atol == 0.0:
        groups: dict[str, tuple[Observation, set[str]]] = {}
        for hypothesis in ordered:
            outcome = experiment.outcome_for(hypothesis)
            key = _canonical_observation(outcome)
            if key not in groups:
                groups[key] = (outcome, set())
            groups[key][1].add(hypothesis)
        return tuple(
            (groups[key][0], frozenset(groups[key][1]))
            for key in sorted(groups)
        )

    remaining = set(ordered)
    components: list[tuple[Observation, frozenset[str]]] = []
    diameter = 2.0 * experiment.observation_atol
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component = {seed}
        remaining.remove(seed)
        while stack:
            current = stack.pop()
            current_outcome = experiment.outcome_for(current)
            neighbors = [
                other
                for other in sorted(remaining)
                if _observation_distance(
                    current_outcome,
                    experiment.outcome_for(other),
                )
                <= diameter
            ]
            for other in neighbors:
                remaining.remove(other)
                component.add(other)
                stack.append(other)
        representative = min(component)
        components.append(
            (
                experiment.outcome_for(representative),
                frozenset(component),
            )
        )

    return tuple(
        sorted(
            components,
            key=lambda item: tuple(sorted(item[1])),
        )
    )


def _policy_payload(policy: DiagnosisPolicy | None) -> object:
    if policy is None:
        return None
    if isinstance(policy, DiagnosisLeaf):
        return {"leaf": list(policy.hypotheses)}
    return {
        "state": list(policy.hypotheses),
        "experiment": policy.experiment,
        "branches": [
            {
                "outcome": branch.outcome,
                "child": _policy_payload(branch.child),
            }
            for branch in policy.branches
        ],
    }


def _result_digest(
    *,
    hypotheses: Sequence[str],
    experiments: Sequence[SemanticExperiment],
    objective: str,
    optimal_cost: float | None,
    policy: DiagnosisPolicy | None,
    unresolved_groups: Sequence[Sequence[str]],
    max_risk: float,
    risk_weight: float,
) -> str:
    payload = {
        "hypotheses": list(hypotheses),
        "experiments": [
            {
                "name": experiment.name,
                "outcomes": [
                    [name, observation]
                    for name, observation in experiment.outcomes
                ],
                "cost": experiment.cost,
                "risk": experiment.risk,
                "probe": experiment.probe,
                "tap_after_factor": experiment.tap_after_factor,
                "observation_atol": experiment.observation_atol,
            }
            for experiment in experiments
        ],
        "objective": objective,
        "optimal_cost": optimal_cost,
        "policy": _policy_payload(policy),
        "unresolved_groups": [list(group) for group in unresolved_groups],
        "max_risk": max_risk,
        "risk_weight": risk_weight,
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _validate_problem(
    hypotheses: Sequence[str],
    experiments: Sequence[SemanticExperiment],
    *,
    objective: str,
    max_risk: float,
    risk_weight: float,
    max_hypotheses: int,
) -> tuple[tuple[str, ...], tuple[SemanticExperiment, ...]]:
    names = tuple(hypotheses)
    if len(names) < 2:
        raise ValueError("at least two hypotheses are required")
    if len(names) > max_hypotheses:
        raise ValueError(
            f"exact adaptive solver is capped at {max_hypotheses} hypotheses"
        )
    if len(set(names)) != len(names):
        raise ValueError("hypothesis names must be unique")
    if objective not in {"worst_case", "uniform_expected"}:
        raise ValueError(
            "objective must be 'worst_case' or 'uniform_expected'"
        )
    if (not isfinite(max_risk)) or max_risk < 0:
        raise ValueError("max_risk must be finite and non-negative")
    if (not isfinite(risk_weight)) or risk_weight < 0:
        raise ValueError("risk_weight must be finite and non-negative")

    items = tuple(experiments)
    if not items:
        raise ValueError("at least one experiment is required")
    experiment_names = tuple(item.name for item in items)
    if len(set(experiment_names)) != len(experiment_names):
        raise ValueError("experiment names must be unique")

    expected = set(names)
    for experiment in items:
        observed = set(experiment.outcome_map)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"experiment {experiment.name!r} hypothesis domain mismatch; "
                f"missing={missing}, extra={extra}"
            )
    return names, items


def _observational_equivalence_groups(
    hypotheses: Sequence[str],
    experiments: Sequence[SemanticExperiment],
) -> tuple[tuple[str, ...], ...]:
    """Return conservative ambiguity components across all experiments."""

    names = tuple(sorted(hypotheses))
    adjacency = {name: set() for name in names}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            never_robustly_separated = all(
                _observation_distance(
                    experiment.outcome_for(left),
                    experiment.outcome_for(right),
                )
                <= 2.0 * experiment.observation_atol
                for experiment in experiments
            )
            if never_robustly_separated:
                adjacency[left].add(right)
                adjacency[right].add(left)

    remaining = set(names)
    groups: list[tuple[str, ...]] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component = {seed}
        remaining.remove(seed)
        while stack:
            current = stack.pop()
            for other in sorted(adjacency[current] & remaining):
                remaining.remove(other)
                component.add(other)
                stack.append(other)
        if len(component) > 1:
            groups.append(tuple(sorted(component)))
    return tuple(sorted(groups))


def solve_optimal_semantic_diagnosis(
    hypotheses: Sequence[str],
    experiments: Sequence[SemanticExperiment],
    *,
    objective: str = "worst_case",
    max_risk: float = 0.0,
    risk_weight: float = 1.0,
    max_hypotheses: int = 18,
) -> OptimalDiagnosisResult:
    """Solve the exact finite adaptive diagnosis problem.

    The Bellman state is the current surviving hypothesis set. At each state the
    solver evaluates every admissible experiment that strictly partitions that
    set and recursively solves each observation branch.

    worst_case minimizes maximum accumulated experiment cost.
    uniform_expected minimizes expected accumulated cost under a uniform prior.

    The returned optimum is exact with respect to the finite supplied
    experiment table, not over probes absent from that table.
    """

    names, items = _validate_problem(
        hypotheses,
        experiments,
        objective=objective,
        max_risk=max_risk,
        risk_weight=risk_weight,
        max_hypotheses=max_hypotheses,
    )
    admissible = tuple(item for item in items if item.risk <= max_risk)
    unresolved = _observational_equivalence_groups(names, admissible)
    if unresolved:
        digest = _result_digest(
            hypotheses=names,
            experiments=items,
            objective=objective,
            optimal_cost=None,
            policy=None,
            unresolved_groups=unresolved,
            max_risk=max_risk,
            risk_weight=risk_weight,
        )
        return OptimalDiagnosisResult(
            status="unidentifiable",
            objective=objective,
            optimal_cost=None,
            hypothesis_names=names,
            experiment_names=tuple(item.name for item in items),
            policy=None,
            unresolved_groups=unresolved,
            bellman_states=0,
            max_risk=max_risk,
            risk_weight=risk_weight,
            digest=digest,
        )

    experiment_by_name = {item.name: item for item in admissible}
    memo: dict[frozenset[str], tuple[float, DiagnosisPolicy]] = {}

    def solve(state: frozenset[str]) -> tuple[float, DiagnosisPolicy]:
        if len(state) == 1:
            return 0.0, DiagnosisLeaf(tuple(sorted(state)))
        if state in memo:
            return memo[state]

        best_cost = float("inf")
        best_name: str | None = None
        best_node: DiagnosisDecision | None = None
        for experiment_name in sorted(experiment_by_name):
            experiment = experiment_by_name[experiment_name]
            parts = _partition(state, experiment)
            if len(parts) <= 1:
                continue

            children: list[
                tuple[Observation, frozenset[str], float, DiagnosisPolicy]
            ] = []
            for outcome, child_state in parts:
                child_cost, child_policy = solve(child_state)
                children.append(
                    (outcome, child_state, child_cost, child_policy)
                )

            immediate = _effective_cost(experiment, risk_weight)
            if objective == "worst_case":
                future = max(
                    child_cost for _, _, child_cost, _ in children
                )
            else:
                denominator = float(len(state))
                future = sum(
                    (len(child_state) / denominator) * child_cost
                    for _, child_state, child_cost, _ in children
                )
            total = immediate + future
            node = DiagnosisDecision(
                hypotheses=tuple(sorted(state)),
                experiment=experiment.name,
                branches=tuple(
                    DiagnosisBranch(outcome=outcome, child=child_policy)
                    for outcome, _, _, child_policy in children
                ),
            )
            if (
                total < best_cost
                or (
                    total == best_cost
                    and (best_name is None or experiment.name < best_name)
                )
            ):
                best_cost = total
                best_name = experiment.name
                best_node = node

        if best_node is None:
            return float("inf"), DiagnosisLeaf(tuple(sorted(state)))

        result = (best_cost, best_node)
        memo[state] = result
        return result

    optimal_cost, policy = solve(frozenset(names))
    if not isfinite(optimal_cost):
        unresolved = (tuple(sorted(names)),)
        digest = _result_digest(
            hypotheses=names,
            experiments=items,
            objective=objective,
            optimal_cost=None,
            policy=None,
            unresolved_groups=unresolved,
            max_risk=max_risk,
            risk_weight=risk_weight,
        )
        return OptimalDiagnosisResult(
            status="unidentifiable",
            objective=objective,
            optimal_cost=None,
            hypothesis_names=names,
            experiment_names=tuple(item.name for item in items),
            policy=None,
            unresolved_groups=unresolved,
            bellman_states=len(memo),
            max_risk=max_risk,
            risk_weight=risk_weight,
            digest=digest,
        )

    digest = _result_digest(
        hypotheses=names,
        experiments=items,
        objective=objective,
        optimal_cost=optimal_cost,
        policy=policy,
        unresolved_groups=(),
        max_risk=max_risk,
        risk_weight=risk_weight,
    )
    return OptimalDiagnosisResult(
        status="optimal",
        objective=objective,
        optimal_cost=optimal_cost,
        hypothesis_names=names,
        experiment_names=tuple(item.name for item in items),
        policy=policy,
        unresolved_groups=(),
        bellman_states=len(memo),
        max_risk=max_risk,
        risk_weight=risk_weight,
        digest=digest,
    )


def _policy_cost_and_validate(
    policy: DiagnosisPolicy,
    state: frozenset[str],
    experiment_by_name: Mapping[str, SemanticExperiment],
    *,
    objective: str,
    risk_weight: float,
) -> float:
    if isinstance(policy, DiagnosisLeaf):
        if tuple(sorted(state)) != policy.hypotheses:
            raise ValueError("leaf state does not match reachable hypotheses")
        if len(state) != 1:
            raise ValueError("diagnostic policy terminates before identification")
        return 0.0

    if tuple(sorted(state)) != policy.hypotheses:
        raise ValueError("decision state does not match reachable hypotheses")
    if policy.experiment not in experiment_by_name:
        raise ValueError("policy references an inadmissible experiment")
    experiment = experiment_by_name[policy.experiment]
    expected_parts = _partition(state, experiment)
    expected_by_key = {
        _canonical_observation(outcome): (outcome, child_state)
        for outcome, child_state in expected_parts
    }
    actual_by_key = {
        _canonical_observation(branch.outcome): branch
        for branch in policy.branches
    }
    if set(expected_by_key) != set(actual_by_key):
        raise ValueError("policy branches do not match experiment outcomes")

    child_costs: list[tuple[int, float]] = []
    for key, (_, child_state) in expected_by_key.items():
        branch = actual_by_key[key]
        child_cost = _policy_cost_and_validate(
            branch.child,
            child_state,
            experiment_by_name,
            objective=objective,
            risk_weight=risk_weight,
        )
        child_costs.append((len(child_state), child_cost))

    immediate = _effective_cost(experiment, risk_weight)
    if objective == "worst_case":
        return immediate + max(cost for _, cost in child_costs)
    denominator = float(len(state))
    return immediate + sum(
        (size / denominator) * cost for size, cost in child_costs
    )


def _independent_optimal_value(
    hypotheses: Sequence[str],
    experiments: Sequence[SemanticExperiment],
    *,
    objective: str,
    max_risk: float,
    risk_weight: float,
) -> float | None:
    """Independent value-only exhaustive verifier."""

    admissible = tuple(item for item in experiments if item.risk <= max_risk)
    cache: dict[frozenset[str], float] = {}

    def value(state: frozenset[str]) -> float:
        if len(state) <= 1:
            return 0.0
        if state in cache:
            return cache[state]

        best = float("inf")
        for experiment in admissible:
            parts = _partition(state, experiment)
            if len(parts) <= 1:
                continue
            child_values = [(child, value(child)) for _, child in parts]
            if any(not isfinite(v) for _, v in child_values):
                continue
            immediate = _effective_cost(experiment, risk_weight)
            if objective == "worst_case":
                candidate = immediate + max(v for _, v in child_values)
            else:
                denominator = float(len(state))
                candidate = immediate + sum(
                    (len(child) / denominator) * v
                    for child, v in child_values
                )
            best = min(best, candidate)

        cache[state] = best
        return best

    result = value(frozenset(hypotheses))
    return result if isfinite(result) else None


def verify_optimal_semantic_diagnosis(
    result: OptimalDiagnosisResult,
    experiments: Sequence[SemanticExperiment],
    *,
    atol: float = 1.0e-12,
) -> DiagnosisVerification:
    """Verify policy structure, cost, digest, and exact finite optimality."""

    if atol < 0:
        raise ValueError("atol must be non-negative")
    items = tuple(experiments)
    names = result.hypothesis_names

    expected_digest = _result_digest(
        hypotheses=names,
        experiments=items,
        objective=result.objective,
        optimal_cost=result.optimal_cost,
        policy=result.policy,
        unresolved_groups=result.unresolved_groups,
        max_risk=result.max_risk,
        risk_weight=result.risk_weight,
    )
    digest_matches = expected_digest == result.digest

    independently_optimal = _independent_optimal_value(
        names,
        items,
        objective=result.objective,
        max_risk=result.max_risk,
        risk_weight=result.risk_weight,
    )

    if result.status == "unidentifiable":
        unresolved = _observational_equivalence_groups(
            names,
            tuple(item for item in items if item.risk <= result.max_risk),
        )
        valid = (
            result.policy is None
            and result.optimal_cost is None
            and unresolved == result.unresolved_groups
            and independently_optimal is None
            and digest_matches
        )
        return DiagnosisVerification(
            valid=valid,
            policy_cost=None,
            independently_optimal_cost=None,
            digest_matches=digest_matches,
            message=(
                "unidentifiability certificate verified"
                if valid
                else "unidentifiability certificate mismatch"
            ),
        )

    if result.policy is None or result.optimal_cost is None:
        return DiagnosisVerification(
            valid=False,
            policy_cost=None,
            independently_optimal_cost=independently_optimal,
            digest_matches=digest_matches,
            message="optimal result is missing policy or cost",
        )

    admissible = {
        item.name: item
        for item in items
        if item.risk <= result.max_risk
    }
    try:
        policy_cost = _policy_cost_and_validate(
            result.policy,
            frozenset(names),
            admissible,
            objective=result.objective,
            risk_weight=result.risk_weight,
        )
    except ValueError as exc:
        return DiagnosisVerification(
            valid=False,
            policy_cost=None,
            independently_optimal_cost=independently_optimal,
            digest_matches=digest_matches,
            message=str(exc),
        )

    optimal_match = (
        independently_optimal is not None
        and abs(policy_cost - independently_optimal) <= atol
        and abs(result.optimal_cost - independently_optimal) <= atol
    )
    valid = digest_matches and optimal_match
    return DiagnosisVerification(
        valid=valid,
        policy_cost=policy_cost,
        independently_optimal_cost=independently_optimal,
        digest_matches=digest_matches,
        message=(
            "policy and exact finite optimality verified"
            if valid
            else "policy cost, optimal value, or digest mismatch"
        ),
    )


def build_transport_experiments(
    hypotheses: Sequence[object],
    *,
    probe_cost: float = 1.0,
    tap_costs: Mapping[str, float] | None = None,
    include_external_output: bool = True,
    round_digits: int = 12,
    observation_atol: float = 0.0,
) -> tuple[SemanticExperiment, ...]:
    """Build joint probe/tap experiments for monomial semantic hypotheses."""

    from .embodied_semantic_transport import MonomialSemanticTransport

    items = tuple(hypotheses)
    if len(items) < 2:
        raise ValueError("at least two transport hypotheses are required")
    factor_names = tuple(factor.name for factor in items[0].factors)
    dimension = items[0].factors[0].transport.dimension

    for item in items:
        if tuple(factor.name for factor in item.factors) != factor_names:
            raise ValueError(
                "all transport hypotheses must use the same factor names"
            )
        if any(
            factor.transport.dimension != dimension
            for factor in item.factors
        ):
            raise ValueError("transport hypothesis dimensions differ")

    costs = {name: 1.0 for name in factor_names[:-1]}
    if tap_costs:
        unknown = set(tap_costs) - set(costs)
        if unknown:
            raise ValueError(
                f"tap_costs references unavailable taps: {sorted(unknown)}"
            )
        for name, value in tap_costs.items():
            numeric = float(value)
            if (not isfinite(numeric)) or numeric < 0:
                raise ValueError("tap costs must be finite and non-negative")
            costs[name] = numeric

    if (not isfinite(probe_cost)) or probe_cost <= 0:
        raise ValueError("probe_cost must be finite and positive")

    prefixes: dict[str, tuple[MonomialSemanticTransport, ...]] = {}
    for item in items:
        prefix = MonomialSemanticTransport.identity(dimension)
        sequence: list[MonomialSemanticTransport] = []
        for factor in item.factors:
            prefix = prefix.then(factor.transport)
            sequence.append(prefix)
        prefixes[item.name] = tuple(sequence)

    def rounded(value: Sequence[float]) -> Observation:
        return tuple(round(float(x), round_digits) for x in value)

    experiments: list[SemanticExperiment] = []
    for basis_index in range(dimension):
        basis = tuple(
            1.0 if index == basis_index else 0.0
            for index in range(dimension)
        )
        probe_name = f"basis[{basis_index}]"

        if include_external_output:
            experiments.append(
                SemanticExperiment(
                    name=f"external::{probe_name}",
                    outcomes=tuple(
                        (
                            item.name,
                            rounded(prefixes[item.name][-1].apply(basis)),
                        )
                        for item in items
                    ),
                    cost=probe_cost,
                    risk=0.0,
                    probe=probe_name,
                    tap_after_factor=None,
                    observation_atol=observation_atol,
                )
            )

        for factor_index, tap_name in enumerate(factor_names[:-1]):
            experiments.append(
                SemanticExperiment(
                    name=f"tap:{tap_name}::{probe_name}",
                    outcomes=tuple(
                        (
                            item.name,
                            rounded(
                                prefixes[item.name][factor_index].apply(basis)
                            ),
                        )
                        for item in items
                    ),
                    cost=probe_cost + costs[tap_name],
                    risk=0.0,
                    probe=probe_name,
                    tap_after_factor=tap_name,
                    observation_atol=observation_atol,
                )
            )

    return tuple(experiments)
