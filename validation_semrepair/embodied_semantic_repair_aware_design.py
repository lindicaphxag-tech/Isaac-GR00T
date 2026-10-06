"""Repair-authority-aware adaptive diagnosis for embodied semantic ABIs.

Generic decision-theoretic troubleshooting and value-of-information are prior
art.  This module narrows the terminal decision to SemRepair's authority model:
diagnosis may stop before the hidden semantic ABI is uniquely identified only
when every surviving hypothesis authorizes the *same exact repair outcome*.

Conversely, hypotheses that are externally observationally equivalent but bind
different factorized repair bundles remain unsafe ambiguity and must be further
instrumented or rejected.

The solver is exact over the finite frozen experiment table and finite
hypothesis set supplied to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
import json
from typing import Mapping, Sequence

from validation_semrepair.embodied_semantic_experiment_design import (
    Observation,
    SemanticExperiment,
)


@dataclass(frozen=True)
class RepairAuthority:
    """Exact downstream authority associated with one semantic hypothesis.

    authority_id must identify the concrete downstream decision, e.g.
    "noop", "reject", or a digest of a verified factorized repair bundle.
    It is deliberately *not* a net end-to-end transform label.
    """

    hypothesis: str
    authority_id: str

    def __post_init__(self) -> None:
        if not self.hypothesis:
            raise ValueError("hypothesis must be non-empty")
        if not self.authority_id:
            raise ValueError("authority_id must be non-empty")


@dataclass(frozen=True)
class BoundRepairAuthority:
    """Canonical implementation/evidence-bound repair authority.

    Two semantically named repairs are the same terminal decision only when
    their concrete bundle, implementations, dependencies, and evidence identity
    are all identical.
    """

    repair_bundle_id: str
    implementation_ids: tuple[str, ...]
    evidence_digest: str
    dependency_digest: str
    authority_version: str = "semrepair-authority-v1"

    def __post_init__(self) -> None:
        fields = (
            self.repair_bundle_id,
            self.evidence_digest,
            self.dependency_digest,
            self.authority_version,
        )
        if any(not value for value in fields):
            raise ValueError("bound authority fields must be non-empty")
        if not self.implementation_ids or any(
            not value for value in self.implementation_ids
        ):
            raise ValueError(
                "bound authority requires concrete implementation identities"
            )

    @property
    def authority_id(self) -> str:
        payload = {
            "authority_version": self.authority_version,
            "repair_bundle_id": self.repair_bundle_id,
            "implementation_ids": list(self.implementation_ids),
            "evidence_digest": self.evidence_digest,
            "dependency_digest": self.dependency_digest,
        }
        return "sha256:" + sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()


def bind_repair_authority(
    hypothesis: str,
    binding: BoundRepairAuthority,
) -> RepairAuthority:
    """Bind one hypothesis to the canonical concrete authority digest."""

    return RepairAuthority(
        hypothesis=hypothesis,
        authority_id=binding.authority_id,
    )


@dataclass(frozen=True)
class RepairDecisionLeaf:
    hypotheses: tuple[str, ...]
    authority_id: str


@dataclass(frozen=True)
class RepairDecisionBranch:
    outcome: Observation
    child: "RepairDecisionPolicy"


@dataclass(frozen=True)
class RepairDecisionNode:
    hypotheses: tuple[str, ...]
    experiment: str
    branches: tuple[RepairDecisionBranch, ...]


RepairDecisionPolicy = RepairDecisionLeaf | RepairDecisionNode


@dataclass(frozen=True)
class RepairAwareDiagnosisResult:
    status: str
    objective: str
    optimal_cost: float | None
    hypothesis_names: tuple[str, ...]
    authority_by_hypothesis: tuple[tuple[str, str], ...]
    policy: RepairDecisionPolicy | None
    unsafe_ambiguity_groups: tuple[tuple[str, ...], ...]
    bellman_states: int
    max_risk: float
    risk_weight: float
    digest: str

    @property
    def complete(self) -> bool:
        return self.status == "optimal" and self.policy is not None


@dataclass(frozen=True)
class RepairAwareVerification:
    valid: bool
    policy_cost: float | None
    independently_optimal_cost: float | None
    digest_matches: bool
    message: str


def _numeric(value: Observation) -> tuple[float, ...] | None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return None
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, tuple):
        out: list[float] = []
        for item in value:
            part = _numeric(item)
            if part is None:
                return None
            out.extend(part)
        return tuple(out)
    return None


def _canonical(value: Observation) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _distance(left: Observation, right: Observation) -> float:
    lhs = _numeric(left)
    rhs = _numeric(right)
    if lhs is None or rhs is None or len(lhs) != len(rhs):
        return 0.0 if _canonical(left) == _canonical(right) else float("inf")
    return max(abs(a - b) for a, b in zip(lhs, rhs, strict=True))


def _partition(
    state: frozenset[str],
    experiment: SemanticExperiment,
) -> tuple[tuple[Observation, frozenset[str]], ...]:
    ordered = tuple(sorted(state))
    if experiment.observation_atol == 0.0:
        groups: dict[str, tuple[Observation, set[str]]] = {}
        for hypothesis in ordered:
            outcome = experiment.outcome_for(hypothesis)
            key = _canonical(outcome)
            if key not in groups:
                groups[key] = (outcome, set())
            groups[key][1].add(hypothesis)
        return tuple(
            (groups[key][0], frozenset(groups[key][1]))
            for key in sorted(groups)
        )

    # Robust semantics: predicted L-infinity balls that overlap are one branch.
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
                if _distance(
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
        sorted(components, key=lambda item: tuple(sorted(item[1])))
    )


def _authority_map(
    hypotheses: Sequence[str],
    authorities: Sequence[RepairAuthority],
) -> dict[str, str]:
    names = tuple(hypotheses)
    if len(names) < 1 or len(set(names)) != len(names):
        raise ValueError("hypothesis names must be non-empty and unique")
    mapping: dict[str, str] = {}
    for item in authorities:
        if item.hypothesis in mapping:
            raise ValueError("duplicate repair authority hypothesis")
        mapping[item.hypothesis] = item.authority_id
    if set(mapping) != set(names):
        raise ValueError("repair authority domain mismatch")
    return mapping


def _terminal_authority(
    state: frozenset[str],
    authority_by_hypothesis: Mapping[str, str],
) -> str | None:
    values = {authority_by_hypothesis[name] for name in state}
    if len(values) == 1:
        return next(iter(values))
    return None


def _effective_cost(experiment: SemanticExperiment, risk_weight: float) -> float:
    return experiment.cost + risk_weight * experiment.risk


def _unsafe_alias_groups(
    hypotheses: Sequence[str],
    experiments: Sequence[SemanticExperiment],
    authority_by_hypothesis: Mapping[str, str],
) -> tuple[tuple[str, ...], ...]:
    """Return observational alias components that disagree on repair authority."""

    names = tuple(sorted(hypotheses))
    adjacency = {name: set() for name in names}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if authority_by_hypothesis[left] == authority_by_hypothesis[right]:
                continue
            never_robustly_separated = all(
                _distance(
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


def _policy_payload(policy: RepairDecisionPolicy | None) -> object:
    if policy is None:
        return None
    if isinstance(policy, RepairDecisionLeaf):
        return {
            "leaf": list(policy.hypotheses),
            "authority_id": policy.authority_id,
        }
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


def _digest(
    *,
    hypotheses: Sequence[str],
    authority_by_hypothesis: Mapping[str, str],
    experiments: Sequence[SemanticExperiment],
    objective: str,
    optimal_cost: float | None,
    policy: RepairDecisionPolicy | None,
    unsafe_ambiguity_groups: Sequence[Sequence[str]],
    max_risk: float,
    risk_weight: float,
) -> str:
    payload = {
        "hypotheses": list(hypotheses),
        "authorities": sorted(authority_by_hypothesis.items()),
        "experiments": [
            {
                "name": item.name,
                "outcomes": list(item.outcomes),
                "cost": item.cost,
                "risk": item.risk,
                "probe": item.probe,
                "tap_after_factor": item.tap_after_factor,
                "observation_atol": item.observation_atol,
            }
            for item in experiments
        ],
        "objective": objective,
        "optimal_cost": optimal_cost,
        "policy": _policy_payload(policy),
        "unsafe_ambiguity_groups": [list(x) for x in unsafe_ambiguity_groups],
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


def solve_repair_aware_semantic_diagnosis(
    hypotheses: Sequence[str],
    experiments: Sequence[SemanticExperiment],
    authorities: Sequence[RepairAuthority],
    *,
    objective: str = "uniform_expected",
    max_risk: float = 0.0,
    risk_weight: float = 1.0,
    max_hypotheses: int = 18,
) -> RepairAwareDiagnosisResult:
    """Minimize diagnostic cost only until repair authority is unique.

    This is an exact Bellman solver over the supplied finite experiment table.
    A leaf may contain multiple semantic hypotheses iff all of them bind the
    same exact authority_id.
    """

    names = tuple(hypotheses)
    if len(names) > max_hypotheses:
        raise ValueError(
            f"exact repair-aware solver is capped at {max_hypotheses} hypotheses"
        )
    authority_map = _authority_map(names, authorities)
    if objective not in {"worst_case", "uniform_expected"}:
        raise ValueError("unsupported objective")
    if not isfinite(max_risk) or max_risk < 0:
        raise ValueError("max_risk must be finite and non-negative")
    if not isfinite(risk_weight) or risk_weight < 0:
        raise ValueError("risk_weight must be finite and non-negative")

    items = tuple(experiments)
    experiment_names = [item.name for item in items]
    if len(experiment_names) != len(set(experiment_names)):
        raise ValueError("experiment names must be unique")
    for item in items:
        if set(item.outcome_map) != set(names):
            raise ValueError(
                f"experiment {item.name!r} hypothesis domain mismatch"
            )

    admissible = tuple(item for item in items if item.risk <= max_risk)
    unsafe = _unsafe_alias_groups(names, admissible, authority_map)
    if unsafe:
        digest = _digest(
            hypotheses=names,
            authority_by_hypothesis=authority_map,
            experiments=items,
            objective=objective,
            optimal_cost=None,
            policy=None,
            unsafe_ambiguity_groups=unsafe,
            max_risk=max_risk,
            risk_weight=risk_weight,
        )
        return RepairAwareDiagnosisResult(
            status="unsafe_unidentifiable",
            objective=objective,
            optimal_cost=None,
            hypothesis_names=names,
            authority_by_hypothesis=tuple(sorted(authority_map.items())),
            policy=None,
            unsafe_ambiguity_groups=unsafe,
            bellman_states=0,
            max_risk=max_risk,
            risk_weight=risk_weight,
            digest=digest,
        )

    by_name = {item.name: item for item in admissible}
    memo: dict[frozenset[str], tuple[float, RepairDecisionPolicy]] = {}

    def solve(state: frozenset[str]) -> tuple[float, RepairDecisionPolicy]:
        terminal = _terminal_authority(state, authority_map)
        if terminal is not None:
            return 0.0, RepairDecisionLeaf(tuple(sorted(state)), terminal)
        if state in memo:
            return memo[state]

        best_cost = float("inf")
        best_name: str | None = None
        best_policy: RepairDecisionNode | None = None

        for name in sorted(by_name):
            experiment = by_name[name]
            parts = _partition(state, experiment)
            if len(parts) <= 1:
                continue

            children: list[
                tuple[Observation, frozenset[str], float, RepairDecisionPolicy]
            ] = []
            feasible = True
            for outcome, child_state in parts:
                child_cost, child_policy = solve(child_state)
                if not isfinite(child_cost):
                    feasible = False
                    break
                children.append(
                    (outcome, child_state, child_cost, child_policy)
                )
            if not feasible:
                continue

            immediate = _effective_cost(experiment, risk_weight)
            if objective == "worst_case":
                future = max(child[2] for child in children)
            else:
                denominator = float(len(state))
                future = sum(
                    (len(child_state) / denominator) * child_cost
                    for _, child_state, child_cost, _ in children
                )
            total = immediate + future
            candidate = RepairDecisionNode(
                hypotheses=tuple(sorted(state)),
                experiment=name,
                branches=tuple(
                    RepairDecisionBranch(outcome, child_policy)
                    for outcome, _, _, child_policy in children
                ),
            )
            if total < best_cost or (
                total == best_cost
                and (best_name is None or name < best_name)
            ):
                best_cost = total
                best_name = name
                best_policy = candidate

        if best_policy is None:
            # This state still contains conflicting authorities and no supplied
            # experiment can resolve them.
            return float("inf"), RepairDecisionLeaf(
                tuple(sorted(state)),
                "__unsafe_unresolved__",
            )

        memo[state] = (best_cost, best_policy)
        return memo[state]

    cost, policy = solve(frozenset(names))
    if not isfinite(cost):
        unresolved = (tuple(sorted(names)),)
        digest = _digest(
            hypotheses=names,
            authority_by_hypothesis=authority_map,
            experiments=items,
            objective=objective,
            optimal_cost=None,
            policy=None,
            unsafe_ambiguity_groups=unresolved,
            max_risk=max_risk,
            risk_weight=risk_weight,
        )
        return RepairAwareDiagnosisResult(
            status="unsafe_unidentifiable",
            objective=objective,
            optimal_cost=None,
            hypothesis_names=names,
            authority_by_hypothesis=tuple(sorted(authority_map.items())),
            policy=None,
            unsafe_ambiguity_groups=unresolved,
            bellman_states=len(memo),
            max_risk=max_risk,
            risk_weight=risk_weight,
            digest=digest,
        )

    digest = _digest(
        hypotheses=names,
        authority_by_hypothesis=authority_map,
        experiments=items,
        objective=objective,
        optimal_cost=cost,
        policy=policy,
        unsafe_ambiguity_groups=(),
        max_risk=max_risk,
        risk_weight=risk_weight,
    )
    return RepairAwareDiagnosisResult(
        status="optimal",
        objective=objective,
        optimal_cost=cost,
        hypothesis_names=names,
        authority_by_hypothesis=tuple(sorted(authority_map.items())),
        policy=policy,
        unsafe_ambiguity_groups=(),
        bellman_states=len(memo),
        max_risk=max_risk,
        risk_weight=risk_weight,
        digest=digest,
    )


def _validate_policy_cost(
    policy: RepairDecisionPolicy,
    state: frozenset[str],
    experiments: Mapping[str, SemanticExperiment],
    authorities: Mapping[str, str],
    *,
    objective: str,
    risk_weight: float,
) -> float:
    if isinstance(policy, RepairDecisionLeaf):
        if tuple(sorted(state)) != policy.hypotheses:
            raise ValueError("leaf hypothesis state mismatch")
        terminal = _terminal_authority(state, authorities)
        if terminal is None or terminal != policy.authority_id:
            raise ValueError("leaf does not bind one exact repair authority")
        return 0.0

    if tuple(sorted(state)) != policy.hypotheses:
        raise ValueError("decision hypothesis state mismatch")
    if policy.experiment not in experiments:
        raise ValueError("policy references unavailable experiment")
    experiment = experiments[policy.experiment]
    expected_parts = _partition(state, experiment)
    if len(expected_parts) != len(policy.branches):
        raise ValueError("policy branch count mismatch")

    children: list[tuple[frozenset[str], float]] = []
    for (expected_outcome, child_state), branch in zip(
        expected_parts,
        policy.branches,
        strict=True,
    ):
        if _canonical(expected_outcome) != _canonical(branch.outcome):
            raise ValueError("policy outcome mismatch")
        child_cost = _validate_policy_cost(
            branch.child,
            child_state,
            experiments,
            authorities,
            objective=objective,
            risk_weight=risk_weight,
        )
        children.append((child_state, child_cost))

    immediate = _effective_cost(experiment, risk_weight)
    if objective == "worst_case":
        return immediate + max(cost for _, cost in children)
    denominator = float(len(state))
    return immediate + sum(
        (len(child_state) / denominator) * cost
        for child_state, cost in children
    )


def _optimal_value_only(
    state: frozenset[str],
    experiments: tuple[SemanticExperiment, ...],
    authorities: Mapping[str, str],
    *,
    objective: str,
    risk_weight: float,
    memo: dict[frozenset[str], float],
) -> float:
    if _terminal_authority(state, authorities) is not None:
        return 0.0
    if state in memo:
        return memo[state]

    best = float("inf")
    for experiment in experiments:
        parts = _partition(state, experiment)
        if len(parts) <= 1:
            continue
        child_values = [
            (child_state, _optimal_value_only(
                child_state,
                experiments,
                authorities,
                objective=objective,
                risk_weight=risk_weight,
                memo=memo,
            ))
            for _, child_state in parts
        ]
        if any(not isfinite(value) for _, value in child_values):
            continue
        immediate = _effective_cost(experiment, risk_weight)
        if objective == "worst_case":
            future = max(value for _, value in child_values)
        else:
            denominator = float(len(state))
            future = sum(
                (len(child_state) / denominator) * value
                for child_state, value in child_values
            )
        best = min(best, immediate + future)
    memo[state] = best
    return best


def verify_repair_aware_semantic_diagnosis(
    result: RepairAwareDiagnosisResult,
    experiments: Sequence[SemanticExperiment],
) -> RepairAwareVerification:
    """Verify digest, authority-safe leaves, and exact finite optimum."""

    authorities = dict(result.authority_by_hypothesis)
    items = tuple(
        item for item in experiments if item.risk <= result.max_risk
    )
    expected_digest = _digest(
        hypotheses=result.hypothesis_names,
        authority_by_hypothesis=authorities,
        experiments=tuple(experiments),
        objective=result.objective,
        optimal_cost=result.optimal_cost,
        policy=result.policy,
        unsafe_ambiguity_groups=result.unsafe_ambiguity_groups,
        max_risk=result.max_risk,
        risk_weight=result.risk_weight,
    )
    digest_matches = expected_digest == result.digest

    independent = _optimal_value_only(
        frozenset(result.hypothesis_names),
        items,
        authorities,
        objective=result.objective,
        risk_weight=result.risk_weight,
        memo={},
    )

    if result.policy is None:
        valid = (
            result.status == "unsafe_unidentifiable"
            and not isfinite(independent)
            and digest_matches
        )
        return RepairAwareVerification(
            valid=valid,
            policy_cost=None,
            independently_optimal_cost=None,
            digest_matches=digest_matches,
            message=(
                "unsafe repair ambiguity independently confirmed"
                if valid
                else "unidentifiability or digest mismatch"
            ),
        )

    try:
        policy_cost = _validate_policy_cost(
            result.policy,
            frozenset(result.hypothesis_names),
            {item.name: item for item in items},
            authorities,
            objective=result.objective,
            risk_weight=result.risk_weight,
        )
    except ValueError as exc:
        return RepairAwareVerification(
            valid=False,
            policy_cost=None,
            independently_optimal_cost=(
                independent if isfinite(independent) else None
            ),
            digest_matches=digest_matches,
            message=str(exc),
        )

    valid = (
        result.status == "optimal"
        and isfinite(independent)
        and abs(policy_cost - independent) <= 1e-12
        and result.optimal_cost is not None
        and abs(result.optimal_cost - independent) <= 1e-12
        and digest_matches
    )
    return RepairAwareVerification(
        valid=valid,
        policy_cost=policy_cost,
        independently_optimal_cost=(
            independent if isfinite(independent) else None
        ),
        digest_matches=digest_matches,
        message=(
            "repair-aware diagnostic optimum verified"
            if valid
            else "policy cost, optimum, authority leaf, or digest mismatch"
        ),
    )
