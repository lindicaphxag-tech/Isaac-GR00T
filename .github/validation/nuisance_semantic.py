"""Exact semantic identification under persistent nuisance state.

This module models a deliberately finite problem.  The hidden world is a pair

    (semantic ABI hypothesis, nuisance state)

where the semantic component is the object we must identify and the nuisance
component may represent a frozen controller gain, contact regime, payload,
calibration state, or other execution condition that changes observations
without changing interface meaning.

A diagnosis is valid only when every remaining world agrees on the semantic
hypothesis.  The nuisance state itself need not be identified.

The nuisance state is *persistent across an adaptive experiment episode*.  That
assumption is intentionally explicit: treating nuisance as independently
adversarial at every probe would be a different and more conservative model.
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


@dataclass(frozen=True, order=True)
class SemanticWorld:
    semantic: str
    nuisance: str

    def __post_init__(self) -> None:
        if not self.semantic or not self.nuisance:
            raise ValueError("semantic and nuisance identifiers must be non-empty")

    @property
    def key(self) -> str:
        return f"{self.semantic}::{self.nuisance}"


@dataclass(frozen=True)
class RobustSemanticExperiment:
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
        if self.probe_cost < 0 or self.tap_cost < 0 or self.risk < 0:
            raise ValueError("experiment costs/risks must be non-negative")
        for value in (self.probe_cost, self.tap_cost, self.risk):
            if not isfinite(float(value)):
                raise ValueError("experiment costs/risks must be finite")
        if len(set(self.evidence_for)) != len(self.evidence_for):
            raise ValueError("evidence_for entries must be unique")
        if any(not item for item in self.evidence_for):
            raise ValueError("evidence_for entries must be non-empty")

    @property
    def cost(self) -> float:
        return float(self.probe_cost + self.tap_cost)


@dataclass(frozen=True)
class RobustLeaf:
    worlds: tuple[str, ...]
    semantics: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class RobustBranch:
    outcome: str
    child: "RobustNode"


@dataclass(frozen=True)
class RobustDecision:
    experiment: str
    probe: str
    tap: str
    branches: tuple[RobustBranch, ...]


RobustNode = RobustLeaf | RobustDecision


@dataclass(frozen=True)
class RobustPlan:
    status: str
    nuisance_model: str
    worlds: tuple[str, ...]
    semantics: tuple[str, ...]
    required_evidence: str | None
    authorized_experiments: tuple[str, ...]
    max_total_risk: float
    worst_case_cost: float
    worst_case_risk: float
    worst_case_depth: int
    root: RobustNode
    full_world_equivalence: tuple[tuple[str, ...], ...]
    authorized_world_equivalence: tuple[tuple[str, ...], ...]
    digest: str

    @property
    def complete(self) -> bool:
        return self.status == "identified"


def _validate(
    worlds: Sequence[SemanticWorld],
    experiments: Sequence[RobustSemanticExperiment],
) -> tuple[tuple[SemanticWorld, ...], tuple[RobustSemanticExperiment, ...]]:
    world_items = tuple(worlds)
    if not world_items:
        raise ValueError("at least one semantic world is required")
    if len({world.key for world in world_items}) != len(world_items):
        raise ValueError("semantic world identities must be unique")
    if len({world.semantic for world in world_items}) < 2:
        raise ValueError("at least two semantic hypotheses are required")

    experiment_items = tuple(experiments)
    if len({item.name for item in experiment_items}) != len(experiment_items):
        raise ValueError("experiment names must be unique")
    expected = {world.key for world in world_items}
    for experiment in experiment_items:
        observed = set(experiment.outcomes)
        if observed != expected:
            raise ValueError(
                f"experiment {experiment.name!r} outcome domain mismatch: "
                f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
            )
    return world_items, experiment_items


def _equivalence_classes(
    worlds: tuple[SemanticWorld, ...],
    experiments: tuple[RobustSemanticExperiment, ...],
) -> tuple[tuple[str, ...], ...]:
    signatures: dict[tuple[str, ...], list[str]] = {}
    for world in worlds:
        signature = tuple(exp.outcomes[world.key] for exp in experiments)
        signatures.setdefault(signature, []).append(world.key)
    return tuple(
        sorted(
            (tuple(sorted(group)) for group in signatures.values()),
            key=lambda group: group,
        )
    )


def _partition(
    subset: frozenset[str],
    experiment: RobustSemanticExperiment,
) -> tuple[tuple[str, frozenset[str]], ...]:
    groups: dict[str, set[str]] = {}
    for key in subset:
        groups.setdefault(experiment.outcomes[key], set()).add(key)
    return tuple(
        sorted(
            ((outcome, frozenset(group)) for outcome, group in groups.items()),
            key=lambda item: item[0],
        )
    )


def _node_payload(node: RobustNode) -> dict:
    if isinstance(node, RobustLeaf):
        return {
            "kind": "leaf",
            "worlds": list(node.worlds),
            "semantics": list(node.semantics),
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


def _digest(
    worlds: tuple[SemanticWorld, ...],
    experiments: tuple[RobustSemanticExperiment, ...],
    *,
    required_evidence: str | None,
    authorized_experiments: tuple[str, ...],
    max_total_risk: Fraction,
    root: RobustNode,
    full_equivalence: tuple[tuple[str, ...], ...],
    authorized_equivalence: tuple[tuple[str, ...], ...],
) -> str:
    payload = {
        "nuisance_model": "persistent",
        "worlds": [
            {"key": world.key, "semantic": world.semantic, "nuisance": world.nuisance}
            for world in sorted(worlds)
        ],
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
        "required_evidence": required_evidence,
        "authorized_experiments": list(authorized_experiments),
        "max_total_risk": str(max_total_risk),
        "full_world_equivalence": [list(group) for group in full_equivalence],
        "authorized_world_equivalence": [
            list(group) for group in authorized_equivalence
        ],
        "tree": _node_payload(root),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class _Solved:
    status: str
    node: RobustNode
    cost: Fraction
    risk: Fraction
    depth: int


def synthesize_nuisance_robust_plan(
    worlds: Sequence[SemanticWorld],
    experiments: Sequence[RobustSemanticExperiment],
    *,
    max_total_risk: float = 0.0,
    required_evidence: str | None = None,
) -> RobustPlan:
    """Identify semantic ABI under a persistent finite nuisance state.

    A leaf is successful when all remaining worlds share the same semantic
    identity, even if multiple nuisance states remain possible.

    Failure modes:
      * nuisance_confounded: some distinct semantics have observationally
        identical persistent worlds across the full experiment language;
      * evidence_scope_limited: full evidence can identify the semantic ABI but
        evidence authorized for the requested claim cannot;
      * budget_limited: claim-authorized evidence is sufficient in principle but
        every complete adaptive strategy exceeds the risk budget.
    """

    world_items, all_items = _validate(worlds, experiments)
    if required_evidence is not None and not required_evidence:
        raise ValueError("required_evidence must be non-empty when provided")
    items = tuple(
        item
        for item in all_items
        if required_evidence is None or required_evidence in item.evidence_for
    )
    authorized_names = tuple(sorted(item.name for item in items))

    world_by_key = {world.key: world for world in world_items}
    risk_budget = _fraction(max_total_risk)
    if risk_budget < 0:
        raise ValueError("max_total_risk must be non-negative")

    full_eq = _equivalence_classes(world_items, all_items)
    auth_eq = (
        _equivalence_classes(world_items, items)
        if items
        else (tuple(sorted(world_by_key)),)
    )

    def class_confounds_semantics(group: tuple[str, ...]) -> bool:
        return len({world_by_key[key].semantic for key in group}) > 1

    full_confounded = tuple(group for group in full_eq if class_confounds_semantics(group))
    auth_confounded = tuple(group for group in auth_eq if class_confounds_semantics(group))

    cost = {item.name: _fraction(item.cost) for item in items}
    risk = {item.name: _fraction(item.risk) for item in items}

    @lru_cache(maxsize=None)
    def solve(subset_tuple: tuple[str, ...], remaining: Fraction) -> _Solved:
        subset = frozenset(subset_tuple)
        semantics = {world_by_key[key].semantic for key in subset}
        if len(semantics) == 1:
            semantic = next(iter(semantics))
            return _Solved(
                "identified",
                RobustLeaf(tuple(sorted(subset)), (semantic,), "semantic_singleton"),
                Fraction(0),
                Fraction(0),
                0,
            )

        candidates: list[tuple[tuple[Fraction, Fraction, int, str], _Solved]] = []
        for experiment in items:
            erisk = risk[experiment.name]
            if erisk > remaining:
                continue
            partitions = _partition(subset, experiment)
            if len(partitions) <= 1:
                continue

            children: list[tuple[str, _Solved]] = []
            feasible = True
            for outcome, child_subset in partitions:
                if child_subset == subset:
                    feasible = False
                    break
                child = solve(tuple(sorted(child_subset)), remaining - erisk)
                if child.status != "identified":
                    feasible = False
                    break
                children.append((outcome, child))
            if not feasible:
                continue

            total_cost = cost[experiment.name] + max(child.cost for _, child in children)
            total_risk = erisk + max(child.risk for _, child in children)
            depth = 1 + max(child.depth for _, child in children)
            node = RobustDecision(
                experiment.name,
                experiment.probe,
                experiment.tap,
                tuple(
                    RobustBranch(outcome, child.node)
                    for outcome, child in sorted(children, key=lambda x: x[0])
                ),
            )
            solved = _Solved("identified", node, total_cost, total_risk, depth)
            candidates.append(((total_cost, total_risk, depth, experiment.name), solved))

        if candidates:
            return min(candidates, key=lambda pair: pair[0])[1]

        def intersects_confounded(groups: tuple[tuple[str, ...], ...]) -> bool:
            for group in groups:
                intersection = set(group) & subset
                if len({world_by_key[key].semantic for key in intersection}) > 1:
                    return True
            return False

        if intersects_confounded(full_confounded):
            status = "nuisance_confounded"
            reason = "persistent_nuisance_semantic_confounding"
        elif required_evidence is not None and intersects_confounded(auth_confounded):
            status = "evidence_scope_limited"
            reason = "claim_evidence_scope_insufficient_under_nuisance"
        else:
            status = "budget_limited"
            reason = "risk_budget_insufficient_under_nuisance"

        return _Solved(
            status,
            RobustLeaf(
                tuple(sorted(subset)),
                tuple(sorted(semantics)),
                reason,
            ),
            Fraction(0),
            Fraction(0),
            0,
        )

    solved = solve(tuple(sorted(world_by_key)), risk_budget)
    digest = _digest(
        world_items,
        all_items,
        required_evidence=required_evidence,
        authorized_experiments=authorized_names,
        max_total_risk=risk_budget,
        root=solved.node,
        full_equivalence=full_eq,
        authorized_equivalence=auth_eq,
    )
    return RobustPlan(
        status=solved.status,
        nuisance_model="persistent",
        worlds=tuple(sorted(world_by_key)),
        semantics=tuple(sorted({world.semantic for world in world_items})),
        required_evidence=required_evidence,
        authorized_experiments=authorized_names,
        max_total_risk=float(risk_budget),
        worst_case_cost=float(solved.cost),
        worst_case_risk=float(solved.risk),
        worst_case_depth=solved.depth,
        root=solved.node,
        full_world_equivalence=full_eq,
        authorized_world_equivalence=auth_eq,
        digest=digest,
    )
