"""Bounded theorem checker for the SemRepair core calculus.

This module is deliberately independent of the production compiler.  It
exhaustively checks a finite semantic universe against four theorem targets:

1. pure-step semantic preservation;
2. non-forgeability of event-owned axes;
3. fail-closed progress totality/exclusivity;
4. minimum-cost repair-path correctness against an independent exhaustive oracle.

This is bounded mechanized evidence, not a proof over the infinite Python
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import product
from typing import Iterable, Mapping, Sequence

from .semrepair_core_calculus import (
    BoundaryStatus,
    CoreStepBlocked,
    CoreType,
    CoreValue,
    EventRule,
    PureRule,
    decide_boundary,
    event_nonforgeability_holds,
    event_step,
    pure_preservation_holds,
    pure_step,
)


@dataclass(frozen=True)
class CostRule:
    name: str
    requires: Mapping[str, str]
    produces: Mapping[str, str]
    cost: int

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True)
class TheoremCheckReport:
    pure_preservation_cases: int
    nonforgeability_cases: int
    progress_cases: int
    shortest_path_graphs: int
    violations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.violations


def _matches_type(value: CoreType, requires: Mapping[str, str]) -> bool:
    return all(getattr(value, field) == expected for field, expected in requires.items())


def _apply_cost_rule(value: CoreType, rule: CostRule) -> CoreType | None:
    if not _matches_type(value, rule.requires):
        return None
    return value.update(rule.produces)


def _assignable(actual: CoreType, target: CoreType) -> bool:
    return all(
        getattr(target, field) is None or getattr(actual, field) == getattr(target, field)
        for field in actual.__dataclass_fields__
    )


def _dijkstra_cost(
    source: CoreType,
    target: CoreType,
    rules: Sequence[CostRule],
    *,
    max_steps: int,
) -> int | None:
    if _assignable(source, target):
        return 0
    queue: list[tuple[int, int, int, CoreType]] = [(0, 0, 0, source)]
    serial = 1
    best = {source: 0}

    while queue:
        cost, steps, _, current = heappop(queue)
        if cost != best.get(current):
            continue
        if _assignable(current, target):
            return cost
        if steps >= max_steps:
            continue
        for rule in rules:
            nxt = _apply_cost_rule(current, rule)
            if nxt is None or nxt == current:
                continue
            new_cost = cost + rule.cost
            if new_cost >= best.get(nxt, 10**9):
                continue
            best[nxt] = new_cost
            heappush(queue, (new_cost, steps + 1, serial, nxt))
            serial += 1
    return None


def _exhaustive_cost(
    source: CoreType,
    target: CoreType,
    rules: Sequence[CostRule],
    *,
    max_steps: int,
) -> int | None:
    best: int | None = 0 if _assignable(source, target) else None
    frontier: list[tuple[CoreType, int, frozenset[CoreType]]] = [
        (source, 0, frozenset((source,)))
    ]
    for _ in range(max_steps):
        nxt_frontier = []
        for current, cost, visited in frontier:
            for rule in rules:
                nxt = _apply_cost_rule(current, rule)
                if nxt is None or nxt in visited:
                    continue
                new_cost = cost + rule.cost
                if _assignable(nxt, target):
                    best = new_cost if best is None else min(best, new_cost)
                nxt_frontier.append((nxt, new_cost, visited | frozenset((nxt,))))
        frontier = nxt_frontier
    return best


def verify_core_theorems() -> TheoremCheckReport:
    violations: list[str] = []
    pure_cases = 0
    event_cases = 0
    progress_cases = 0
    graph_cases = 0

    representations = ("axis_angle", "euler_xyz")
    provenances = ("requested", "executed")
    freshness_values = ("stale", "fresh")

    # T1: every admitted pure step preserves canonical meaning and lineage.
    pure_rules = (
        PureRule(
            "axis-to-euler",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
        ),
        PureRule(
            "retime",
            requires={"clock": "dataset_global"},
            produces={"clock": "episode_local"},
            evidence_keys=("origin",),
        ),
    )
    for representation, provenance, freshness in product(
        representations, provenances, freshness_values
    ):
        value = CoreValue(
            CoreType(
                representation=representation,
                provenance=provenance,
                freshness=freshness,
                clock="dataset_global",
            ),
            meaning_token=f"{representation}:{provenance}:{freshness}",
        )
        for rule in pure_rules:
            try:
                step = pure_step(value, rule, evidence={"origin": 0})
            except CoreStepBlocked:
                continue
            pure_cases += 1
            if not pure_preservation_holds(step):
                violations.append(f"pure preservation failed: {rule.name}")

    # T2: every event-owned semantic change requires receipt evidence and lineage.
    event_rules = (
        EventRule(
            "execute",
            requires={"provenance": "requested"},
            produces={"provenance": "executed"},
            receipt_keys=("action_id",),
        ),
        EventRule(
            "observe",
            requires={"freshness": "stale"},
            produces={"freshness": "fresh"},
            receipt_keys=("sample_id",),
        ),
    )
    for provenance, freshness in product(provenances, freshness_values):
        value = CoreValue(
            CoreType(provenance=provenance, freshness=freshness),
            meaning_token=f"{provenance}:{freshness}",
        )
        for rule in event_rules:
            if not all(
                getattr(value.semantic_type, field) == expected
                for field, expected in rule.requires.items()
            ):
                continue
            # Missing receipt must block.
            try:
                event_step(value, rule, receipt={})
            except CoreStepBlocked:
                pass
            else:
                violations.append(f"event rule admitted without receipt: {rule.name}")

            receipt = {rule.receipt_keys[0]: "witness"}
            step = event_step(value, rule, receipt=receipt)
            event_cases += 1
            if not event_nonforgeability_holds(step):
                violations.append(f"event nonforgeability failed: {rule.name}")

    # T3: progress classifier is total and returns exactly one status.
    source = CoreType(representation="axis_angle")
    target = CoreType(representation="euler_xyz")
    statuses = set(BoundaryStatus)
    for repair_paths, obligations, survivors in product(
        (0, 1, 2, 3),
        ((), ("frame_tf",)),
        (0, 1, 2, 3),
    ):
        decision = decide_boundary(
            source,
            target,
            repair_paths=repair_paths,
            missing_obligations=obligations,
            inference_survivors=survivors,
        )
        progress_cases += 1
        if decision.status not in statuses:
            violations.append("progress returned status outside closed outcome set")

    accept = decide_boundary(source, source, repair_paths=0)
    progress_cases += 1
    if accept.status is not BoundaryStatus.ACCEPT:
        violations.append("assignable boundary did not ACCEPT")

    # T4: Dijkstra result matches independent exhaustive minimum on finite graphs.
    base_types = (
        CoreType(representation="axis_angle", unit="rad"),
        CoreType(representation="quaternion", unit="rad"),
        CoreType(representation="euler_xyz", unit="rad"),
        CoreType(representation="euler_xyz", unit="deg"),
    )
    candidate_rules = (
        CostRule(
            "axis-to-quat",
            {"representation": "axis_angle"},
            {"representation": "quaternion"},
            1,
        ),
        CostRule(
            "quat-to-euler",
            {"representation": "quaternion"},
            {"representation": "euler_xyz"},
            1,
        ),
        CostRule(
            "axis-to-euler-direct",
            {"representation": "axis_angle"},
            {"representation": "euler_xyz"},
            3,
        ),
        CostRule(
            "rad-to-deg",
            {"unit": "rad"},
            {"unit": "deg"},
            2,
        ),
    )

    # Enumerate all non-empty rule subsets.  This includes disconnected and
    # alternative-path graphs.
    for mask in range(1, 1 << len(candidate_rules)):
        rules = tuple(
            rule for i, rule in enumerate(candidate_rules) if mask & (1 << i)
        )
        for source_type, target_type in product(base_types, repeat=2):
            graph_cases += 1
            dijkstra = _dijkstra_cost(source_type, target_type, rules, max_steps=4)
            exhaustive = _exhaustive_cost(source_type, target_type, rules, max_steps=4)
            if dijkstra != exhaustive:
                violations.append(
                    "minimum-cost mismatch: "
                    f"source={source_type} target={target_type} "
                    f"dijkstra={dijkstra} exhaustive={exhaustive}"
                )

    return TheoremCheckReport(
        pure_preservation_cases=pure_cases,
        nonforgeability_cases=event_cases,
        progress_cases=progress_cases,
        shortest_path_graphs=graph_cases,
        violations=tuple(violations),
    )


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run bounded theorem checks for SemRepair core")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = verify_core_theorems()
    payload = {
        "checker": "semrepair-core-theorems/v0.1",
        "valid": report.valid,
        "pure_preservation_cases": report.pure_preservation_cases,
        "nonforgeability_cases": report.nonforgeability_cases,
        "progress_cases": report.progress_cases,
        "shortest_path_graphs": report.shortest_path_graphs,
        "violations": list(report.violations),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(payload)
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()