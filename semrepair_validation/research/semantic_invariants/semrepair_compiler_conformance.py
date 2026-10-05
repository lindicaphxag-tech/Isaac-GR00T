"""Differential conformance between the production semantic compiler and an
independent finite exhaustive oracle.

The oracle intentionally does not call the production Dijkstra implementation.
It enumerates simple semantic-type paths, tracks context-evidence obligations,
and classifies each boundary as:

- accept
- unique-repair
- ambiguous
- obligation
- reject

The checker then compares that classification with the production
synthesize_unique_adapter_plan behavior across every non-empty subset of a small
adapter universe and every source/target pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

from .embodied_semantic_types import (
    AmbiguousSemanticRepair,
    NoSemanticRepair,
    SemanticAdapter,
    SemanticTensorType,
    is_assignable,
    synthesize_unique_adapter_plan,
)


@dataclass(frozen=True)
class OraclePlan:
    names: tuple[str, ...]
    cost: float
    result: SemanticTensorType


@dataclass(frozen=True)
class OracleDecision:
    status: str
    cost: float | None = None
    alternatives: tuple[tuple[str, ...], ...] = ()
    obligations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConformanceReport:
    cases_checked: int
    accept_cases: int
    unique_repair_cases: int
    ambiguous_cases: int
    obligation_cases: int
    reject_cases: int
    violations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.violations


def _enumerate_simple_paths(
    source: SemanticTensorType,
    target: SemanticTensorType,
    adapters: Sequence[SemanticAdapter],
    *,
    evidence: Mapping[str, object] | None,
    max_steps: int,
) -> tuple[list[OraclePlan], set[str]]:
    available = set((evidence or {}).keys())
    solutions: list[OraclePlan] = []
    obligations: set[str] = set()

    frontier: list[
        tuple[SemanticTensorType, tuple[str, ...], float, frozenset[SemanticTensorType]]
    ] = [(source, (), 0.0, frozenset((source,)))]

    for depth in range(max_steps + 1):
        nxt_frontier = []
        for current, names, cost, visited in frontier:
            if is_assignable(current, target):
                solutions.append(OraclePlan(names, cost, current))
                continue
            if depth >= max_steps:
                continue

            for adapter in adapters:
                if not adapter.applies_to(current):
                    continue
                missing = set(adapter.required_evidence) - available
                if missing:
                    obligations.update(missing)
                    continue
                nxt = adapter.apply_type(current)
                if nxt == current or nxt in visited:
                    continue
                nxt_frontier.append(
                    (
                        nxt,
                        names + (adapter.name,),
                        cost + adapter.cost,
                        visited | frozenset((nxt,)),
                    )
                )
        frontier = nxt_frontier

    return solutions, obligations


def oracle_decision(
    source: SemanticTensorType,
    target: SemanticTensorType,
    adapters: Sequence[SemanticAdapter],
    *,
    evidence: Mapping[str, object] | None = None,
    max_steps: int = 6,
) -> OracleDecision:
    if is_assignable(source, target):
        return OracleDecision("accept", cost=0.0, alternatives=((),))

    solutions, obligations = _enumerate_simple_paths(
        source,
        target,
        adapters,
        evidence=evidence,
        max_steps=max_steps,
    )

    if solutions:
        min_cost = min(item.cost for item in solutions)
        minimal = sorted(
            {item.names for item in solutions if item.cost == min_cost}
        )
        if len(minimal) == 1:
            return OracleDecision(
                "unique-repair",
                cost=min_cost,
                alternatives=(minimal[0],),
            )
        return OracleDecision(
            "ambiguous",
            cost=min_cost,
            alternatives=tuple(minimal),
        )

    if obligations:
        return OracleDecision(
            "obligation",
            obligations=tuple(sorted(obligations)),
        )
    return OracleDecision("reject")


def production_decision(
    source: SemanticTensorType,
    target: SemanticTensorType,
    adapters: Sequence[SemanticAdapter],
    *,
    evidence: Mapping[str, object] | None = None,
    max_steps: int = 6,
) -> OracleDecision:
    try:
        plan = synthesize_unique_adapter_plan(
            source,
            target,
            adapters,
            evidence=evidence,
            max_steps=max_steps,
        )
    except AmbiguousSemanticRepair as exc:
        return OracleDecision(
            "ambiguous",
            alternatives=tuple(sorted(tuple(path) for path in exc.alternatives)),
        )
    except NoSemanticRepair as exc:
        if exc.proof_obligations:
            return OracleDecision(
                "obligation",
                obligations=tuple(sorted(exc.proof_obligations)),
            )
        return OracleDecision("reject")

    names = tuple(item.name for item in plan.adapters)
    return OracleDecision(
        "accept" if not names else "unique-repair",
        cost=plan.total_cost,
        alternatives=(names,),
    )


def _adapter_universe() -> tuple[SemanticAdapter, ...]:
    return (
        SemanticAdapter(
            "axis-to-quat",
            requires={"representation": "axis_angle"},
            produces={"representation": "quaternion"},
            cost=1.0,
        ),
        SemanticAdapter(
            "quat-to-euler",
            requires={"representation": "quaternion"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
        SemanticAdapter(
            "axis-to-euler-direct",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=3.0,
        ),
        SemanticAdapter(
            "axis-to-euler-alt",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=3.0,
        ),
        SemanticAdapter(
            "rad-to-deg",
            requires={"unit": "rad"},
            produces={"unit": "deg"},
            cost=2.0,
        ),
        SemanticAdapter(
            "retime",
            requires={"clock": "dataset_global"},
            produces={"clock": "episode_local"},
            cost=1.0,
            witness_keys=("episode_origin",),
        ),
    )


def verify_compiler_conformance() -> ConformanceReport:
    violations: list[str] = []
    counts = {
        "accept": 0,
        "unique-repair": 0,
        "ambiguous": 0,
        "obligation": 0,
        "reject": 0,
    }
    cases = 0

    types = (
        SemanticTensorType(
            role="action",
            entity="ee_rotation_delta",
            representation="axis_angle",
            unit="rad",
            clock="dataset_global",
            provenance="requested",
        ),
        SemanticTensorType(
            role="action",
            entity="ee_rotation_delta",
            representation="quaternion",
            unit="rad",
            clock="dataset_global",
            provenance="requested",
        ),
        SemanticTensorType(
            role="action",
            entity="ee_rotation_delta",
            representation="euler_xyz",
            unit="rad",
            clock="dataset_global",
            provenance="requested",
        ),
        SemanticTensorType(
            role="action",
            entity="ee_rotation_delta",
            representation="euler_xyz",
            unit="deg",
            clock="episode_local",
            provenance="requested",
        ),
    )
    universe = _adapter_universe()

    evidence_variants = (
        {},
        {"episode_origin": 0.0},
    )

    # Every non-empty adapter subset, all source/target pairs, both evidence states.
    for mask in range(1, 1 << len(universe)):
        adapters = tuple(
            item for index, item in enumerate(universe) if mask & (1 << index)
        )
        for source, target, evidence in product(types, types, evidence_variants):
            oracle = oracle_decision(
                source,
                target,
                adapters,
                evidence=evidence,
                max_steps=5,
            )
            production = production_decision(
                source,
                target,
                adapters,
                evidence=evidence,
                max_steps=5,
            )
            cases += 1
            counts[oracle.status] += 1

            if oracle.status != production.status:
                violations.append(
                    "classification mismatch "
                    f"source={source!r} target={target!r} "
                    f"adapters={[a.name for a in adapters]!r} evidence={evidence!r} "
                    f"oracle={oracle!r} production={production!r}"
                )
                continue

            if oracle.status in {"accept", "unique-repair"}:
                if oracle.cost != production.cost:
                    violations.append(
                        "cost mismatch "
                        f"source={source!r} target={target!r} "
                        f"oracle={oracle.cost!r} production={production.cost!r}"
                    )
                if oracle.alternatives != production.alternatives:
                    violations.append(
                        "path mismatch "
                        f"source={source!r} target={target!r} "
                        f"oracle={oracle.alternatives!r} "
                        f"production={production.alternatives!r}"
                    )

            elif oracle.status == "ambiguous":
                if oracle.alternatives != production.alternatives:
                    violations.append(
                        "ambiguity set mismatch "
                        f"source={source!r} target={target!r} "
                        f"oracle={oracle.alternatives!r} "
                        f"production={production.alternatives!r}"
                    )

            elif oracle.status == "obligation":
                if oracle.obligations != production.obligations:
                    violations.append(
                        "obligation mismatch "
                        f"source={source!r} target={target!r} "
                        f"oracle={oracle.obligations!r} "
                        f"production={production.obligations!r}"
                    )

    # Non-forgeable provenance must never be repairable by pure adapters.
    source = SemanticTensorType(
        role="action",
        entity="joint_command",
        representation="joint_position",
        provenance="requested",
    )
    target = source.updated(provenance="executed")
    cases += 1
    try:
        SemanticAdapter(
            "fake-execution",
            requires={"provenance": "requested"},
            produces={"provenance": "executed"},
        )
    except ValueError:
        counts["reject"] += 1
    else:
        violations.append("production type system admitted a provenance-forging adapter")

    return ConformanceReport(
        cases_checked=cases,
        accept_cases=counts["accept"],
        unique_repair_cases=counts["unique-repair"],
        ambiguous_cases=counts["ambiguous"],
        obligation_cases=counts["obligation"],
        reject_cases=counts["reject"],
        violations=tuple(violations),
    )


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Differentially check production SemRepair search against an exhaustive oracle"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = verify_compiler_conformance()
    payload = {
        "checker": "semrepair-production-conformance/v0.1",
        "valid": report.valid,
        "cases_checked": report.cases_checked,
        "accept_cases": report.accept_cases,
        "unique_repair_cases": report.unique_repair_cases,
        "ambiguous_cases": report.ambiguous_cases,
        "obligation_cases": report.obligation_cases,
        "reject_cases": report.reject_cases,
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