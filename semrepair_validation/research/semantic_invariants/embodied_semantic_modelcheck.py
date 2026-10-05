"""Finite model checks for the embodied semantic compiler core.

This is not a substitute for a paper proof. It is an executable oracle for the
prototype's two central finite-state claims:

1. adapter synthesis returns a minimum-cost semantic repair path;
2. pure adapters cannot create non-forgeable provenance/freshness semantics.

The brute-force search is intentionally independent of the production Dijkstra
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

from .embodied_semantic_types import (
    NON_FORGEABLE_FIELDS,
    NoSemanticRepair,
    SemanticAdapter,
    SemanticTensorType,
    is_assignable,
    synthesize_adapter_plan,
)


@dataclass(frozen=True)
class ModelCheckReport:
    checked_pairs: int
    reachable_pairs: int
    unreachable_pairs: int
    optimality_violations: int
    nonforgeable_violations: int

    @property
    def passed(self) -> bool:
        return (
            self.optimality_violations == 0
            and self.nonforgeable_violations == 0
        )


def _bruteforce_min_cost(
    source: SemanticTensorType,
    target: SemanticTensorType,
    adapters: Sequence[SemanticAdapter],
    *,
    max_steps: int,
) -> float | None:
    if is_assignable(source, target):
        return 0.0

    best: float | None = None

    def visit(
        current: SemanticTensorType,
        cost: float,
        steps: int,
        visited: frozenset[SemanticTensorType],
    ) -> None:
        nonlocal best
        if best is not None and cost >= best:
            return
        if is_assignable(current, target):
            best = cost
            return
        if steps >= max_steps:
            return

        for adapter in adapters:
            if adapter.required_evidence:
                continue
            if not adapter.applies_to(current):
                continue
            nxt = adapter.apply_type(current)
            if nxt == current or nxt in visited:
                continue
            visit(
                nxt,
                cost + adapter.cost,
                steps + 1,
                visited | frozenset((nxt,)),
            )

    visit(source, 0.0, 0, frozenset((source,)))
    return best


def check_repair_optimality(
    semantic_types: Sequence[SemanticTensorType],
    adapters: Sequence[SemanticAdapter],
    *,
    max_steps: int = 5,
) -> ModelCheckReport:
    checked = 0
    reachable = 0
    unreachable = 0
    optimality_violations = 0

    for source, target in product(semantic_types, repeat=2):
        checked += 1
        oracle = _bruteforce_min_cost(
            source,
            target,
            adapters,
            max_steps=max_steps,
        )
        try:
            plan = synthesize_adapter_plan(
                source,
                target,
                adapters,
                max_steps=max_steps,
            )
        except NoSemanticRepair:
            if oracle is None:
                unreachable += 1
            else:
                optimality_violations += 1
            continue

        if oracle is None:
            optimality_violations += 1
            continue

        reachable += 1
        if abs(plan.total_cost - oracle) > 1.0e-12:
            optimality_violations += 1

    nonforgeable_violations = 0
    for adapter in adapters:
        for field in NON_FORGEABLE_FIELDS:
            before = adapter.requires.get(field)
            after = adapter.produces.get(field)
            if after is not None and before != after:
                nonforgeable_violations += 1

    return ModelCheckReport(
        checked_pairs=checked,
        reachable_pairs=reachable,
        unreachable_pairs=unreachable,
        optimality_violations=optimality_violations,
        nonforgeable_violations=nonforgeable_violations,
    )
