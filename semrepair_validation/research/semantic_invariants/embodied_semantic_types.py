"""Semantic type-and-effect prototype for embodied learning pipelines.

This module models semantic metadata erased by ordinary shape/dtype checks:
coordinate frame, action representation, physical unit, time basis, episode
scope, provenance, ordering, and embodiment identity.

The central safety rule is that semantic repair is not ordinary casting:

- pure representation/unit conversions may be synthesized when certified;
- context-sensitive conversions require explicit evidence;
- event-owned semantics such as requested -> executed provenance are
  non-forgeable and can only be produced by an explicit owner-boundary event.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from heapq import heappop, heappush
from itertools import count
from typing import Mapping, Sequence


SEMANTIC_FIELDS = (
    "role",
    "entity",
    "frame",
    "representation",
    "mode",
    "convention",
    "unit",
    "clock",
    "scope",
    "freshness",
    "provenance",
    "ordering",
    "embodiment",
)

NON_FORGEABLE_FIELDS = frozenset({"provenance", "freshness"})


def _exact_adapter_cost(value: float) -> Decimal:
    """Map the public numeric cost to an exact decimal search weight."""
    cost = Decimal(str(value))
    if not cost.is_finite() or cost < 0:
        raise ValueError("adapter cost must be finite and non-negative")
    return cost


@dataclass(frozen=True)
class SemanticTensorType:
    role: str
    entity: str
    frame: str | None = None
    representation: str | None = None
    mode: str | None = None
    convention: str | None = None
    unit: str | None = None
    clock: str | None = None
    scope: str | None = None
    freshness: str | None = None
    provenance: str | None = None
    ordering: str | None = None
    embodiment: str | None = None

    def updated(self, **changes: str | None) -> "SemanticTensorType":
        unknown = set(changes) - set(SEMANTIC_FIELDS)
        if unknown:
            raise ValueError(f"unknown semantic field(s): {sorted(unknown)}")
        return replace(self, **changes)


@dataclass(frozen=True)
class SemanticMismatch:
    field: str
    actual: str | None
    expected: str | None


def semantic_mismatches(
    actual: SemanticTensorType,
    expected: SemanticTensorType,
) -> tuple[SemanticMismatch, ...]:
    """Return mismatches; unspecified consumer axes are unconstrained."""
    out: list[SemanticMismatch] = []
    for field in SEMANTIC_FIELDS:
        want = getattr(expected, field)
        if want is None:
            continue
        have = getattr(actual, field)
        if have != want:
            out.append(SemanticMismatch(field, have, want))
    return tuple(out)


def is_assignable(actual: SemanticTensorType, expected: SemanticTensorType) -> bool:
    return not semantic_mismatches(actual, expected)


@dataclass(frozen=True)
class SemanticAdapter:
    """One explicit semantics-preserving conversion owned by the host stack.

    witness_keys names concrete evidence required by context-sensitive
    conversions such as an episode origin, frame transform, or ordering map.

    proof_required is retained for compatibility. If it is set without explicit
    witness keys, the adapter gets an implicit proof:<adapter-name> obligation.
    A boolean allow flag never discharges that obligation.
    """

    name: str
    requires: Mapping[str, str]
    produces: Mapping[str, str]
    cost: float = 1.0
    effects: tuple[str, ...] = ()
    proof_required: bool = False
    witness_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = (set(self.requires) | set(self.produces)) - set(SEMANTIC_FIELDS)
        if unknown:
            raise ValueError(f"unknown semantic field(s): {sorted(unknown)}")
        _exact_adapter_cost(self.cost)

        for field in NON_FORGEABLE_FIELDS & set(self.produces):
            before = self.requires.get(field)
            after = self.produces[field]
            if before != after:
                raise ValueError(
                    f"adapter {self.name!r} cannot forge non-forgeable field "
                    f"{field!r}: {before!r}->{after!r}; use SemanticEventTransition"
                )

    @property
    def required_evidence(self) -> tuple[str, ...]:
        if self.witness_keys:
            return tuple(dict.fromkeys(self.witness_keys))
        if self.proof_required:
            return (f"proof:{self.name}",)
        return ()

    def applies_to(self, value_type: SemanticTensorType) -> bool:
        return all(getattr(value_type, key) == value for key, value in self.requires.items())

    def apply_type(self, value_type: SemanticTensorType) -> SemanticTensorType:
        if not self.applies_to(value_type):
            raise ValueError(f"adapter {self.name!r} does not apply")
        return value_type.updated(**dict(self.produces))


@dataclass(frozen=True)
class SemanticEventTransition:
    """Owner-boundary transition that may create non-forgeable semantics."""

    name: str
    requires: Mapping[str, str]
    produces: Mapping[str, str]
    receipt_keys: tuple[str, ...]
    effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = (set(self.requires) | set(self.produces)) - set(SEMANTIC_FIELDS)
        if unknown:
            raise ValueError(f"unknown semantic field(s): {sorted(unknown)}")
        if not self.receipt_keys:
            raise ValueError("event transitions require at least one receipt key")

    def applies_to(self, value_type: SemanticTensorType) -> bool:
        return all(getattr(value_type, key) == value for key, value in self.requires.items())


@dataclass(frozen=True)
class EventTransitionResult:
    transition: str
    result: SemanticTensorType
    evidence_used: tuple[str, ...]
    effects: tuple[str, ...]


class MissingSemanticEvidence(RuntimeError):
    def __init__(self, message: str, *, obligations: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.obligations = tuple(obligations)


def apply_event_transition(
    value_type: SemanticTensorType,
    transition: SemanticEventTransition,
    *,
    receipt: Mapping[str, object],
) -> EventTransitionResult:
    """Apply a runtime event transition only with explicit owner evidence."""
    if not transition.applies_to(value_type):
        raise ValueError(f"event transition {transition.name!r} does not apply")

    missing = tuple(key for key in transition.receipt_keys if key not in receipt)
    if missing:
        raise MissingSemanticEvidence(
            f"event transition {transition.name!r} is missing owner evidence: {missing!r}",
            obligations=missing,
        )

    return EventTransitionResult(
        transition=transition.name,
        result=value_type.updated(**dict(transition.produces)),
        evidence_used=tuple(transition.receipt_keys),
        effects=tuple(transition.effects),
    )


@dataclass(frozen=True)
class AdapterPlan:
    source: SemanticTensorType
    target: SemanticTensorType
    adapters: tuple[SemanticAdapter, ...]
    result: SemanticTensorType
    total_cost: float
    effects: tuple[str, ...]
    evidence_used: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return is_assignable(self.result, self.target)


class NoSemanticRepair(RuntimeError):
    def __init__(self, message: str, *, proof_obligations: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.proof_obligations = tuple(proof_obligations)


def synthesize_adapter_plan(
    source: SemanticTensorType,
    target: SemanticTensorType,
    adapters: Sequence[SemanticAdapter],
    *,
    evidence: Mapping[str, object] | None = None,
    allow_proof_required: bool = False,
    max_steps: int = 6,
) -> AdapterPlan:
    """Find the least-cost explicit semantic repair path with Dijkstra search.

    allow_proof_required is retained only for call-site compatibility. It no
    longer authorizes a context-dependent repair by itself: concrete evidence
    must discharge every adapter obligation.
    """

    del allow_proof_required
    available_evidence = set((evidence or {}).keys())

    if is_assignable(source, target):
        return AdapterPlan(source, target, (), source, 0.0, (), ())

    serial = count()
    queue = [(Decimal("0"), 0, next(serial), source, (), ())]
    best: dict[SemanticTensorType, Decimal] = {source: Decimal("0")}
    blocked_obligations: set[str] = set()

    while queue:
        cost, steps, _, current, path, effects = heappop(queue)
        if cost != best.get(current):
            continue
        if is_assignable(current, target):
            evidence_used = tuple(
                dict.fromkeys(
                    key for adapter in path for key in adapter.required_evidence
                )
            )
            return AdapterPlan(
                source,
                target,
                path,
                current,
                float(cost),
                effects,
                evidence_used,
            )
        if steps >= max_steps:
            continue

        for adapter in adapters:
            if not adapter.applies_to(current):
                continue

            missing = set(adapter.required_evidence) - available_evidence
            if missing:
                blocked_obligations.update(missing)
                continue

            nxt = adapter.apply_type(current)
            if nxt == current:
                continue
            new_cost = cost + _exact_adapter_cost(adapter.cost)
            if new_cost >= best.get(nxt, Decimal("Infinity")):
                continue
            best[nxt] = new_cost
            heappush(
                queue,
                (
                    new_cost,
                    steps + 1,
                    next(serial),
                    nxt,
                    path + (adapter,),
                    effects + adapter.effects,
                ),
            )

    mismatch = ", ".join(
        f"{m.field}: {m.actual!r}->{m.expected!r}"
        for m in semantic_mismatches(source, target)
    )
    obligations = tuple(sorted(blocked_obligations))
    suffix = f"; missing evidence={obligations!r}" if obligations else ""
    raise NoSemanticRepair(
        f"no explicit semantic repair path ({mismatch}){suffix}",
        proof_obligations=obligations,
    )


class AmbiguousSemanticRepair(NoSemanticRepair):
    """Raised when multiple equal-cost explicit adapter plans remain valid."""

    def __init__(
        self,
        message: str,
        *,
        alternatives: Sequence[Sequence[str]],
    ) -> None:
        super().__init__(message)
        self.alternatives = tuple(tuple(path) for path in alternatives)


def synthesize_unique_adapter_plan(
    source: SemanticTensorType,
    target: SemanticTensorType,
    adapters: Sequence[SemanticAdapter],
    *,
    evidence: Mapping[str, object] | None = None,
    max_steps: int = 6,
) -> AdapterPlan:
    """Find the unique least-cost semantic adapter path or fail closed.

    Unlike synthesize_adapter_plan(), this research-grade variant retains every
    equal-cost path that reaches a type assignable to the consumer contract.
    Enumeration order is never used as semantic evidence.
    """

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    available_evidence = set((evidence or {}).keys())
    if is_assignable(source, target):
        return AdapterPlan(source, target, (), source, 0.0, (), ())

    serial = count()
    queue = [
        (Decimal("0"), 0, next(serial), source, (), frozenset((source,)))
    ]
    best: dict[SemanticTensorType, Decimal] = {source: Decimal("0")}
    blocked_obligations: set[str] = set()
    seen_paths: set[tuple[SemanticTensorType, tuple[str, ...]]] = {(source, ())}
    minimum_cost: Decimal | None = None
    solutions: list[AdapterPlan] = []

    while queue:
        cost, steps, _, current, path, visited_types = heappop(queue)

        if minimum_cost is not None and cost > minimum_cost:
            break

        known = best.get(current)
        if known is not None and cost > known:
            continue

        if is_assignable(current, target):
            if minimum_cost is None:
                minimum_cost = cost
            if cost == minimum_cost:
                effects = tuple(
                    effect for adapter in path for effect in adapter.effects
                )
                evidence_used = tuple(
                    dict.fromkeys(
                        key
                        for adapter in path
                        for key in adapter.required_evidence
                    )
                )
                plan = AdapterPlan(
                    source,
                    target,
                    path,
                    current,
                    float(cost),
                    effects,
                    evidence_used,
                )
                signature = tuple(adapter.name for adapter in path)
                if signature not in {
                    tuple(adapter.name for adapter in item.adapters)
                    for item in solutions
                }:
                    solutions.append(plan)
            continue

        if steps >= max_steps:
            continue

        for adapter in adapters:
            if not adapter.applies_to(current):
                continue

            missing = set(adapter.required_evidence) - available_evidence
            if missing:
                blocked_obligations.update(missing)
                continue

            nxt = adapter.apply_type(current)
            if nxt == current or nxt in visited_types:
                # Canonical repairs are simple paths in semantic-type space.
                # With non-negative costs, removing a repeated-type cycle never
                # increases cost or changes the remaining suffix semantics.
                continue

            new_cost = cost + _exact_adapter_cost(adapter.cost)
            if minimum_cost is not None and new_cost > minimum_cost:
                continue

            previous = best.get(nxt)
            if previous is not None and new_cost > previous:
                continue
            if previous is None or new_cost < previous:
                best[nxt] = new_cost

            new_path = path + (adapter,)
            signature = (nxt, tuple(item.name for item in new_path))
            if signature in seen_paths:
                continue
            seen_paths.add(signature)
            heappush(
                queue,
                (
                    new_cost,
                    steps + 1,
                    next(serial),
                    nxt,
                    new_path,
                    visited_types | frozenset((nxt,)),
                ),
            )

    solutions.sort(
        key=lambda item: tuple(adapter.name for adapter in item.adapters)
    )

    if len(solutions) == 1:
        return solutions[0]

    if len(solutions) > 1:
        alternatives = tuple(
            tuple(adapter.name for adapter in plan.adapters)
            for plan in solutions
        )
        raise AmbiguousSemanticRepair(
            "multiple equal-cost semantic repair paths remain valid; "
            "acquire a discriminating witness before installation",
            alternatives=alternatives,
        )

    mismatch = ", ".join(
        f"{m.field}: {m.actual!r}->{m.expected!r}"
        for m in semantic_mismatches(source, target)
    )
    obligations = tuple(sorted(blocked_obligations))
    suffix = f"; missing evidence={obligations!r}" if obligations else ""
    raise NoSemanticRepair(
        f"no unique explicit semantic repair path ({mismatch}){suffix}",
        proof_obligations=obligations,
    )


@dataclass(frozen=True)
class ComponentSignature:
    name: str
    accepts: SemanticTensorType
    produces: SemanticTensorType


@dataclass(frozen=True)
class PipelineStep:
    component: str
    input_type: SemanticTensorType
    repair: AdapterPlan
    output_type: SemanticTensorType


@dataclass(frozen=True)
class PipelineTypingResult:
    start: SemanticTensorType
    steps: tuple[PipelineStep, ...]
    final: SemanticTensorType
    effects: tuple[str, ...]
    evidence_used: tuple[str, ...]


def type_and_repair_pipeline(
    start: SemanticTensorType,
    components: Sequence[ComponentSignature],
    adapters: Sequence[SemanticAdapter],
    *,
    evidence: Mapping[str, object] | None = None,
    allow_proof_required: bool = False,
) -> PipelineTypingResult:
    """Type-check component composition and synthesize local adapters."""
    current = start
    typed_steps: list[PipelineStep] = []
    accumulated_effects: list[str] = []
    accumulated_evidence: list[str] = []

    for component in components:
        plan = synthesize_adapter_plan(
            current,
            component.accepts,
            adapters,
            evidence=evidence,
            allow_proof_required=allow_proof_required,
        )
        accumulated_effects.extend(plan.effects)
        accumulated_evidence.extend(plan.evidence_used)
        typed_steps.append(
            PipelineStep(
                component=component.name,
                input_type=plan.result,
                repair=plan,
                output_type=component.produces,
            )
        )
        current = component.produces

    return PipelineTypingResult(
        start=start,
        steps=tuple(typed_steps),
        final=current,
        effects=tuple(accumulated_effects),
        evidence_used=tuple(dict.fromkeys(accumulated_evidence)),
    )