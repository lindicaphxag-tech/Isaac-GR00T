"""Independent small-step calculus model for SemRepair.

This file is intentionally separate from the production compiler.  It is an
executable formal model used to test the intended preservation/progress shape
before mechanizing the core in a proof assistant.

The model distinguishes:
- pure representation steps, which must preserve a canonical meaning token;
- evidence-gated context steps, which also preserve canonical meaning;
- event refinements, which may create non-forgeable semantics only when a
  receipt is present.

It does not prove the full Python implementation correct.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Sequence


NON_FORGEABLE = frozenset({"provenance", "freshness"})


@dataclass(frozen=True)
class CoreType:
    representation: str | None = None
    unit: str | None = None
    clock: str | None = None
    scope: str | None = None
    provenance: str | None = None
    freshness: str | None = None
    ordering: str | None = None

    def update(self, changes: Mapping[str, str]) -> "CoreType":
        unknown = set(changes) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown core field(s): {sorted(unknown)}")
        return replace(self, **dict(changes))


@dataclass(frozen=True)
class CoreValue:
    semantic_type: CoreType
    meaning_token: str
    event_lineage: tuple[str, ...] = ()


@dataclass(frozen=True)
class PureRule:
    name: str
    requires: Mapping[str, str]
    produces: Mapping[str, str]
    evidence_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field, value in self.produces.items():
            if field in NON_FORGEABLE and self.requires.get(field) != value:
                raise ValueError(
                    f"pure rule {self.name!r} cannot forge {field!r}"
                )


@dataclass(frozen=True)
class EventRule:
    name: str
    requires: Mapping[str, str]
    produces: Mapping[str, str]
    receipt_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.receipt_keys:
            raise ValueError("event rule requires receipt evidence")


class StepKind(str, Enum):
    PURE = "pure"
    EVENT = "event"


@dataclass(frozen=True)
class StepResult:
    kind: StepKind
    rule: str
    before: CoreValue
    after: CoreValue
    evidence_used: tuple[str, ...]


class CoreStepBlocked(RuntimeError):
    def __init__(self, message: str, *, obligations: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.obligations = tuple(obligations)


def _matches(value: CoreValue, requires: Mapping[str, str]) -> bool:
    return all(
        getattr(value.semantic_type, field) == expected
        for field, expected in requires.items()
    )


def pure_step(
    value: CoreValue,
    rule: PureRule,
    *,
    evidence: Mapping[str, object] | None = None,
) -> StepResult:
    if not _matches(value, rule.requires):
        raise CoreStepBlocked(f"pure rule {rule.name!r} precondition mismatch")
    available = set((evidence or {}).keys())
    missing = tuple(key for key in rule.evidence_keys if key not in available)
    if missing:
        raise CoreStepBlocked(
            f"pure rule {rule.name!r} lacks evidence",
            obligations=missing,
        )

    after = CoreValue(
        semantic_type=value.semantic_type.update(rule.produces),
        meaning_token=value.meaning_token,
        event_lineage=value.event_lineage,
    )
    return StepResult(
        kind=StepKind.PURE,
        rule=rule.name,
        before=value,
        after=after,
        evidence_used=tuple(rule.evidence_keys),
    )


def event_step(
    value: CoreValue,
    rule: EventRule,
    *,
    receipt: Mapping[str, object],
) -> StepResult:
    if not _matches(value, rule.requires):
        raise CoreStepBlocked(f"event rule {rule.name!r} precondition mismatch")
    missing = tuple(key for key in rule.receipt_keys if key not in receipt)
    if missing:
        raise CoreStepBlocked(
            f"event rule {rule.name!r} lacks receipt",
            obligations=missing,
        )

    receipt_id = "|".join(f"{key}={receipt[key]!r}" for key in rule.receipt_keys)
    after = CoreValue(
        semantic_type=value.semantic_type.update(rule.produces),
        # An event refinement denotes a new physical event/value.  The lineage,
        # not silent relabeling, explains the semantic change.
        meaning_token=f"{value.meaning_token}->{rule.name}[{receipt_id}]",
        event_lineage=value.event_lineage + (rule.name,),
    )
    return StepResult(
        kind=StepKind.EVENT,
        rule=rule.name,
        before=value,
        after=after,
        evidence_used=tuple(rule.receipt_keys),
    )


def pure_preservation_holds(step: StepResult) -> bool:
    if step.kind is not StepKind.PURE:
        raise ValueError("preservation predicate expects a pure step")
    return (
        step.before.meaning_token == step.after.meaning_token
        and step.before.event_lineage == step.after.event_lineage
    )


def event_nonforgeability_holds(step: StepResult) -> bool:
    if step.kind is not StepKind.EVENT:
        raise ValueError("event predicate expects an event step")

    changed = {
        field
        for field in NON_FORGEABLE
        if getattr(step.before.semantic_type, field)
        != getattr(step.after.semantic_type, field)
    }
    if not changed:
        return True
    return bool(step.evidence_used) and len(step.after.event_lineage) == len(
        step.before.event_lineage
    ) + 1


class BoundaryStatus(str, Enum):
    ACCEPT = "accept"
    REPAIR = "repair"
    OBLIGATION = "obligation"
    AMBIGUOUS = "ambiguous"
    REJECT = "reject"


@dataclass(frozen=True)
class BoundaryDecision:
    status: BoundaryStatus
    detail: str = ""


def decide_boundary(
    source: CoreType,
    target: CoreType,
    *,
    repair_paths: int,
    missing_obligations: Sequence[str] = (),
    inference_survivors: int = 1,
) -> BoundaryDecision:
    """Independent progress classifier for one abstract boundary."""

    constrained = [
        field
        for field in source.__dataclass_fields__
        if getattr(target, field) is not None
    ]
    if all(getattr(source, field) == getattr(target, field) for field in constrained):
        return BoundaryDecision(BoundaryStatus.ACCEPT)

    if inference_survivors > 1:
        return BoundaryDecision(BoundaryStatus.AMBIGUOUS)

    if missing_obligations:
        return BoundaryDecision(
            BoundaryStatus.OBLIGATION,
            ",".join(sorted(missing_obligations)),
        )

    if repair_paths == 1:
        return BoundaryDecision(BoundaryStatus.REPAIR)

    if repair_paths > 1:
        return BoundaryDecision(BoundaryStatus.AMBIGUOUS)

    return BoundaryDecision(BoundaryStatus.REJECT)
