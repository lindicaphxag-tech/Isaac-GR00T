"""Triangulated physical-effect evidence for Semantic Effect Commit.

A single acknowledgement or cache may be wrong for the same reason as the
executor. This module resolves physical completion from independent semantic
observation planes and then feeds the result into the ambiguity-aware commit
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

from .semantic_effect_commit import (
    CommitOutcome,
    EffectEvidence,
    SemanticEffectIntent,
    SemanticEffectRuntime,
)

State = TypeVar("State")


@dataclass(frozen=True)
class EffectObservation(Generic[State]):
    name: str
    plane: str
    observed_state: State
    fresh: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("observation name must be non-empty")
        if not self.plane:
            raise ValueError("observation plane must be non-empty")


@dataclass(frozen=True)
class EffectEvidenceSummary:
    fresh_planes: tuple[str, ...]
    postcondition_planes: tuple[str, ...]
    precondition_planes: tuple[str, ...]
    contradictory_planes: tuple[str, ...]


def summarize_effect_evidence(
    intent: SemanticEffectIntent[State],
    observations: Sequence[EffectObservation[State]],
) -> EffectEvidenceSummary:
    """Evaluate pre/postconditions once per independent observation plane."""
    by_plane: dict[str, list[tuple[bool, bool]]] = {}
    for observation in observations:
        if not observation.fresh:
            continue
        by_plane.setdefault(observation.plane, []).append(
            (
                bool(intent.precondition(observation.observed_state)),
                bool(intent.postcondition(observation.observed_state)),
            )
        )

    fresh_planes = []
    post_planes = []
    pre_planes = []
    contradictory = []

    for plane, values in sorted(by_plane.items()):
        fresh_planes.append(plane)
        unique = set(values)
        if len(unique) != 1:
            contradictory.append(plane)
            continue

        pre_holds, post_holds = values[0]
        if pre_holds and post_holds:
            contradictory.append(plane)
        elif post_holds:
            post_planes.append(plane)
        elif pre_holds:
            pre_planes.append(plane)

    return EffectEvidenceSummary(
        fresh_planes=tuple(fresh_planes),
        postcondition_planes=tuple(post_planes),
        precondition_planes=tuple(pre_planes),
        contradictory_planes=tuple(contradictory),
    )


def resolve_with_triangulated_evidence(
    runtime: SemanticEffectRuntime[State],
    intent: SemanticEffectIntent[State],
    observations: Sequence[EffectObservation[State]],
    *,
    min_commit_planes: int = 2,
    min_abort_planes: int = 1,
    executor_acknowledged_success: bool = False,
    executor_acknowledged_abort: bool = False,
) -> tuple[CommitOutcome, EffectEvidenceSummary]:
    """Resolve an effect using independent physical-semantic evidence."""
    if min_commit_planes < 1:
        raise ValueError("min_commit_planes must be positive")
    if min_abort_planes < 1:
        raise ValueError("min_abort_planes must be positive")
    if not observations:
        raise ValueError("at least one observation is required")

    summary = summarize_effect_evidence(intent, observations)

    if len(summary.postcondition_planes) >= min_commit_planes:
        representative = next(
            item.observed_state
            for item in observations
            if item.fresh and item.plane in summary.postcondition_planes
        )
        outcome = runtime.resolve(
            intent,
            EffectEvidence(
                observed_state=representative,
                executor_acknowledged_success=executor_acknowledged_success,
                observation_fresh=True,
            ),
        )
        return outcome, summary

    if (
        executor_acknowledged_abort
        and len(summary.precondition_planes) >= min_abort_planes
    ):
        representative = next(
            item.observed_state
            for item in observations
            if item.fresh and item.plane in summary.precondition_planes
        )
        outcome = runtime.resolve(
            intent,
            EffectEvidence(
                observed_state=representative,
                executor_acknowledged_abort=True,
                observation_fresh=True,
            ),
        )
        return outcome, summary

    outcome = runtime.resolve(
        intent,
        EffectEvidence(
            observed_state=observations[0].observed_state,
            executor_acknowledged_success=False,
            executor_acknowledged_abort=False,
            observation_fresh=False,
        ),
    )
    return outcome, summary