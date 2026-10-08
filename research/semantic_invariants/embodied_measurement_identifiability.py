"""Counterfactual audit for semantic measurement identifiability.

A measurement channel can be perfectly deterministic and still be unable to
identify the semantic variable that a repair claim depends on.  This module
performs a conservative collision search:

- if two semantically distinct counterfactuals produce the same observable
  within tolerance, identifiability is disproved for that channel;
- failure to find a collision in a finite bank is *not* a proof of global
  identifiability.

The positive authority gate therefore still requires an external semantic
anchor through MeasurementQualificationCertificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Generic, Sequence, TypeVar


Semantic = TypeVar("Semantic")
Observable = TypeVar("Observable")


@dataclass(frozen=True)
class CounterfactualSemanticCase(Generic[Semantic]):
    case_id: str
    semantics: Semantic


@dataclass(frozen=True)
class MeasurementCollision:
    left_case: str
    right_case: str
    semantic_distance: float
    observable_distance: float


@dataclass(frozen=True)
class IdentifiabilityAudit:
    channel_id: str
    case_count: int
    pair_count: int
    collisions: tuple[MeasurementCollision, ...]
    disproved: bool
    digest: str

    @property
    def status(self) -> str:
        return "fail" if self.disproved else "undetermined"


def _digest(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def audit_measurement_identifiability(
    *,
    channel_id: str,
    cases: Sequence[CounterfactualSemanticCase[Semantic]],
    observe: Callable[[Semantic], Observable],
    semantic_distance: Callable[[Semantic, Semantic], float],
    observable_distance: Callable[[Observable, Observable], float],
    semantic_change_threshold: float = 0.0,
    observable_tolerance: float = 1.0e-12,
) -> IdentifiabilityAudit:
    """Search a finite counterfactual bank for semantic measurement collisions.

    A collision is a pair whose latent semantics differ materially while the
    measurement channel cannot distinguish their observables.

    The result is deliberately one-sided:
      collision found -> identifiability FAIL
      no collision     -> identifiability UNDETERMINED

    Positive qualification requires an independent semantic anchor elsewhere.
    """

    if not channel_id:
        raise ValueError("channel_id must be non-empty")
    if len(cases) < 2:
        raise ValueError("at least two counterfactual cases are required")
    if semantic_change_threshold < 0:
        raise ValueError("semantic_change_threshold must be non-negative")
    if observable_tolerance < 0:
        raise ValueError("observable_tolerance must be non-negative")

    ids = [item.case_id for item in cases]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("counterfactual case ids must be unique and non-empty")

    observed = {item.case_id: observe(item.semantics) for item in cases}
    collisions: list[MeasurementCollision] = []
    pair_count = 0

    for i, left in enumerate(cases):
        for right in cases[i + 1 :]:
            pair_count += 1
            sdist = float(semantic_distance(left.semantics, right.semantics))
            if sdist <= semantic_change_threshold:
                continue
            odist = float(
                observable_distance(
                    observed[left.case_id],
                    observed[right.case_id],
                )
            )
            if odist <= observable_tolerance:
                collisions.append(
                    MeasurementCollision(
                        left_case=left.case_id,
                        right_case=right.case_id,
                        semantic_distance=sdist,
                        observable_distance=odist,
                    )
                )

    payload = {
        "schema_version": 1,
        "channel_id": channel_id,
        "case_ids": ids,
        "case_count": len(cases),
        "pair_count": pair_count,
        "semantic_change_threshold": semantic_change_threshold,
        "observable_tolerance": observable_tolerance,
        "collisions": [
            {
                "left_case": item.left_case,
                "right_case": item.right_case,
                "semantic_distance": item.semantic_distance,
                "observable_distance": item.observable_distance,
            }
            for item in collisions
        ],
    }

    return IdentifiabilityAudit(
        channel_id=channel_id,
        case_count=len(cases),
        pair_count=pair_count,
        collisions=tuple(collisions),
        disproved=bool(collisions),
        digest=_digest(payload),
    )
