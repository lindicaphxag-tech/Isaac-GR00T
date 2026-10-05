"""Factorial analysis of interacting semantic repairs in embodied pipelines.

A physically wrong pipeline can look deceptively healthy when two semantic
faults partially cancel.  In that setting, repairing either boundary alone can
make end-to-end error worse even though the repair is locally correct.

This module treats candidate repairs as factors in a complete repair lattice.
It detects non-monotone repair edges, strict compensating bundles, and
higher-order interaction terms.  The output is evidence-bound by a canonical
digest so the classification cannot silently outlive the measurements that
produced it.

The analysis is intentionally metric-agnostic.  It only assumes the supplied
metric has a declared optimization direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from math import isfinite
from typing import Literal, Sequence


Objective = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class RepairOutcome:
    repairs: frozenset[str]
    value: float
    evidence_id: str

    def __post_init__(self) -> None:
        if not isfinite(self.value):
            raise ValueError("repair outcome must be finite")
        if not self.evidence_id:
            raise ValueError("repair outcome requires an evidence_id")


@dataclass(frozen=True)
class RepairEdgeViolation:
    before: tuple[str, ...]
    added_repair: str
    after: tuple[str, ...]
    before_loss: float
    after_loss: float

    @property
    def regression(self) -> float:
        return self.after_loss - self.before_loss


@dataclass(frozen=True)
class CompensatingRepairBundle:
    repairs: tuple[str, ...]
    baseline_loss: float
    repaired_loss: float
    proper_subset_losses: tuple[tuple[tuple[str, ...], float], ...]
    masking_mode: Literal["partial", "complete"]

    @property
    def closure_gain(self) -> float:
        return self.baseline_loss - self.repaired_loss

    @property
    def behaviorally_masked(self) -> bool:
        return self.masking_mode == "complete"


@dataclass(frozen=True)
class RepairInteractionCertificate:
    subject: str
    metric: str
    objective: Objective
    repairs: tuple[str, ...]
    outcomes: tuple[RepairOutcome, ...]
    edge_violations: tuple[RepairEdgeViolation, ...]
    compensating_bundles: tuple[CompensatingRepairBundle, ...]
    mobius_terms: tuple[tuple[tuple[str, ...], float], ...]
    digest: str

    @property
    def has_repair_paradox(self) -> bool:
        return bool(self.compensating_bundles)

    def interaction_for(self, repairs: Sequence[str]) -> float:
        key = tuple(sorted(repairs))
        for subset, value in self.mobius_terms:
            if subset == key:
                return value
        raise KeyError(key)


def _powerset(items: tuple[str, ...]) -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset(combo)
        for size in range(len(items) + 1)
        for combo in combinations(items, size)
    )


def _canonical_loss(value: float, objective: Objective) -> float:
    if objective == "minimize":
        return value
    if objective == "maximize":
        return -value
    raise ValueError(f"unknown objective: {objective!r}")


def _canonical_payload(
    *,
    subject: str,
    metric: str,
    objective: Objective,
    repairs: tuple[str, ...],
    outcomes: tuple[RepairOutcome, ...],
) -> bytes:
    payload = {
        "subject": subject,
        "metric": metric,
        "objective": objective,
        "repairs": list(repairs),
        "outcomes": [
            {
                "repairs": sorted(item.repairs),
                "value": item.value,
                "evidence_id": item.evidence_id,
            }
            for item in outcomes
        ],
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def analyze_repair_lattice(
    *,
    subject: str,
    metric: str,
    objective: Objective,
    repairs: Sequence[str],
    outcomes: Sequence[RepairOutcome],
    tolerance: float = 0.0,
) -> RepairInteractionCertificate:
    """Analyze a complete factorial repair lattice.

    A *strict compensating bundle* is a set of two or more repairs for which:

    1. applying the whole set is no worse than the baseline metric; and
    2. every non-empty proper subset is worse than baseline.

    Two modes are distinguished:

    - partial masking: the fully repaired set improves on a still-imperfect
      faulty baseline;
    - complete masking: the faulty baseline and fully repaired system are
      behaviorally equivalent within tolerance, while every partial repair is
      worse.

    The second case is especially dangerous: task success can make a
    semantically wrong pipeline indistinguishable from a correct one until one
    boundary is fixed, replaced, or reused in a different stack.

    The complete power set is required. Missing cells fail closed rather than
    being imputed.
    """

    if not subject:
        raise ValueError("subject must be non-empty")
    if not metric:
        raise ValueError("metric must be non-empty")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    repair_names = tuple(sorted(dict.fromkeys(repairs)))
    if not repair_names:
        raise ValueError("at least one repair is required")
    if len(repair_names) != len(tuple(repairs)):
        raise ValueError("repair names must be unique")

    expected = set(_powerset(repair_names))
    by_subset: dict[frozenset[str], RepairOutcome] = {}
    for item in outcomes:
        unknown = item.repairs - set(repair_names)
        if unknown:
            raise ValueError(f"outcome references unknown repairs: {sorted(unknown)}")
        if item.repairs in by_subset:
            raise ValueError(f"duplicate outcome for subset: {sorted(item.repairs)}")
        by_subset[item.repairs] = item

    missing = expected - set(by_subset)
    extra = set(by_subset) - expected
    if missing or extra:
        raise ValueError(
            "repair lattice must be complete; "
            f"missing={sorted(map(sorted, missing))}, "
            f"extra={sorted(map(sorted, extra))}"
        )

    ordered_outcomes = tuple(
        by_subset[subset]
        for subset in sorted(expected, key=lambda s: (len(s), tuple(sorted(s))))
    )
    loss = {
        subset: _canonical_loss(item.value, objective)
        for subset, item in by_subset.items()
    }
    baseline = loss[frozenset()]

    edge_violations: list[RepairEdgeViolation] = []
    for subset in expected:
        for repair in repair_names:
            if repair in subset:
                continue
            after = subset | {repair}
            if loss[after] > loss[subset] + tolerance:
                edge_violations.append(
                    RepairEdgeViolation(
                        before=tuple(sorted(subset)),
                        added_repair=repair,
                        after=tuple(sorted(after)),
                        before_loss=loss[subset],
                        after_loss=loss[after],
                    )
                )

    bundles: list[CompensatingRepairBundle] = []
    for subset in expected:
        if len(subset) < 2:
            continue
        if loss[subset] > baseline + tolerance:
            continue

        proper_nonempty = [
            other
            for other in expected
            if other and other < subset
        ]
        if proper_nonempty and all(
            loss[other] > baseline + tolerance
            for other in proper_nonempty
        ):
            bundles.append(
                CompensatingRepairBundle(
                    repairs=tuple(sorted(subset)),
                    baseline_loss=baseline,
                    repaired_loss=loss[subset],
                    proper_subset_losses=tuple(
                        (tuple(sorted(other)), loss[other])
                        for other in sorted(
                            proper_nonempty,
                            key=lambda s: (len(s), tuple(sorted(s))),
                        )
                    ),
                    masking_mode=(
                        "partial"
                        if loss[subset] < baseline - tolerance
                        else "complete"
                    ),
                )
            )

    # Möbius transform of the canonical-loss set function.  Order-1 terms are
    # main effects; order >=2 terms quantify non-additive interaction.
    mobius: list[tuple[tuple[str, ...], float]] = []
    for subset in sorted(expected, key=lambda s: (len(s), tuple(sorted(s)))):
        if not subset:
            continue
        total = 0.0
        subset_items = tuple(sorted(subset))
        for size in range(len(subset_items) + 1):
            for combo in combinations(subset_items, size):
                inner = frozenset(combo)
                sign = -1.0 if (len(subset) - len(inner)) % 2 else 1.0
                total += sign * loss[inner]
        mobius.append((subset_items, total))

    digest = sha256(
        _canonical_payload(
            subject=subject,
            metric=metric,
            objective=objective,
            repairs=repair_names,
            outcomes=ordered_outcomes,
        )
    ).hexdigest()

    return RepairInteractionCertificate(
        subject=subject,
        metric=metric,
        objective=objective,
        repairs=repair_names,
        outcomes=ordered_outcomes,
        edge_violations=tuple(
            sorted(
                edge_violations,
                key=lambda item: (item.before, item.added_repair),
            )
        ),
        compensating_bundles=tuple(
            sorted(bundles, key=lambda item: (len(item.repairs), item.repairs))
        ),
        mobius_terms=tuple(mobius),
        digest=digest,
    )
