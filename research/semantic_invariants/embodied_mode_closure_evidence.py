"""Evidence-triangulated configuration/runtime semantic closure.

A declared runtime mode is not considered closed merely because it falls
through a default branch.  Intentional baseline/no-op semantics need evidence
from independent planes, such as real configuration use and documentation.

This prevents a shallow enum-reference checker from misclassifying both
ABSOLUTE and DELTA in GR00T-style action representations.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping, Sequence

from .embodied_mode_closure_source import (
    EnumModeClosure,
    extract_enum_members,
    extract_runtime_handled_modes,
)


@dataclass(frozen=True)
class ModeEvidence:
    mode: str
    planes: tuple[str, ...]

    @property
    def independent_planes(self) -> int:
        return len(set(self.planes))


@dataclass(frozen=True)
class TriangulatedModeClosure:
    closure: EnumModeClosure
    usage_counts: tuple[tuple[str, int], ...]
    default_evidence: tuple[ModeEvidence, ...]
    baseline_candidates: tuple[str, ...]
    evidence_insufficient: tuple[str, ...]


def _enum_member(node: ast.AST, enum_name: str) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == enum_name
    ):
        return node.attr
    return None


def extract_enum_usage_counts(
    sources: Sequence[str],
    enum_name: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in sources:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            member = _enum_member(node, enum_name)
            if member is not None:
                counts[member] = counts.get(member, 0) + 1
    return counts


def infer_triangulated_mode_closure(
    *,
    enum_source: str,
    runtime_source: str,
    usage_sources: Sequence[str],
    enum_name: str,
    baseline_evidence: Mapping[str, Sequence[str]] | None = None,
    explicit_rejected: Sequence[str] = (),
    minimum_default_planes: int = 2,
) -> TriangulatedModeClosure:
    """Infer closure while requiring independent evidence for fall-through defaults."""

    if minimum_default_planes < 1:
        raise ValueError("minimum_default_planes must be >= 1")

    declared = extract_enum_members(enum_source, enum_name)
    handled = extract_runtime_handled_modes(runtime_source, enum_name)
    usage = extract_enum_usage_counts(usage_sources, enum_name)
    baseline_evidence = dict(baseline_evidence or {})

    unknown_evidence = set(baseline_evidence) - set(declared)
    if unknown_evidence:
        raise ValueError(
            "baseline evidence mentions undeclared modes: "
            + ", ".join(sorted(unknown_evidence))
        )

    evidence_rows = tuple(
        ModeEvidence(mode, tuple(dict.fromkeys(str(x) for x in planes)))
        for mode, planes in sorted(baseline_evidence.items())
    )
    explicit_default = {
        row.mode
        for row in evidence_rows
        if row.independent_planes >= minimum_default_planes
    }

    rejected = frozenset(explicit_rejected)
    unknown_rejected = rejected - declared
    if unknown_rejected:
        raise ValueError(
            "rejected modes are undeclared: " + ", ".join(sorted(unknown_rejected))
        )

    unresolved = declared - handled - explicit_default - rejected
    baseline_candidates = tuple(
        sorted(
            mode
            for mode in declared - handled
            if usage.get(mode, 0) > 0 and mode not in explicit_default
        )
    )
    evidence_insufficient = tuple(
        sorted(
            row.mode
            for row in evidence_rows
            if row.independent_planes < minimum_default_planes
        )
    )

    closure = EnumModeClosure(
        enum_name=enum_name,
        declared=declared,
        handled=handled & declared,
        explicit_default=frozenset(explicit_default),
        rejected=rejected,
        unresolved=frozenset(unresolved),
    )
    return TriangulatedModeClosure(
        closure=closure,
        usage_counts=tuple(sorted((mode, usage.get(mode, 0)) for mode in declared)),
        default_evidence=evidence_rows,
        baseline_candidates=baseline_candidates,
        evidence_insufficient=evidence_insufficient,
    )