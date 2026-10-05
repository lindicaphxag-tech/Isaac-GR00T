"""Independent proof checker for pure embodied semantic adapter plans.

The compiler is treated as an untrusted producer.  This verifier does not call
the compiler's Dijkstra/repair routines.  It independently replays the selected
adapter path and performs a bounded exhaustive search to check minimum-cost
uniqueness under the exact registry/evidence snapshot committed by the
certificate.

Scope: this kernel verifies the *pure adapter* segment from refined_source to
target.  Runtime inference/refinement evidence has its own owner-boundary trust
requirements and is deliberately not forged or re-derived here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .embodied_compilation_certificate import (
    SemanticCompilationCertificate,
    adapter_registry_digest,
    evidence_identity_digest,
    verify_compilation_certificate,
)
from .embodied_semantic_types import (
    SemanticAdapter,
    SemanticTensorType,
    is_assignable,
)


@dataclass(frozen=True)
class CompilationProofCheck:
    valid: bool
    reasons: tuple[str, ...]
    selected_path: tuple[str, ...]
    selected_cost: float | None
    cheaper_paths: tuple[tuple[str, ...], ...]
    equal_cost_alternatives: tuple[tuple[str, ...], ...]


def _type_from_record(record: Mapping[str, object]) -> SemanticTensorType:
    allowed = {
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
    }
    unknown = set(record) - allowed
    if unknown:
        raise ValueError(f"unknown semantic type field(s): {sorted(unknown)!r}")
    return SemanticTensorType(**dict(record))  # type: ignore[arg-type]


def _registry_by_name(
    adapters: Sequence[SemanticAdapter],
) -> tuple[dict[str, SemanticAdapter], tuple[str, ...]]:
    by_name: dict[str, SemanticAdapter] = {}
    duplicates: list[str] = []
    for adapter in adapters:
        if adapter.name in by_name:
            duplicates.append(adapter.name)
        else:
            by_name[adapter.name] = adapter
    return by_name, tuple(sorted(set(duplicates)))


def _replay_path(
    source: SemanticTensorType,
    path: Sequence[str],
    *,
    registry: Mapping[str, SemanticAdapter],
    evidence_keys: set[str],
) -> tuple[SemanticTensorType | None, float, tuple[str, ...]]:
    current = source
    total_cost = 0.0
    reasons: list[str] = []

    for index, name in enumerate(path):
        adapter = registry.get(name)
        if adapter is None:
            reasons.append(f"path[{index}] references unknown adapter {name!r}")
            return None, total_cost, tuple(reasons)

        missing = set(adapter.required_evidence) - evidence_keys
        if missing:
            reasons.append(
                f"path[{index}] adapter {name!r} lacks evidence {tuple(sorted(missing))!r}"
            )
            return None, total_cost, tuple(reasons)

        if not adapter.applies_to(current):
            reasons.append(
                f"path[{index}] adapter {name!r} does not apply to current semantic type"
            )
            return None, total_cost, tuple(reasons)

        current = adapter.apply_type(current)
        total_cost += float(adapter.cost)

    return current, total_cost, tuple(reasons)


def _enumerate_admissible_solutions(
    source: SemanticTensorType,
    target: SemanticTensorType,
    *,
    adapters: Sequence[SemanticAdapter],
    evidence_keys: set[str],
    max_steps: int,
    max_cost: float,
) -> tuple[tuple[float, tuple[str, ...]], ...]:
    """Independent bounded exhaustive search, intentionally not Dijkstra."""
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    solutions: set[tuple[float, tuple[str, ...]]] = set()
    stack: list[
        tuple[
            SemanticTensorType,
            tuple[str, ...],
            float,
            int,
            frozenset[SemanticTensorType],
        ]
    ] = [(source, (), 0.0, 0, frozenset((source,)))]

    while stack:
        current, path, cost, steps, visited_types = stack.pop()

        if cost > max_cost:
            continue

        if is_assignable(current, target):
            solutions.add((cost, path))
            # A zero-cost cycle after reaching the target cannot improve the
            # semantic repair proof, so do not expand target-reaching states.
            continue

        if steps >= max_steps:
            continue

        for adapter in adapters:
            if set(adapter.required_evidence) - evidence_keys:
                continue
            if not adapter.applies_to(current):
                continue

            nxt = adapter.apply_type(current)
            if nxt == current or nxt in visited_types:
                continue

            new_cost = cost + float(adapter.cost)
            if new_cost > max_cost:
                continue

            stack.append(
                (
                    nxt,
                    path + (adapter.name,),
                    new_cost,
                    steps + 1,
                    visited_types | frozenset((nxt,)),
                )
            )

    return tuple(sorted(solutions, key=lambda item: (item[0], item[1])))


def verify_pure_adapter_certificate(
    certificate: SemanticCompilationCertificate,
    *,
    adapters: Sequence[SemanticAdapter],
    evidence_identity: Mapping[str, str] | None = None,
    max_steps: int = 6,
    atol: float = 1.0e-12,
) -> CompilationProofCheck:
    """Independently verify an installable pure-adapter compilation certificate."""

    reasons: list[str] = []
    cheaper: list[tuple[str, ...]] = []
    equal: list[tuple[str, ...]] = []
    evidence_identity = dict(evidence_identity or {})

    if not verify_compilation_certificate(certificate):
        reasons.append("certificate integrity digest does not match")

    if certificate.repair_candidate is not None:
        reasons.append("repair_candidate certificates are not installable pure-adapter proofs")

    expected_registry_digest = adapter_registry_digest(adapters)
    if certificate.registry_digest != expected_registry_digest:
        reasons.append("adapter registry digest has drifted")

    expected_evidence_digest = evidence_identity_digest(evidence_identity)
    if certificate.evidence_digest != expected_evidence_digest:
        reasons.append("context evidence identity digest has drifted")

    registry, duplicates = _registry_by_name(adapters)
    if duplicates:
        reasons.append(f"adapter registry contains duplicate names: {duplicates!r}")

    try:
        source = _type_from_record(certificate.refined_source)
        target = _type_from_record(certificate.target)
    except (TypeError, ValueError) as exc:
        reasons.append(f"certificate semantic type record is invalid: {exc}")
        return CompilationProofCheck(
            valid=False,
            reasons=tuple(reasons),
            selected_path=certificate.selected_adapter_path,
            selected_cost=certificate.selected_cost,
            cheaper_paths=(),
            equal_cost_alternatives=(),
        )

    replayed, replay_cost, replay_reasons = _replay_path(
        source,
        certificate.selected_adapter_path,
        registry=registry,
        evidence_keys=set(evidence_identity),
    )
    reasons.extend(replay_reasons)

    if replayed is not None and not is_assignable(replayed, target):
        reasons.append("selected adapter path does not reach an assignable target type")

    if certificate.selected_cost is None:
        reasons.append("installable pure-adapter certificate lacks selected_cost")
        selected_cost = replay_cost
    else:
        selected_cost = float(certificate.selected_cost)
        if abs(replay_cost - selected_cost) > atol:
            reasons.append(
                f"selected cost mismatch: replay={replay_cost!r}, certificate={selected_cost!r}"
            )

    if not duplicates and not replay_reasons and selected_cost >= 0:
        solutions = _enumerate_admissible_solutions(
            source,
            target,
            adapters=adapters,
            evidence_keys=set(evidence_identity),
            max_steps=max_steps,
            max_cost=selected_cost + atol,
        )
        chosen = certificate.selected_adapter_path
        for cost, path in solutions:
            if cost < selected_cost - atol:
                cheaper.append(path)
            elif abs(cost - selected_cost) <= atol and path != chosen:
                equal.append(path)

        if cheaper:
            reasons.append(
                f"selected path is not minimum cost; cheaper paths={tuple(cheaper)!r}"
            )
        if equal:
            reasons.append(
                "selected path is not uniquely minimum cost; "
                f"equal alternatives={tuple(equal)!r}"
            )

        if not any(
            abs(cost - selected_cost) <= atol and path == chosen
            for cost, path in solutions
        ):
            reasons.append("selected path is absent from independent admissible solution set")

    return CompilationProofCheck(
        valid=not reasons,
        reasons=tuple(reasons),
        selected_path=certificate.selected_adapter_path,
        selected_cost=certificate.selected_cost,
        cheaper_paths=tuple(cheaper),
        equal_cost_alternatives=tuple(equal),
    )