"""Bridge static source semantics into the embodied semantic compiler.

The source extractor yields auditable facts.  This module turns those facts into
partial SemanticTensorType values without silently overwriting contradictions.

The bridge is intentionally conservative:
- source evidence only fills unknown fields unless it agrees with an existing
  explicit type;
- contradictory source facts keep the boundary unresolved;
- every inferred field retains the source rule and line that justified it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .embodied_semantic_compiler import CompilationResult, compile_semantic_boundary
from .embodied_semantic_types import SemanticAdapter, SemanticTensorType
from .embodied_source_semantics import (
    SemanticFact,
    SourceSemanticRule,
    infer_source_semantic_facts,
)


class ConflictingSourceSemantics(RuntimeError):
    pass


@dataclass(frozen=True)
class InferredFieldEvidence:
    field: str
    value: str
    rule_id: str
    line: int
    callee: str


@dataclass(frozen=True)
class SourceTypeInference:
    semantic_type: SemanticTensorType
    evidence: tuple[InferredFieldEvidence, ...]


@dataclass(frozen=True)
class SourceBoundaryCompilation:
    producer: SourceTypeInference
    consumer: SourceTypeInference
    compilation: CompilationResult


def infer_type_from_source(
    base: SemanticTensorType,
    source: str,
    rules: Sequence[SourceSemanticRule],
) -> SourceTypeInference:
    """Fill unknown semantic fields from source facts, fail on contradictions."""

    facts = infer_source_semantic_facts(source, rules)
    grouped: dict[str, list[SemanticFact]] = {}
    for fact in facts:
        grouped.setdefault(fact.field, []).append(fact)

    current = base
    evidence: list[InferredFieldEvidence] = []

    for field, field_facts in sorted(grouped.items()):
        if not hasattr(current, field):
            continue
        values = {fact.value for fact in field_facts}
        if len(values) != 1:
            detail = ", ".join(
                f"{fact.value}@L{fact.location.line}:{fact.rule_id}"
                for fact in field_facts
            )
            raise ConflictingSourceSemantics(
                f"source gives conflicting semantics for {field!r}: {detail}"
            )

        value = next(iter(values))
        existing = getattr(current, field)
        if existing is not None and existing != value:
            raise ConflictingSourceSemantics(
                f"explicit {field!r}={existing!r} conflicts with source {value!r}"
            )

        if existing is None:
            current = current.updated(**{field: value})

        # Preserve every supporting source fact rather than collapsing provenance.
        for fact in field_facts:
            evidence.append(
                InferredFieldEvidence(
                    field=field,
                    value=fact.value,
                    rule_id=fact.rule_id,
                    line=fact.location.line,
                    callee=fact.callee,
                )
            )

    return SourceTypeInference(
        semantic_type=current,
        evidence=tuple(evidence),
    )


def compile_source_boundary(
    *,
    producer_source: str,
    consumer_source: str,
    producer_base: SemanticTensorType,
    consumer_base: SemanticTensorType,
    rules: Sequence[SourceSemanticRule],
    adapters: Sequence[SemanticAdapter] = (),
    adapter_evidence: Mapping[str, object] | None = None,
) -> SourceBoundaryCompilation:
    """Infer both endpoint types from source and compile the resulting boundary."""

    producer = infer_type_from_source(producer_base, producer_source, rules)
    consumer = infer_type_from_source(consumer_base, consumer_source, rules)

    compilation = compile_semantic_boundary(
        producer.semantic_type,
        consumer.semantic_type,
        adapters=adapters,
        adapter_evidence=adapter_evidence,
    )
    return SourceBoundaryCompilation(
        producer=producer,
        consumer=consumer,
        compilation=compilation,
    )
