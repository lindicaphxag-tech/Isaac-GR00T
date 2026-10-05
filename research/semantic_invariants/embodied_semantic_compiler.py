"""End-to-end semantic compilation for embodied pipeline boundaries.

The compiler combines four mechanisms:
1. active inference for unknown semantic fields;
2. runtime refinement for event-owned semantics;
3. least-cost certified adapter synthesis for explicit conversions;
4. bounded counterexample-driven repair synthesis for pure numerical mismatches.

It fails closed on ambiguity, missing proof evidence, event semantics without a
refinement receipt, or repair families that are not explicitly justified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .embodied_refinement import RefinementResult
from .embodied_repair_synthesis import (
    RepairExample,
    RepairProgram,
    build_vector_repair_catalog,
    synthesize_minimal_repair,
)
from .embodied_semantic_inference import (
    InferenceResult,
    SemanticHypothesis,
    SemanticProbe,
    active_semantic_inference,
)
from .embodied_semantic_types import (
    NON_FORGEABLE_FIELDS,
    AdapterPlan,
    AmbiguousSemanticRepair,
    NoSemanticRepair,
    SemanticAdapter,
    SemanticTensorType,
    semantic_mismatches,
    synthesize_unique_adapter_plan,
)


class SemanticCompilationError(RuntimeError):
    pass


class SemanticProofObligationError(SemanticCompilationError):
    def __init__(self, message: str, *, obligations: Sequence[str]) -> None:
        super().__init__(message)
        self.obligations = tuple(obligations)


@dataclass(frozen=True)
class FieldInferenceSpec:
    field: str
    hypotheses: Mapping[str, SemanticHypothesis]
    probes: tuple[SemanticProbe, ...]
    observe: Callable[[SemanticProbe], object]
    max_risk: float = 0.0


@dataclass(frozen=True)
class FieldInferenceRecord:
    field: str
    result: InferenceResult
    inferred_value: str | None


@dataclass(frozen=True)
class CompilationResult:
    source: SemanticTensorType
    inferred_source: SemanticTensorType
    refined_source: SemanticTensorType
    target: SemanticTensorType
    inference: tuple[FieldInferenceRecord, ...]
    refinements: tuple[RefinementResult, ...]
    adapter_plan: AdapterPlan | None
    repair_candidate: RepairProgram | None
    repair_families: tuple[str, ...]
    effects: tuple[str, ...]

    @property
    def status(self) -> str:
        if self.repair_candidate is not None:
            return "repair_candidate"
        changed = (
            self.source != self.inferred_source
            or self.inferred_source != self.refined_source
            or bool(self.adapter_plan and self.adapter_plan.adapters)
        )
        return "repaired" if changed else "accepted"


def _infer_fields(
    source: SemanticTensorType,
    specs: Sequence[FieldInferenceSpec],
) -> tuple[SemanticTensorType, tuple[FieldInferenceRecord, ...]]:
    current = source
    records: list[FieldInferenceRecord] = []

    for spec in specs:
        if not hasattr(current, spec.field):
            raise SemanticCompilationError(f"unknown semantic field: {spec.field}")
        if getattr(current, spec.field) is not None:
            continue

        result = active_semantic_inference(
            list(spec.hypotheses.values()),
            spec.probes,
            observe=spec.observe,
            max_risk=spec.max_risk,
        )
        if result.status != "identified":
            records.append(FieldInferenceRecord(spec.field, result, None))
            raise SemanticCompilationError(
                f"semantic inference for {spec.field!r} is {result.status}: "
                f"{result.survivors}"
            )

        hypothesis_name = result.survivors[0]
        if hypothesis_name not in spec.hypotheses:
            raise SemanticCompilationError(
                f"inference returned unknown hypothesis {hypothesis_name!r}"
            )
        current = current.updated(**{spec.field: hypothesis_name})
        records.append(FieldInferenceRecord(spec.field, result, hypothesis_name))

    return current, tuple(records)


def _repair_families(
    source: SemanticTensorType,
    target: SemanticTensorType,
) -> tuple[str, ...]:
    """Map a narrow semantic mismatch to a bounded numerical repair DSL."""

    mismatches = semantic_mismatches(source, target)
    families: list[str] = []
    for mismatch in mismatches:
        if mismatch.field in NON_FORGEABLE_FIELDS:
            return ()

        if mismatch.field == "representation":
            representations = {source.representation, target.representation}
            if representations <= {"axis_angle", "euler_xyz"}:
                families.append("rotation-representation")
            else:
                return ()
        elif mismatch.field == "ordering":
            families.append("ordering")
        elif mismatch.field == "convention":
            conventions = {source.convention, target.convention}
            if conventions <= {
                "positive_action_negative_rotation",
                "positive_action_positive_rotation",
            }:
                families.append("sign")
            else:
                return ()
        elif mismatch.field == "unit":
            families.append("unit")
        elif mismatch.field == "frame":
            if "velocity" in source.entity and "velocity" in target.entity:
                families.append("rigid-body-frame")
            else:
                return ()
        else:
            return ()

    return tuple(dict.fromkeys(families))


def _synthesize_candidate(
    source: SemanticTensorType,
    target: SemanticTensorType,
    examples: Sequence[RepairExample],
    *,
    max_depth: int,
    atol: float,
) -> tuple[RepairProgram, tuple[str, ...]]:
    families = _repair_families(source, target)
    if not families:
        raise SemanticCompilationError(
            "semantic boundary has no registered bounded repair family"
        )
    if not examples:
        raise SemanticCompilationError(
            "semantic repair requires behavioral witnesses; refusing to guess"
        )

    dimensions = {
        len(vector)
        for example in examples
        for vector in (example.observed, example.expected)
    }
    if len(dimensions) != 1:
        raise SemanticCompilationError(
            "repair witnesses disagree on vector dimensionality"
        )

    result = synthesize_minimal_repair(
        examples,
        build_vector_repair_catalog(next(iter(dimensions))),
        max_depth=max_depth,
        allowed_families=set(families),
        atol=atol,
    )
    if result.program is None:
        raise SemanticCompilationError(
            "no bounded semantic repair fits the supplied witnesses"
        )
    if result.status != "unique":
        raise SemanticCompilationError(
            "semantic repair is ambiguous; acquire a discriminating counterexample"
        )
    return result.program, families


def compile_semantic_boundary(
    source: SemanticTensorType,
    target: SemanticTensorType,
    *,
    adapters: Sequence[SemanticAdapter] = (),
    inference_specs: Sequence[FieldInferenceSpec] = (),
    refinements: Sequence[RefinementResult] = (),
    adapter_evidence: Mapping[str, object] | None = None,
    repair_examples: Sequence[RepairExample] = (),
    max_repair_depth: int = 2,
    repair_atol: float = 1.0e-8,
    allow_proof_required: bool = False,
) -> CompilationResult:
    """Compile one producer->consumer boundary or fail closed.

    A numerical repair synthesized from witnesses is returned only as a
    repair_candidate.  Runtime installation remains forbidden until an
    independent verifier promotes that program to verified status.
    """

    inferred, inference_records = _infer_fields(source, inference_specs)

    current = inferred
    accepted_refinements: list[RefinementResult] = []
    refinement_effects: list[str] = []
    for refinement in refinements:
        if refinement.before != current:
            raise SemanticCompilationError(
                "refinement evidence was issued for a different semantic value"
            )
        current = refinement.after
        accepted_refinements.append(refinement)
        refinement_effects.extend(refinement.effects)

    mismatches = semantic_mismatches(current, target)
    event_fields = tuple(
        item.field for item in mismatches if item.field in NON_FORGEABLE_FIELDS
    )
    if event_fields:
        raise SemanticCompilationError(
            "event-owned semantic mismatch requires runtime refinement at the "
            f"owner boundary: {event_fields}"
        )

    try:
        # The research-grade compiler never lets Dijkstra enumeration order
        # choose between equally cheap physical meanings.
        plan = synthesize_unique_adapter_plan(
            current,
            target,
            adapters,
            evidence=adapter_evidence,
        )
    except AmbiguousSemanticRepair as exc:
        alternatives = tuple(" -> ".join(path) or "identity" for path in exc.alternatives)
        raise SemanticCompilationError(
            "explicit semantic repair is ambiguous; acquire a discriminating "
            f"witness before installation: {alternatives!r}"
        ) from exc
    except NoSemanticRepair as exc:
        if exc.proof_obligations:
            raise SemanticProofObligationError(
                "semantic boundary requires explicit context evidence",
                obligations=exc.proof_obligations,
            ) from exc

        if repair_examples:
            candidate, families = _synthesize_candidate(
                current,
                target,
                repair_examples,
                max_depth=max_repair_depth,
                atol=repair_atol,
            )
            return CompilationResult(
                source=source,
                inferred_source=inferred,
                refined_source=current,
                target=target,
                inference=inference_records,
                refinements=tuple(accepted_refinements),
                adapter_plan=None,
                repair_candidate=candidate,
                repair_families=families,
                effects=tuple(refinement_effects) + ("repair-candidate",),
            )

        mismatch = ", ".join(
            f"{item.field}={item.actual!r}->{item.expected!r}"
            for item in semantic_mismatches(current, target)
        )
        raise SemanticCompilationError(
            f"semantic boundary cannot be compiled ({mismatch})"
        ) from exc

    return CompilationResult(
        source=source,
        inferred_source=inferred,
        refined_source=current,
        target=target,
        inference=inference_records,
        refinements=tuple(accepted_refinements),
        adapter_plan=plan,
        repair_candidate=None,
        repair_families=(),
        effects=tuple(refinement_effects) + plan.effects,
    )