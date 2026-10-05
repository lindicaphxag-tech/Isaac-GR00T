"""Counterexample-guided synthesis of embodied semantic interface repairs.

The repair layer is intentionally dependency-light. It searches over a small,
physically meaningful adapter DSL rather than generating arbitrary code. A
candidate is accepted only when it satisfies the frozen semantic examples and an
external counterexample oracle cannot falsify it.

This is a method prototype, not evidence that any specific upstream issue has
been repaired prospectively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product
from math import asin, atan2, cos, sin, sqrt
from typing import Any, Callable, Mapping, Sequence


Vector = tuple[float, ...]
Context = Mapping[str, Any]


def _close(actual: Vector, expected: Vector, atol: float) -> bool:
    return len(actual) == len(expected) and all(
        abs(float(a) - float(b)) <= atol
        for a, b in zip(actual, expected, strict=True)
    )


def _cross(a: Sequence[float], b: Sequence[float]) -> Vector:
    return (
        float(a[1] * b[2] - a[2] * b[1]),
        float(a[2] * b[0] - a[0] * b[2]),
        float(a[0] * b[1] - a[1] * b[0]),
    )


def _axis_angle_to_matrix(value: Vector):
    x, y, z = value
    angle = sqrt(x * x + y * y + z * z)
    if angle < 1.0e-14:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    x, y, z = x / angle, y / angle, z / angle
    c, s, one_minus_c = cos(angle), sin(angle), 1.0 - cos(angle)
    return (
        (
            c + x * x * one_minus_c,
            x * y * one_minus_c - z * s,
            x * z * one_minus_c + y * s,
        ),
        (
            y * x * one_minus_c + z * s,
            c + y * y * one_minus_c,
            y * z * one_minus_c - x * s,
        ),
        (
            z * x * one_minus_c - y * s,
            z * y * one_minus_c + x * s,
            c + z * z * one_minus_c,
        ),
    )


def _matrix_to_euler_xyz(matrix) -> Vector:
    # Convention matches Rx(x) @ Ry(y) @ Rz(z).
    y = asin(max(-1.0, min(1.0, float(matrix[0][2]))))
    cy = cos(y)
    if abs(cy) > 1.0e-10:
        x = atan2(-matrix[1][2], matrix[2][2])
        z = atan2(-matrix[0][1], matrix[0][0])
    else:
        # Gimbal-lock convention: choose z = 0 and absorb rotation into x.
        z = 0.0
        x = atan2(matrix[2][1], matrix[1][1])
    return (float(x), float(y), float(z))


@dataclass(frozen=True)
class RepairExample:
    observed: Vector
    expected: Vector
    context: Context = field(default_factory=dict)
    label: str = ""


@dataclass(frozen=True)
class RepairPrimitive:
    name: str
    family: str
    cost: int
    apply_fn: Callable[[Vector, Context], Vector]
    implementation_id: str | None = None

    def apply(self, value: Vector, context: Context) -> Vector:
        return tuple(float(item) for item in self.apply_fn(value, context))


@dataclass(frozen=True)
class RepairProgram:
    operations: tuple[RepairPrimitive, ...] = ()

    @property
    def name(self) -> str:
        return "identity" if not self.operations else " -> ".join(
            operation.name for operation in self.operations
        )

    @property
    def cost(self) -> int:
        # Prefer a single physically meaningful repair over a chain of equally
        # cheap patches. The extra composition penalty is deliberate.
        return sum(operation.cost for operation in self.operations) + max(
            0, len(self.operations) - 1
        )

    def apply(self, value: Vector, context: Context | None = None) -> Vector:
        output = tuple(float(item) for item in value)
        context = context or {}
        for operation in self.operations:
            output = operation.apply(output, context)
        return output


@dataclass(frozen=True)
class SynthesisResult:
    status: str
    program: RepairProgram | None
    alternatives: tuple[RepairProgram, ...]
    examples: tuple[RepairExample, ...]


@dataclass(frozen=True)
class CEGISResult:
    status: str
    program: RepairProgram | None
    rounds: int
    examples: tuple[RepairExample, ...]
    last_alternatives: tuple[str, ...]


def build_vector_repair_catalog(dimension: int) -> tuple[RepairPrimitive, ...]:
    """Build a bounded, interpretable repair DSL for one vector width."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")

    primitives: list[RepairPrimitive] = []
    identity_order = tuple(range(dimension))

    for order in permutations(range(dimension)):
        if order == identity_order:
            continue
        primitives.append(
            RepairPrimitive(
                name=f"permute{order}",
                family="ordering",
                cost=1,
                apply_fn=lambda value, context, order=order: tuple(
                    value[index] for index in order
                ),
                implementation_id=f"builtin-v0:permute:{order}",
            )
        )

    positive = tuple(1.0 for _ in range(dimension))
    for signs in product((-1.0, 1.0), repeat=dimension):
        if signs == positive:
            continue
        primitives.append(
            RepairPrimitive(
                name=f"sign{tuple(int(sign) for sign in signs)}",
                family="sign",
                cost=1,
                apply_fn=lambda value, context, signs=signs: tuple(
                    item * sign
                    for item, sign in zip(value, signs, strict=True)
                ),
                implementation_id=f"builtin-v0:sign:{signs}",
            )
        )

    for scale in (0.001, 0.01, 0.1, 10.0, 100.0, 1000.0):
        primitives.append(
            RepairPrimitive(
                name=f"scale({scale:g})",
                family="unit",
                cost=1,
                apply_fn=lambda value, context, scale=scale: tuple(
                    scale * item for item in value
                ),
                implementation_id=f"builtin-v0:scale:{scale:g}",
            )
        )

    if dimension == 3:
        primitives.extend(
            [
                RepairPrimitive(
                    name="axis-angle->euler-xyz",
                    family="rotation-representation",
                    cost=2,
                    apply_fn=lambda value, context: _matrix_to_euler_xyz(
                        _axis_angle_to_matrix(value)
                    ),
                    implementation_id="builtin-v0:axis-angle-to-euler-xyz",
                ),
                RepairPrimitive(
                    name="com-velocity->link-velocity",
                    family="rigid-body-frame",
                    cost=2,
                    apply_fn=lambda value, context: tuple(
                        item - shift
                        for item, shift in zip(
                            value,
                            _cross(context["angular"], context["com_offset"]),
                            strict=True,
                        )
                    ),
                    implementation_id="builtin-v0:com-velocity-to-link-velocity",
                ),
                RepairPrimitive(
                    name="link-velocity->com-velocity",
                    family="rigid-body-frame",
                    cost=2,
                    apply_fn=lambda value, context: tuple(
                        item + shift
                        for item, shift in zip(
                            value,
                            _cross(context["angular"], context["com_offset"]),
                            strict=True,
                        )
                    ),
                    implementation_id="builtin-v0:link-velocity-to-com-velocity",
                ),
            ]
        )

    return tuple(primitives)


def enumerate_programs(
    catalog: Sequence[RepairPrimitive],
    *,
    max_depth: int = 2,
    allowed_families: set[str] | None = None,
) -> tuple[RepairProgram, ...]:
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")

    operations = [
        item
        for item in catalog
        if allowed_families is None or item.family in allowed_families
    ]

    programs = [RepairProgram()]
    if max_depth >= 1:
        programs.extend(RepairProgram((operation,)) for operation in operations)

    if max_depth >= 2:
        for left in operations:
            for right in operations:
                if left.name == right.name:
                    continue
                programs.append(RepairProgram((left, right)))

    # Search is complexity-first and deterministic.
    return tuple(
        sorted(
            programs,
            key=lambda item: (item.cost, len(item.operations), item.name),
        )
    )


def program_fits(
    program: RepairProgram,
    examples: Sequence[RepairExample],
    *,
    atol: float = 1.0e-8,
) -> bool:
    try:
        return all(
            _close(
                program.apply(example.observed, example.context),
                example.expected,
                atol,
            )
            for example in examples
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def synthesize_minimal_repair(
    examples: Sequence[RepairExample],
    catalog: Sequence[RepairPrimitive],
    *,
    max_depth: int = 2,
    allowed_families: set[str] | None = None,
    atol: float = 1.0e-8,
) -> SynthesisResult:
    """Return all minimum-cost repair programs consistent with the evidence."""
    if not examples:
        raise ValueError("at least one repair example is required")

    matching = [
        program
        for program in enumerate_programs(
            catalog,
            max_depth=max_depth,
            allowed_families=allowed_families,
        )
        if program_fits(program, examples, atol=atol)
    ]
    if not matching:
        return SynthesisResult(
            status="no_solution",
            program=None,
            alternatives=(),
            examples=tuple(examples),
        )

    minimum_cost = matching[0].cost
    minimal = tuple(
        program for program in matching if program.cost == minimum_cost
    )
    return SynthesisResult(
        status="unique" if len(minimal) == 1 else "ambiguous",
        program=minimal[0],
        alternatives=minimal,
        examples=tuple(examples),
    )


def cegis_repair(
    seed_examples: Sequence[RepairExample],
    catalog: Sequence[RepairPrimitive],
    *,
    counterexample_finder: Callable[[RepairProgram], RepairExample | None],
    max_depth: int = 2,
    allowed_families: set[str] | None = None,
    atol: float = 1.0e-8,
    max_rounds: int = 20,
) -> CEGISResult:
    """Counterexample-guided repair synthesis with conservative failure modes."""
    examples = list(seed_examples)
    if not examples:
        raise ValueError("at least one seed example is required")

    last_alternatives: tuple[str, ...] = ()
    for round_index in range(1, max_rounds + 1):
        result = synthesize_minimal_repair(
            examples,
            catalog,
            max_depth=max_depth,
            allowed_families=allowed_families,
            atol=atol,
        )
        if result.program is None:
            return CEGISResult(
                status="no_solution",
                program=None,
                rounds=round_index,
                examples=tuple(examples),
                last_alternatives=(),
            )

        last_alternatives = tuple(item.name for item in result.alternatives)
        counterexample = counterexample_finder(result.program)
        if counterexample is None:
            return CEGISResult(
                status="verified",
                program=result.program,
                rounds=round_index,
                examples=tuple(examples),
                last_alternatives=last_alternatives,
            )

        duplicate = any(
            example.observed == counterexample.observed
            and example.expected == counterexample.expected
            and dict(example.context) == dict(counterexample.context)
            for example in examples
        )
        if duplicate:
            return CEGISResult(
                status="stalled",
                program=None,
                rounds=round_index,
                examples=tuple(examples),
                last_alternatives=last_alternatives,
            )
        examples.append(counterexample)

    return CEGISResult(
        status="budget_exhausted",
        program=None,
        rounds=max_rounds,
        examples=tuple(examples),
        last_alternatives=last_alternatives,
    )


def bank_counterexample_finder(
    bank: Sequence[RepairExample],
    *,
    atol: float = 1.0e-8,
) -> Callable[[RepairProgram], RepairExample | None]:
    """Build a deterministic verifier over a frozen held-out evidence bank."""

    def find(program: RepairProgram) -> RepairExample | None:
        for example in bank:
            if not _close(
                program.apply(example.observed, example.context),
                example.expected,
                atol,
            ):
                return example
        return None

    return find