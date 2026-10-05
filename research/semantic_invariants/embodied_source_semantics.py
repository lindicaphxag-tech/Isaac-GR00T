"""Static semantic evidence extraction from Python robotics code.

The extractor is rule-driven rather than repository-specific. Project adapters
register the semantic meaning of API calls that already encode physical
conventions in source code; the extractor turns those calls into auditable
semantic facts and compares producer/consumer evidence across a boundary.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SourceLocation:
    line: int
    column: int


@dataclass(frozen=True)
class SemanticFact:
    field: str
    value: str
    rule_id: str
    callee: str
    location: SourceLocation
    evidence_kind: str = "source"


@dataclass(frozen=True)
class SourceSemanticRule:
    rule_id: str
    callee_suffix: str
    facts: Mapping[str, str]
    required_string_args: Mapping[int, str] | None = None
    required_string_kwargs: Mapping[str, str] | None = None


@dataclass(frozen=True)
class SemanticBoundaryConflict:
    field: str
    producer_value: str
    consumer_value: str
    producer_fact: SemanticFact
    consumer_fact: SemanticFact


def _callee_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _callee_name(node.value)
        return node.attr if base is None else f"{base}.{node.attr}"
    return None


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _rule_matches(call: ast.Call, callee: str, rule: SourceSemanticRule) -> bool:
    if not (callee == rule.callee_suffix or callee.endswith("." + rule.callee_suffix)):
        return False

    for index, expected in (rule.required_string_args or {}).items():
        if index >= len(call.args):
            return False
        if _literal_string(call.args[index]) != expected:
            return False

    keywords = {item.arg: item.value for item in call.keywords if item.arg is not None}
    for name, expected in (rule.required_string_kwargs or {}).items():
        if name not in keywords:
            return False
        if _literal_string(keywords[name]) != expected:
            return False

    return True


def infer_source_semantic_facts(
    source: str,
    rules: Sequence[SourceSemanticRule],
) -> tuple[SemanticFact, ...]:
    tree = ast.parse(source)
    facts: list[SemanticFact] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _callee_name(node.func)
        if callee is None:
            continue
        for rule in rules:
            if not _rule_matches(node, callee, rule):
                continue
            for field, value in rule.facts.items():
                facts.append(
                    SemanticFact(
                        field=field,
                        value=value,
                        rule_id=rule.rule_id,
                        callee=callee,
                        location=SourceLocation(node.lineno, node.col_offset),
                    )
                )

    return tuple(
        sorted(
            facts,
            key=lambda item: (
                item.location.line,
                item.location.column,
                item.rule_id,
                item.field,
                item.value,
            ),
        )
    )


def latest_facts_by_field(facts: Iterable[SemanticFact]) -> dict[str, SemanticFact]:
    out: dict[str, SemanticFact] = {}
    for fact in facts:
        previous = out.get(fact.field)
        if previous is None or (fact.location.line, fact.location.column, fact.rule_id) >= (
            previous.location.line, previous.location.column, previous.rule_id
        ):
            out[fact.field] = fact
    return out


def compare_boundary_semantics(
    producer_facts: Sequence[SemanticFact],
    consumer_facts: Sequence[SemanticFact],
) -> tuple[SemanticBoundaryConflict, ...]:
    producer = latest_facts_by_field(producer_facts)
    consumer = latest_facts_by_field(consumer_facts)
    conflicts: list[SemanticBoundaryConflict] = []

    for field in sorted(set(producer) & set(consumer)):
        left = producer[field]
        right = consumer[field]
        if left.value == right.value:
            continue
        conflicts.append(
            SemanticBoundaryConflict(
                field=field,
                producer_value=left.value,
                consumer_value=right.value,
                producer_fact=left,
                consumer_fact=right,
            )
        )
    return tuple(conflicts)


DEFAULT_ROTATION_RULES = (
    SourceSemanticRule(
        rule_id="rotation/euler-xyz-input",
        callee_suffix="euler_angles_to_matrix",
        required_string_args={1: "XYZ"},
        facts={"representation": "euler_xyz"},
    ),
    SourceSemanticRule(
        rule_id="rotation/axis-angle-output",
        callee_suffix="compact_axis_angle_from_quaternion",
        facts={"representation": "axis_angle"},
    ),
    SourceSemanticRule(
        rule_id="rotation/quaternion-to-matrix-input",
        callee_suffix="quaternion_to_matrix",
        facts={"representation": "quaternion"},
    ),
)