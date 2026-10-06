"""Static semantic-closure extraction for enum-backed runtime modes.

The extractor is deliberately conservative. It separates:
- declared enum modes;
- modes explicitly handled by runtime branches;
- modes explicitly designated as the runtime default/baseline;
- modes explicitly rejected;
- unresolved modes.

An unresolved mode is not automatically a bug. This distinction is required for
interfaces such as GR00T action representations where ABSOLUTE may be the
documented baseline while DELTA can otherwise silently share the fall-through
path.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EnumModeClosure:
    enum_name: str
    declared: frozenset[str]
    handled: frozenset[str]
    explicit_default: frozenset[str]
    rejected: frozenset[str]
    unresolved: frozenset[str]

    @property
    def closed(self) -> bool:
        return not self.unresolved


def _enum_member(node: ast.AST, enum_name: str) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == enum_name
    ):
        return node.attr
    return None


def extract_enum_members(source: str, enum_name: str) -> frozenset[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != enum_name:
            continue
        bases = {
            base.id
            for base in node.bases
            if isinstance(base, ast.Name)
        }
        if "Enum" not in bases:
            continue
        members = {
            stmt.targets[0].id
            for stmt in node.body
            if isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and not stmt.targets[0].id.startswith("_")
        }
        return frozenset(members)
    return frozenset()


def extract_runtime_handled_modes(
    source: str,
    enum_name: str,
) -> frozenset[str]:
    """Extract enum members compared in if/elif or matched in case clauses."""

    tree = ast.parse(source)
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for operand in operands:
                member = _enum_member(operand, enum_name)
                if member is not None:
                    found.add(member)
        elif isinstance(node, ast.MatchValue):
            member = _enum_member(node.value, enum_name)
            if member is not None:
                found.add(member)
        elif isinstance(node, ast.MatchOr):
            for pattern in node.patterns:
                if isinstance(pattern, ast.MatchValue):
                    member = _enum_member(pattern.value, enum_name)
                    if member is not None:
                        found.add(member)

    return frozenset(found)


def infer_enum_mode_closure(
    *,
    enum_source: str,
    runtime_source: str,
    enum_name: str,
    explicit_default: Iterable[str] = (),
    explicit_rejected: Iterable[str] = (),
) -> EnumModeClosure:
    declared = extract_enum_members(enum_source, enum_name)
    handled = extract_runtime_handled_modes(runtime_source, enum_name)
    default = frozenset(explicit_default)
    rejected = frozenset(explicit_rejected)

    unknown_annotations = (default | rejected) - declared
    if unknown_annotations:
        raise ValueError(
            "closure annotations mention undeclared modes: "
            + ", ".join(sorted(unknown_annotations))
        )

    unresolved = declared - handled - default - rejected
    return EnumModeClosure(
        enum_name=enum_name,
        declared=declared,
        handled=handled & declared,
        explicit_default=default,
        rejected=rejected,
        unresolved=unresolved,
    )