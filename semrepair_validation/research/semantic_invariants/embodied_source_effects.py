"""Static semantic-effect extraction for embodied execution paths.

This layer complements value-type inference. Two paths may expose compatible
types yet apply different semantics-changing effects before the same owner
boundary. Rules are explicit and auditable; absence of a rule remains unknown
rather than being guessed.

A source-effect asymmetry is evidence for a parity investigation, not by itself
proof of a defect.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SourceEffectRule:
    rule_id: str
    callee_suffix: str
    effect: str
    details: Mapping[str, str] | None = None
    required_string_args: Mapping[int, str] | None = None
    required_string_kwargs: Mapping[str, str] | None = None


@dataclass(frozen=True)
class SourceEffectFact:
    effect: str
    rule_id: str
    callee: str
    line: int
    column: int
    details: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class EffectParityResult:
    reference_effects: tuple[str, ...]
    alternative_effects: tuple[str, ...]
    shared_effects: tuple[str, ...]
    reference_only: tuple[str, ...]
    alternative_only: tuple[str, ...]

    @property
    def symmetric(self) -> bool:
        return not self.reference_only and not self.alternative_only


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


def _matches(call: ast.Call, callee: str, rule: SourceEffectRule) -> bool:
    if not (callee == rule.callee_suffix or callee.endswith("." + rule.callee_suffix)):
        return False

    for index, expected in (rule.required_string_args or {}).items():
        if index >= len(call.args) or _literal_string(call.args[index]) != expected:
            return False

    kwargs = {item.arg: item.value for item in call.keywords if item.arg is not None}
    for key, expected in (rule.required_string_kwargs or {}).items():
        if key not in kwargs or _literal_string(kwargs[key]) != expected:
            return False

    return True


def infer_source_effects(
    source: str,
    rules: Sequence[SourceEffectRule],
) -> tuple[SourceEffectFact, ...]:
    tree = ast.parse(source)
    facts: list[SourceEffectFact] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _callee_name(node.func)
        if callee is None:
            continue
        for rule in rules:
            if not _matches(node, callee, rule):
                continue
            facts.append(
                SourceEffectFact(
                    effect=rule.effect,
                    rule_id=rule.rule_id,
                    callee=callee,
                    line=node.lineno,
                    column=node.col_offset,
                    details=tuple(sorted((rule.details or {}).items())),
                )
            )
    return tuple(
        sorted(
            facts,
            key=lambda item: (item.line, item.column, item.rule_id, item.effect),
        )
    )


def compare_effect_signatures(
    reference: Sequence[SourceEffectFact],
    alternative: Sequence[SourceEffectFact],
) -> EffectParityResult:
    """Compare unique semantic-effect labels across two source paths.

    Multiplicity is intentionally ignored in v1 because repeated helper calls
    may occur in different loop structures. A future trace-aware generation can
    track cardinality/order when that semantic claim is needed.
    """

    left = {item.effect for item in reference}
    right = {item.effect for item in alternative}
    return EffectParityResult(
        reference_effects=tuple(sorted(left)),
        alternative_effects=tuple(sorted(right)),
        shared_effects=tuple(sorted(left & right)),
        reference_only=tuple(sorted(left - right)),
        alternative_only=tuple(sorted(right - left)),
    )


DEFAULT_PATH_EFFECT_RULES = (
    SourceEffectRule(
        rule_id="image/resize-policy-shape",
        callee_suffix="resize_robot_observation_image",
        effect="image_resize_to_policy_shape",
        details={"interpolation": "bilinear", "align_corners": "false"},
    ),
    SourceEffectRule(
        rule_id="pipeline/preprocess",
        callee_suffix="_preprocessor",
        effect="policy_preprocess",
    ),
    SourceEffectRule(
        rule_id="pipeline/preprocess-public",
        callee_suffix="preprocessor",
        effect="policy_preprocess",
    ),
)