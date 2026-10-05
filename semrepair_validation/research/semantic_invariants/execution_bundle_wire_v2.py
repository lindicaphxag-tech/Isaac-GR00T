"""Language-neutral SemRepair execution proof bundle v0.2.

v0.2 removes binary floating-point values from the hashed wire surface. Adapter
costs and producer-selected costs are carried as canonical decimal strings, and
minimum-cost verification uses Decimal arithmetic. This prevents a semantic tie
such as 0.1 + 0.2 versus 0.3 from being turned into a false unique minimum by
binary floating-point rounding.

The verifier is intentionally independent from the production compiler/search
implementation. Producer certificate digests are retained as opaque dependency
identities; language-neutral integrity is provided by v0.2 wire digests.

The wire digests are deterministic integrity/dependency receipts, not digital
signatures and not issuer authentication.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata


BUNDLE_SCHEMA = "semrepair-execution-proof-bundle/v0.2"
CANONICALIZATION_PROFILE = "semrepair-wire-c14n/v0.1"
SPEC_VERSION = "0.2"
DECIMAL_TAG = "$semrepair_decimal"

SEMANTIC_FIELDS = (
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
)
NON_FORGEABLE_FIELDS = frozenset({"provenance", "freshness"})
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DECIMAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")


def canonical_decimal(value: object) -> str:
    """Return one exponent-free decimal spelling for a finite numeric value."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a decimal value")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid decimal value: {value!r}") from exc
    if not number.is_finite():
        raise ValueError("wire decimals must be finite")
    if number == 0:
        return "0"
    text = format(number.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    if not DECIMAL_RE.fullmatch(text):
        raise ValueError(f"non-canonical decimal: {text!r}")
    return text


def _parse_decimal(value: object, *, nonnegative: bool = False) -> Decimal:
    if not isinstance(value, str) or not DECIMAL_RE.fullmatch(value):
        raise ValueError(f"expected canonical decimal string, got {value!r}")
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError("wire decimals must be finite")
    if nonnegative and number < 0:
        raise ValueError("wire decimal must be non-negative")
    if canonical_decimal(number) != value:
        raise ValueError(f"non-canonical decimal spelling: {value!r}")
    return number


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _normalize_wire(value: object) -> object:
    """Normalize arbitrary JSON-compatible evidence without leaving floats."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, (float, Decimal)):
        return {DECIMAL_TAG: canonical_decimal(value)}
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise ValueError("wire object keys must be strings")
            key = _normalize_text(raw_key)
            if key in normalized:
                raise ValueError("NFC normalization produced a duplicate object key")
            normalized[key] = _normalize_wire(raw_value)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_wire(item) for item in value]
    raise ValueError(f"unsupported wire value type: {type(value).__name__}")


def canonical_wire_json(payload: object) -> str:
    """Serialize the v0.1 wire canonicalization profile.

    The accepted surface contains null/bool/integer/string/array/object only.
    Floating-point values must first be represented as canonical decimal strings
    or tagged decimal objects.
    """

    normalized = _normalize_wire(payload)

    def emit(value: object) -> str:
        if value is None:
            return "null"
        if value is True:
            return "true"
        if value is False:
            return "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if isinstance(value, list):
            return "[" + ",".join(emit(item) for item in value) + "]"
        if isinstance(value, dict):
            keys = sorted(value)
            return "{" + ",".join(
                emit(key) + ":" + emit(value[key]) for key in keys
            ) + "}"
        raise TypeError(f"non-wire value after normalization: {type(value).__name__}")

    return emit(normalized)


def wire_digest(payload: object) -> str:
    return sha256(canonical_wire_json(payload).encode("utf-8")).hexdigest()


def _semantic_record(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        raw = dict(value)
    else:
        raw = dict(vars(value))
    return {field: _normalize_wire(raw.get(field)) for field in SEMANTIC_FIELDS}


def _adapter_record(adapter: object) -> dict[str, object]:
    if isinstance(adapter, Mapping):
        get = adapter.get
        name = get("name")
        requires = get("requires") or {}
        produces = get("produces") or {}
        cost = get("cost_decimal", get("cost", "1"))
        effects = get("effects") or []
        required_evidence = get("required_evidence") or get("witness_keys") or []
    else:
        name = getattr(adapter, "name")
        requires = getattr(adapter, "requires")
        produces = getattr(adapter, "produces")
        cost = getattr(adapter, "cost")
        effects = getattr(adapter, "effects")
        required_evidence = getattr(adapter, "required_evidence", ())

    return {
        "name": _normalize_text(str(name)),
        "requires": _normalize_wire(dict(requires)),
        "produces": _normalize_wire(dict(produces)),
        "cost_decimal": canonical_decimal(cost),
        "effects": _normalize_wire(list(effects)),
        "required_evidence": _normalize_wire(list(required_evidence)),
    }


def _compilation_record(certificate: object) -> dict[str, object]:
    selected_cost = getattr(certificate, "selected_cost")
    return {
        "schema": str(getattr(certificate, "schema")),
        "compiler_version": str(getattr(certificate, "compiler_version")),
        "producer_registry_digest": str(getattr(certificate, "registry_digest")),
        "producer_evidence_digest": str(getattr(certificate, "evidence_digest")),
        "source": _semantic_record(getattr(certificate, "source")),
        "target": _semantic_record(getattr(certificate, "target")),
        "inferred_source": _semantic_record(getattr(certificate, "inferred_source")),
        "refined_source": _semantic_record(getattr(certificate, "refined_source")),
        "selected_adapter_path": _normalize_wire(
            list(getattr(certificate, "selected_adapter_path"))
        ),
        "producer_selected_cost_decimal": (
            None if selected_cost is None else canonical_decimal(selected_cost)
        ),
        "repair_candidate": _normalize_wire(getattr(certificate, "repair_candidate")),
        "repair_families": _normalize_wire(list(getattr(certificate, "repair_families"))),
        "refinement_records": _normalize_wire(
            list(getattr(certificate, "refinement_records"))
        ),
        "effects": _normalize_wire(list(getattr(certificate, "effects"))),
        "producer_decision_digest": str(getattr(certificate, "decision_digest")),
    }


def _effect_record(certificate: object) -> dict[str, object]:
    return {
        "schema": str(getattr(certificate, "schema")),
        "effect_id": str(getattr(certificate, "effect_id")),
        "action_name": str(getattr(certificate, "action_name")),
        "effect_class": str(getattr(certificate, "effect_class")),
        "dependency_version": str(getattr(certificate, "dependency_version")),
        "policy": _normalize_wire(dict(getattr(certificate, "policy"))),
        "semantic_contract_id": str(getattr(certificate, "semantic_contract_id")),
        "issuer_version": str(getattr(certificate, "issuer_version")),
        "producer_decision_digest": str(getattr(certificate, "decision_digest")),
    }


def export_execution_bundle_v2(
    compilation_certificate: object,
    effect_certificate: object,
    intent: object,
    *,
    semantic_contract_id: str,
    adapters: Sequence[object],
    evidence_identity: Mapping[str, str] | None = None,
    max_steps: int = 6,
) -> dict[str, object]:
    if not semantic_contract_id:
        raise ValueError("semantic_contract_id must be non-empty")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    adapter_records = [_adapter_record(adapter) for adapter in adapters]
    adapter_records.sort(key=lambda item: (str(item["name"]), wire_digest(item)))

    compilation = _compilation_record(compilation_certificate)
    effect = _effect_record(effect_certificate)
    effect_class = getattr(
        getattr(intent, "effect_class"), "value", getattr(intent, "effect_class")
    )
    intent_record = {
        "effect_id": str(getattr(intent, "effect_id")),
        "action_name": str(getattr(intent, "action_name")),
        "effect_class": str(effect_class),
        "dependency_version": str(getattr(intent, "dependency_version")),
    }
    evidence_record = _normalize_wire(dict(sorted(dict(evidence_identity or {}).items())))

    bindings = {
        "adapter_registry_digest": wire_digest(adapter_records),
        "evidence_identity_digest": wire_digest(evidence_record),
        "compilation_digest": wire_digest(compilation),
        "effect_digest": wire_digest(effect),
    }

    payload: dict[str, object] = {
        "schema": BUNDLE_SCHEMA,
        "spec_version": SPEC_VERSION,
        "canonicalization_profile": CANONICALIZATION_PROFILE,
        "semantic_contract_id": _normalize_text(semantic_contract_id),
        "compilation_certificate": compilation,
        "effect_certificate": effect,
        "intent": intent_record,
        "adapter_registry": adapter_records,
        "evidence_identity": evidence_record,
        "verification_profile": {"max_steps": max_steps},
        "bindings": bindings,
    }
    return {**payload, "bundle_digest": wire_digest(payload)}


def _assignable(actual: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    for field in SEMANTIC_FIELDS:
        want = target.get(field)
        if want is not None and actual.get(field) != want:
            return False
    return True


def _apply_adapter(
    current: Mapping[str, Any],
    adapter: Mapping[str, Any],
    *,
    evidence_keys: set[str],
) -> dict[str, Any] | None:
    requires = adapter.get("requires")
    produces = adapter.get("produces")
    required_evidence = adapter.get("required_evidence")
    if not isinstance(requires, Mapping) or not isinstance(produces, Mapping):
        return None
    if not isinstance(required_evidence, list):
        return None
    if not all(isinstance(item, str) for item in required_evidence):
        return None
    if not set(required_evidence).issubset(evidence_keys):
        return None
    if any(current.get(key) != value for key, value in requires.items()):
        return None

    unknown = (set(requires) | set(produces)) - set(SEMANTIC_FIELDS)
    if unknown:
        return None
    for field in NON_FORGEABLE_FIELDS & set(produces):
        if requires.get(field) != produces.get(field):
            return None

    result = dict(current)
    result.update(dict(produces))
    return result


def _enumerate_solutions(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    adapters: Sequence[Mapping[str, Any]],
    *,
    evidence_keys: set[str],
    max_steps: int,
    max_cost: Decimal | None = None,
) -> list[tuple[Decimal, tuple[str, ...]]]:
    if _assignable(source, target):
        return [(Decimal("0"), ())]

    def key(value: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple((field, value.get(field)) for field in SEMANTIC_FIELDS)

    solutions: set[tuple[Decimal, tuple[str, ...]]] = set()
    stack: list[
        tuple[
            dict[str, Any],
            tuple[str, ...],
            Decimal,
            int,
            frozenset[tuple[tuple[str, Any], ...]],
        ]
    ] = [(dict(source), (), Decimal("0"), 0, frozenset((key(source),)))]

    while stack:
        current, path, cost, steps, visited = stack.pop()
        if max_cost is not None and cost > max_cost:
            continue
        if _assignable(current, target):
            solutions.add((cost, path))
            continue
        if steps >= max_steps:
            continue

        for adapter in adapters:
            nxt = _apply_adapter(current, adapter, evidence_keys=evidence_keys)
            if nxt is None:
                continue
            nxt_key = key(nxt)
            if nxt_key == key(current) or nxt_key in visited:
                continue
            try:
                adapter_cost = _parse_decimal(
                    adapter.get("cost_decimal"), nonnegative=True
                )
            except ValueError:
                continue
            new_cost = cost + adapter_cost
            if max_cost is not None and new_cost > max_cost:
                continue
            stack.append(
                (
                    nxt,
                    path + (str(adapter.get("name")),),
                    new_cost,
                    steps + 1,
                    visited | frozenset((nxt_key,)),
                )
            )

    return sorted(solutions, key=lambda item: (item[0], item[1]))


def _valid_hex64(value: object) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def verify_execution_bundle_v2(record: Mapping[str, Any]) -> dict[str, object]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    checks["bundle_schema"] = record.get("schema") == BUNDLE_SCHEMA
    checks["spec_version"] = record.get("spec_version") == SPEC_VERSION
    checks["canonicalization_profile"] = (
        record.get("canonicalization_profile") == CANONICALIZATION_PROFILE
    )

    outer = dict(record)
    claimed_bundle_digest = outer.pop("bundle_digest", None)
    checks["bundle_digest"] = (
        _valid_hex64(claimed_bundle_digest)
        and wire_digest(outer) == claimed_bundle_digest
    )

    compilation = record.get("compilation_certificate")
    effect = record.get("effect_certificate")
    intent = record.get("intent")
    adapters = record.get("adapter_registry")
    evidence = record.get("evidence_identity")
    profile = record.get("verification_profile")
    bindings = record.get("bindings")

    structural = (
        isinstance(compilation, Mapping)
        and isinstance(effect, Mapping)
        and isinstance(intent, Mapping)
        and isinstance(adapters, list)
        and all(isinstance(item, Mapping) for item in adapters)
        and isinstance(evidence, Mapping)
        and isinstance(profile, Mapping)
        and isinstance(bindings, Mapping)
    )
    checks["structure"] = structural
    if not structural:
        errors.append("bundle structure is malformed")
        return {
            "schema": BUNDLE_SCHEMA,
            "valid": False,
            "checks": checks,
            "errors": errors,
        }

    checks["registry_wire_digest"] = (
        bindings.get("adapter_registry_digest") == wire_digest(adapters)
    )
    checks["evidence_wire_digest"] = (
        bindings.get("evidence_identity_digest") == wire_digest(evidence)
    )
    checks["compilation_wire_digest"] = (
        bindings.get("compilation_digest") == wire_digest(compilation)
    )
    checks["effect_wire_digest"] = (
        bindings.get("effect_digest") == wire_digest(effect)
    )

    names = [item.get("name") for item in adapters]
    checks["unique_adapter_names"] = (
        all(isinstance(name, str) and name for name in names)
        and len(names) == len(set(names))
    )

    adapter_semantics_valid = True
    for index, adapter in enumerate(adapters):
        requires = adapter.get("requires")
        produces = adapter.get("produces")
        if not isinstance(requires, Mapping) or not isinstance(produces, Mapping):
            adapter_semantics_valid = False
            errors.append(f"adapter[{index}] has malformed semantic maps")
            continue
        unknown = (set(requires) | set(produces)) - set(SEMANTIC_FIELDS)
        if unknown:
            adapter_semantics_valid = False
            errors.append(
                f"adapter[{index}] uses unknown semantic fields: {tuple(sorted(unknown))!r}"
            )
        try:
            _parse_decimal(adapter.get("cost_decimal"), nonnegative=True)
        except ValueError as exc:
            adapter_semantics_valid = False
            errors.append(f"adapter[{index}] cost: {exc}")
        for field in NON_FORGEABLE_FIELDS & set(produces):
            if requires.get(field) != produces.get(field):
                adapter_semantics_valid = False
                errors.append(
                    f"adapter[{index}] forges non-forgeable field {field!r}"
                )
    checks["adapter_semantics_valid"] = adapter_semantics_valid

    checks["installable_compilation"] = compilation.get("repair_candidate") is None

    producer_comp_digest = compilation.get("producer_decision_digest")
    producer_effect_digest = effect.get("producer_decision_digest")
    checks["producer_digest_identity_shape"] = (
        _valid_hex64(producer_comp_digest) and _valid_hex64(producer_effect_digest)
    )

    contract_id = record.get("semantic_contract_id")
    checks["semantic_contract_binding"] = (
        isinstance(contract_id, str)
        and bool(contract_id)
        and effect.get("semantic_contract_id") == contract_id
    )

    checks["effect_matches_intent"] = (
        effect.get("effect_id") == intent.get("effect_id")
        and effect.get("action_name") == intent.get("action_name")
        and effect.get("effect_class") == intent.get("effect_class")
        and effect.get("dependency_version") == intent.get("dependency_version")
    )
    checks["compilation_effect_dependency"] = (
        _valid_hex64(producer_comp_digest)
        and effect.get("dependency_version") == producer_comp_digest
        and intent.get("dependency_version") == producer_comp_digest
    )

    max_steps = profile.get("max_steps")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 0:
        checks["verification_profile"] = False
        errors.append("verification_profile.max_steps must be a non-negative integer")
        max_steps = 0
    else:
        checks["verification_profile"] = True

    selected_path = compilation.get("selected_adapter_path")
    refined_source = compilation.get("refined_source")
    target = compilation.get("target")
    path_shape = (
        isinstance(selected_path, list)
        and all(isinstance(item, str) for item in selected_path)
        and isinstance(refined_source, Mapping)
        and isinstance(target, Mapping)
    )
    checks["selected_path_shape"] = path_shape

    selected_cost = Decimal("0")
    replay_valid = path_shape
    current = dict(refined_source) if isinstance(refined_source, Mapping) else {}
    by_name = {str(item.get("name")): item for item in adapters}
    evidence_keys = {str(key) for key in evidence.keys()}

    if path_shape:
        for name in selected_path:
            adapter = by_name.get(name)
            if adapter is None:
                replay_valid = False
                errors.append(f"selected adapter {name!r} is absent from registry")
                break
            nxt = _apply_adapter(current, adapter, evidence_keys=evidence_keys)
            if nxt is None:
                replay_valid = False
                errors.append(f"selected adapter {name!r} is not admissible")
                break
            try:
                selected_cost += _parse_decimal(
                    adapter.get("cost_decimal"), nonnegative=True
                )
            except ValueError:
                replay_valid = False
                errors.append(f"selected adapter {name!r} has invalid decimal cost")
                break
            current = nxt

    checks["selected_path_replay"] = (
        replay_valid
        and isinstance(target, Mapping)
        and _assignable(current, target)
    )

    minimum_unique = False
    if checks["selected_path_replay"]:
        solutions = _enumerate_solutions(
            refined_source,
            target,
            adapters,
            evidence_keys=evidence_keys,
            max_steps=max_steps,
            max_cost=selected_cost,
        )
        minimum = min((cost for cost, _ in solutions), default=None)
        min_paths = [
            path for cost, path in solutions if minimum is not None and cost == minimum
        ]
        minimum_unique = (
            minimum is not None
            and selected_cost == minimum
            and min_paths == [tuple(selected_path)]
        )
    checks["minimum_cost_unique"] = minimum_unique
    if not minimum_unique:
        errors.append("selected adapter path is not the unique exact-decimal minimum")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        errors.extend(f"check failed: {name}" for name in failed)

    return {
        "schema": BUNDLE_SCHEMA,
        "valid": not failed,
        "checks": checks,
        "errors": list(dict.fromkeys(errors)),
        "selected_cost_decimal": canonical_decimal(selected_cost),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a SemRepair execution proof bundle v0.2"
    )
    parser.add_argument("--input", required=True, help="proof-bundle JSON path")
    parser.add_argument("--json", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = verify_execution_bundle_v2(payload)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
