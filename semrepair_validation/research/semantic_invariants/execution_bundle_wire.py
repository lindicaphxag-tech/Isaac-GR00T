"""Language-neutral SemRepair execution proof bundles.

The wire verifier intentionally re-implements certificate and adapter-path
checks over plain JSON-compatible mappings. It does not invoke the production
compiler/search code. This makes the verification algorithm portable to
non-Python consumers.

Integrity digests provide deterministic tamper detection and environment
binding. They are NOT digital signatures and do not authenticate the issuer.
"""

from __future__ import annotations

from hashlib import sha256
import json
from math import isclose
from typing import Any, Mapping, Sequence


BUNDLE_SCHEMA = "semrepair-execution-proof-bundle/v0.1"
COMPILATION_SCHEMA = "embodied-semantic-compilation-certificate/v0.1"
EFFECT_SCHEMA = "semantic-effect-commit-certificate/v0.1"
SPEC_VERSION = "0.2"

NON_FORGEABLE_FIELDS = frozenset({"provenance", "freshness"})

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


def canonical_json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _adapter_record(adapter: object) -> dict[str, object]:
    return {
        "name": str(getattr(adapter, "name")),
        "requires": dict(sorted(dict(getattr(adapter, "requires")).items())),
        "produces": dict(sorted(dict(getattr(adapter, "produces")).items())),
        "cost": float(getattr(adapter, "cost")),
        "effects": list(getattr(adapter, "effects")),
        "required_evidence": list(getattr(adapter, "required_evidence")),
    }


def _compilation_record(certificate: object) -> dict[str, object]:
    return {
        "schema": getattr(certificate, "schema"),
        "compiler_version": getattr(certificate, "compiler_version"),
        "registry_digest": getattr(certificate, "registry_digest"),
        "evidence_digest": getattr(certificate, "evidence_digest"),
        "source": dict(getattr(certificate, "source")),
        "target": dict(getattr(certificate, "target")),
        "inferred_source": dict(getattr(certificate, "inferred_source")),
        "refined_source": dict(getattr(certificate, "refined_source")),
        "selected_adapter_path": list(getattr(certificate, "selected_adapter_path")),
        "selected_cost": getattr(certificate, "selected_cost"),
        "repair_candidate": getattr(certificate, "repair_candidate"),
        "repair_families": list(getattr(certificate, "repair_families")),
        "refinement_records": list(getattr(certificate, "refinement_records")),
        "effects": list(getattr(certificate, "effects")),
        "decision_digest": getattr(certificate, "decision_digest"),
    }


def _effect_record(certificate: object) -> dict[str, object]:
    return {
        "schema": getattr(certificate, "schema"),
        "effect_id": getattr(certificate, "effect_id"),
        "action_name": getattr(certificate, "action_name"),
        "effect_class": getattr(certificate, "effect_class"),
        "dependency_version": getattr(certificate, "dependency_version"),
        "policy": dict(getattr(certificate, "policy")),
        "semantic_contract_id": getattr(certificate, "semantic_contract_id"),
        "issuer_version": getattr(certificate, "issuer_version"),
        "decision_digest": getattr(certificate, "decision_digest"),
    }


def export_execution_bundle(
    compilation_certificate: object,
    effect_certificate: object,
    intent: object,
    *,
    semantic_contract_id: str,
    adapters: Sequence[object],
    evidence_identity: Mapping[str, str] | None = None,
    max_steps: int = 6,
    atol: float = 1.0e-12,
) -> dict[str, object]:
    """Export a canonical JSON-compatible proof bundle."""
    if not semantic_contract_id:
        raise ValueError("semantic_contract_id must be non-empty")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if atol < 0:
        raise ValueError("atol must be non-negative")

    adapter_records = [_adapter_record(item) for item in adapters]
    adapter_records.sort(key=lambda item: (str(item["name"]), canonical_json_digest(item)))

    effect_class = getattr(getattr(intent, "effect_class"), "value", getattr(intent, "effect_class"))

    payload: dict[str, object] = {
        "schema": BUNDLE_SCHEMA,
        "spec_version": SPEC_VERSION,
        "semantic_contract_id": semantic_contract_id,
        "compilation_certificate": _compilation_record(compilation_certificate),
        "effect_certificate": _effect_record(effect_certificate),
        "intent": {
            "effect_id": getattr(intent, "effect_id"),
            "action_name": getattr(intent, "action_name"),
            "effect_class": effect_class,
            "dependency_version": getattr(intent, "dependency_version"),
        },
        "adapter_registry": adapter_records,
        "evidence_identity": dict(sorted(dict(evidence_identity or {}).items())),
        "verification_profile": {
            "max_steps": max_steps,
            "atol": float(atol),
        },
    }
    return {**payload, "bundle_digest": canonical_json_digest(payload)}


def _compilation_payload(record: Mapping[str, Any]) -> dict[str, object]:
    return {
        "schema": record.get("schema"),
        "compiler_version": record.get("compiler_version"),
        "registry_digest": record.get("registry_digest"),
        "evidence_digest": record.get("evidence_digest"),
        "source": record.get("source"),
        "target": record.get("target"),
        "inferred_source": record.get("inferred_source"),
        "refined_source": record.get("refined_source"),
        "selected_adapter_path": record.get("selected_adapter_path"),
        "selected_cost": record.get("selected_cost"),
        "repair_candidate": record.get("repair_candidate"),
        "repair_families": record.get("repair_families"),
        "refinement_records": record.get("refinement_records"),
        "effects": record.get("effects"),
    }


def _effect_payload(record: Mapping[str, Any]) -> dict[str, object]:
    return {
        "schema": record.get("schema"),
        "effect_id": record.get("effect_id"),
        "action_name": record.get("action_name"),
        "effect_class": record.get("effect_class"),
        "dependency_version": record.get("dependency_version"),
        "policy": record.get("policy"),
        "semantic_contract_id": record.get("semantic_contract_id"),
        "issuer_version": record.get("issuer_version"),
    }


def _registry_digest(records: Sequence[Mapping[str, Any]]) -> str:
    normalized = []
    for item in records:
        normalized.append(
            {
                "name": item.get("name"),
                "requires": dict(sorted(dict(item.get("requires") or {}).items())),
                "produces": dict(sorted(dict(item.get("produces") or {}).items())),
                "cost": float(item.get("cost", 0.0)),
                "effects": list(item.get("effects") or []),
                "required_evidence": list(item.get("required_evidence") or []),
            }
        )
    normalized.sort(key=lambda item: (str(item["name"]), canonical_json_digest(item)))
    return canonical_json_digest(normalized)


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
    requires = dict(adapter.get("requires") or {})
    if any(current.get(key) != value for key, value in requires.items()):
        return None
    required_evidence = set(adapter.get("required_evidence") or [])
    if required_evidence - evidence_keys:
        return None
    nxt = dict(current)
    nxt.update(dict(adapter.get("produces") or {}))
    return nxt


def _enumerate_solutions(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    adapters: Sequence[Mapping[str, Any]],
    *,
    evidence_keys: set[str],
    max_steps: int,
    max_cost: float,
) -> list[tuple[float, tuple[str, ...]]]:
    """Independent exhaustive simple-path enumeration over semantic types."""
    if _assignable(source, target):
        return [(0.0, ())]

    def key(value: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple((field, value.get(field)) for field in SEMANTIC_FIELDS)

    solutions: set[tuple[float, tuple[str, ...]]] = set()
    stack: list[tuple[dict[str, Any], tuple[str, ...], float, int, frozenset[tuple[tuple[str, Any], ...]]]] = [
        (dict(source), (), 0.0, 0, frozenset((key(source),)))
    ]

    while stack:
        current, path, cost, steps, visited = stack.pop()
        if cost > max_cost:
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
            new_cost = cost + float(adapter.get("cost", 0.0))
            if new_cost > max_cost:
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


def verify_execution_bundle(record: Mapping[str, Any]) -> dict[str, object]:
    """Verify a wire bundle without calling SemRepair compiler/search code."""
    errors: list[str] = []
    checks: dict[str, bool] = {}

    checks["bundle_schema"] = record.get("schema") == BUNDLE_SCHEMA
    if not checks["bundle_schema"]:
        errors.append("unsupported bundle schema")

    checks["spec_version"] = record.get("spec_version") == SPEC_VERSION
    if not checks["spec_version"]:
        errors.append("unsupported semantic spec version")

    outer = dict(record)
    claimed_bundle_digest = outer.pop("bundle_digest", None)
    checks["bundle_digest"] = (
        isinstance(claimed_bundle_digest, str)
        and canonical_json_digest(outer) == claimed_bundle_digest
    )
    if not checks["bundle_digest"]:
        errors.append("bundle digest mismatch")

    compilation = record.get("compilation_certificate")
    effect = record.get("effect_certificate")
    intent = record.get("intent")
    adapters = record.get("adapter_registry")
    evidence_identity = record.get("evidence_identity")
    profile = record.get("verification_profile")

    if not isinstance(compilation, Mapping):
        errors.append("compilation_certificate must be an object")
        compilation = {}
    if not isinstance(effect, Mapping):
        errors.append("effect_certificate must be an object")
        effect = {}
    if not isinstance(intent, Mapping):
        errors.append("intent must be an object")
        intent = {}
    if not isinstance(adapters, list) or not all(isinstance(item, Mapping) for item in adapters):
        errors.append("adapter_registry must be an array of objects")
        adapters = []
    if not isinstance(evidence_identity, Mapping):
        errors.append("evidence_identity must be an object")
        evidence_identity = {}
    if not isinstance(profile, Mapping):
        errors.append("verification_profile must be an object")
        profile = {}

    checks["compilation_schema"] = compilation.get("schema") == COMPILATION_SCHEMA
    if not checks["compilation_schema"]:
        errors.append("unsupported compilation certificate schema")

    comp_digest = compilation.get("decision_digest")
    checks["compilation_digest"] = (
        isinstance(comp_digest, str)
        and canonical_json_digest(_compilation_payload(compilation)) == comp_digest
    )
    if not checks["compilation_digest"]:
        errors.append("compilation certificate digest mismatch")

    checks["registry_digest"] = (
        compilation.get("registry_digest") == _registry_digest(adapters)
    )
    if not checks["registry_digest"]:
        errors.append("adapter registry digest mismatch")

    checks["evidence_digest"] = (
        compilation.get("evidence_digest")
        == canonical_json_digest(dict(sorted(dict(evidence_identity).items())))
    )
    if not checks["evidence_digest"]:
        errors.append("evidence identity digest mismatch")

    names = [str(item.get("name")) for item in adapters]
    checks["unique_adapter_names"] = len(names) == len(set(names))
    if not checks["unique_adapter_names"]:
        errors.append("adapter registry contains duplicate names")

    adapter_records_valid = True
    for index, adapter in enumerate(adapters):
        requires = dict(adapter.get("requires") or {})
        produces = dict(adapter.get("produces") or {})
        unknown = (set(requires) | set(produces)) - set(SEMANTIC_FIELDS)
        if unknown:
            adapter_records_valid = False
            errors.append(
                f"adapter[{index}] uses unknown semantic fields: {tuple(sorted(unknown))!r}"
            )
        try:
            cost = float(adapter.get("cost", 0.0))
        except (TypeError, ValueError):
            cost = -1.0
        if cost < 0:
            adapter_records_valid = False
            errors.append(f"adapter[{index}] has negative or invalid cost")
        for field in NON_FORGEABLE_FIELDS & set(produces):
            if requires.get(field) != produces.get(field):
                adapter_records_valid = False
                errors.append(
                    f"adapter[{index}] forges non-forgeable field {field!r}"
                )
    checks["adapter_semantics_valid"] = adapter_records_valid

    checks["installable_compilation"] = compilation.get("repair_candidate") is None
    if not checks["installable_compilation"]:
        errors.append("compilation certificate still contains an unverified repair candidate")

    max_steps = profile.get("max_steps", 6)
    atol = profile.get("atol", 1.0e-12)
    if not isinstance(max_steps, int) or max_steps < 0:
        errors.append("verification_profile.max_steps must be a non-negative integer")
        max_steps = 0
    if not isinstance(atol, (int, float)) or float(atol) < 0:
        errors.append("verification_profile.atol must be non-negative")
        atol = 0.0
    atol = float(atol)

    selected_path = compilation.get("selected_adapter_path")
    selected_cost = compilation.get("selected_cost")
    refined_source = compilation.get("refined_source")
    target = compilation.get("target")

    path_ok = (
        isinstance(selected_path, list)
        and all(isinstance(item, str) for item in selected_path)
        and isinstance(refined_source, Mapping)
        and isinstance(target, Mapping)
        and isinstance(selected_cost, (int, float))
    )
    if not path_ok:
        errors.append("compilation path/source/target/cost record is malformed")
    else:
        by_name = {str(item.get("name")): item for item in adapters}
        current = dict(refined_source)
        replay_cost = 0.0
        replay_valid = True
        for name in selected_path:
            adapter = by_name.get(name)
            if adapter is None:
                replay_valid = False
                errors.append(f"selected path references unknown adapter {name!r}")
                break
            nxt = _apply_adapter(current, adapter, evidence_keys=set(evidence_identity))
            if nxt is None:
                replay_valid = False
                errors.append(f"selected adapter {name!r} is not admissible")
                break
            current = nxt
            replay_cost += float(adapter.get("cost", 0.0))

        checks["selected_path_replay"] = (
            replay_valid and _assignable(current, target)
        )
        if not checks["selected_path_replay"]:
            errors.append("selected adapter path does not reach target semantics")

        checks["selected_cost"] = isclose(
            replay_cost,
            float(selected_cost),
            rel_tol=0.0,
            abs_tol=atol,
        )
        if not checks["selected_cost"]:
            errors.append("selected adapter cost mismatch")

        solutions = _enumerate_solutions(
            refined_source,
            target,
            adapters,
            evidence_keys=set(evidence_identity),
            max_steps=max_steps,
            max_cost=float(selected_cost) + atol,
        )
        minimum = min((cost for cost, _ in solutions), default=None)
        selected_tuple = tuple(selected_path)
        min_paths = [
            path
            for cost, path in solutions
            if minimum is not None and isclose(cost, minimum, rel_tol=0.0, abs_tol=atol)
        ]
        checks["minimum_cost_unique"] = (
            minimum is not None
            and isclose(float(selected_cost), minimum, rel_tol=0.0, abs_tol=atol)
            and min_paths == [selected_tuple]
        )
        if not checks["minimum_cost_unique"]:
            errors.append("selected adapter path is not the unique minimum-cost solution")

    checks["effect_schema"] = effect.get("schema") == EFFECT_SCHEMA
    if not checks["effect_schema"]:
        errors.append("unsupported effect certificate schema")

    effect_digest = effect.get("decision_digest")
    checks["effect_digest"] = (
        isinstance(effect_digest, str)
        and canonical_json_digest(_effect_payload(effect)) == effect_digest
    )
    if not checks["effect_digest"]:
        errors.append("effect certificate digest mismatch")

    intent_fields = ("effect_id", "action_name", "effect_class", "dependency_version")
    checks["effect_matches_intent"] = all(
        effect.get(field) == intent.get(field) for field in intent_fields
    )
    if not checks["effect_matches_intent"]:
        errors.append("effect certificate does not match intent identity")

    checks["semantic_contract_binding"] = (
        effect.get("semantic_contract_id") == record.get("semantic_contract_id")
    )
    if not checks["semantic_contract_binding"]:
        errors.append("semantic contract id mismatch")

    checks["compilation_effect_dependency"] = (
        isinstance(comp_digest, str)
        and effect.get("dependency_version") == comp_digest
        and intent.get("dependency_version") == comp_digest
    )
    if not checks["compilation_effect_dependency"]:
        errors.append("effect dependency is not bound to compilation decision")

    return {
        "schema": BUNDLE_SCHEMA,
        "valid": not errors and all(checks.values()),
        "errors": errors,
        "checks": checks,
        "summary": {
            "semantic_contract_id": record.get("semantic_contract_id"),
            "selected_adapter_path": selected_path,
            "adapter_count": len(adapters),
            "evidence_identity_count": len(evidence_identity),
        },
    }


def dumps_execution_bundle(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Verify a language-neutral SemRepair execution proof bundle"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    record = json.loads(args.input.read_text(encoding="utf-8"))
    report = verify_execution_bundle(record)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"valid={report['valid']} "
            f"checks={sum(report['checks'].values())}/{len(report['checks'])}"
        )
        for error in report["errors"]:
            print(f"ERROR: {error}")

    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()