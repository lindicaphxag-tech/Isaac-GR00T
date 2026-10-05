"""Portable CLI for source-to-semantic compilation in external robot stacks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .embodied_semantic_source_bridge import (
    ConflictingSourceSemantics,
    compile_source_boundary,
)
from .embodied_semantic_types import (
    SemanticAdapter,
    SemanticTensorType,
)
from .embodied_source_semantics import SourceSemanticRule
from .embodied_semantic_compiler import SemanticCompilationError


def _semantic_type(payload: dict[str, Any]) -> SemanticTensorType:
    return SemanticTensorType(**payload)


def _rule(payload: dict[str, Any]) -> SourceSemanticRule:
    return SourceSemanticRule(
        rule_id=str(payload["rule_id"]),
        callee_suffix=str(payload["callee_suffix"]),
        facts=dict(payload["facts"]),
        required_string_args={
            int(key): str(value)
            for key, value in dict(payload.get("required_string_args", {})).items()
        }
        or None,
        required_string_kwargs={
            str(key): str(value)
            for key, value in dict(payload.get("required_string_kwargs", {})).items()
        }
        or None,
    )


def _adapter(payload: dict[str, Any]) -> SemanticAdapter:
    return SemanticAdapter(
        name=str(payload["name"]),
        requires=dict(payload.get("requires", {})),
        produces=dict(payload.get("produces", {})),
        cost=float(payload.get("cost", 1.0)),
        effects=tuple(payload.get("effects", ())),
        proof_required=bool(payload.get("proof_required", False)),
        witness_keys=tuple(payload.get("witness_keys", ())),
    )


def _fact_payload(item) -> dict[str, Any]:
    return {
        "field": item.field,
        "value": item.value,
        "rule_id": item.rule_id,
        "line": item.line,
        "callee": item.callee,
    }


def run_source_manifest(
    manifest_path: Path,
    *,
    evidence: dict[str, object] | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Compile one source boundary described by a repository-local JSON manifest."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")

    root = source_root if source_root is not None else manifest_path.parent
    producer_path = root / str(payload["producer"]["path"])
    consumer_path = root / str(payload["consumer"]["path"])

    rules = tuple(_rule(item) for item in payload.get("rules", ()))
    adapters = tuple(_adapter(item) for item in payload.get("adapters", ()))

    result = compile_source_boundary(
        producer_source=producer_path.read_text(encoding="utf-8"),
        consumer_source=consumer_path.read_text(encoding="utf-8"),
        producer_base=_semantic_type(dict(payload["producer"]["base_type"])),
        consumer_base=_semantic_type(dict(payload["consumer"]["base_type"])),
        rules=rules,
        adapters=adapters,
        adapter_evidence=evidence,
    )

    plan = result.compilation.adapter_plan
    return {
        "schema_version": 1,
        "status": result.compilation.status,
        "producer": {
            "path": str(payload["producer"]["path"]),
            "semantic_type": asdict(result.producer.semantic_type),
            "evidence": [_fact_payload(item) for item in result.producer.evidence],
        },
        "consumer": {
            "path": str(payload["consumer"]["path"]),
            "semantic_type": asdict(result.consumer.semantic_type),
            "evidence": [_fact_payload(item) for item in result.consumer.evidence],
        },
        "repair": None
        if plan is None
        else {
            "adapters": [adapter.name for adapter in plan.adapters],
            "total_cost": plan.total_cost,
            "effects": list(plan.effects),
            "evidence_used": list(plan.evidence_used),
        },
        "claim_boundary": (
            "source-level semantic compilation result; project-native behavioral "
            "validation remains required before runtime deployment"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer and compile a semantic boundary from repository source"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evidence-json", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Optional repository root used to resolve producer/consumer paths.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    evidence = None
    if args.evidence_json is not None:
        evidence = json.loads(args.evidence_json.read_text(encoding="utf-8"))

    try:
        report = run_source_manifest(
            args.manifest,
            evidence=evidence,
            source_root=args.source_root,
        )
    except (
        ConflictingSourceSemantics,
        SemanticCompilationError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        if args.json:
            print(json.dumps({"status": "rejected", "error": str(exc)}, indent=2))
        else:
            print(f"REJECTED: {exc}")
        raise SystemExit(2) from exc

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        repair = report["repair"]
        if repair is None:
            print(f"{report['status']}: no repair required")
        else:
            print(
                f"{report['status']}: "
                + " -> ".join(repair["adapters"])
            )


if __name__ == "__main__":
    main()