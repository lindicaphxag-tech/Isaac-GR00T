"""CLI for source-level execution-path semantic-effect parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embodied_source_effects import (
    DEFAULT_PATH_EFFECT_RULES,
    compare_effect_signatures,
    infer_source_effects,
)


def _fact(item):
    return {
        "effect": item.effect,
        "rule_id": item.rule_id,
        "callee": item.callee,
        "line": item.line,
        "details": dict(item.details),
    }


def _read_bundle(paths: tuple[Path, ...]) -> str:
    if not paths:
        raise ValueError("execution path must contain at least one source file")
    return "\n\n".join(path.read_text(encoding="utf-8") for path in paths)


def run_effect_parity(
    *,
    reference_sources: tuple[Path, ...],
    alternative_sources: tuple[Path, ...],
) -> dict[str, object]:
    reference = infer_source_effects(
        _read_bundle(reference_sources),
        DEFAULT_PATH_EFFECT_RULES,
    )
    alternative = infer_source_effects(
        _read_bundle(alternative_sources),
        DEFAULT_PATH_EFFECT_RULES,
    )
    comparison = compare_effect_signatures(reference, alternative)
    return {
        "schema_version": 1,
        "reference": [_fact(item) for item in reference],
        "alternative": [_fact(item) for item in alternative],
        "shared_effects": list(comparison.shared_effects),
        "reference_only": list(comparison.reference_only),
        "alternative_only": list(comparison.alternative_only),
        "symmetric": comparison.symmetric,
        "claim_boundary": (
            "static source-effect asymmetry only; owner-boundary behavioral "
            "parity must be tested before classifying a defect"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare semantic-effect signatures of two execution paths"
    )
    parser.add_argument(
        "--reference-source",
        required=True,
        type=Path,
        action="append",
        dest="reference_sources",
        help="Source file in the reference execution path; may be repeated.",
    )
    parser.add_argument(
        "--alternative-source",
        required=True,
        type=Path,
        action="append",
        dest="alternative_sources",
        help="Source file in the alternative execution path; may be repeated.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_effect_parity(
        reference_sources=tuple(args.reference_sources),
        alternative_sources=tuple(args.alternative_sources),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"symmetric={report['symmetric']} "
            f"reference_only={report['reference_only']} "
            f"alternative_only={report['alternative_only']}"
        )


if __name__ == "__main__":
    main()