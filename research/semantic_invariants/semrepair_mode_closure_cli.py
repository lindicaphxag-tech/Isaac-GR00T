"""CLI for source-derived enum/runtime semantic closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embodied_mode_closure_source import infer_enum_mode_closure


def run_mode_closure(
    *,
    enum_source: Path,
    runtime_source: Path,
    enum_name: str,
    explicit_default: tuple[str, ...] = (),
    explicit_rejected: tuple[str, ...] = (),
) -> dict[str, object]:
    result = infer_enum_mode_closure(
        enum_source=enum_source.read_text(encoding="utf-8"),
        runtime_source=runtime_source.read_text(encoding="utf-8"),
        enum_name=enum_name,
        explicit_default=explicit_default,
        explicit_rejected=explicit_rejected,
    )
    return {
        "schema_version": 1,
        "enum_name": result.enum_name,
        "declared": sorted(result.declared),
        "handled": sorted(result.handled),
        "explicit_default": sorted(result.explicit_default),
        "rejected": sorted(result.rejected),
        "unresolved": sorted(result.unresolved),
        "closed": result.closed,
        "claim_boundary": (
            "source-derived semantic closure; unresolved modes require independent "
            "semantic evidence before being classified as defects"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer enum-backed runtime semantic closure from source"
    )
    parser.add_argument("--enum-source", required=True, type=Path)
    parser.add_argument("--runtime-source", required=True, type=Path)
    parser.add_argument("--enum-name", required=True)
    parser.add_argument("--default", action="append", default=[])
    parser.add_argument("--rejected", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_mode_closure(
        enum_source=args.enum_source,
        runtime_source=args.runtime_source,
        enum_name=args.enum_name,
        explicit_default=tuple(args.default),
        explicit_rejected=tuple(args.rejected),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"{report['enum_name']}: closed={report['closed']} "
            f"unresolved={report['unresolved']}"
        )


if __name__ == "__main__":
    main()