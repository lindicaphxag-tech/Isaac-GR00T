"""Validate a SemRepair external-adoption manifest conservatively."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_ALLOWED_INTEGRATIONS = {
    "native-regression",
    "source-boundary-ci",
    "runtime-refinement",
}
_ALLOWED_STATUS = {"candidate", "maintained"}


def validate_adoption_manifest(
    payload: dict[str, Any],
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []

    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    repository = payload.get("repository")
    if not isinstance(repository, str) or repository.count("/") != 1:
        errors.append("repository must be owner/name")

    integration_type = payload.get("integration_type")
    if integration_type not in _ALLOWED_INTEGRATIONS:
        errors.append("invalid integration_type")

    semantic_layer = payload.get("semantic_layer")
    if not isinstance(semantic_layer, str) or not semantic_layer.strip():
        errors.append("semantic_layer must be non-empty")

    contract_ids = payload.get("contract_ids")
    if (
        not isinstance(contract_ids, list)
        or not contract_ids
        or any(not isinstance(item, str) or not item for item in contract_ids)
        or len(set(contract_ids)) != len(contract_ids)
    ):
        errors.append("contract_ids must be a non-empty unique string list")

    maintained_path = payload.get("maintained_path")
    if not isinstance(maintained_path, str) or not maintained_path:
        errors.append("maintained_path must be non-empty")
    elif repository_root is not None:
        resolved = (repository_root / maintained_path).resolve()
        root = repository_root.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            errors.append("maintained_path escapes repository_root")
        else:
            if not resolved.exists():
                errors.append(f"maintained_path does not exist: {maintained_path}")

    status = payload.get("status")
    if status not in _ALLOWED_STATUS:
        errors.append("status must be candidate or maintained")

    claim = payload.get("claim_boundary")
    if not isinstance(claim, str) or not claim.strip():
        errors.append("claim_boundary must be non-empty")

    return {
        "valid": not errors,
        "errors": errors,
        "summary": {
            "repository": repository,
            "integration_type": integration_type,
            "semantic_layer": semantic_layer,
            "contracts": len(contract_ids) if isinstance(contract_ids, list) else 0,
            "status": status,
            "counts_as_external_adoption": False,
        },
        "claim_boundary": (
            "A locally valid manifest is not external adoption evidence. "
            "External adoption requires the named independent repository to "
            "retain the declared integration."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a SemRepair adoption manifest"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = validate_adoption_manifest(
        payload,
        repository_root=args.repository_root,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"{report['summary']['repository']}: "
            f"{report['summary']['status']} / valid={report['valid']}"
        )
        for error in report["errors"]:
            print(f"ERROR: {error}")

    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
