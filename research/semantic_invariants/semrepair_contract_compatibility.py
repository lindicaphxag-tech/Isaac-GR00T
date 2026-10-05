"""Compatibility rules for versioned SemRepair embodied contracts.

The policy is intentionally conservative. Within one major version, a contract
may gain aliases or executable support, but its canonical semantic relation and
identity cannot silently change.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_ID_RE = re.compile(r"^(?P<path>embodied/.+)@(?P<major>\d+)\.(?P<minor>\d+)$")


def parse_contract_id(value: str) -> tuple[str, int, int]:
    match = _ID_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid canonical contract id: {value!r}")
    return match.group("path"), int(match.group("major")), int(match.group("minor"))


def compare_contract(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    old_id = str(old.get("canonical_id", ""))
    new_id = str(new.get("canonical_id", ""))

    try:
        old_path, old_major, old_minor = parse_contract_id(old_id)
        new_path, new_major, new_minor = parse_contract_id(new_id)
    except ValueError as exc:
        return {"compatible": False, "errors": [str(exc)]}

    if old_path != new_path:
        errors.append("canonical contract path changed")

    if new_major < old_major or (new_major == old_major and new_minor < old_minor):
        errors.append("contract version moved backwards")

    if new_major == old_major:
        if old.get("relation") != new.get("relation"):
            errors.append("semantic relation changed without a major-version bump")

        old_aliases = set(old.get("aliases", []))
        new_aliases = set(new.get("aliases", []))
        removed = sorted(old_aliases - new_aliases)
        if removed:
            errors.append(f"compatibility aliases removed without major bump: {removed!r}")

        if old.get("status") == "executable" and new.get("status") != "executable":
            errors.append("executable contract downgraded within the same major version")

    return {
        "compatible": not errors,
        "errors": errors,
        "old": old_id,
        "new": new_id,
        "requires_major_bump": bool(errors),
    }


def compare_registries(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_by_path: dict[str, dict[str, Any]] = {}
    new_by_path: dict[str, dict[str, Any]] = {}

    def index(payload: dict[str, Any], target: dict[str, dict[str, Any]]) -> list[str]:
        errors: list[str] = []
        for contract in payload.get("contracts", []):
            try:
                path, _, _ = parse_contract_id(str(contract.get("canonical_id", "")))
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if path in target:
                errors.append(f"duplicate canonical contract path: {path}")
            target[path] = contract
        return errors

    errors = index(old, old_by_path) + index(new, new_by_path)
    reports: list[dict[str, Any]] = []

    for path, old_contract in sorted(old_by_path.items()):
        new_contract = new_by_path.get(path)
        if new_contract is None:
            errors.append(f"contract removed from registry without an explicit successor: {path}")
            continue
        report = compare_contract(old_contract, new_contract)
        reports.append(report)
        errors.extend(f"{path}: {item}" for item in report["errors"])

    return {
        "compatible": not errors,
        "errors": errors,
        "checked_contracts": len(reports),
        "new_contracts": sorted(set(new_by_path) - set(old_by_path)),
        "contracts": reports,
        "claim_boundary": (
            "Compatibility here protects canonical semantic identity and public "
            "contract surface. It does not prove behavioral equivalence of arbitrary adapters."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SemRepair contract-registry compatibility")
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    old = json.loads(args.old.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))
    report = compare_registries(old, new)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"compatible={report['compatible']} checked={report['checked_contracts']}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
    if not report["compatible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
