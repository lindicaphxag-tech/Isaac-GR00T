"""Command-line analysis for complete factorial semantic-repair evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .embodied_repair_interactions import RepairOutcome, analyze_repair_lattice


def analyze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")

    outcomes_raw = payload.get("outcomes")
    if not isinstance(outcomes_raw, list) or not outcomes_raw:
        raise ValueError("outcomes must be a non-empty list")

    outcomes = tuple(
        RepairOutcome(
            repairs=frozenset(str(x) for x in item["repairs"]),
            value=float(item["value"]),
            evidence_id=str(item["evidence_id"]),
        )
        for item in outcomes_raw
    )

    cert = analyze_repair_lattice(
        subject=str(payload["subject"]),
        metric=str(payload["metric"]),
        objective=str(payload["objective"]),
        repairs=tuple(str(x) for x in payload["repairs"]),
        outcomes=outcomes,
        tolerance=float(payload.get("tolerance", 0.0)),
    )

    return {
        "schema_version": 1,
        "subject": cert.subject,
        "metric": cert.metric,
        "objective": cert.objective,
        "repairs": list(cert.repairs),
        "has_repair_paradox": cert.has_repair_paradox,
        "edge_violations": [
            {
                "before": list(edge.before),
                "added_repair": edge.added_repair,
                "after": list(edge.after),
                "before_loss": edge.before_loss,
                "after_loss": edge.after_loss,
                "regression": edge.regression,
            }
            for edge in cert.edge_violations
        ],
        "compensating_bundles": [
            {
                "repairs": list(bundle.repairs),
                "baseline_loss": bundle.baseline_loss,
                "repaired_loss": bundle.repaired_loss,
                "proper_subset_losses": [
                    {"repairs": list(names), "loss": loss}
                    for names, loss in bundle.proper_subset_losses
                ],
                "masking_mode": bundle.masking_mode,
                "closure_gain": bundle.closure_gain,
                "behaviorally_masked": bundle.behaviorally_masked,
            }
            for bundle in cert.compensating_bundles
        ],
        "mobius_terms": [
            {"repairs": list(names), "value": value}
            for names, value in cert.mobius_terms
        ],
        "digest": cert.digest,
        "claim_boundary": (
            "This command classifies supplied factorial evidence. It does not "
            "establish that the measurements are independent, prospective, or "
            "maintainer-retained."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("input root must be a JSON object")

    result = analyze_payload(payload)
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"subject: {result['subject']}")
        print(f"repair paradox: {result['has_repair_paradox']}")
        for bundle in result["compensating_bundles"]:
            names = ", ".join(bundle["repairs"])
            print(
                f"bundle [{names}] mode={bundle['masking_mode']} "
                f"closure_gain={bundle['closure_gain']}"
            )
        print(f"digest: {result['digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
