import json
from pathlib import Path

import pytest

from research.semantic_invariants.semrepair_repair_lattice_cli import analyze_payload


ROOT = Path(__file__).resolve().parents[1]


def test_public_maniskill_factorial_example_is_classified_as_partial_masking():
    payload = json.loads(
        (ROOT / "examples/maniskill_pr1472_pr1495_factorial.json").read_text(
            encoding="utf-8"
        )
    )

    result = analyze_payload(payload)

    assert result["has_repair_paradox"]
    assert len(result["compensating_bundles"]) == 1
    bundle = result["compensating_bundles"][0]
    assert bundle["repairs"] == [
        "pr1472-controller-sign",
        "pr1495-xyz-euler",
    ]
    assert bundle["masking_mode"] == "partial"
    assert not bundle["behaviorally_masked"]

    singleton = {
        row["added_repair"]: row["regression"]
        for row in result["edge_violations"]
        if row["before"] == []
    }
    assert singleton["pr1472-controller-sign"] > 50.0
    assert singleton["pr1495-xyz-euler"] > 50.0

    pair = next(
        row
        for row in result["mobius_terms"]
        if set(row["repairs"])
        == {"pr1472-controller-sign", "pr1495-xyz-euler"}
    )
    assert pair["value"] < -100.0
    assert result["digest"]


def test_cli_payload_requires_complete_factorial_lattice():
    payload = {
        "schema_version": 1,
        "subject": "missing cell",
        "metric": "error",
        "objective": "minimize",
        "repairs": ["a", "b"],
        "outcomes": [
            {"repairs": [], "value": 1.0, "evidence_id": "baseline"},
            {"repairs": ["a"], "value": 2.0, "evidence_id": "a"},
            {"repairs": ["a", "b"], "value": 0.0, "evidence_id": "ab"},
        ],
    }

    with pytest.raises(ValueError, match="complete"):
        analyze_payload(payload)
