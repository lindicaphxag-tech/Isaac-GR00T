from pathlib import Path

from research.semantic_invariants.semrepair_adoption_cli import (
    build_adoption_manifest,
    render_native_regression,
)


def test_adoption_manifest_is_explicitly_candidate_not_self_awarded():
    manifest = build_adoption_manifest(
        repository="example/robot-stack",
        contract_id="embodied/provenance/executed-action@0.2",
        integration_type="native-regression",
        maintained_path="tests/test_action_provenance.py",
        semantic_layer="dataset/safety provenance",
    )

    assert manifest["status"] == "candidate"
    assert manifest["repository"] == "example/robot-stack"
    assert manifest["contract_ids"] == [
        "embodied/provenance/executed-action@0.2"
    ]
    assert "only when maintained" in manifest["claim_boundary"]


def test_native_regression_scaffold_never_silently_passes():
    text = render_native_regression(
        "embodied/representation/controller-roundtrip@0.2"
    )
    assert "NotImplementedError" in text
    assert "CONTRACT_ID" in text
