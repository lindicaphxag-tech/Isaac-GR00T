from pathlib import Path

from research.semantic_invariants.semrepair_adoption_check import (
    validate_adoption_manifest,
)


def _manifest():
    return {
        "schema_version": 1,
        "repository": "example/robot-stack",
        "integration_type": "native-regression",
        "semantic_layer": "controller/action-representation",
        "contract_ids": ["embodied/representation/controller-roundtrip@0.2"],
        "maintained_path": "tests/test_semantic_contract.py",
        "status": "candidate",
        "claim_boundary": "candidate only",
    }


def test_local_manifest_never_counts_as_external_adoption(tmp_path: Path):
    path = tmp_path / "tests" / "test_semantic_contract.py"
    path.parent.mkdir(parents=True)
    path.write_text("def test_placeholder(): pass\n", encoding="utf-8")

    report = validate_adoption_manifest(
        _manifest(),
        repository_root=tmp_path,
    )

    assert report["valid"]
    assert report["summary"]["counts_as_external_adoption"] is False
    assert "not external adoption evidence" in report["claim_boundary"]


def test_missing_maintained_path_fails_validation(tmp_path: Path):
    report = validate_adoption_manifest(
        _manifest(),
        repository_root=tmp_path,
    )
    assert not report["valid"]
    assert any("does not exist" in error for error in report["errors"])


def test_path_escape_fails_closed(tmp_path: Path):
    manifest = _manifest()
    manifest["maintained_path"] = "../outside.py"
    report = validate_adoption_manifest(
        manifest,
        repository_root=tmp_path,
    )
    assert not report["valid"]
    assert any("escapes" in error for error in report["errors"])
