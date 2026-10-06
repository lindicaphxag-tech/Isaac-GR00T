import json
from pathlib import Path

from research.semantic_invariants.embodied_repair_interactions import (
    RepairOutcome,
    analyze_repair_lattice,
    authorize_repair_subset,
)


EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "evidence"
    / "maniskill_compensating_repairs_v1.json"
)


def _certificate():
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    outcomes = tuple(
        RepairOutcome(
            repairs=frozenset(cell["repairs"]),
            value=float(cell["value"]),
            evidence_id=cell["evidence_id"],
        )
        for cell in payload["cells"]
    )
    return payload, analyze_repair_lattice(
        subject=payload["subject"],
        metric=payload["metric"]["name"],
        objective=payload["metric"]["objective"],
        repairs=("controller-sign", "converter-representation"),
        outcomes=outcomes,
    )


def test_public_maniskill_factorial_evidence_is_strict_compensating_bundle():
    payload, cert = _certificate()
    assert payload["public_run"]["conclusion"] == "success"
    assert cert.has_repair_paradox
    assert len(cert.compensating_bundles) == 1
    bundle = cert.compensating_bundles[0]
    assert bundle.repairs == ("controller-sign", "converter-representation")
    assert bundle.masking_mode == "complete"


def test_public_maniskill_evidence_enforces_repair_set_atomicity():
    _, cert = _certificate()
    converter = authorize_repair_subset(cert, ("converter-representation",))
    controller = authorize_repair_subset(cert, ("controller-sign",))
    composed = authorize_repair_subset(
        cert, ("controller-sign", "converter-representation")
    )
    assert not converter.authorized
    assert not controller.authorized
    assert composed.authorized
    assert any("incomplete subset" in reason for reason in converter.reasons)
    assert any("incomplete subset" in reason for reason in controller.reasons)


def test_real_stack_capsule_never_claims_external_adoption_or_prospective_i2():
    payload, _ = _certificate()
    assert payload["adoption_credit"] is False
    assert payload["prospective_i2_credit"] is False
