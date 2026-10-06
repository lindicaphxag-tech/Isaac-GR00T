import json
from pathlib import Path

from research.semantic_invariants.embodied_repair_interactions import (
    RepairOutcome,
    analyze_repair_lattice,
    authorize_repair_subset_across_replicates,
)


EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "embodied_correctness"
    / "evidence"
    / "maniskill_compensating_repairs_v2.json"
)


def _certificates():
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    certificates = []
    for replicate in payload["replicates"]:
        outcomes = tuple(
            RepairOutcome(
                repairs=frozenset(cell["repairs"]),
                value=float(cell["value"]),
                evidence_id=cell["evidence_id"],
            )
            for cell in replicate["cells"]
        )
        certificates.append(
            analyze_repair_lattice(
                subject=payload["subject"],
                metric=payload["metric"]["name"],
                objective=payload["metric"]["objective"],
                repairs=("controller-sign", "converter-representation"),
                outcomes=outcomes,
            )
        )
    return payload, tuple(certificates)


def test_singleton_regressions_are_robust_across_public_replicates():
    payload, certificates = _certificates()
    assert len(certificates) == 2

    converter = authorize_repair_subset_across_replicates(
        certificates, ("converter-representation",)
    )
    controller = authorize_repair_subset_across_replicates(
        certificates, ("controller-sign",)
    )

    assert not converter.authorized
    assert not controller.authorized
    assert converter.unanimous
    assert controller.unanimous
    assert payload["robust_findings"]["singleton_converter_regresses_in_all_replicates"]
    assert payload["robust_findings"]["singleton_controller_regresses_in_all_replicates"]


def test_composed_task_success_is_not_robust_enough_to_authorize():
    payload, certificates = _certificates()

    composed = authorize_repair_subset_across_replicates(
        certificates,
        ("controller-sign", "converter-representation"),
    )

    assert not composed.authorized
    assert not composed.unanimous
    assert any("regresses" in reason for reason in composed.reasons)
    assert payload["robust_findings"]["strict_compensating_bundle_unanimous"] is False
    assert (
        payload["robust_findings"]["task_success_authorization_for_composed"]
        == "fail_closed_ambiguous"
    )


def test_real_stack_replication_never_claims_adoption_or_prospective_i2():
    payload, _ = _certificates()
    assert payload["adoption_credit"] is False
    assert payload["prospective_i2_credit"] is False
    assert payload["adaptive_converter_followup"]["decision"] == "do_not_promote_upstream_yet"
