from research.semantic_invariants.embodied_behavioral_evidence import (
    build_behavioral_evidence,
)


def test_behavioral_evidence_covers_three_distinct_semantic_failure_classes():
    report = build_behavioral_evidence()
    rows = {item["family"]: item for item in report["comparisons"]}

    assert set(rows) == {
        "joint-order",
        "rotation-representation-plus-sign",
        "sensor-freshness",
    }
    assert report["all_improve"]

    assert rows["joint-order"]["repaired_metric"] == 0.0
    assert rows["sensor-freshness"]["repaired_metric"] == 0.0
    assert rows["rotation-representation-plus-sign"]["repaired_metric"] < 1.0e-5


def test_behavioral_report_does_not_overclaim_real_robot_validation():
    report = build_behavioral_evidence()
    assert "not a robotics benchmark" in report["claim_boundary"]
    assert "not evidence of real-stack" in report["claim_boundary"]
