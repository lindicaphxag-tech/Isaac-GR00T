import pytest

from casj.selective import SelectiveRepairRecord, evaluate_selective_repair


def test_selective_repair_metrics_separate_harm_from_coverage():
    records = [
        SelectiveRepairRecord(True, 1.0, 0.2, 0.0, label="accepted-good"),
        SelectiveRepairRecord(False, 1.0, 2.0, 0.0, label="rejected-harm"),
        SelectiveRepairRecord(False, 1.0, 0.5, 0.0, label="rejected-benefit"),
        SelectiveRepairRecord(True, 1.0, 1.5, 0.0, label="accepted-harm"),
    ]

    report = evaluate_selective_repair(records, requery_cost_ratios=(0.2,))

    assert report["coverage"] == pytest.approx(0.5)
    assert report["raw_harm_rate"] == pytest.approx(0.5)
    assert report["false_accept_count"] == 1
    assert report["false_accept_rate_given_acceptance"] == pytest.approx(0.5)
    assert report["harmful_repair_rejection_rate"] == pytest.approx(0.5)
    assert report["beneficial_repair_rejection_rate"] == pytest.approx(1.0)
    assert report["fallback_integrity_rate_on_rejection"] == pytest.approx(1.0)
    assert report["accepted_mean_raw_to_stale_ratio"] == pytest.approx(0.85)

    point = report["cost_aware_risk_curve"][0]
    assert point["selective_normalized_risk"] == pytest.approx(0.525)
    assert point["always_raw_normalized_risk"] == pytest.approx(1.05)
    assert point["always_fallback_normalized_risk"] == pytest.approx(0.2)
    assert point["stale_normalized_risk"] == pytest.approx(1.0)
    assert point["query_savings_vs_always_fallback"] == pytest.approx(0.5)


def test_all_refusal_is_safe_but_has_zero_coverage():
    records = [
        SelectiveRepairRecord(False, 2.0, 10.0, 0.0, True, "a"),
        SelectiveRepairRecord(False, 4.0, 8.0, 0.0, True, "b"),
    ]

    report = evaluate_selective_repair(records, requery_cost_ratios=(0.0, 0.25))

    assert report["coverage"] == 0.0
    assert report["requery_rate"] == 1.0
    assert report["raw_harm_rate"] == 1.0
    assert report["false_accept_rate_given_acceptance"] is None
    assert report["false_accept_wilson95_upper_given_acceptance"] is None
    assert report["harmful_repair_rejection_rate"] == 1.0
    assert report["fallback_integrity_rate_on_rejection"] == 1.0
    assert report["cost_aware_risk_curve"][0]["selective_normalized_risk"] == 0.0
    assert report["cost_aware_risk_curve"][1]["selective_normalized_risk"] == 0.25


@pytest.mark.parametrize(
    "kwargs",
    [
        {"records": []},
        {
            "records": [SelectiveRepairRecord(False, 1.0, 1.0, 0.0)],
            "requery_cost_ratios": (-0.1,),
        },
        {"records": [SelectiveRepairRecord(False, -1.0, 1.0, 0.0)]},
    ],
)
def test_selective_repair_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        evaluate_selective_repair(**kwargs)
