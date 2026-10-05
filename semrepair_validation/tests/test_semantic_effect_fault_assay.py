from research.semantic_invariants.semantic_effect_fault_assay import run_fault_assay


def test_fault_assay_exposes_retry_safety_liveness_tradeoff():
    report = run_fault_assay()
    aggregate = report["aggregate"]

    retry = aggregate["retry_on_timeout"]
    sec = aggregate["semantic_effect_commit"]
    fail_stop = aggregate["fail_stop"]

    assert retry["duplicate_effects"] >= 3
    assert retry["unsafe_retries"] >= 3

    assert sec["duplicate_effects"] == 0
    assert sec["unsafe_retries"] == 0
    assert sec["task_completions"] >= fail_stop["task_completions"]


def test_semantic_effect_commit_recovers_definite_abort_but_blocks_unresolved_success():
    report = run_fault_assay()
    rows = [
        row for row in report["outcomes"]
        if row["policy"] == "semantic_effect_commit"
    ]
    by_name = {row["scenario"]: row for row in rows}

    definite_abort = by_name["definite_abort"]
    assert definite_abort["logical_complete"]
    assert definite_abort["retries"] == 1
    assert definite_abort["duplicate_effects"] == 0

    restart = by_name["success_then_runtime_restart"]
    assert restart["blocked"]
    assert restart["retries"] == 0
    assert restart["duplicate_effects"] == 0


def test_correlated_or_incomplete_evidence_fails_closed():
    report = run_fault_assay()
    rows = {
        row["scenario"]: row
        for row in report["outcomes"]
        if row["policy"] == "semantic_effect_commit"
    }

    assert rows["success_ack_lost_one_plane_stale"]["blocked"]
    assert rows["success_correlated_camera_false_negative"]["blocked"]