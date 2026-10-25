from research.semantic_invariants.semantic_runtime_release_gate import evaluate_runtime_release_gate


def test_release_gate_exercises_complete_authority_chain():
    report = evaluate_runtime_release_gate(model_depth=3)

    assert report["passed"]
    assert report["passed_checks"] == report["total_checks"]
    assert report["total_checks"] >= 7
    assert report["model_traces_checked"] > 0