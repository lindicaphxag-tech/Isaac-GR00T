from research.semantic_invariants.semrepair_core_theorem_check import verify_core_theorems


def test_bounded_core_theorem_checker_has_no_counterexample():
    report = verify_core_theorems()

    assert report.valid
    assert report.violations == ()
    assert report.pure_preservation_cases > 0
    assert report.nonforgeability_cases > 0
    assert report.progress_cases > 20
    assert report.shortest_path_graphs >= 200