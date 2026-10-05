from research.semantic_invariants.semrepair_compiler_conformance import (
    verify_compiler_conformance,
)


def test_production_compiler_matches_independent_exhaustive_oracle():
    report = verify_compiler_conformance()

    assert report.valid
    assert report.violations == ()
    assert report.cases_checked > 1000
    assert report.accept_cases > 0
    assert report.unique_repair_cases > 0
    assert report.ambiguous_cases > 0
    assert report.obligation_cases > 0
    assert report.reject_cases > 0