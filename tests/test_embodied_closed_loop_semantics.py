from research.semantic_invariants.embodied_closed_loop_semantics import (
    run_composed_rotation_assay,
    synthesize_verified_composed_repair,
)


def test_full_composed_repair_recovers_canonical_closed_loop_trajectory():
    results = run_composed_rotation_assay()
    full = results["full_composed"]

    assert full.steps_to_tolerance == 8
    assert full.final_error_degrees < 1.0e-5
    assert full.max_deviation_from_reference_degrees == 0.0


def test_unrepaired_and_representation_only_pipelines_diverge():
    results = run_composed_rotation_assay()

    assert results["none"].final_error_degrees > 70.0
    assert results["representation_only"].final_error_degrees > 70.0
    assert results["none"].steps_to_tolerance is None
    assert results["representation_only"].steps_to_tolerance is None


def test_sign_only_can_hide_representation_bug_behind_task_success():
    results = run_composed_rotation_assay()
    sign_only = results["sign_only"]
    full = results["full_composed"]

    assert sign_only.steps_to_tolerance == 9
    assert sign_only.final_error_degrees < 0.01
    assert sign_only.max_deviation_from_reference_degrees > 0.4
    assert sign_only.path_length_degrees > full.path_length_degrees


def test_closed_loop_assay_distinguishes_task_success_from_semantic_equivalence():
    results = run_composed_rotation_assay()

    assert results["sign_only"].final_error_degrees < 0.01
    assert results["sign_only"].max_deviation_from_reference_degrees > 0.4
    assert results["full_composed"].max_deviation_from_reference_degrees == 0.0

def test_compiler_synthesizes_the_unique_verified_composed_runtime_program():
    program = synthesize_verified_composed_repair()

    assert program.name == "sign(-1, -1, -1) -> axis-angle->euler-xyz"


def test_compiler_mediated_closed_loop_matches_canonical_full_repair():
    results = run_composed_rotation_assay()
    generated = results["compiler_mediated"]
    reference = results["full_composed"]

    assert generated.steps_to_tolerance == reference.steps_to_tolerance
    assert generated.final_error_degrees < 1.0e-5
    assert generated.max_deviation_from_reference_degrees < 1.0e-5
    assert abs(generated.path_length_degrees - reference.path_length_degrees) < 1.0e-8