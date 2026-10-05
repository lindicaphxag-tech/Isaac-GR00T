from research.semantic_invariants.embodied_freshness_closed_loop import (
    run_freshness_feedback,
)


def test_stale_sensor_semantics_change_closed_loop_trajectory():
    broken = run_freshness_feedback(guarded=False, horizon=6)
    guarded = run_freshness_feedback(guarded=True, horizon=6)

    assert broken.states != guarded.states
    assert broken.path_length > guarded.path_length
    assert broken.final_error > guarded.final_error


def test_freshness_guard_requires_owner_resampling_not_casting():
    guarded = run_freshness_feedback(guarded=True, horizon=6)

    assert guarded.final_error == 0.0
    assert guarded.path_length == 1.0
    assert guarded.resamples == 6
    assert guarded.states[1:] == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
