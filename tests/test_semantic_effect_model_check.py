from research.semantic_invariants.semantic_effect_model_check import (
    bounded_model_check,
    run_trace,
)
from research.semantic_invariants.semantic_effect_commit import EffectClass


def test_crash_after_dispatch_is_safe_for_irreversible_effect():
    result = run_trace(
        EffectClass.IRREVERSIBLE,
        ("prepare", "dispatch", "restart", "prepare", "dispatch"),
    )

    assert result.safe
    assert result.dispatch_count == 1


def test_ambiguous_irreversible_effect_is_not_redispatched():
    result = run_trace(
        EffectClass.IRREVERSIBLE,
        ("prepare", "dispatch", "observe_stale", "restart", "dispatch"),
    )

    assert result.safe
    assert result.dispatch_count == 1


def test_bounded_state_space_has_no_unsafe_non_idempotent_redispatch():
    report = bounded_model_check(max_depth=4)

    assert report["safe"]
    assert report["violations"] == []
    assert report["traces_checked"] > 4000