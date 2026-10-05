from research.semantic_invariants.semantic_effect_commit import (
    EffectClass,
    EffectStatus,
    RuntimeDecision,
    SemanticEffectIntent,
    SemanticEffectRuntime,
)
from research.semantic_invariants.semantic_effect_evidence import (
    EffectObservation,
    resolve_with_triangulated_evidence,
    summarize_effect_evidence,
)


def _intent():
    return SemanticEffectIntent(
        effect_id="handoff:task-2",
        action_name="handoff_token",
        effect_class=EffectClass.IRREVERSIBLE,
        dependency_version="plan-v1",
        precondition=lambda state: state["holder"] == "robot",
        postcondition=lambda state: state["holder"] == "human",
    )


def _dispatched():
    runtime = SemanticEffectRuntime()
    intent = _intent()
    runtime.prepare(intent, current_state={"holder": "robot"}, current_dependency_version="plan-v1")
    runtime.mark_dispatched(intent.effect_id)
    return runtime, intent


def test_two_independent_postcondition_planes_commit_irreversible_effect():
    runtime, intent = _dispatched()
    outcome, summary = resolve_with_triangulated_evidence(
        runtime,
        intent,
        (
            EffectObservation("vision", "camera", {"holder": "human"}),
            EffectObservation("gripper force", "proprioception", {"holder": "human"}),
        ),
        min_commit_planes=2,
    )
    assert outcome.status == EffectStatus.COMMITTED
    assert outcome.decision == RuntimeDecision.ADVANCE
    assert summary.postcondition_planes == ("camera", "proprioception")


def test_two_correlated_caches_from_same_plane_do_not_fake_triangulation():
    runtime, intent = _dispatched()
    outcome, summary = resolve_with_triangulated_evidence(
        runtime,
        intent,
        (
            EffectObservation("public cache A", "public_api", {"holder": "human"}),
            EffectObservation("public cache B", "public_api", {"holder": "human"}),
        ),
        min_commit_planes=2,
    )
    assert summary.postcondition_planes == ("public_api",)
    assert outcome.status == EffectStatus.AMBIGUOUS
    assert outcome.decision == RuntimeDecision.BLOCK


def test_success_ack_without_independent_physical_evidence_does_not_commit():
    runtime, intent = _dispatched()
    outcome, _ = resolve_with_triangulated_evidence(
        runtime,
        intent,
        (EffectObservation("executor cache", "executor", {"holder": "human"}),),
        min_commit_planes=2,
        executor_acknowledged_success=True,
    )
    assert outcome.status == EffectStatus.AMBIGUOUS
    assert outcome.decision == RuntimeDecision.BLOCK


def test_stale_second_plane_does_not_count_toward_commit_quorum():
    runtime, intent = _dispatched()
    outcome, summary = resolve_with_triangulated_evidence(
        runtime,
        intent,
        (
            EffectObservation("camera", "camera", {"holder": "human"}),
            EffectObservation("force", "proprioception", {"holder": "human"}, fresh=False),
        ),
        min_commit_planes=2,
    )
    assert summary.postcondition_planes == ("camera",)
    assert outcome.status == EffectStatus.AMBIGUOUS


def test_same_plane_disagreement_is_quarantined():
    intent = _intent()
    summary = summarize_effect_evidence(
        intent,
        (
            EffectObservation("cache A", "public_api", {"holder": "human"}),
            EffectObservation("cache B", "public_api", {"holder": "robot"}),
        ),
    )
    assert summary.contradictory_planes == ("public_api",)
    assert summary.postcondition_planes == ()
    assert summary.precondition_planes == ()


def test_confirmed_abort_plus_independent_precondition_allows_retry():
    runtime, intent = _dispatched()
    outcome, summary = resolve_with_triangulated_evidence(
        runtime,
        intent,
        (EffectObservation("camera", "camera", {"holder": "robot"}),),
        min_commit_planes=2,
        min_abort_planes=1,
        executor_acknowledged_abort=True,
    )
    assert summary.precondition_planes == ("camera",)
    assert outcome.status == EffectStatus.ABORTED
    assert outcome.decision == RuntimeDecision.RETRY