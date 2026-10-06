import itertools

import pytest

from research.semantic_invariants.semrepair_core_calculus import (
    BoundaryStatus,
    CoreStepBlocked,
    CoreType,
    CoreValue,
    EventRule,
    PureRule,
    decide_boundary,
    event_nonforgeability_holds,
    event_step,
    pure_preservation_holds,
    pure_step,
)


def test_pure_adapter_preserves_canonical_meaning_token():
    value = CoreValue(
        CoreType(representation="axis_angle", provenance="requested"),
        meaning_token="R_target",
    )
    rule = PureRule(
        "axis-to-euler",
        requires={"representation": "axis_angle"},
        produces={"representation": "euler_xyz"},
    )

    step = pure_step(value, rule)

    assert pure_preservation_holds(step)
    assert step.after.semantic_type.representation == "euler_xyz"
    assert step.after.meaning_token == "R_target"


def test_pure_rule_cannot_forge_executed_provenance():
    with pytest.raises(ValueError, match="cannot forge"):
        PureRule(
            "fake-execute",
            requires={"provenance": "requested"},
            produces={"provenance": "executed"},
        )


def test_event_refinement_requires_receipt_and_records_lineage():
    value = CoreValue(
        CoreType(provenance="requested"),
        meaning_token="action-request-7",
    )
    rule = EventRule(
        "controller-execute",
        requires={"provenance": "requested"},
        produces={"provenance": "executed"},
        receipt_keys=("action_id", "controller_id"),
    )

    with pytest.raises(CoreStepBlocked) as blocked:
        event_step(value, rule, receipt={"action_id": "7"})
    assert blocked.value.obligations == ("controller_id",)

    step = event_step(
        value,
        rule,
        receipt={"action_id": "7", "controller_id": "arm"},
    )

    assert event_nonforgeability_holds(step)
    assert step.after.semantic_type.provenance == "executed"
    assert step.after.event_lineage == ("controller-execute",)
    assert step.after.meaning_token != step.before.meaning_token


def test_context_pure_step_requires_explicit_evidence():
    value = CoreValue(
        CoreType(clock="dataset_global", scope="dataset"),
        meaning_token="frame-17",
    )
    rule = PureRule(
        "retime",
        requires={"clock": "dataset_global", "scope": "dataset"},
        produces={"clock": "episode_local", "scope": "episode"},
        evidence_keys=("episode_origin",),
    )

    with pytest.raises(CoreStepBlocked) as blocked:
        pure_step(value, rule)
    assert blocked.value.obligations == ("episode_origin",)

    step = pure_step(value, rule, evidence={"episode_origin": 10.0})
    assert pure_preservation_holds(step)


def test_progress_classifier_is_total_over_finite_abstract_cases():
    source = CoreType(representation="axis_angle")
    target = CoreType(representation="euler_xyz")

    observed = set()
    for repair_paths, obligations, survivors in itertools.product(
        (0, 1, 2),
        ((), ("frame_tf",)),
        (1, 2),
    ):
        decision = decide_boundary(
            source,
            target,
            repair_paths=repair_paths,
            missing_obligations=obligations,
            inference_survivors=survivors,
        )
        observed.add(decision.status)

    assert observed == {
        BoundaryStatus.REPAIR,
        BoundaryStatus.OBLIGATION,
        BoundaryStatus.AMBIGUOUS,
        BoundaryStatus.REJECT,
    }

    assert (
        decide_boundary(source, source, repair_paths=0).status
        is BoundaryStatus.ACCEPT
    )


def test_nonforgeable_transition_space_is_exhaustively_guarded():
    values = [
        CoreValue(
            CoreType(provenance=provenance, freshness=freshness),
            meaning_token=f"{provenance}:{freshness}",
        )
        for provenance in ("requested", "executed")
        for freshness in ("stale", "fresh")
    ]

    for value in values:
        for field, before, after in (
            ("provenance", "requested", "executed"),
            ("freshness", "stale", "fresh"),
        ):
            if getattr(value.semantic_type, field) != before:
                continue
            with pytest.raises(ValueError):
                PureRule(
                    f"forge-{field}",
                    requires={field: before},
                    produces={field: after},
                )
