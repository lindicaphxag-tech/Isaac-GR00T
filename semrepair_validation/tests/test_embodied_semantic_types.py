import pytest

from research.semantic_invariants.embodied_semantic_types import (
    ComponentSignature,
    AmbiguousSemanticRepair,
    MissingSemanticEvidence,
    NoSemanticRepair,
    SemanticAdapter,
    SemanticEventTransition,
    SemanticTensorType,
    apply_event_transition,
    is_assignable,
    semantic_mismatches,
    synthesize_adapter_plan,
    synthesize_unique_adapter_plan,
    type_and_repair_pipeline,
)


def _rotation(rep: str) -> SemanticTensorType:
    return SemanticTensorType(
        role="action",
        entity="ee_rotation_delta",
        frame="root_aligned_body",
        representation=rep,
        unit="rad",
        scope="control_step",
        provenance="requested",
        embodiment="panda",
    )


def test_axis_angle_to_euler_requires_explicit_adapter_chain():
    source = _rotation("axis_angle")
    target = _rotation("euler_xyz")
    adapters = [
        SemanticAdapter(
            "axis-angle-to-quaternion",
            requires={"representation": "axis_angle"},
            produces={"representation": "quaternion"},
            effects=("representation-conversion",),
        ),
        SemanticAdapter(
            "quaternion-to-euler-xyz",
            requires={"representation": "quaternion"},
            produces={"representation": "euler_xyz"},
            effects=("representation-conversion",),
        ),
    ]

    plan = synthesize_adapter_plan(source, target, adapters)

    assert plan.passed
    assert [adapter.name for adapter in plan.adapters] == [
        "axis-angle-to-quaternion",
        "quaternion-to-euler-xyz",
    ]
    assert plan.evidence_used == ()


def test_requested_action_cannot_be_cast_into_executed_action():
    requested = SemanticTensorType(
        role="action",
        entity="joint_command",
        representation="joint_position",
        unit="rad",
        provenance="requested",
    )
    executed = requested.updated(provenance="executed")

    assert not is_assignable(requested, executed)
    assert semantic_mismatches(requested, executed)[0].field == "provenance"

    with pytest.raises(ValueError, match="cannot forge non-forgeable field"):
        SemanticAdapter(
            "fake-execution-cast",
            requires={"provenance": "requested"},
            produces={"provenance": "executed"},
        )

    with pytest.raises(NoSemanticRepair):
        synthesize_adapter_plan(requested, executed, [])


def test_stale_observation_cannot_be_cast_fresh():
    stale = SemanticTensorType(
        role="observation",
        entity="camera_frame",
        freshness="stale",
        provenance="observed",
    )

    with pytest.raises(ValueError, match="cannot forge non-forgeable field"):
        SemanticAdapter(
            "fake-refresh",
            requires={"freshness": "stale"},
            produces={"freshness": "fresh"},
        )


def test_context_sensitive_retiming_requires_concrete_episode_origin():
    global_time = SemanticTensorType(
        role="observation",
        entity="camera_frame",
        clock="dataset_global",
        scope="dataset",
        provenance="observed",
    )
    episode_time = global_time.updated(clock="episode_local", scope="episode")
    retime = SemanticAdapter(
        "retime-with-episode-origin",
        requires={"clock": "dataset_global", "scope": "dataset"},
        produces={"clock": "episode_local", "scope": "episode"},
        witness_keys=("episode_origin",),
        effects=("retimed",),
    )

    with pytest.raises(NoSemanticRepair) as blocked:
        synthesize_adapter_plan(global_time, episode_time, [retime])
    assert blocked.value.proof_obligations == ("episode_origin",)

    # Legacy boolean permission alone must not bypass a missing witness.
    with pytest.raises(NoSemanticRepair):
        synthesize_adapter_plan(
            global_time,
            episode_time,
            [retime],
            allow_proof_required=True,
        )

    plan = synthesize_adapter_plan(
        global_time,
        episode_time,
        [retime],
        evidence={"episode_origin": 12.5},
    )
    assert plan.passed
    assert plan.effects == ("retimed",)
    assert plan.evidence_used == ("episode_origin",)


def test_legacy_proof_required_becomes_explicit_obligation():
    source = SemanticTensorType(
        role="state",
        entity="joint_position",
        ordering="urdf",
    )
    target = source.updated(ordering="policy")
    adapter = SemanticAdapter(
        "legacy-reorder",
        requires={"ordering": "urdf"},
        produces={"ordering": "policy"},
        proof_required=True,
    )

    with pytest.raises(NoSemanticRepair) as blocked:
        synthesize_adapter_plan(source, target, [adapter], allow_proof_required=True)

    obligation = "proof:legacy-reorder"
    assert blocked.value.proof_obligations == (obligation,)

    plan = synthesize_adapter_plan(
        source,
        target,
        [adapter],
        evidence={obligation: {"mapping": [2, 0, 1]}},
    )
    assert plan.passed
    assert plan.evidence_used == (obligation,)


def test_executed_action_requires_owner_boundary_receipt():
    requested = SemanticTensorType(
        role="action",
        entity="joint_command",
        representation="joint_position",
        unit="rad",
        provenance="requested",
    )
    execution = SemanticEventTransition(
        "controller-execution",
        requires={"provenance": "requested"},
        produces={"provenance": "executed"},
        receipt_keys=("action_id", "controller_id", "executed_at"),
        effects=("execute",),
    )

    with pytest.raises(MissingSemanticEvidence) as blocked:
        apply_event_transition(
            requested,
            execution,
            receipt={"action_id": "a-17"},
        )
    assert blocked.value.obligations == ("controller_id", "executed_at")

    result = apply_event_transition(
        requested,
        execution,
        receipt={
            "action_id": "a-17",
            "controller_id": "arm-controller",
            "executed_at": 42.0,
        },
    )

    assert result.result.provenance == "executed"
    assert result.evidence_used == ("action_id", "controller_id", "executed_at")
    assert result.effects == ("execute",)


def test_fresh_observation_requires_sensor_sample_receipt():
    stale = SemanticTensorType(
        role="observation",
        entity="camera_frame",
        freshness="stale",
        provenance="observed",
    )
    refresh = SemanticEventTransition(
        "camera-sample",
        requires={"freshness": "stale", "provenance": "observed"},
        produces={"freshness": "fresh"},
        receipt_keys=("sensor_id", "sample_time"),
        effects=("observe",),
    )

    with pytest.raises(MissingSemanticEvidence):
        apply_event_transition(stale, refresh, receipt={"sensor_id": "cam0"})

    result = apply_event_transition(
        stale,
        refresh,
        receipt={"sensor_id": "cam0", "sample_time": 1.25},
    )
    assert result.result.freshness == "fresh"
    assert result.effects == ("observe",)


def test_joint_order_requires_named_reorder():
    source = SemanticTensorType(
        role="state",
        entity="joint_position",
        representation="vector",
        unit="rad",
        ordering="urdf-order",
        embodiment="robot-a",
    )
    target = source.updated(ordering="policy-order")
    reorder = SemanticAdapter(
        "urdf-to-policy-joint-order",
        requires={"ordering": "urdf-order", "embodiment": "robot-a"},
        produces={"ordering": "policy-order"},
        effects=("reordered",),
    )

    plan = synthesize_adapter_plan(source, target, [reorder])
    assert plan.passed
    assert plan.adapters == (reorder,)


def test_pipeline_repairs_representation_without_hiding_provenance():
    source = _rotation("axis_angle")
    controller_input = _rotation("euler_xyz")
    controller_output = SemanticTensorType(
        role="action",
        entity="joint_target",
        representation="joint_position",
        unit="rad",
        scope="control_step",
        provenance="requested",
        ordering="controller-order",
        embodiment="panda",
    )
    components = [
        ComponentSignature(
            name="pd-ee-pose-controller",
            accepts=controller_input,
            produces=controller_output,
        )
    ]
    adapters = [
        SemanticAdapter(
            "axis-angle-to-quaternion",
            requires={"representation": "axis_angle"},
            produces={"representation": "quaternion"},
        ),
        SemanticAdapter(
            "quaternion-to-euler-xyz",
            requires={"representation": "quaternion"},
            produces={"representation": "euler_xyz"},
        ),
    ]

    result = type_and_repair_pipeline(source, components, adapters)

    assert result.steps[0].repair.passed
    assert result.final == controller_output
    assert result.final.provenance == "requested"
    assert result.evidence_used == ()

def test_unique_adapter_synthesis_rejects_equal_cost_semantic_ambiguity():
    source = _rotation("axis_angle")
    target = _rotation("euler_xyz")
    adapters = [
        SemanticAdapter(
            "direct-a",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
        SemanticAdapter(
            "direct-b",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
    ]

    with pytest.raises(AmbiguousSemanticRepair) as blocked:
        synthesize_unique_adapter_plan(source, target, adapters)

    assert blocked.value.alternatives == (("direct-a",), ("direct-b",))


def test_unique_adapter_synthesis_prefers_strictly_cheaper_semantic_path():
    source = _rotation("axis_angle")
    target = _rotation("euler_xyz")
    adapters = [
        SemanticAdapter(
            "direct",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
        SemanticAdapter(
            "axis-to-quat",
            requires={"representation": "axis_angle"},
            produces={"representation": "quaternion"},
            cost=1.0,
        ),
        SemanticAdapter(
            "quat-to-euler",
            requires={"representation": "quaternion"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
    ]

    plan = synthesize_unique_adapter_plan(source, target, adapters)

    assert [adapter.name for adapter in plan.adapters] == ["direct"]
    assert plan.total_cost == 1.0

def test_unique_adapter_synthesis_ignores_zero_cost_semantic_cycles():
    source = _rotation("axis_angle")
    target = _rotation("euler_xyz")
    adapters = [
        SemanticAdapter(
            "axis-to-quat-free",
            requires={"representation": "axis_angle"},
            produces={"representation": "quaternion"},
            cost=0.0,
        ),
        SemanticAdapter(
            "quat-to-axis-free",
            requires={"representation": "quaternion"},
            produces={"representation": "axis_angle"},
            cost=0.0,
        ),
        SemanticAdapter(
            "direct",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
    ]

    plan = synthesize_unique_adapter_plan(source, target, adapters)

    assert [adapter.name for adapter in plan.adapters] == ["direct"]
    assert plan.total_cost == 1.0


def test_unique_adapter_synthesis_treats_decimal_equal_paths_as_ambiguous():
    source = _rotation("axis_angle")
    target = _rotation("euler_xyz")
    adapters = [
        SemanticAdapter(
            "direct",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=0.3,
        ),
        SemanticAdapter(
            "axis-to-quat",
            requires={"representation": "axis_angle"},
            produces={"representation": "quaternion"},
            cost=0.1,
        ),
        SemanticAdapter(
            "quat-to-euler",
            requires={"representation": "quaternion"},
            produces={"representation": "euler_xyz"},
            cost=0.2,
        ),
    ]

    with pytest.raises(AmbiguousSemanticRepair) as blocked:
        synthesize_unique_adapter_plan(source, target, adapters)

    assert blocked.value.alternatives == (
        ("axis-to-quat", "quat-to-euler"),
        ("direct",),
    )


@pytest.mark.parametrize("bad_cost", [float("inf"), float("-inf"), float("nan")])
def test_adapter_cost_must_be_finite(bad_cost):
    with pytest.raises(ValueError, match="finite and non-negative"):
        SemanticAdapter(
            "bad-cost",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=bad_cost,
        )
