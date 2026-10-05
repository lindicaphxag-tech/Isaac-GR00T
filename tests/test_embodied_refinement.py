import pytest

from research.semantic_invariants.embodied_refinement import (
    InvalidSemanticEvidence,
    SemanticEvidence,
    refine_episode_time,
    refine_executed_action,
    refine_sensor_freshness,
    refinement_satisfies,
)
from research.semantic_invariants.embodied_semantic_types import SemanticTensorType


def test_requested_action_requires_real_execution_receipt():
    requested = SemanticTensorType(
        role="action",
        entity="joint_command",
        representation="joint_position",
        unit="rad",
        provenance="requested",
    )
    receipt = SemanticEvidence(
        kind="execution_receipt",
        subject_id="action-7",
        claims={"executed": True, "controller_id": "arm-controller"},
        issuer="controller",
    )
    target = requested.updated(provenance="executed")

    result = refine_executed_action(requested, receipt)

    assert result.after.provenance == "executed"
    assert refinement_satisfies(result, target)
    assert len(result.evidence_digest) == 64


def test_requested_action_cannot_be_refined_by_arbitrary_metadata():
    requested = SemanticTensorType(
        role="action",
        entity="joint_command",
        provenance="requested",
    )
    fake = SemanticEvidence(
        kind="metadata",
        subject_id="action-7",
        claims={"executed": True, "controller_id": "arm-controller"},
        issuer="logger",
    )

    with pytest.raises(InvalidSemanticEvidence):
        refine_executed_action(requested, fake)


def test_sensor_freshness_requires_sample_after_due_boundary():
    observation = SemanticTensorType(
        role="observation",
        entity="camera_frame",
        clock="simulation",
        freshness="stale",
        provenance="observed",
    )
    stale = SemanticEvidence(
        kind="sensor_sample",
        subject_id="frame-1",
        claims={"sample_time": 9.9},
        issuer="camera",
    )
    fresh = SemanticEvidence(
        kind="sensor_sample",
        subject_id="frame-2",
        claims={"sample_time": 10.1},
        issuer="camera",
    )

    with pytest.raises(InvalidSemanticEvidence):
        refine_sensor_freshness(observation, stale, due_time=10.0)

    result = refine_sensor_freshness(observation, fresh, due_time=10.0)
    assert result.after.freshness == "fresh"


def test_episode_local_time_requires_origin_witness():
    value = SemanticTensorType(
        role="observation",
        entity="camera_frame",
        clock="dataset_global",
        scope="dataset",
    )
    evidence = SemanticEvidence(
        kind="episode_origin",
        subject_id="episode-3",
        claims={"episode_id": 3, "origin_time": 120.0},
        issuer="dataset-index",
    )

    result = refine_episode_time(value, evidence)

    assert result.after.clock == "episode_local"
    assert result.after.scope == "episode"
    assert result.effects == ("retime",)
