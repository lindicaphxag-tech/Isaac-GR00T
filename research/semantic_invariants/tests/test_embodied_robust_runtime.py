"""End-to-end no-hardware test: robust semantic belief -> durable HMAC effect gate."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from research.semantic_invariants.embodied_diagnostic_episode_store import (
    DiagnosticEpisodeStore,
    diagnostic_observation_mac,
)
from research.semantic_invariants.embodied_diagnostic_execution import (
    DiagnosticExecutionRejected,
    DiagnosticObservedStep,
)
from research.semantic_invariants.embodied_robust_diagnosis import (
    synthesize_robust_experiment_plan,
)
from research.semantic_invariants.embodied_semantic_experiment_design import (
    SemanticExperiment,
)
from research.semantic_invariants.embodied_semantic_observability import (
    SemanticDiagnosisHypothesis,
)
from research.semantic_invariants.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
)


SENSOR_KEY = b"test-only-independent-sensor-secret-not-for-production!"
EPISODE = "robot/episode/robust-001"


def _robust_plan(*, conflicting=False, source_suffix="v1"):
    hypotheses = tuple(
        SemanticDiagnosisHypothesis(
            name,
            (SemanticTransportFactor(
                "delta-action-chart",
                MonomialSemanticTransport((0,), (scale,)),
                f"source:{source_suffix}/{name}",
            ),),
        )
        for name, scale in (
            ("alias-A", 1.0), ("alias-B", 1.0), ("other", 2.0)
        )
    )
    return synthesize_robust_experiment_plan(
        hypotheses,
        (SemanticExperiment("calibrated-probe", (1.0,), risk=0.3),),
        epsilon=0.1, risk_budget=0.3, max_probe_risk=0.3,
        authorities={
            "alias-A": "sha256:concrete-adapter-v1",
            "alias-B": ("sha256:incompatible-adapter"
                        if conflicting else "sha256:concrete-adapter-v1"),
            "other": "sha256:second-adapter-v1",
        },
    )


def _start(store, plan):
    store.create_episode(
        episode_id=EPISODE,
        plan=plan,
        trusted_problem_digest=plan.digest,
        trusted_tree_commitment=plan.digest,
    )


def _mac(token, observation):
    # Represents a trusted sensor-side signer; the source proposal
    # and repair generator must NOT hold SENSOR_KEY in actual deployment.
    return diagnostic_observation_mac(
        SENSOR_KEY,
        episode_id=EPISODE, step_index=0,
        reservation_token=token, observation=observation,
    )


def test_noisy_authority_decision_commits_only_authenticated_single_dispatch(tmp_path):
    plan = _robust_plan()
    store = DiagnosticEpisodeStore(
        tmp_path / "robust.sqlite3", trusted_evidence_key=SENSOR_KEY
    )
    _start(store, plan)
    grant = store.reserve_next(episode_id=EPISODE, plan=plan)
    assert grant.step_index == 0
    assert grant.risk_after_probe == pytest.approx(0.3)
    assert len(grant.reservation_token) == 64
    assert grant.experiment.name == "calibrated-probe"
    measured = DiagnosticObservedStep(
        "calibrated-probe", (1.05,), "sensor/external/001"
    )
    result = store.commit_observation(
        episode_id=EPISODE, plan=plan,
        observation=measured, reservation_token=grant.reservation_token,
        evidence_mac=_mac(grant.reservation_token, measured),
    )
    assert result.status == "DONE"
    assert result.risk_charged == pytest.approx(0.3)
    with pytest.raises(DiagnosticExecutionRejected, match="unconsumed"):
        store.reserve_next(episode_id=EPISODE, plan=plan)

    restart = DiagnosticEpisodeStore(
        tmp_path / "robust.sqlite3", trusted_evidence_key=SENSOR_KEY
    )
    assert restart.snapshot(episode_id=EPISODE).status == "DONE"
    with pytest.raises(DiagnosticExecutionRejected, match="duplicate episode"):
        _start(restart, plan)


def test_calibrated_sensor_bound_is_enforced_after_valid_hmac(tmp_path):
    plan = _robust_plan()
    store = DiagnosticEpisodeStore(
        tmp_path / "robust.sqlite3", trusted_evidence_key=SENSOR_KEY
    )
    _start(store, plan)
    grant = store.reserve_next(episode_id=EPISODE, plan=plan)
    # Even a correctly authenticated sensor reading does not automatically
    # count as valid *semantic* evidence if it violates the frozen noise model.
    impossible = DiagnosticObservedStep(
        "calibrated-probe", (1.45,), "sensor/external/002"
    )
    with pytest.raises(DiagnosticExecutionRejected, match="error bound"):
        store.commit_observation(
            episode_id=EPISODE, plan=plan,
            observation=impossible, reservation_token=grant.reservation_token,
            evidence_mac=_mac(grant.reservation_token, impossible),
        )
    assert store.snapshot(episode_id=EPISODE).status == "RESERVED"
    with pytest.raises(DiagnosticExecutionRejected, match="unconsumed"):
        store.reserve_next(episode_id=EPISODE, plan=plan)
    store.abort(episode_id=EPISODE)
    assert store.snapshot(episode_id=EPISODE).status == "ABORTED"


def test_upgraded_source_or_conflicting_repair_authority_revokes_frozen_plan(tmp_path):
    plan = _robust_plan()
    store = DiagnosticEpisodeStore(
        tmp_path / "robust.sqlite3", trusted_evidence_key=SENSOR_KEY
    )
    _start(store, plan)
    source_changed = _robust_plan(source_suffix="v2")
    assert source_changed.digest != plan.digest
    with pytest.raises(DiagnosticExecutionRejected, match="plan identity"):
        store.reserve_next(episode_id=EPISODE, plan=source_changed)
    conflicting = _robust_plan(conflicting=True)
    assert not conflicting.complete
    assert conflicting.digest != plan.digest
    with pytest.raises(DiagnosticExecutionRejected, match="plan identity"):
        store.reserve_next(episode_id=EPISODE, plan=conflicting)


def test_racing_workers_cannot_duplicate_noisy_physical_probe(tmp_path):
    plan = _robust_plan()
    path = tmp_path / "robust.sqlite3"
    _start(DiagnosticEpisodeStore(path, trusted_evidence_key=SENSOR_KEY), plan)

    def worker(_):
        try:
            return DiagnosticEpisodeStore(
                path, trusted_evidence_key=SENSOR_KEY
            ).reserve_next(episode_id=EPISODE, plan=plan)
        except DiagnosticExecutionRejected:
            return None

    with ThreadPoolExecutor(max_workers=8) as workers:
        accepted = list(workers.map(worker, range(20)))
    assert sum(item is not None for item in accepted) == 1
