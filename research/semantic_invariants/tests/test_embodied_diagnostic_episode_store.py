from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from research.semantic_invariants.embodied_diagnostic_episode_store import (
    DiagnosticEpisodeStore,
    diagnostic_observation_mac,
)
from research.semantic_invariants.embodied_diagnostic_execution import (
    DiagnosticExecutionRejected,
    DiagnosticObservedStep,
    diagnostic_tree_commitment,
)
from research.semantic_invariants.embodied_semantic_experiment_design import (
    SemanticExperiment,
    synthesize_optimal_experiment_plan,
)
from research.semantic_invariants.embodied_semantic_observability import (
    SemanticDiagnosisHypothesis,
)
from research.semantic_invariants.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
)


def _plan():
    identity = MonomialSemanticTransport.identity(2)
    x_scaled = MonomialSemanticTransport((0, 1), (2.0, 1.0))
    y_scaled = MonomialSemanticTransport((0, 1), (1.0, 2.0))
    hs = tuple(
        SemanticDiagnosisHypothesis(name, (SemanticTransportFactor("chart", t, name),))
        for name, t in (
            ("identity", identity), ("x-scaled", x_scaled), ("y-scaled", y_scaled)
        )
    )
    exps = (
        SemanticExperiment("probe-x", (1., 0.), risk=0.6),
        SemanticExperiment("probe-y", (0., 1.), risk=0.6),
    )
    return synthesize_optimal_experiment_plan(
        hs, exps, max_risk=0.6, total_risk_budget=1.2
    )


def _start(store, plan, episode="robotA/trial42"):
    store.create_episode(
        episode_id=episode, plan=plan, trusted_problem_digest=plan.digest,
        trusted_tree_commitment=diagnostic_tree_commitment(plan),
    )


def _identity_obs(node, evidence_id):
    cl = next(c for c in node.observation_classes if "identity" in c.hypotheses)
    return DiagnosticObservedStep(node.experiment.name, tuple(cl.signature), evidence_id)


TRUSTED_SENSOR_KEY = b"static-fixture-only-not-a-deployment-secret!"


def _store(path, *, key=TRUSTED_SENSOR_KEY):
    return DiagnosticEpisodeStore(path, trusted_evidence_key=key)


def _sensor_mac(step_index, token, observation):
    """Fixture only: production MACs originate outside the repair generator."""
    return diagnostic_observation_mac(
        TRUSTED_SENSOR_KEY,
        episode_id="robotA/trial42",
        step_index=step_index,
        reservation_token=token,
        observation=observation,
    )


def test_one_physical_probe_reservation_survives_restart_and_replay(tmp_path):
    plan = _plan()
    db = tmp_path / "episodes.sqlite3"
    first_store = _store(db)
    _start(first_store, plan)
    grant = first_store.reserve_next(episode_id="robotA/trial42", plan=plan)
    assert grant.step_index == 0
    assert grant.risk_after_probe == pytest.approx(0.6)
    assert isinstance(grant.reservation_token, str) and len(grant.reservation_token) == 64

    # A simulator crash cannot grant a second physical effect for this step.
    restarted = _store(db)
    snap = restarted.snapshot(episode_id="robotA/trial42")
    assert snap.status == "RESERVED"
    assert snap.risk_charged == pytest.approx(0.6)
    with pytest.raises(DiagnosticExecutionRejected, match="unconsumed"):
        restarted.reserve_next(episode_id="robotA/trial42", plan=plan)
    with pytest.raises(DiagnosticExecutionRejected, match="duplicate episode"):
        _start(restarted, plan)

    first = _identity_obs(plan.root, "evidence/live/0001")
    s1 = restarted.commit_observation(
        episode_id="robotA/trial42", plan=plan, observation=first,
        reservation_token=grant.reservation_token, evidence_mac=_sensor_mac(0, grant.reservation_token, first),
    )
    assert s1.status == "READY"
    assert s1.risk_charged == pytest.approx(0.6)
    with pytest.raises(DiagnosticExecutionRejected, match="no outstanding"):
        restarted.commit_observation(
            episode_id="robotA/trial42", plan=plan, observation=first,
            reservation_token=grant.reservation_token,
            evidence_mac=_sensor_mac(0, grant.reservation_token, first),
        )

    grant2 = restarted.reserve_next(episode_id="robotA/trial42", plan=plan)
    assert grant2.step_index == 1
    assert grant2.risk_after_probe == pytest.approx(1.2)
    assert grant2.reservation_token != grant.reservation_token
    cl = next(c for c in plan.root.observation_classes if "identity" in c.hypotheses)
    child = plan.root.children[plan.root.observation_classes.index(cl)]
    second = _identity_obs(child, "evidence/live/0002")
    s2 = restarted.commit_observation(
        episode_id="robotA/trial42", plan=plan, observation=second,
        reservation_token=grant2.reservation_token,
        evidence_mac=_sensor_mac(1, grant2.reservation_token, second),
    )
    assert s2.status == "DONE"
    assert s2.risk_charged == pytest.approx(1.2)
    with pytest.raises(DiagnosticExecutionRejected, match="unconsumed"):
        restarted.reserve_next(episode_id="robotA/trial42", plan=plan)


def test_competing_workers_receive_at_most_one_reservation(tmp_path):
    plan = _plan()
    path = tmp_path / "racing.sqlite3"
    trusted = _store(path)
    _start(trusted, plan)

    def attempt(_):
        try:
            return _store(path).reserve_next(
                episode_id="robotA/trial42", plan=plan
            )
        except DiagnosticExecutionRejected:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(24)))
    assert sum(item is not None for item in results) == 1
    assert trusted.snapshot(episode_id="robotA/trial42").risk_charged == pytest.approx(0.6)


def test_forged_observation_does_not_release_pending_reservation(tmp_path):
    plan = _plan()
    store = _store(tmp_path / "cases.sqlite3")
    _start(store, plan)
    reservation = store.reserve_next(episode_id="robotA/trial42", plan=plan)
    bad = (
        DiagnosticObservedStep("wrong-probe", (1., 0.), "sensor/A"),
        DiagnosticObservedStep(plan.root.experiment.name, (999.,), "sensor/B"),
        DiagnosticObservedStep(plan.root.experiment.name, (float("nan"),), "sensor/C"),
    )
    for observation in bad:
        with pytest.raises(DiagnosticExecutionRejected):
            store.commit_observation(
                episode_id="robotA/trial42", plan=plan, observation=observation,
                reservation_token=reservation.reservation_token,
                evidence_mac="bad-sensor-mac",
            )
        assert store.snapshot(episode_id="robotA/trial42").status == "RESERVED"
    store.abort(episode_id="robotA/trial42")
    assert store.snapshot(episode_id="robotA/trial42").status == "ABORTED"
    with pytest.raises(DiagnosticExecutionRejected):
        store.reserve_next(episode_id="robotA/trial42", plan=plan)


def test_changed_plan_is_not_allowed_to_splice_into_active_episode(tmp_path):
    plan = _plan()
    store = _store(tmp_path / "cases.sqlite3")
    _start(store, plan)
    mutated = replace(plan, root=replace(plan.root, experiment=replace(
        plan.root.experiment, risk=0.0,
    )))
    with pytest.raises(DiagnosticExecutionRejected, match="plan identity"):
        store.reserve_next(episode_id="robotA/trial42", plan=mutated)
    assert store.snapshot(episode_id="robotA/trial42").risk_charged == 0.0


def test_store_refuses_nonpersistent_db_and_untrusted_plan(tmp_path):
    with pytest.raises(ValueError, match="persistent"):
        _store(":memory:")
    plan = _plan()
    store = _store(tmp_path / "episodes.sqlite3")
    with pytest.raises(DiagnosticExecutionRejected, match="untrusted"):
        store.create_episode(
            episode_id="untrusted", plan=plan,
            trusted_problem_digest="0"*64,
            trusted_tree_commitment=diagnostic_tree_commitment(plan),
        )


def test_wrong_token_or_wrong_key_cannot_commit_physical_effect(tmp_path):
    plan = _plan()
    store = _store(tmp_path / "nonce.sqlite3")
    _start(store, plan)
    grant = store.reserve_next(episode_id="robotA/trial42", plan=plan)
    observation = _identity_obs(plan.root, "evidence/live/0001")
    valid_mac = _sensor_mac(0, grant.reservation_token, observation)

    with pytest.raises(DiagnosticExecutionRejected, match="token"):
        store.commit_observation(
            episode_id="robotA/trial42", plan=plan, observation=observation,
            reservation_token="previous-step-token", evidence_mac=valid_mac,
        )
    with pytest.raises(DiagnosticExecutionRejected, match="HMAC"):
        store.commit_observation(
            episode_id="robotA/trial42", plan=plan, observation=observation,
            reservation_token=grant.reservation_token,
            evidence_mac="forged-no-trusted-sensor-key",
        )
    with pytest.raises(DiagnosticExecutionRejected, match="HMAC"):
        store.commit_observation(
            episode_id="robotA/trial42", plan=plan,
            observation=replace(observation, evidence_id="spoofed"),
            reservation_token=grant.reservation_token,
            evidence_mac=valid_mac,
        )
    with pytest.raises(DiagnosticExecutionRejected, match="key changed"):
        _store(
            tmp_path / "nonce.sqlite3", key=b"second-fixture-sensor-key-32-bytes!!"
        ).commit_observation(
            episode_id="robotA/trial42", plan=plan, observation=observation,
            reservation_token=grant.reservation_token,
            evidence_mac=valid_mac,
        )

    with pytest.raises(DiagnosticExecutionRejected, match="key changed"):
        _store(
            tmp_path / "nonce.sqlite3", key=b"second-fixture-sensor-key-32-bytes!!"
        ).abort(episode_id="robotA/trial42")
    assert store.snapshot(episode_id="robotA/trial42").status == "RESERVED"
    result = store.commit_observation(
        episode_id="robotA/trial42", plan=plan, observation=observation,
        reservation_token=grant.reservation_token, evidence_mac=valid_mac,
    )
    assert result.status == "READY"


def test_sensor_key_must_be_provisioned_by_trusted_bootstrap(tmp_path):
    with pytest.raises(ValueError, match="at least 32"):
        _store(tmp_path / "short-key.sqlite3", key=b"weak")
    with pytest.raises(TypeError):
        DiagnosticEpisodeStore(tmp_path / "missing-key.sqlite3")
