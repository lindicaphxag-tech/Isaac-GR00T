from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from research.semantic_invariants.embodied_diagnostic_episode_store import (
    DiagnosticEpisodeStore,
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


def test_one_physical_probe_reservation_survives_restart_and_replay(tmp_path):
    plan = _plan()
    db = tmp_path / "episodes.sqlite3"
    first_store = DiagnosticEpisodeStore(db)
    _start(first_store, plan)
    grant = first_store.reserve_next(episode_id="robotA/trial42", plan=plan)
    assert grant.step_index == 0
    assert grant.risk_after_probe == pytest.approx(0.6)

    # A simulator crash cannot grant a second physical effect for this step.
    restarted = DiagnosticEpisodeStore(db)
    snap = restarted.snapshot(episode_id="robotA/trial42")
    assert snap.status == "RESERVED"
    assert snap.risk_charged == pytest.approx(0.6)
    with pytest.raises(DiagnosticExecutionRejected, match="unconsumed"):
        restarted.reserve_next(episode_id="robotA/trial42", plan=plan)
    with pytest.raises(DiagnosticExecutionRejected, match="duplicate episode"):
        _start(restarted, plan)

    first = _identity_obs(plan.root, "evidence/live/0001")
    s1 = restarted.commit_observation(
        episode_id="robotA/trial42", plan=plan, observation=first
    )
    assert s1.status == "READY"
    assert s1.risk_charged == pytest.approx(0.6)
    with pytest.raises(DiagnosticExecutionRejected, match="no outstanding"):
        restarted.commit_observation(
            episode_id="robotA/trial42", plan=plan, observation=first
        )

    grant2 = restarted.reserve_next(episode_id="robotA/trial42", plan=plan)
    assert grant2.step_index == 1
    assert grant2.risk_after_probe == pytest.approx(1.2)
    cl = next(c for c in plan.root.observation_classes if "identity" in c.hypotheses)
    child = plan.root.children[plan.root.observation_classes.index(cl)]
    second = _identity_obs(child, "evidence/live/0002")
    s2 = restarted.commit_observation(
        episode_id="robotA/trial42", plan=plan, observation=second
    )
    assert s2.status == "DONE"
    assert s2.risk_charged == pytest.approx(1.2)
    with pytest.raises(DiagnosticExecutionRejected, match="unconsumed"):
        restarted.reserve_next(episode_id="robotA/trial42", plan=plan)


def test_competing_workers_receive_at_most_one_reservation(tmp_path):
    plan = _plan()
    path = tmp_path / "racing.sqlite3"
    trusted = DiagnosticEpisodeStore(path)
    _start(trusted, plan)

    def attempt(_):
        try:
            return DiagnosticEpisodeStore(path).reserve_next(
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
    store = DiagnosticEpisodeStore(tmp_path / "cases.sqlite3")
    _start(store, plan)
    store.reserve_next(episode_id="robotA/trial42", plan=plan)
    bad = (
        DiagnosticObservedStep("wrong-probe", (1., 0.), "sensor/A"),
        DiagnosticObservedStep(plan.root.experiment.name, (999.,), "sensor/B"),
        DiagnosticObservedStep(plan.root.experiment.name, (float("nan"),), "sensor/C"),
    )
    for observation in bad:
        with pytest.raises(DiagnosticExecutionRejected):
            store.commit_observation(
                episode_id="robotA/trial42", plan=plan, observation=observation
            )
        assert store.snapshot(episode_id="robotA/trial42").status == "RESERVED"
    store.abort(episode_id="robotA/trial42")
    assert store.snapshot(episode_id="robotA/trial42").status == "ABORTED"
    with pytest.raises(DiagnosticExecutionRejected):
        store.reserve_next(episode_id="robotA/trial42", plan=plan)


def test_changed_plan_is_not_allowed_to_splice_into_active_episode(tmp_path):
    plan = _plan()
    store = DiagnosticEpisodeStore(tmp_path / "cases.sqlite3")
    _start(store, plan)
    mutated = replace(plan, root=replace(plan.root, experiment=replace(
        plan.root.experiment, risk=0.0,
    )))
    with pytest.raises(DiagnosticExecutionRejected, match="plan identity"):
        store.reserve_next(episode_id="robotA/trial42", plan=mutated)
    assert store.snapshot(episode_id="robotA/trial42").risk_charged == 0.0


def test_store_refuses_nonpersistent_db_and_untrusted_plan(tmp_path):
    with pytest.raises(ValueError, match="persistent"):
        DiagnosticEpisodeStore(":memory:")
    plan = _plan()
    store = DiagnosticEpisodeStore(tmp_path / "episodes.sqlite3")
    with pytest.raises(DiagnosticExecutionRejected, match="untrusted"):
        store.create_episode(
            episode_id="untrusted", plan=plan,
            trusted_problem_digest="0"*64,
            trusted_tree_commitment=diagnostic_tree_commitment(plan),
        )
