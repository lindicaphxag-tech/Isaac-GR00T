"""Adversarial software handoff tests: NO real robot actuation occurs."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import json
import sqlite3

import pytest

from research.semantic_invariants.embodied_repair_handoff import (
    RepairDispatchStore,
)
from research.semantic_invariants.embodied_diagnostic_episode_store import (
    diagnostic_observation_mac,
)
from research.semantic_invariants.embodied_diagnostic_execution import (
    DiagnosticExecutionRejected, DiagnosticObservedStep,
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
    MonomialSemanticTransport, SemanticTransportFactor,
)


KEY = b"fixture-test-sensor-HMAC-secret-not-deployment-credential"
EP = "robot/fault/episode-42"
BUNDLE_A = b"actual-byte-content-of-adapter-a-v1"
BUNDLE_B = b"actual-byte-content-of-adapter-b-v1"


def authority(payload: bytes) -> str:
    return "sha256:" + sha256(payload).hexdigest()


def plan(*, source="frozen-v1", pseudo=False):
    worlds = tuple(
        SemanticDiagnosisHypothesis(
            name, (
                SemanticTransportFactor(
                    "action-chart",
                    MonomialSemanticTransport((0,), (scale,)),
                    f"{source}/{name}",
                ),
            ),
        )
        for name, scale in (("alias1", 1.0), ("alias2", 1.0), ("distinct", 2.0))
    )
    return synthesize_robust_experiment_plan(
        worlds,
        (SemanticExperiment("probe", (1.0,), risk=0.25),),
        epsilon=0.1, risk_budget=0.25, max_probe_risk=0.25,
        authorities={
            "alias1": "sha256:not-real-bundle" if pseudo else authority(BUNDLE_A),
            "alias2": "sha256:not-real-bundle" if pseudo else authority(BUNDLE_A),
            "distinct": authority(BUNDLE_B),
        },
    )


def finished_store(tmp_path, *, episode_id=EP, test_plan=None, observed=1.0):
    p = test_plan or plan()
    db = tmp_path / "handoff.sqlite3"
    store = RepairDispatchStore(db, trusted_evidence_key=KEY)
    store.create_episode(
        episode_id=episode_id, plan=p,
        trusted_problem_digest=p.digest,
        trusted_tree_commitment=p.digest,
    )
    probe = store.reserve_next(episode_id=episode_id, plan=p)
    observation = DiagnosticObservedStep("probe", (observed,), "trusted/sensor/e1")
    mac = diagnostic_observation_mac(
        KEY, episode_id=episode_id,
        step_index=0, reservation_token=probe.reservation_token,
        observation=observation,
    )
    assert store.commit_observation(
        episode_id=episode_id, plan=p, observation=observation,
        reservation_token=probe.reservation_token, evidence_mac=mac,
    ).status == "DONE"
    return db, store, p


def test_one_source_bound_payload_handoff_never_reissued_after_restart(tmp_path):
    db, store, p = finished_store(tmp_path)
    issued = store.reserve_repair_once(episode_id=EP, plan=p, payload=BUNDLE_A)
    assert issued.one_time_token and len(issued.one_time_token) == 64
    assert issued.concrete_authority_id == authority(BUNDLE_A)
    assert issued.payload_sha256 == sha256(BUNDLE_A).hexdigest()
    assert issued.status == "RESERVED_UNCONFIRMED"
    snapshot = store.handoff_snapshot(episode_id=EP)
    assert not hasattr(snapshot, "one_time_token")
    assert snapshot.token_sha256 == sha256(
        issued.one_time_token.encode("ascii")
    ).hexdigest()
    assert issued.one_time_token != snapshot.token_sha256
    restarted = RepairDispatchStore(db, trusted_evidence_key=KEY)
    assert restarted.handoff_snapshot(episode_id=EP) == snapshot
    with pytest.raises(DiagnosticExecutionRejected, match="already reserved"):
        restarted.reserve_repair_once(
            episode_id=EP, plan=p, payload=BUNDLE_A
        )


def test_wrong_bundle_and_descriptive_fake_sha_do_not_authorize(tmp_path):
    _, store, p = finished_store(tmp_path)
    with pytest.raises(DiagnosticExecutionRejected, match="bytes do not match"):
        store.reserve_repair_once(episode_id=EP, plan=p, payload=BUNDLE_B)
    with pytest.raises(DiagnosticExecutionRejected, match="payload must"):
        store.reserve_repair_once(episode_id=EP, plan=p, payload=b"")
    # Rejected attempts must not burn the legitimate episode's ability to
    # deliver the correct artifact, unlike a successful reservation.
    issued = store.reserve_repair_once(
        episode_id=EP, plan=p, payload=BUNDLE_A
    )
    assert issued.status == "RESERVED_UNCONFIRMED"


def test_no_dispatch_from_pseudo_hash_authority_identifier(tmp_path):
    p = plan(pseudo=True)
    assert p.complete
    _, store, p = finished_store(tmp_path, test_plan=p)
    with pytest.raises(DiagnosticExecutionRejected, match="bytes do not match"):
        store.reserve_repair_once(episode_id=EP, plan=p, payload=BUNDLE_A)


def test_no_dispatch_before_authenticated_diagnosis_is_done(tmp_path):
    p = plan()
    store = RepairDispatchStore(tmp_path / "unresolved.sqlite3", trusted_evidence_key=KEY)
    store.create_episode(
        episode_id=EP, plan=p,
        trusted_problem_digest=p.digest,
        trusted_tree_commitment=p.digest,
    )
    with pytest.raises(DiagnosticExecutionRejected, match="requires a finished"):
        store.reserve_repair_once(episode_id=EP, plan=p, payload=BUNDLE_A)
    store.reserve_next(episode_id=EP, plan=p)
    with pytest.raises(DiagnosticExecutionRejected, match="requires a finished"):
        store.reserve_repair_once(episode_id=EP, plan=p, payload=BUNDLE_A)


def test_changed_source_tree_revokes_handoff_even_if_payload_same(tmp_path):
    _, store, p = finished_store(tmp_path)
    changed = plan(source="different-controller-commit")
    assert changed.digest != p.digest
    with pytest.raises(DiagnosticExecutionRejected, match="plan identity"):
        store.reserve_repair_once(episode_id=EP, plan=changed, payload=BUNDLE_A)


def test_tampered_trace_and_matching_forged_repair_cannot_dispatch(tmp_path):
    db, store, p = finished_store(tmp_path)
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT trace_json FROM diagnostic_episode WHERE episode_id=?",
            (EP,),
        ).fetchone()
        trace = json.loads(row[0])
        trace[0]["observation_signature"] = [2.0]
        conn.execute(
            "UPDATE diagnostic_episode SET trace_json=?, "
            "resolved_authority_id=? WHERE episode_id=?",
            (json.dumps(trace), authority(BUNDLE_B), EP),
        )
    with pytest.raises(DiagnosticExecutionRejected, match="sensor HMAC receipt"):
        store.reserve_repair_once(episode_id=EP, plan=p, payload=BUNDLE_B)
    with pytest.raises(DiagnosticExecutionRejected, match="no repair handoff"):
        store.handoff_snapshot(episode_id=EP)


def test_anonymized_replay_or_missing_mac_receipt_blocks_repair(tmp_path):
    db, store, p = finished_store(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE diagnostic_episode SET sensor_receipts_json='[]' "
            "WHERE episode_id=?",
            (EP,),
        )
    with pytest.raises(DiagnosticExecutionRejected, match="missing persisted signed"):
        store.reserve_repair_once(episode_id=EP, plan=p, payload=BUNDLE_A)


def test_repair_reservation_is_atomic_against_24_competing_workers(tmp_path):
    db, _, p = finished_store(tmp_path)

    def attempt(_):
        store = RepairDispatchStore(db, trusted_evidence_key=KEY)
        try:
            return store.reserve_repair_once(
                episode_id=EP, plan=p, payload=BUNDLE_A
            )
        except DiagnosticExecutionRejected:
            return None

    with ThreadPoolExecutor(max_workers=12) as workers:
        results = list(workers.map(attempt, range(24)))
    reservations = [x for x in results if x is not None]
    assert len(reservations) == 1
    assert reservations[0].status == "RESERVED_UNCONFIRMED"
    assert RepairDispatchStore(
        db, trusted_evidence_key=KEY
    ).handoff_snapshot(episode_id=EP).token_sha256 == sha256(
        reservations[0].one_time_token.encode("ascii")
    ).hexdigest()


def test_one_handoff_is_scoped_to_episode_and_cannot_authorize_other_episode(tmp_path):
    db, store, p = finished_store(tmp_path)
    other = EP + "/second"
    store.create_episode(
        episode_id=other, plan=p,
        trusted_problem_digest=p.digest,
        trusted_tree_commitment=p.digest,
    )
    with pytest.raises(DiagnosticExecutionRejected, match="requires a finished"):
        store.reserve_repair_once(episode_id=other, plan=p, payload=BUNDLE_A)
    reserved = store.reserve_repair_once(episode_id=EP, plan=p, payload=BUNDLE_A)
    assert reserved.episode_id == EP
    assert reserved.one_time_token not in str(
        store.handoff_snapshot(episode_id=EP)
    )


def test_no_actuation_completion_or_replay_token_retrieval_api(tmp_path):
    db, store, p = finished_store(tmp_path)
    token = store.reserve_repair_once(
        episode_id=EP, plan=p, payload=BUNDLE_A
    ).one_time_token
    restarted = RepairDispatchStore(db, trusted_evidence_key=KEY)
    view = restarted.handoff_snapshot(episode_id=EP)
    assert view.status == "RESERVED_UNCONFIRMED"
    assert token not in repr(view)
    assert not hasattr(restarted, "actuate")
    assert not hasattr(restarted, "execute_repair")
    assert not hasattr(restarted, "mark_success")
