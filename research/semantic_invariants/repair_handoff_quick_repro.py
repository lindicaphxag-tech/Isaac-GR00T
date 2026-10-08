"""Pure stdlib source-pinned one-use repair handoff falsification.

  python -m research.semantic_invariants.repair_handoff_quick_repro

The test-only 'repair' is inert bytes. NO robot, sim, model, physical risk
measurement, deployment key or real actuator exists in this demonstration.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .embodied_repair_handoff import RepairDispatchStore
from .embodied_diagnostic_episode_store import diagnostic_observation_mac
from .embodied_diagnostic_execution import (
    DiagnosticExecutionRejected, DiagnosticObservedStep,
)
from .embodied_robust_diagnosis import synthesize_robust_experiment_plan
from .embodied_semantic_experiment_design import SemanticExperiment
from .robust_diagnosis_quick_repro import _world


def run() -> dict:
    correct_payload = b"EXAMPLE ONLY inert adapter byte payload version A"
    other_payload = b"EXAMPLE ONLY inert adapter byte payload version B"
    authority = lambda data: "sha256:" + sha256(data).hexdigest()
    plan = synthesize_robust_experiment_plan(
        (_world("a", 1.0), _world("b", 2.0)),
        (SemanticExperiment("source-bound-probe", (1.0,), risk=0.1),),
        epsilon=0.1, risk_budget=0.1, max_probe_risk=0.1,
        authorities={"a": authority(correct_payload), "b": authority(other_payload)},
    )
    assert plan.complete
    demo_key = b"PUBLIC-TEST-HMAC-SIGNER-NOT-A-REAL-SENSOR-SECRET!!!!"
    episode_id = "fixture/one-use/episode1"
    with TemporaryDirectory(prefix="repair_handoff_demo_") as root:
        db = Path(root) / "episode.sqlite3"
        store = RepairDispatchStore(db, trusted_evidence_key=demo_key)
        store.create_episode(
            episode_id=episode_id, plan=plan,
            trusted_problem_digest=plan.digest,
            trusted_tree_commitment=plan.digest,
        )
        probe = store.reserve_next(episode_id=episode_id, plan=plan)
        signed = DiagnosticObservedStep(
            "source-bound-probe", (1.0,), "fixture-only-sensor1"
        )
        mac = diagnostic_observation_mac(
            demo_key, episode_id=episode_id, step_index=0,
            reservation_token=probe.reservation_token, observation=signed,
        )
        done = store.commit_observation(
            episode_id=episode_id, plan=plan, observation=signed,
            reservation_token=probe.reservation_token, evidence_mac=mac,
        )
        assert done.status == "DONE"
        wrong_payload_refused = False
        try:
            store.reserve_repair_once(
                episode_id=episode_id, plan=plan, payload=other_payload
            )
        except DiagnosticExecutionRejected:
            wrong_payload_refused = True
        assert wrong_payload_refused
        first = store.reserve_repair_once(
            episode_id=episode_id, plan=plan, payload=correct_payload
        )
        restarted = RepairDispatchStore(db, trusted_evidence_key=demo_key)
        duplicate_refused = False
        try:
            restarted.reserve_repair_once(
                episode_id=episode_id, plan=plan, payload=correct_payload
            )
        except DiagnosticExecutionRejected:
            duplicate_refused = True
        assert duplicate_refused
        inspect = restarted.handoff_snapshot(episode_id=episode_id)
        assert inspect.status == "RESERVED_UNCONFIRMED"
        assert sha256(first.one_time_token.encode("ascii")).hexdigest() == (
            inspect.token_sha256
        )
        assert not hasattr(inspect, "one_time_token")
    return {
        "model": "inert bytes, bounded-noise 1D diagnostic and test HMAC key",
        "source_bound_real_sha256_payload": first.payload_sha256,
        "wrong_payload_refused": wrong_payload_refused,
        "single_repair_handoff_reserved": first.status == "RESERVED_UNCONFIRMED",
        "restart_duplicate_handoff_refused": duplicate_refused,
        "raw_one_use_token_persisted_or_recoverable": False,
        "actual_robot_repair_executed": False,
        "physical_effect_exactly_once_proven": False,
        "independently_reviewed": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
