"""Zero-hardware, standard-library proof of persisted sensor-receipt authenticity.

Command:
  python -m research.semantic_invariants.signed_repair_receipt_quick_repro

Intentionally reproduces a *software HMAC substitution attack* against a
SQLite database. This is a deterministic fixture, not real sensor data or
real robot actuation. Reproduction requires no account, models or GPU.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from .embodied_diagnostic_episode_store import (
    DiagnosticEpisodeStore,
    diagnostic_observation_mac,
)
from .embodied_diagnostic_execution import (
    DiagnosticExecutionRejected, DiagnosticObservedStep,
)
from .embodied_robust_diagnosis import synthesize_robust_experiment_plan
from .embodied_semantic_experiment_design import SemanticExperiment
from .robust_diagnosis_quick_repro import _world


def reproduce() -> dict[str, object]:
    plan = synthesize_robust_experiment_plan(
        (_world("world-A", 1.0), _world("world-B", 2.0)),
        (SemanticExperiment("probe", (1.0,), risk=0.1),),
        epsilon=0.1,
        risk_budget=0.1,
        max_probe_risk=0.1,
        authorities={
            "world-A": "sha256:fixture-repair-A",
            "world-B": "sha256:fixture-repair-B",
        },
    )
    assert plan.complete
    test_only_secret = b"public-fixture-signing-key-not-a-production-secret!!"
    episode_id = "public-repro/sensor-proof"
    with TemporaryDirectory(prefix="semrepair_signed_") as root:
        db = Path(root) / "test.sqlite3"
        store = DiagnosticEpisodeStore(db, trusted_evidence_key=test_only_secret)
        store.create_episode(
            episode_id=episode_id,
            plan=plan,
            trusted_problem_digest=plan.digest,
            trusted_tree_commitment=plan.digest,
        )
        grant = store.reserve_next(episode_id=episode_id, plan=plan)
        signed = DiagnosticObservedStep(
            "probe", (1.0,), "demo-only-no-real-sensor"
        )
        mac = diagnostic_observation_mac(
            test_only_secret,
            episode_id=episode_id,
            step_index=0,
            reservation_token=grant.reservation_token,
            observation=signed,
        )
        committed = store.commit_observation(
            episode_id=episode_id,
            plan=plan, observation=signed,
            reservation_token=grant.reservation_token,
            evidence_mac=mac,
        )
        assert committed.status == "DONE"
        restarted = DiagnosticEpisodeStore(
            db, trusted_evidence_key=test_only_secret
        )
        accepted = restarted.verified_authority(
            episode_id=episode_id, plan=plan
        )
        assert accepted.authority_id == "sha256:fixture-repair-A"

        with sqlite3.connect(db) as con:
            old_trace, old_authority, old_receipts = con.execute(
                "SELECT trace_json, resolved_authority_id, sensor_receipts_json "
                "FROM diagnostic_episode WHERE episode_id=?",
                (episode_id,),
            ).fetchone()
            # Replaces both the observed world AND terminal repair with a
            # mutually consistent alternative, without possessing MAC key.
            fake = json.loads(old_trace)
            fake[0]["observation_signature"] = [2.0]
            con.execute(
                "UPDATE diagnostic_episode SET trace_json=?, "
                "resolved_authority_id=? WHERE episode_id=?",
                (json.dumps(fake), "sha256:fixture-repair-B", episode_id),
            )
        forged_consistent_authority_rejected = False
        try:
            restarted.verified_authority(
                episode_id=episode_id, plan=plan
            )
        except DiagnosticExecutionRejected:
            forged_consistent_authority_rejected = True
        assert forged_consistent_authority_rejected

        with sqlite3.connect(db) as con:
            con.execute(
                "UPDATE diagnostic_episode SET trace_json=?, "
                "resolved_authority_id=?, sensor_receipts_json='[]' "
                "WHERE episode_id=?",
                (old_trace, old_authority, episode_id),
            )
        missing_receipt_rejected = False
        try:
            restarted.verified_authority(
                episode_id=episode_id, plan=plan
            )
        except DiagnosticExecutionRejected:
            missing_receipt_rejected = True
        assert missing_receipt_rejected
        assert len(json.loads(old_receipts)) == 1

    return {
        "model": "finite 1D semantic transports and test-only HMAC signer",
        "verified_real_sensor_origin": False,
        "verified_real_robot_actuation": False,
        "intact_signed_repair_identity": accepted.authority_id,
        "forged_alternative_trace_plus_authority_refused": (
            forged_consistent_authority_rejected
        ),
        "legacy_missing_mac_receipt_refused": missing_receipt_rejected,
        "physical_safety_claim": False,
        "upstream_adoption_claim": False,
    }


if __name__ == "__main__":
    print(json.dumps(reproduce(), sort_keys=True, indent=2))
