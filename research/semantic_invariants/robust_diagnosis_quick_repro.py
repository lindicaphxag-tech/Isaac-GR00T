"""One-command, CPU-only falsification capsule for bounded-noise SemRepair.

Usage:
    python -m research.semantic_invariants.robust_diagnosis_quick_repro

The output is a deterministic JSON *mechanism witness*, not learned-policy
performance, third-party validation, or physical exactly-once execution.
"""
from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path
import json

from .embodied_diagnostic_episode_store import (
    DiagnosticEpisodeStore,
    diagnostic_observation_mac,
)
from .embodied_diagnostic_execution import DiagnosticExecutionRejected, DiagnosticObservedStep
from .embodied_robust_diagnosis import (
    RobustResolution,
    RobustObservation,
    advance_robust_diagnosis,
    synthesize_robust_experiment_plan,
)
from .embodied_semantic_experiment_design import SemanticExperiment, _signature
from .embodied_semantic_observability import SemanticDiagnosisHypothesis
from .embodied_semantic_transport import MonomialSemanticTransport, SemanticTransportFactor


def _world(name: str, scale: float) -> SemanticDiagnosisHypothesis:
    return SemanticDiagnosisHypothesis(
        name,
        (SemanticTransportFactor(
            "controller-chart",
            MonomialSemanticTransport((0,), (scale,)),
            f"frozen-{name}-evidence",
        ),),
    )


def run_capsule() -> dict[str, object]:
    # 1. A naive bucket boundary *creates* apparent separation between
    # predictions separated by only 0.002, far below 2*epsilon.
    false_bucket_separator = (
        _signature((1.049,), atol=0.1) != _signature((1.051,), atol=0.1)
    )
    noise_probe = (SemanticExperiment("sensor-probe", (1.0,), risk=0.3),)
    overlapping = synthesize_robust_experiment_plan(
        (_world("low", 1.049), _world("high", 1.051)),
        noise_probe, epsilon=0.1, risk_budget=0.3, max_probe_risk=0.3,
    )
    assert false_bucket_separator and not overlapping.complete

    # 2. Exact concrete-authority equality allows a decision despite
    # non-identifiability of two distinct hidden ABI hypotheses.
    worlds = (
        _world("masked-world-A", 1.0),
        _world("masked-world-B", 1.0),
        _world("different-repair", 2.0),
    )
    common = "sha256:test-placeholder-identical-verified-bundle"
    authorities = {
        "masked-world-A": common,
        "masked-world-B": common,
        "different-repair": "sha256:test-placeholder-other-bundle",
    }
    action_plan = synthesize_robust_experiment_plan(
        worlds, noise_probe, epsilon=0.1, risk_budget=0.3,
        max_probe_risk=0.3, authorities=authorities,
    )
    assert action_plan.complete and action_plan.root.experiment is not None
    observation = RobustObservation("sensor-probe", (1.05,))
    decision = advance_robust_diagnosis(
        plan=action_plan, trusted_plan_digest=action_plan.digest,
        observations=(observation,),
    )
    assert isinstance(decision, RobustResolution)
    assert decision.consistent_hypotheses == (
        "masked-world-A", "masked-world-B"
    ) and decision.authority_id == common
    assert decision.spent_risk == 0.3

    conflicting = synthesize_robust_experiment_plan(
        worlds, noise_probe, epsilon=0.1, risk_budget=0.3,
        max_probe_risk=0.3, authorities={
            **authorities,
            "masked-world-B": "sha256:test-placeholder-different-repair",
        },
    )
    assert not conflicting.complete

    # 3. Route the same raw noisy observation through a durable physically
    # reserved probe; use a test-only pretend sensor signer.
    demo_key = b"test-fixture-key-not-production-sensor-identity!!"
    with TemporaryDirectory(prefix="semrepair_repro_") as td:
        store = DiagnosticEpisodeStore(
            Path(td) / "episode.sqlite3",
            trusted_evidence_key=demo_key,
        )
        episode = "cpu-proof/episode001"
        store.create_episode(
            episode_id=episode, plan=action_plan,
            trusted_problem_digest=action_plan.digest,
            trusted_tree_commitment=action_plan.digest,
        )
        grant = store.reserve_next(episode_id=episode, plan=action_plan)
        received = DiagnosticObservedStep(
            "sensor-probe", observation.measured, "fixture:noise-limited-sensor"
        )
        mac = diagnostic_observation_mac(
            demo_key, episode_id=episode, step_index=0,
            reservation_token=grant.reservation_token,
            observation=received,
        )
        commit = store.commit_observation(
            episode_id=episode, plan=action_plan, observation=received,
            reservation_token=grant.reservation_token, evidence_mac=mac,
        )
        assert commit.status == "DONE" and commit.risk_charged == 0.3

        repeated_effect_denied = False
        try:
            store.reserve_next(episode_id=episode, plan=action_plan)
        except DiagnosticExecutionRejected:
            repeated_effect_denied = True
        assert repeated_effect_denied

    return {
        "model": "finite monomial semantic transports; deterministic bounded error",
        "input_noise_bound": 0.1,
        "false_quantization_separator_detected": false_bucket_separator,
        "noise_overlapping_full_id_refused": not overlapping.complete,
        "same_authority_without_unique_world_accepted": True,
        "surviving_worlds": list(decision.consistent_hypotheses),
        "conflicting_authority_refused": not conflicting.complete,
        "worst_path_risk": action_plan.worst_path_risk,
        "actual_path_risk": decision.spent_risk,
        "authenticated_reservation_finished": commit.status == "DONE",
        "duplicate_effect_reservation_denied": repeated_effect_denied,
        "physical_robot_experiment": False,
        "upstream_maintainer_adoption": False,
    }


if __name__ == "__main__":
    print(json.dumps(run_capsule(), sort_keys=True, indent=2))
