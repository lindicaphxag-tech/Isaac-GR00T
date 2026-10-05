"""Deterministic fault-injection assay for physical-effect execution semantics.

The assay compares common retry/recovery policies on side-effecting embodied
actions. It is intentionally lightweight and state-based so the protocol can be
stress-tested without robot hardware or a heavy simulator.

The claim is not that this toy world represents all robotics. Its purpose is to
make the safety/liveness tradeoff explicit under controlled ambiguity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from .semantic_effect_certificate import EffectCommitPolicy, issue_effect_certificate
from .semantic_effect_commit import (
    EffectClass,
    EffectStatus,
    InMemoryEffectLedger,
    RuntimeDecision,
    SemanticEffectIntent,
    SemanticEffectRuntime,
)
from .semantic_effect_evidence import EffectObservation
from .semantic_effect_runtime import CertifiedSemanticEffectRuntime


@dataclass(frozen=True)
class FaultScenario:
    name: str
    first_dispatch_applies_effect: bool
    ack_delivered: bool
    explicit_abort: bool
    camera_fresh: bool
    proprio_fresh: bool
    camera_correct: bool = True
    proprio_correct: bool = True
    restart_before_resolution: bool = False


@dataclass(frozen=True)
class PolicyOutcome:
    policy: str
    scenario: str
    physical_effect_count: int
    logical_complete: bool
    blocked: bool
    retries: int
    unsafe_retries: int
    duplicate_effects: int
    omitted_effect: bool


def default_scenarios() -> tuple[FaultScenario, ...]:
    return (
        FaultScenario(
            name="success_ack_lost_two_fresh_planes",
            first_dispatch_applies_effect=True,
            ack_delivered=False,
            explicit_abort=False,
            camera_fresh=True,
            proprio_fresh=True,
        ),
        FaultScenario(
            name="success_ack_lost_one_plane_stale",
            first_dispatch_applies_effect=True,
            ack_delivered=False,
            explicit_abort=False,
            camera_fresh=True,
            proprio_fresh=False,
        ),
        FaultScenario(
            name="failed_dispatch_ack_lost",
            first_dispatch_applies_effect=False,
            ack_delivered=False,
            explicit_abort=False,
            camera_fresh=True,
            proprio_fresh=True,
        ),
        FaultScenario(
            name="definite_abort",
            first_dispatch_applies_effect=False,
            ack_delivered=False,
            explicit_abort=True,
            camera_fresh=True,
            proprio_fresh=True,
        ),
        FaultScenario(
            name="success_then_runtime_restart",
            first_dispatch_applies_effect=True,
            ack_delivered=False,
            explicit_abort=False,
            camera_fresh=False,
            proprio_fresh=False,
            restart_before_resolution=True,
        ),
        FaultScenario(
            name="success_correlated_camera_false_negative",
            first_dispatch_applies_effect=True,
            ack_delivered=False,
            explicit_abort=False,
            camera_fresh=True,
            proprio_fresh=True,
            camera_correct=False,
            proprio_correct=True,
        ),
    )


def _observed_state(*, physical_done: bool, correct: bool) -> dict[str, int]:
    observed = physical_done if correct else not physical_done
    return {"done": int(observed)}


def _intent(effect_class: EffectClass = EffectClass.IRREVERSIBLE):
    return SemanticEffectIntent(
        effect_id="dispense:assay",
        action_name="dispense_one",
        effect_class=effect_class,
        dependency_version="plan-v1",
        precondition=lambda state: state["done"] == 0,
        postcondition=lambda state: state["done"] == 1,
    )


def _finalize(
    *,
    policy: str,
    scenario: FaultScenario,
    physical_effect_count: int,
    logical_complete: bool,
    blocked: bool,
    retries: int,
    unsafe_retries: int,
) -> PolicyOutcome:
    return PolicyOutcome(
        policy=policy,
        scenario=scenario.name,
        physical_effect_count=physical_effect_count,
        logical_complete=logical_complete,
        blocked=blocked,
        retries=retries,
        unsafe_retries=unsafe_retries,
        duplicate_effects=max(0, physical_effect_count - 1),
        omitted_effect=physical_effect_count == 0,
    )


def retry_on_timeout(scenario: FaultScenario) -> PolicyOutcome:
    physical = int(scenario.first_dispatch_applies_effect)
    retries = 0
    unsafe = 0
    complete = scenario.ack_delivered and scenario.first_dispatch_applies_effect

    if not scenario.ack_delivered:
        retries += 1
        if scenario.first_dispatch_applies_effect:
            unsafe += 1
        # Model a retry whose executor succeeds.
        physical += 1
        complete = True

    return _finalize(
        policy="retry_on_timeout",
        scenario=scenario,
        physical_effect_count=physical,
        logical_complete=complete,
        blocked=False,
        retries=retries,
        unsafe_retries=unsafe,
    )


def fail_stop(scenario: FaultScenario) -> PolicyOutcome:
    physical = int(scenario.first_dispatch_applies_effect)
    complete = scenario.ack_delivered and scenario.first_dispatch_applies_effect
    return _finalize(
        policy="fail_stop",
        scenario=scenario,
        physical_effect_count=physical,
        logical_complete=complete,
        blocked=not complete,
        retries=0,
        unsafe_retries=0,
    )


def ledger_only(scenario: FaultScenario) -> PolicyOutcome:
    # Durable dispatch record prevents duplicate re-execution, but without
    # physical evidence the workflow cannot safely infer completion.
    physical = int(scenario.first_dispatch_applies_effect)
    complete = scenario.ack_delivered and scenario.first_dispatch_applies_effect
    return _finalize(
        policy="ledger_only",
        scenario=scenario,
        physical_effect_count=physical,
        logical_complete=complete,
        blocked=not complete,
        retries=0,
        unsafe_retries=0,
    )


def semantic_effect_commit(scenario: FaultScenario) -> PolicyOutcome:
    intent = _intent()
    contract_id = "embodied/effect/dispense@0.1"
    certificate = issue_effect_certificate(
        intent,
        policy=EffectCommitPolicy(min_commit_planes=2, min_abort_planes=1),
        semantic_contract_id=contract_id,
    )
    ledger = InMemoryEffectLedger()
    certified = CertifiedSemanticEffectRuntime(SemanticEffectRuntime(ledger))

    certified.prepare(
        intent,
        certificate,
        semantic_contract_id=contract_id,
        current_state={"done": 0},
        current_dependency_version="plan-v1",
    )
    certified.mark_dispatched(intent, certificate, semantic_contract_id=contract_id)

    physical = int(scenario.first_dispatch_applies_effect)

    if scenario.restart_before_resolution:
        certified = CertifiedSemanticEffectRuntime(SemanticEffectRuntime(ledger))
        resumed = certified.prepare(
            intent,
            certificate,
            semantic_contract_id=contract_id,
            current_state={"done": physical},
            current_dependency_version="plan-v1",
        )
        return _finalize(
            policy="semantic_effect_commit",
            scenario=scenario,
            physical_effect_count=physical,
            logical_complete=resumed.decision == RuntimeDecision.ADVANCE,
            blocked=resumed.decision in {RuntimeDecision.BLOCK, RuntimeDecision.RECONCILE},
            retries=0,
            unsafe_retries=0,
        )

    observations = (
        EffectObservation(
            "camera",
            "camera",
            _observed_state(
                physical_done=bool(physical),
                correct=scenario.camera_correct,
            ),
            fresh=scenario.camera_fresh,
        ),
        EffectObservation(
            "proprioception",
            "proprioception",
            _observed_state(
                physical_done=bool(physical),
                correct=scenario.proprio_correct,
            ),
            fresh=scenario.proprio_fresh,
        ),
    )

    resolution = certified.resolve(
        intent,
        certificate,
        observations,
        semantic_contract_id=contract_id,
        executor_acknowledged_success=(
            scenario.ack_delivered and scenario.first_dispatch_applies_effect
        ),
        executor_acknowledged_abort=scenario.explicit_abort,
    )
    outcome = resolution.outcome
    retries = 0

    if outcome.decision == RuntimeDecision.RETRY:
        retries = 1
        # Retry occurs only after definite abort under this irreversible intent.
        physical += 1
        # The assay models the retry executor as successful with fresh evidence.
        # A real runtime would run the full protocol again under a new attempt.
        complete = True
    else:
        complete = outcome.decision == RuntimeDecision.ADVANCE

    return _finalize(
        policy="semantic_effect_commit",
        scenario=scenario,
        physical_effect_count=physical,
        logical_complete=complete,
        blocked=outcome.decision in {RuntimeDecision.BLOCK, RuntimeDecision.RECONCILE},
        retries=retries,
        unsafe_retries=0,
    )


POLICIES: tuple[Callable[[FaultScenario], PolicyOutcome], ...] = (
    retry_on_timeout,
    fail_stop,
    ledger_only,
    semantic_effect_commit,
)


def run_fault_assay(
    scenarios: Iterable[FaultScenario] | None = None,
) -> dict[str, object]:
    scenarios = tuple(scenarios or default_scenarios())
    outcomes = tuple(
        policy(scenario)
        for scenario in scenarios
        for policy in POLICIES
    )

    aggregate = {}
    for policy in tuple(item.__name__ for item in POLICIES):
        rows = [item for item in outcomes if item.policy == policy]
        aggregate[policy] = {
            "scenarios": len(rows),
            "task_completions": sum(item.logical_complete for item in rows),
            "blocked": sum(item.blocked for item in rows),
            "retries": sum(item.retries for item in rows),
            "unsafe_retries": sum(item.unsafe_retries for item in rows),
            "duplicate_effects": sum(item.duplicate_effects for item in rows),
            "omitted_effect_scenarios": sum(item.omitted_effect for item in rows),
        }

    return {
        "assay": "semantic-effect-fault-injection-v0",
        "scenario_count": len(scenarios),
        "outcomes": [asdict(item) for item in outcomes],
        "aggregate": aggregate,
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run embodied effect fault-injection assay")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_fault_assay()

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for policy, summary in report["aggregate"].items():
            print(policy, summary)


if __name__ == "__main__":
    main()