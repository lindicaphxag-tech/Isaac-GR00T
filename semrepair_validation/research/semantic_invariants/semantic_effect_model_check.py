"""Finite-state model checking for Semantic Effect Commit.

The checker explores short event traces over the executable runtime itself.
It is deliberately small and dependency-free: the goal is to catch unsafe
state-machine transitions such as redispatching an unresolved irreversible
physical effect after crash/replay.

This is bounded verification, not a proof over arbitrary environment dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .semantic_effect_commit import (
    EffectClass,
    EffectEvidence,
    EffectStatus,
    InMemoryEffectLedger,
    RuntimeDecision,
    SemanticEffectIntent,
    SemanticEffectRuntime,
)


EVENTS = (
    "prepare",
    "dispatch",
    "restart",
    "observe_commit",
    "observe_abort",
    "observe_stale",
)


@dataclass(frozen=True)
class TraceResult:
    effect_class: EffectClass
    trace: tuple[str, ...]
    safe: bool
    violation: str | None
    final_status: str | None
    dispatch_count: int


def _intent(effect_class: EffectClass) -> SemanticEffectIntent[dict[str, int]]:
    return SemanticEffectIntent(
        effect_id=f"effect:{effect_class.value}",
        action_name="physical_effect",
        effect_class=effect_class,
        dependency_version="v1",
        precondition=lambda state: state["done"] == 0,
        postcondition=lambda state: state["done"] == 1,
    )


def run_trace(effect_class: EffectClass, trace: tuple[str, ...]) -> TraceResult:
    ledger = InMemoryEffectLedger()
    runtime = SemanticEffectRuntime(ledger)
    intent = _intent(effect_class)
    violation = None

    for event in trace:
        try:
            if event == "prepare":
                outcome = runtime.prepare(
                    intent,
                    current_state={"done": 0},
                    current_dependency_version="v1",
                )
                record = ledger.get(intent.effect_id)
                if (
                    record is not None
                    and record.dispatch_count > 0
                    and effect_class != EffectClass.IDEMPOTENT
                    and outcome.decision in {RuntimeDecision.DISPATCH, RuntimeDecision.RETRY}
                    and record.status not in {EffectStatus.ABORTED}
                ):
                    violation = "non-idempotent unresolved effect became redispatchable"

            elif event == "dispatch":
                before = ledger.get(intent.effect_id)
                before_count = before.dispatch_count if before else 0
                runtime.mark_dispatched(intent.effect_id)
                after = ledger.get(intent.effect_id)
                if (
                    effect_class != EffectClass.IDEMPOTENT
                    and before is not None
                    and before_count > 0
                    and before.status in {EffectStatus.DISPATCHED, EffectStatus.AMBIGUOUS}
                    and after is not None
                    and after.dispatch_count > before_count
                ):
                    violation = "non-idempotent unresolved effect was dispatched twice"

            elif event == "restart":
                runtime = SemanticEffectRuntime(ledger)
                record = ledger.get(intent.effect_id)
                if record is not None:
                    outcome = runtime.prepare(
                        intent,
                        current_state={"done": 0},
                        current_dependency_version="v1",
                    )
                    if (
                        record.dispatch_count > 0
                        and effect_class != EffectClass.IDEMPOTENT
                        and record.status in {EffectStatus.DISPATCHED, EffectStatus.AMBIGUOUS}
                        and outcome.decision in {RuntimeDecision.DISPATCH, RuntimeDecision.RETRY}
                    ):
                        violation = "restart made unresolved non-idempotent effect redispatchable"

            elif event == "observe_commit":
                runtime.resolve(
                    intent,
                    EffectEvidence(
                        observed_state={"done": 1},
                        observation_fresh=True,
                    ),
                )

            elif event == "observe_abort":
                runtime.resolve(
                    intent,
                    EffectEvidence(
                        observed_state={"done": 0},
                        executor_acknowledged_abort=True,
                        observation_fresh=True,
                    ),
                )

            elif event == "observe_stale":
                runtime.resolve(
                    intent,
                    EffectEvidence(
                        observed_state={"done": 0},
                        observation_fresh=False,
                    ),
                )

            else:  # pragma: no cover
                raise AssertionError(event)

        except (KeyError, ValueError):
            # Invalid event order is a rejected transition, not a safety failure.
            pass

        if violation is not None:
            break

    record = ledger.get(intent.effect_id)
    return TraceResult(
        effect_class=effect_class,
        trace=trace,
        safe=violation is None,
        violation=violation,
        final_status=record.status.value if record is not None else None,
        dispatch_count=record.dispatch_count if record is not None else 0,
    )


def bounded_model_check(*, max_depth: int = 5) -> dict[str, object]:
    if max_depth < 1:
        raise ValueError("max_depth must be positive")

    checked = 0
    violations: list[dict[str, object]] = []
    max_dispatch = {item.value: 0 for item in EffectClass}

    for effect_class in EffectClass:
        for depth in range(1, max_depth + 1):
            for trace in product(EVENTS, repeat=depth):
                result = run_trace(effect_class, trace)
                checked += 1
                max_dispatch[effect_class.value] = max(
                    max_dispatch[effect_class.value],
                    result.dispatch_count,
                )
                if not result.safe:
                    violations.append(
                        {
                            "effect_class": effect_class.value,
                            "trace": list(result.trace),
                            "violation": result.violation,
                            "dispatch_count": result.dispatch_count,
                            "final_status": result.final_status,
                        }
                    )

    return {
        "checker": "semantic-effect-commit-bounded-v0",
        "max_depth": max_depth,
        "events": list(EVENTS),
        "traces_checked": checked,
        "violations": violations,
        "safe": not violations,
        "max_dispatch_count": max_dispatch,
    }


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Bounded model check Semantic Effect Commit")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = bounded_model_check(max_depth=args.depth)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"checked={report['traces_checked']} safe={report['safe']} "
            f"violations={len(report['violations'])}"
        )

    if not report["safe"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()