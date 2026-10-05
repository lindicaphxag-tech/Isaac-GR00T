from research.semantic_invariants.semantic_effect_commit import (
    EffectClass,
    EffectEvidence,
    EffectStatus,
    InMemoryEffectLedger,
    RuntimeDecision,
    SemanticEffectIntent,
    SemanticEffectRuntime,
)


def _intent(effect_class=EffectClass.IRREVERSIBLE, effect_id="dispense:task-7:item-3"):
    return SemanticEffectIntent(
        effect_id=effect_id,
        action_name="dispense_one",
        effect_class=effect_class,
        dependency_version="plan-v3",
        precondition=lambda state: state["dispensed"] == 0,
        postcondition=lambda state: state["dispensed"] == 1,
    )


def test_lost_ack_after_physical_success_advances_without_second_dispatch():
    runtime = SemanticEffectRuntime()
    intent = _intent()

    prepared = runtime.prepare(
        intent,
        current_state={"dispensed": 0},
        current_dependency_version="plan-v3",
    )
    assert prepared.decision == RuntimeDecision.DISPATCH

    runtime.mark_dispatched(intent.effect_id)

    resolved = runtime.resolve(
        intent,
        EffectEvidence(
            observed_state={"dispensed": 1},
            executor_acknowledged_success=False,
            observation_fresh=True,
        ),
    )

    assert resolved.status == EffectStatus.COMMITTED
    assert resolved.decision == RuntimeDecision.ADVANCE
    assert resolved.dispatch_count == 1

    resumed = runtime.prepare(
        intent,
        current_state={"dispensed": 1},
        current_dependency_version="plan-v3",
    )
    assert resumed.decision == RuntimeDecision.ADVANCE
    assert resumed.dispatch_count == 1


def test_restart_after_dispatch_before_resolution_never_blindly_retries_irreversible_effect():
    ledger = InMemoryEffectLedger()
    intent = _intent()

    first = SemanticEffectRuntime(ledger)
    first.prepare(intent, current_state={"dispensed": 0}, current_dependency_version="plan-v3")
    first.mark_dispatched(intent.effect_id)

    restarted = SemanticEffectRuntime(ledger)
    outcome = restarted.prepare(
        intent,
        current_state={"dispensed": 0},
        current_dependency_version="plan-v3",
    )

    assert outcome.status == EffectStatus.DISPATCHED
    assert outcome.decision == RuntimeDecision.BLOCK
    assert outcome.dispatch_count == 1


def test_restart_after_dispatch_reconciles_reversible_effect():
    ledger = InMemoryEffectLedger()
    intent = _intent(effect_class=EffectClass.REVERSIBLE, effect_id="move-tray:slot-a")

    first = SemanticEffectRuntime(ledger)
    first.prepare(intent, current_state={"dispensed": 0}, current_dependency_version="plan-v3")
    first.mark_dispatched(intent.effect_id)

    restarted = SemanticEffectRuntime(ledger)
    outcome = restarted.prepare(
        intent,
        current_state={"dispensed": 0},
        current_dependency_version="plan-v3",
    )

    assert outcome.decision == RuntimeDecision.RECONCILE
    assert outcome.dispatch_count == 1


def test_restart_after_dispatch_can_retry_only_idempotent_effect():
    ledger = InMemoryEffectLedger()
    intent = _intent(effect_class=EffectClass.IDEMPOTENT, effect_id="set-light:on")

    first = SemanticEffectRuntime(ledger)
    first.prepare(intent, current_state={"dispensed": 0}, current_dependency_version="plan-v3")
    first.mark_dispatched(intent.effect_id)

    restarted = SemanticEffectRuntime(ledger)
    outcome = restarted.prepare(
        intent,
        current_state={"dispensed": 0},
        current_dependency_version="plan-v3",
    )

    assert outcome.decision == RuntimeDecision.RETRY


def test_irreversible_ambiguous_completion_blocks_blind_retry():
    runtime = SemanticEffectRuntime()
    intent = _intent()

    runtime.prepare(intent, current_state={"dispensed": 0}, current_dependency_version="plan-v3")
    runtime.mark_dispatched(intent.effect_id)

    outcome = runtime.resolve(
        intent,
        EffectEvidence(
            observed_state={"dispensed": 0},
            executor_acknowledged_success=False,
            executor_acknowledged_abort=False,
            observation_fresh=False,
        ),
    )

    assert outcome.status == EffectStatus.AMBIGUOUS
    assert outcome.decision == RuntimeDecision.BLOCK

    try:
        runtime.mark_dispatched(intent.effect_id)
    except ValueError as error:
        assert "ambiguous non-idempotent" in str(error)
    else:
        raise AssertionError("ambiguous irreversible effect must not be redispatched")


def test_idempotent_ambiguous_completion_allows_retry():
    runtime = SemanticEffectRuntime()
    intent = _intent(effect_class=EffectClass.IDEMPOTENT, effect_id="set-light:on")

    runtime.prepare(intent, current_state={"dispensed": 0}, current_dependency_version="plan-v3")
    runtime.mark_dispatched(intent.effect_id)

    outcome = runtime.resolve(
        intent,
        EffectEvidence(observed_state={"dispensed": 0}, observation_fresh=False),
    )

    assert outcome.status == EffectStatus.AMBIGUOUS
    assert outcome.decision == RuntimeDecision.RETRY


def test_reversible_ambiguous_completion_requires_reconciliation():
    runtime = SemanticEffectRuntime()
    intent = _intent(effect_class=EffectClass.REVERSIBLE, effect_id="move-tray:slot-a")

    runtime.prepare(intent, current_state={"dispensed": 0}, current_dependency_version="plan-v3")
    runtime.mark_dispatched(intent.effect_id)

    outcome = runtime.resolve(
        intent,
        EffectEvidence(observed_state={"dispensed": 0}, observation_fresh=False),
    )

    assert outcome.decision == RuntimeDecision.RECONCILE


def test_stale_plan_dependency_is_rejected_before_dispatch():
    runtime = SemanticEffectRuntime()
    intent = _intent()

    outcome = runtime.prepare(
        intent,
        current_state={"dispensed": 0},
        current_dependency_version="plan-v4",
    )

    assert outcome.status == EffectStatus.ABORTED
    assert outcome.decision == RuntimeDecision.BLOCK
    assert runtime.ledger.get(intent.effect_id) is None


def test_failed_semantic_precondition_is_rejected_before_dispatch():
    runtime = SemanticEffectRuntime()
    intent = _intent()

    outcome = runtime.prepare(
        intent,
        current_state={"dispensed": 1},
        current_dependency_version="plan-v3",
    )

    assert outcome.status == EffectStatus.ABORTED
    assert outcome.decision == RuntimeDecision.BLOCK


def test_durable_ledger_prevents_reexecution_after_runtime_restart():
    ledger = InMemoryEffectLedger()
    intent = _intent()

    first = SemanticEffectRuntime(ledger)
    first.prepare(intent, current_state={"dispensed": 0}, current_dependency_version="plan-v3")
    first.mark_dispatched(intent.effect_id)
    first.resolve(intent, EffectEvidence(observed_state={"dispensed": 1}))

    restarted = SemanticEffectRuntime(ledger)
    outcome = restarted.prepare(
        intent,
        current_state={"dispensed": 1},
        current_dependency_version="plan-v3",
    )

    assert outcome.status == EffectStatus.COMMITTED
    assert outcome.decision == RuntimeDecision.ADVANCE
    assert outcome.dispatch_count == 1


def test_same_effect_id_cannot_be_rebound_to_different_action_semantics():
    ledger = InMemoryEffectLedger()
    runtime = SemanticEffectRuntime(ledger)
    first = _intent()
    runtime.prepare(first, current_state={"dispensed": 0}, current_dependency_version="plan-v3")

    conflicting = SemanticEffectIntent(
        effect_id=first.effect_id,
        action_name="drop_item",
        effect_class=EffectClass.IRREVERSIBLE,
        dependency_version="plan-v3",
        precondition=lambda state: True,
        postcondition=lambda state: True,
    )

    try:
        runtime.ledger.put(
            runtime.ledger.get(first.effect_id).__class__(
                effect_id=conflicting.effect_id,
                action_name=conflicting.action_name,
                effect_class=conflicting.effect_class,
                dependency_version=conflicting.dependency_version,
                status=EffectStatus.PREPARED,
            )
        )
    except ValueError as error:
        assert "cannot be reused" in str(error)
    else:
        raise AssertionError("effect identity rebinding must be rejected")


def test_same_effect_id_cannot_change_dependency_version_on_resume():
    ledger = InMemoryEffectLedger()
    runtime = SemanticEffectRuntime(ledger)
    first = _intent()
    runtime.prepare(first, current_state={"dispensed": 0}, current_dependency_version="plan-v3")

    rebound = SemanticEffectIntent(
        effect_id=first.effect_id,
        action_name=first.action_name,
        effect_class=first.effect_class,
        dependency_version="plan-v4",
        precondition=first.precondition,
        postcondition=first.postcondition,
    )

    try:
        runtime.prepare(
            rebound,
            current_state={"dispensed": 0},
            current_dependency_version="plan-v4",
        )
    except ValueError as error:
        assert "dependency version" in str(error)
    else:
        raise AssertionError("stable effect identity must bind dependency version")