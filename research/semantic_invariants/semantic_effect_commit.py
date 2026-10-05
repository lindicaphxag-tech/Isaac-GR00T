"""Semantic Effect Commit for embodied actions.

This module models physical actions as externally visible effects whose completion
may become ambiguous after timeouts, crashes, lost acknowledgements, or replay.

The protocol deliberately does NOT claim universal exactly-once execution.
Instead it separates logical intent identity, semantic pre/postconditions,
execution acknowledgement, effect observation, and retry safety.

Safety rule: once a non-idempotent effect has been dispatched, restart/replay
must never silently redispatch it without first resolving or preserving the
completion ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Generic, Mapping, TypeVar


State = TypeVar("State")


class EffectStatus(str, Enum):
    PREPARED = "prepared"
    DISPATCHED = "dispatched"
    COMMITTED = "committed"
    ABORTED = "aborted"
    AMBIGUOUS = "ambiguous"


class EffectClass(str, Enum):
    IDEMPOTENT = "idempotent"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


class RuntimeDecision(str, Enum):
    DISPATCH = "dispatch"
    ADVANCE = "advance"
    RETRY = "retry"
    RECONCILE = "reconcile"
    COMPENSATE = "compensate"
    BLOCK = "block"


@dataclass(frozen=True)
class SemanticEffectIntent(Generic[State]):
    effect_id: str
    action_name: str
    effect_class: EffectClass
    dependency_version: str
    precondition: Callable[[State], bool]
    postcondition: Callable[[State], bool]

    def __post_init__(self) -> None:
        if not self.effect_id:
            raise ValueError("effect_id must be non-empty")
        if not self.action_name:
            raise ValueError("action_name must be non-empty")
        if not self.dependency_version:
            raise ValueError("dependency_version must be non-empty")


@dataclass(frozen=True)
class EffectEvidence(Generic[State]):
    observed_state: State
    executor_acknowledged_success: bool = False
    executor_acknowledged_abort: bool = False
    observation_fresh: bool = True

    def __post_init__(self) -> None:
        if self.executor_acknowledged_success and self.executor_acknowledged_abort:
            raise ValueError("executor cannot acknowledge both success and abort")


@dataclass(frozen=True)
class EffectRecord:
    effect_id: str
    action_name: str
    effect_class: EffectClass
    dependency_version: str
    status: EffectStatus
    dispatch_count: int = 0

    def __post_init__(self) -> None:
        if self.dispatch_count < 0:
            raise ValueError("dispatch_count must be non-negative")


class InMemoryEffectLedger:
    def __init__(self) -> None:
        self._records: dict[str, EffectRecord] = {}

    def get(self, effect_id: str) -> EffectRecord | None:
        return self._records.get(effect_id)

    def put(self, record: EffectRecord) -> None:
        current = self._records.get(record.effect_id)
        if current is not None:
            if current.action_name != record.action_name:
                raise ValueError("stable effect_id cannot be reused for a different action")
            if current.effect_class != record.effect_class:
                raise ValueError("stable effect_id cannot change effect class")
            if current.dependency_version != record.dependency_version:
                raise ValueError("stable effect_id cannot change dependency version")
        self._records[record.effect_id] = record

    def snapshot(self) -> Mapping[str, EffectRecord]:
        return dict(self._records)


@dataclass(frozen=True)
class CommitOutcome:
    status: EffectStatus
    decision: RuntimeDecision
    reason: str
    dispatch_count: int


class SemanticEffectRuntime(Generic[State]):
    """Reference state machine for ambiguity-aware embodied execution."""

    def __init__(self, ledger: InMemoryEffectLedger | None = None) -> None:
        self.ledger = ledger or InMemoryEffectLedger()

    def prepare(
        self,
        intent: SemanticEffectIntent[State],
        *,
        current_state: State,
        current_dependency_version: str,
    ) -> CommitOutcome:
        existing = self.ledger.get(intent.effect_id)
        if existing is not None:
            self._validate_existing_intent(existing, intent)
            return self._resume_decision(existing)

        if current_dependency_version != intent.dependency_version:
            return CommitOutcome(
                status=EffectStatus.ABORTED,
                decision=RuntimeDecision.BLOCK,
                reason="dependency version changed before dispatch",
                dispatch_count=0,
            )

        if not intent.precondition(current_state):
            return CommitOutcome(
                status=EffectStatus.ABORTED,
                decision=RuntimeDecision.BLOCK,
                reason="semantic precondition failed",
                dispatch_count=0,
            )

        record = EffectRecord(
            effect_id=intent.effect_id,
            action_name=intent.action_name,
            effect_class=intent.effect_class,
            dependency_version=intent.dependency_version,
            status=EffectStatus.PREPARED,
            dispatch_count=0,
        )
        self.ledger.put(record)
        return CommitOutcome(
            status=EffectStatus.PREPARED,
            decision=RuntimeDecision.DISPATCH,
            reason="intent prepared",
            dispatch_count=0,
        )

    def mark_dispatched(self, effect_id: str) -> EffectRecord:
        """Persist the transition immediately before/with physical dispatch.

        A previously ambiguous non-idempotent effect cannot be redispatched via
        this primitive; the caller must reconcile it first.
        """
        record = self._require(effect_id)
        if record.status == EffectStatus.COMMITTED:
            return record
        if record.status == EffectStatus.AMBIGUOUS and record.effect_class != EffectClass.IDEMPOTENT:
            raise ValueError("cannot blindly redispatch ambiguous non-idempotent effect")
        if record.status not in {
            EffectStatus.PREPARED,
            EffectStatus.DISPATCHED,
            EffectStatus.AMBIGUOUS,
        }:
            raise ValueError(f"cannot dispatch effect from status {record.status.value}")

        # Repeating mark_dispatched while already DISPATCHED means the caller is
        # attempting another physical dispatch without resolving the first one.
        if record.status == EffectStatus.DISPATCHED and record.effect_class != EffectClass.IDEMPOTENT:
            raise ValueError("cannot redispatch unresolved non-idempotent effect")

        updated = replace(
            record,
            status=EffectStatus.DISPATCHED,
            dispatch_count=record.dispatch_count + 1,
        )
        self.ledger.put(updated)
        return updated

    def resolve(
        self,
        intent: SemanticEffectIntent[State],
        evidence: EffectEvidence[State],
    ) -> CommitOutcome:
        record = self._require(intent.effect_id)
        self._validate_existing_intent(record, intent)

        if record.status == EffectStatus.COMMITTED:
            return CommitOutcome(
                status=EffectStatus.COMMITTED,
                decision=RuntimeDecision.ADVANCE,
                reason="effect already committed",
                dispatch_count=record.dispatch_count,
            )

        post_holds = evidence.observation_fresh and intent.postcondition(evidence.observed_state)
        pre_holds = evidence.observation_fresh and intent.precondition(evidence.observed_state)

        if evidence.executor_acknowledged_success and post_holds:
            return self._store_and_decide(
                record,
                EffectStatus.COMMITTED,
                RuntimeDecision.ADVANCE,
                "executor success and semantic postcondition agree",
            )

        if post_holds:
            return self._store_and_decide(
                record,
                EffectStatus.COMMITTED,
                RuntimeDecision.ADVANCE,
                "physical effect observed despite missing acknowledgement",
            )

        if evidence.executor_acknowledged_abort and pre_holds:
            return self._store_and_decide(
                record,
                EffectStatus.ABORTED,
                RuntimeDecision.RETRY,
                "executor abort and unchanged semantic state agree",
            )

        return self._resolve_ambiguity(record)

    def _resolve_ambiguity(self, record: EffectRecord) -> CommitOutcome:
        if record.effect_class == EffectClass.IDEMPOTENT:
            decision = RuntimeDecision.RETRY
            reason = "completion ambiguous but effect declared idempotent"
        elif record.effect_class == EffectClass.REVERSIBLE:
            decision = RuntimeDecision.RECONCILE
            reason = "completion ambiguous; reconcile or compensate before retry"
        else:
            decision = RuntimeDecision.BLOCK
            reason = "irreversible effect completion ambiguous; blind retry prohibited"

        return self._store_and_decide(
            record,
            EffectStatus.AMBIGUOUS,
            decision,
            reason,
        )

    def _resume_decision(self, record: EffectRecord) -> CommitOutcome:
        if record.status == EffectStatus.COMMITTED:
            decision = RuntimeDecision.ADVANCE
            reason = "durable ledger proves effect already committed"
        elif record.status == EffectStatus.ABORTED:
            decision = RuntimeDecision.RETRY
            reason = "durable ledger records definite abort"
        elif record.status == EffectStatus.PREPARED:
            decision = RuntimeDecision.DISPATCH
            reason = "prepared effect has not been recorded as dispatched"
        elif record.status in {EffectStatus.DISPATCHED, EffectStatus.AMBIGUOUS}:
            # A crash after DISPATCHED but before resolution is precisely the
            # ambiguous-completion case. Never convert it back to fresh DISPATCH
            # for a non-idempotent physical effect.
            if record.effect_class == EffectClass.IDEMPOTENT:
                decision = RuntimeDecision.RETRY
                reason = "unresolved dispatch is retryable only because effect is idempotent"
            elif record.effect_class == EffectClass.REVERSIBLE:
                decision = RuntimeDecision.RECONCILE
                reason = "unresolved dispatch requires reconciliation before retry"
            else:
                decision = RuntimeDecision.BLOCK
                reason = "unresolved irreversible dispatch blocks blind replay"
        else:  # pragma: no cover - defensive enum exhaustiveness
            raise AssertionError(record.status)

        return CommitOutcome(
            status=record.status,
            decision=decision,
            reason=reason,
            dispatch_count=record.dispatch_count,
        )

    def _validate_existing_intent(
        self,
        record: EffectRecord,
        intent: SemanticEffectIntent[State],
    ) -> None:
        if record.action_name != intent.action_name:
            raise ValueError("existing effect_id is bound to a different action")
        if record.effect_class != intent.effect_class:
            raise ValueError("existing effect_id is bound to a different effect class")
        if record.dependency_version != intent.dependency_version:
            raise ValueError("existing effect_id is bound to a different dependency version")

    def _store_and_decide(
        self,
        record: EffectRecord,
        status: EffectStatus,
        decision: RuntimeDecision,
        reason: str,
    ) -> CommitOutcome:
        updated = replace(record, status=status)
        self.ledger.put(updated)
        return CommitOutcome(
            status=status,
            decision=decision,
            reason=reason,
            dispatch_count=updated.dispatch_count,
        )

    def _require(self, effect_id: str) -> EffectRecord:
        record = self.ledger.get(effect_id)
        if record is None:
            raise KeyError(f"unknown effect_id: {effect_id}")
        return record