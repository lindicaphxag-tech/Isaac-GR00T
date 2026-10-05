from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from .runtime_abi import (
    ActionChart,
    CASJProbeBatch,
    FrozenChunkPolicy,
    collect_two_scale_probes,
)


@runtime_checkable
class SlotSupportInterventionChart(Protocol):
    """Apply one physical-support intervention to one temporal observation slot."""

    @property
    def num_supports(self) -> int:
        ...

    @property
    def support_dim(self) -> int:
        ...

    def intervene_slot(
        self,
        observation_slot: Any,
        *,
        direction: int,
        coefficients: np.ndarray,
        magnitude: float,
    ) -> Any:
        ...


@dataclass(frozen=True)
class TemporalSupportInterventionChart:
    """Lift a per-slot support intervention into a history with explicit scope.

    The affected slots are part of the intervention identity.  This prevents a
    physical event at the current time from silently rewriting past policy
    observations.
    """

    slot_chart: SlotSupportInterventionChart
    affected_slots: tuple[int, ...]

    @property
    def num_supports(self) -> int:
        return int(self.slot_chart.num_supports)

    @property
    def support_dim(self) -> int:
        return int(self.slot_chart.support_dim)

    def normalized_slots(self, history_length: int) -> tuple[int, ...]:
        if history_length <= 0:
            raise ValueError("history must be non-empty")
        normalized: list[int] = []
        for raw in self.affected_slots:
            index = int(raw)
            if index < 0:
                index += history_length
            if index < 0 or index >= history_length:
                raise ValueError(
                    f"history slot {raw} is outside length {history_length}"
                )
            if index not in normalized:
                normalized.append(index)
        if not normalized:
            raise ValueError("at least one temporal slot must be affected")
        return tuple(sorted(normalized))

    def intervene(
        self,
        observation: Sequence[Any],
        *,
        direction: int,
        coefficients: np.ndarray,
        magnitude: float,
    ) -> list[Any]:
        history = list(observation)
        slots = self.normalized_slots(len(history))
        out = list(history)
        for index in slots:
            out[index] = self.slot_chart.intervene_slot(
                history[index],
                direction=direction,
                coefficients=coefficients,
                magnitude=magnitude,
            )
        return out


def event_suffix_slots(history_length: int, event_slot: int) -> tuple[int, ...]:
    """Return policy-history slots causally downstream of an event onset."""
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    index = int(event_slot)
    if index < 0:
        index += history_length
    if index < 0 or index >= history_length:
        raise ValueError("event_slot is outside the policy history")
    return tuple(range(index, history_length))


@dataclass(frozen=True)
class TemporalCASJProbeDecomposition:
    """Two-scale probe batches for each independently perturbed history slot."""

    slot_indices: tuple[int, ...]
    probes: tuple[CASJProbeBatch, ...]

    @property
    def total_query_count(self) -> int:
        return sum(probe.query_count for probe in self.probes)


def collect_temporal_slot_probes(
    *,
    policy: FrozenChunkPolicy,
    action_chart: ActionChart,
    slot_chart: SlotSupportInterventionChart,
    observation: Sequence[Any],
    randomness: Any,
    codes: np.ndarray,
    fine_epsilon: float | np.ndarray,
    coarse_epsilon: float | np.ndarray,
    slot_indices: Sequence[int] | None = None,
    max_pairing_error: float = 1e-6,
) -> TemporalCASJProbeDecomposition:
    """Measure support response one temporal observation slot at a time.

    This diagnostic intentionally spends separate paired queries per slot.  It
    identifies whether a policy's support dependence is concentrated in the
    current observation or stored in temporal context.  It is not claimed as a
    query-efficient runtime path.
    """
    history = list(observation)
    if not history:
        raise ValueError("observation history must be non-empty")
    raw_slots = tuple(range(len(history))) if slot_indices is None else tuple(slot_indices)
    normalizer = TemporalSupportInterventionChart(slot_chart, raw_slots)
    slots = normalizer.normalized_slots(len(history))

    batches = []
    for slot in slots:
        chart = TemporalSupportInterventionChart(slot_chart, (slot,))
        batches.append(
            collect_two_scale_probes(
                policy=policy,
                action_chart=action_chart,
                support_chart=chart,
                observation=history,
                randomness=randomness,
                codes=codes,
                fine_epsilon=fine_epsilon,
                coarse_epsilon=coarse_epsilon,
                max_pairing_error=max_pairing_error,
            )
        )
    return TemporalCASJProbeDecomposition(
        slot_indices=slots,
        probes=tuple(batches),
    )
