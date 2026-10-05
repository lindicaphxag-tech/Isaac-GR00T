from dataclasses import dataclass

import numpy as np

from casj.temporal_scope import (
    TemporalSupportInterventionChart,
    collect_temporal_slot_probes,
    event_suffix_slots,
)


@dataclass
class ArraySlotChart:
    num_supports: int = 1
    support_dim: int = 1

    def intervene_slot(
        self, observation_slot, *, direction, coefficients, magnitude
    ):
        assert direction == 0
        out = np.asarray(observation_slot, dtype=float).copy()
        out[0] += float(coefficients[0]) * magnitude
        return out


class IdentityChart:
    def decode(self, raw_chunk, *, observation):
        return np.asarray(raw_chunk, dtype=float)

    def local_response(self, changed, baseline):
        return np.asarray(changed) - np.asarray(baseline)

    def pairing_error(self, first, second):
        return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))


class HistoryWeightedPolicy:
    def query(self, observation, *, randomness):
        del randomness
        x = np.asarray(observation, dtype=float)[:, 0]
        # One one-dimensional action position.  Current-slot support dependence
        # is four times stronger than the previous-slot dependence.
        return np.asarray([[x[0] + 4.0 * x[1]]], dtype=float)


def test_event_suffix_slots_never_rewrites_pre_event_history():
    assert event_suffix_slots(4, 2) == (2, 3)
    assert event_suffix_slots(4, -1) == (3,)


def test_temporal_chart_changes_only_declared_slots():
    history = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
    chart = TemporalSupportInterventionChart(ArraySlotChart(), (1, -1))
    out = chart.intervene(
        history,
        direction=0,
        coefficients=np.ones(1),
        magnitude=0.5,
    )
    np.testing.assert_allclose(out[0], [1.0])
    np.testing.assert_allclose(out[1], [2.5])
    np.testing.assert_allclose(out[2], [3.5])
    # Input history is immutable from the chart's perspective.
    np.testing.assert_allclose(history[1], [2.0])


def test_temporal_probe_decomposes_memory_and_current_support_dependence():
    history = [np.array([2.0]), np.array([3.0])]
    result = collect_temporal_slot_probes(
        policy=HistoryWeightedPolicy(),
        action_chart=IdentityChart(),
        slot_chart=ArraySlotChart(),
        observation=history,
        randomness=0,
        codes=np.ones((1, 1)),
        fine_epsilon=1e-4,
        coarse_epsilon=2e-4,
    )
    assert result.slot_indices == (0, 1)
    previous = result.probes[0].fine_first[0, 0, 0, 0]
    current = result.probes[1].fine_first[0, 0, 0, 0]
    np.testing.assert_allclose(previous, 1.0, atol=1e-10)
    np.testing.assert_allclose(current, 4.0, atol=1e-10)
    assert result.total_query_count == 12
