"""Closed-loop assay for non-forgeable sensor freshness semantics.

Unlike representation errors, stale observations cannot be repaired by a
numerical cast.  The only valid intervention is to obtain a new sample from the
sensor owner boundary and refine the semantic type with that evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .embodied_refinement import SemanticEvidence, refine_sensor_freshness
from .embodied_semantic_types import SemanticTensorType


@dataclass
class IntegratorPlant:
    state: float
    target: float

    def step(self, action: float) -> float:
        self.state += float(action)
        return self.state


@dataclass(frozen=True)
class FreshnessRollout:
    mode: str
    states: tuple[float, ...]
    observations: tuple[float, ...]
    actions: tuple[float, ...]
    final_error: float
    path_length: float
    resamples: int


def run_freshness_feedback(
    *,
    guarded: bool,
    horizon: int = 6,
    gain: float = 1.0,
) -> FreshnessRollout:
    """Run feedback with a one-step-lag cache.

    The unguarded path blindly consumes the cached previous observation.  The
    guarded path treats that cache as semantically stale and obtains a fresh
    owner sample before control.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    plant = IntegratorPlant(state=0.0, target=1.0)
    observation_type = SemanticTensorType(
        role="observation",
        entity="joint_position",
        unit="rad",
        clock="control_step",
        scope="control_step",
        freshness="stale",
        provenance="observed",
    )

    cached = plant.state
    states = [plant.state]
    observations: list[float] = []
    actions: list[float] = []
    resamples = 0

    for step in range(horizon):
        if guarded:
            fresh_value = plant.state
            evidence = SemanticEvidence(
                kind="sensor_sample",
                subject_id=f"joint-sensor:{step}",
                claims={"sample_time": float(step), "value": fresh_value},
                issuer="joint-sensor",
            )
            refinement = refine_sensor_freshness(
                observation_type,
                evidence,
                due_time=float(step),
            )
            if refinement.after.freshness != "fresh":
                raise AssertionError("sensor owner failed to issue fresh semantics")
            observed = fresh_value
            resamples += 1
        else:
            observed = cached

        action = gain * (plant.target - observed)
        previous_state = plant.state
        plant.step(action)

        # The cache update is intentionally delayed by one control step.
        cached = previous_state
        states.append(plant.state)
        observations.append(observed)
        actions.append(action)

    path_length = sum(
        abs(right - left)
        for left, right in zip(states, states[1:], strict=True)
    )
    return FreshnessRollout(
        mode="freshness-guarded" if guarded else "stale-unguarded",
        states=tuple(states),
        observations=tuple(observations),
        actions=tuple(actions),
        final_error=abs(plant.target - plant.state),
        path_length=path_length,
        resamples=resamples,
    )
