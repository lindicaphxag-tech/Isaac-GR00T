"""Deterministic closed-loop smoke for semantic interface corruption and repair.

This is not a robotics benchmark. It is a causal executable witness showing
that a semantic type mismatch can change closed-loop behavior while shape and
dtype remain valid, and that executing the compiler-selected adapter restores
the intended behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

from .embodied_runtime_adapters import RuntimeAdapter, execute_adapter_plan
from .embodied_semantic_types import (
    SemanticAdapter,
    SemanticTensorType,
    synthesize_unique_adapter_plan,
)


Vector = tuple[float, ...]


def _norm(value: Sequence[float]) -> float:
    return sqrt(sum(component * component for component in value))


@dataclass
class LinearJointReachEnv:
    state: Vector
    target: Vector

    def step(self, controller_action: Vector) -> Vector:
        if len(controller_action) != len(self.state):
            raise ValueError("action/state dimension mismatch")
        self.state = tuple(
            position + delta
            for position, delta in zip(self.state, controller_action)
        )
        return self.state

    @property
    def error(self) -> float:
        return _norm(
            tuple(
                target - position
                for position, target in zip(self.state, self.target)
            )
        )


def policy_action_in_order(
    state: Vector,
    target: Vector,
    policy_order: Sequence[int],
) -> Vector:
    canonical = tuple(target[i] - state[i] for i in range(len(state)))
    return tuple(canonical[index] for index in policy_order)


def permutation_transform(
    source_order: Sequence[int],
    target_order: Sequence[int],
):
    """Return a value transform between two index-label orderings."""

    source_position = {
        label: position for position, label in enumerate(source_order)
    }
    if set(source_position) != set(target_order):
        raise ValueError("source and target orders must contain the same labels")

    def transform(value: Vector) -> Vector:
        if len(value) != len(source_order):
            raise ValueError("value/order dimension mismatch")
        return tuple(value[source_position[label]] for label in target_order)

    return transform


@dataclass(frozen=True)
class ClosedLoopOutcome:
    final_error: float
    trajectory: tuple[Vector, ...]
    adapter_applied: tuple[str, ...]


def run_joint_order_case(
    *,
    repaired: bool,
    steps: int = 3,
) -> ClosedLoopOutcome:
    policy_order = (1, 0)
    controller_order = (0, 1)
    env = LinearJointReachEnv(
        state=(0.0, 0.0),
        target=(1.0, -0.5),
    )
    trajectory: list[Vector] = [env.state]
    applied: tuple[str, ...] = ()

    source_type = SemanticTensorType(
        role="action",
        entity="joint_delta",
        representation="vector",
        unit="rad",
        ordering="policy-order",
        provenance="requested",
        embodiment="two-joint-proxy",
    )
    target_type = source_type.updated(ordering="controller-order")
    semantic_adapter = SemanticAdapter(
        "policy-to-controller-order",
        requires={
            "ordering": "policy-order",
            "embodiment": "two-joint-proxy",
        },
        produces={"ordering": "controller-order"},
        effects=("reorder",),
    )
    plan = synthesize_unique_adapter_plan(
        source_type,
        target_type,
        [semantic_adapter],
    )
    runtime = RuntimeAdapter[Vector](
        name="policy-to-controller-order",
        transform=permutation_transform(policy_order, controller_order),
        certification="proxy/permutation-bijection",
    )

    for _ in range(steps):
        policy_action = policy_action_in_order(
            env.state,
            env.target,
            policy_order,
        )
        if repaired:
            execution = execute_adapter_plan(
                policy_action,
                plan,
                {runtime.name: runtime},
            )
            controller_action = execution.value
            applied = execution.applied
        else:
            controller_action = policy_action
        trajectory.append(env.step(controller_action))

    return ClosedLoopOutcome(
        final_error=env.error,
        trajectory=tuple(trajectory),
        adapter_applied=applied,
    )