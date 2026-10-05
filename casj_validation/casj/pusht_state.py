"""Deterministic PushT state capture and reconstruction for paired rollouts.

PushT's visible rigid-body state is not the whole episode state: Gymnasium also
tracks elapsed steps, and Pymunk retains contact state in its Space. Rewriting
body fields in an existing Space therefore does not create an independent
paired rollout. This adapter resets the wrapper, rebuilds the simulator Space,
restores every dynamic body field, and verifies the public state before use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PushTSnapshot:
    agent_position: np.ndarray
    agent_velocity: np.ndarray
    block_position: np.ndarray
    block_angle: float
    block_velocity: np.ndarray
    block_angular_velocity: float

    def shifted_block(
        self,
        delta_xy: np.ndarray,
        delta_angle: float = 0.0,
    ) -> "PushTSnapshot":
        return PushTSnapshot(
            agent_position=self.agent_position.copy(),
            agent_velocity=self.agent_velocity.copy(),
            block_position=self.block_position + np.asarray(delta_xy, dtype=float),
            block_angle=float(self.block_angle + delta_angle),
            block_velocity=self.block_velocity.copy(),
            block_angular_velocity=float(self.block_angular_velocity),
        )


def _vec2(body_value: Any) -> np.ndarray:
    return np.asarray([body_value[0], body_value[1]], dtype=np.float64)


def capture_snapshot(env: Any) -> PushTSnapshot:
    """Capture the public dynamic body state used by the PushT policy."""
    unwrapped = env.unwrapped
    return PushTSnapshot(
        agent_position=_vec2(unwrapped.agent.position),
        agent_velocity=_vec2(unwrapped.agent.velocity),
        block_position=_vec2(unwrapped.block.position),
        block_angle=float(unwrapped.block.angle),
        block_velocity=_vec2(unwrapped.block.velocity),
        block_angular_velocity=float(unwrapped.block.angular_velocity),
    )


def restore_snapshot(env: Any, snapshot: PushTSnapshot) -> None:
    """Rebuild a clean PushT episode at ``snapshot`` or fail closed.

    ``env.reset`` resets Gymnasium wrapper counters and RNG state. PushT 0.1.6's
    reset path advances Pymunk once while installing ``reset_to_state`` and
    cannot restore velocity, so this adapter rebuilds a fresh Space after reset
    and writes the full dynamic state before the next physics step.
    """
    env.reset(seed=0)
    unwrapped = env.unwrapped
    rebuild_space = getattr(unwrapped, "_setup", None)
    if not callable(rebuild_space):
        raise RuntimeError(
            "unsupported gym-pusht state API: expected PushTEnv._setup to rebuild Space"
        )

    rebuild_space()
    unwrapped.agent.position = snapshot.agent_position.tolist()
    unwrapped.agent.velocity = snapshot.agent_velocity.tolist()
    # Pymunk rotates bodies about their center of gravity. Set angle first,
    # then position, or restoring a nonzero angle shifts the block pose.
    unwrapped.block.angle = float(snapshot.block_angle)
    unwrapped.block.position = snapshot.block_position.tolist()
    unwrapped.block.velocity = snapshot.block_velocity.tolist()
    unwrapped.block.angular_velocity = float(snapshot.block_angular_velocity)
    unwrapped.space.reindex_shapes_for_body(unwrapped.agent)
    unwrapped.space.reindex_shapes_for_body(unwrapped.block)
    unwrapped.n_contact_points = 0
    if hasattr(unwrapped, "_last_action"):
        unwrapped._last_action = None

    restored = capture_snapshot(env)
    for field in (
        "agent_position",
        "agent_velocity",
        "block_position",
        "block_velocity",
    ):
        if not np.array_equal(getattr(restored, field), getattr(snapshot, field)):
            raise RuntimeError(f"PushT state restoration mismatch in {field}")
    if restored.block_angle != snapshot.block_angle:
        raise RuntimeError("PushT state restoration mismatch in block_angle")
    if restored.block_angular_velocity != snapshot.block_angular_velocity:
        raise RuntimeError(
            "PushT state restoration mismatch in block_angular_velocity"
        )

    elapsed_steps = getattr(env, "_elapsed_steps", 0)
    if elapsed_steps not in (None, 0):
        raise RuntimeError(
            f"Gymnasium episode counter did not reset: elapsed_steps={elapsed_steps}"
        )