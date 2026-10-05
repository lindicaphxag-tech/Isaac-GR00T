#!/usr/bin/env python3
"""Native MuJoCo witness for completely masked semantic boundary faults.

Two independent joint-order faults can cancel before physical dispatch:
producer swap -> dispatch swap. The resulting control vector is identical to the
fully correct pipeline for every command, so no endpoint trajectory observation
can distinguish the two chains. Repairing only one boundary exposes the
remaining swap and changes the physical trajectory.

This is a controlled semantic-corruption assay, not a claim of an upstream
MuJoCo defect.
"""

from __future__ import annotations

import json
from math import cos, sin

import mujoco
import numpy as np

from research.semantic_invariants.embodied_repair_interactions import (
    RepairOutcome,
    analyze_repair_lattice,
)


XML = r"""
<mujoco model="semrepair_masking">
  <option timestep="0.01" gravity="0 0 0" integrator="RK4"/>
  <worldbody>
    <body name="axis0" pos="0 0 0">
      <joint name="j0" type="hinge" axis="1 0 0" damping="0.3"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.03" mass="1.0"/>
    </body>
    <body name="axis1" pos="0.4 0 0">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.04" mass="1.7"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="a0" joint="j0" gear="1.0" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="a1" joint="j1" gear="1.4" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def _swap(value: np.ndarray) -> np.ndarray:
    return value[[1, 0]]


def _command(step: int) -> np.ndarray:
    return np.array(
        [
            0.52 + 0.19 * sin(step * 0.047),
            -0.31 + 0.17 * cos(step * 0.071),
        ],
        dtype=np.float64,
    )


def rollout(*, producer_swap: bool, dispatch_swap: bool, steps: int = 240) -> np.ndarray:
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    trajectory = []

    for step in range(steps):
        value = _command(step)
        if producer_swap:
            value = _swap(value)
        if dispatch_swap:
            value = _swap(value)
        data.ctrl[:] = value
        mujoco.mj_step(model, data)
        trajectory.append(data.qpos.copy())

    return np.asarray(trajectory)


def main() -> int:
    # The fully correct trajectory is the semantic oracle.
    correct = rollout(producer_swap=False, dispatch_swap=False)

    # Faulty baseline: two wrong order conventions cancel exactly before ctrl.
    both_faulty = rollout(producer_swap=True, dispatch_swap=True)

    # Starting from the faulty baseline, fix exactly one boundary.
    producer_repaired_only = rollout(producer_swap=False, dispatch_swap=True)
    dispatch_repaired_only = rollout(producer_swap=True, dispatch_swap=False)

    # Both repairs is the same physical configuration as correct.
    both_repaired = rollout(producer_swap=False, dispatch_swap=False)

    def error(candidate: np.ndarray) -> float:
        return float(np.max(np.abs(candidate - correct)))

    errors = {
        "both_faulty": error(both_faulty),
        "producer_repaired_only": error(producer_repaired_only),
        "dispatch_repaired_only": error(dispatch_repaired_only),
        "both_repaired": error(both_repaired),
    }

    # Exact cancellation happens before MuJoCo receives ctrl, so the two
    # trajectories must be byte-for-byte equal under deterministic execution.
    assert np.array_equal(both_faulty, correct)
    assert np.array_equal(both_repaired, correct)
    assert errors["producer_repaired_only"] > 1e-3
    assert errors["dispatch_repaired_only"] > 1e-3

    interaction = analyze_repair_lattice(
        subject="native MuJoCo joint-order producer -> dispatch chain",
        metric="trajectory_linf_vs_correct",
        objective="minimize",
        repairs=("producer-order-repair", "dispatch-order-repair"),
        outcomes=(
            RepairOutcome(
                frozenset(),
                errors["both_faulty"],
                "native-mujoco/both-faulty-cancelled",
            ),
            RepairOutcome(
                frozenset(("producer-order-repair",)),
                errors["producer_repaired_only"],
                "native-mujoco/producer-repaired-only",
            ),
            RepairOutcome(
                frozenset(("dispatch-order-repair",)),
                errors["dispatch_repaired_only"],
                "native-mujoco/dispatch-repaired-only",
            ),
            RepairOutcome(
                frozenset(("producer-order-repair", "dispatch-order-repair")),
                errors["both_repaired"],
                "native-mujoco/both-repaired",
            ),
        ),
        tolerance=1e-12,
    )

    assert interaction.has_repair_paradox
    assert len(interaction.compensating_bundles) == 1
    bundle = interaction.compensating_bundles[0]
    assert bundle.masking_mode == "complete"
    assert bundle.behaviorally_masked

    result = {
        "schema_version": 1,
        "mujoco_version": mujoco.__version__,
        "steps": int(correct.shape[0]),
        "errors": errors,
        "baseline_equals_correct_exactly": bool(np.array_equal(both_faulty, correct)),
        "both_repaired_equals_correct_exactly": bool(
            np.array_equal(both_repaired, correct)
        ),
        "masking_mode": bundle.masking_mode,
        "repair_interaction_digest": interaction.digest,
        "claim_boundary": (
            "Controlled native-MuJoCo semantic-corruption assay. It proves a "
            "real physics execution can hide exactly canceling boundary faults; "
            "it is not an upstream MuJoCo bug or external adoption."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
