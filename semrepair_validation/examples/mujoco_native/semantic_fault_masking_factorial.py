"""Native MuJoCo assay for compensating semantic faults.

This experiment starts from the public SemRepair 0.3.0rc1 snapshot and injects
two independent semantic ordering faults around a real MuJoCo controller path:

A. producer packing fault: (joint_x, joint_y) -> (joint_y, joint_x)
B. dispatch mapping fault: model slot 0/1 -> actuator_y/actuator_x

With both faults present the two permutations cancel, so end-to-end task error
is nearly indistinguishable from the fully correct pipeline.  Repairing either
boundary alone exposes the remaining fault and makes the task fail.

This is a controlled physical-simulator fault-interaction assay, not an
upstream MuJoCo defect and not external adoption.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt

import mujoco

from research.semantic_invariants.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
    analyze_transport_cancellation,
)


MODEL_XML = """
<mujoco model="semantic_fault_masking">
  <option timestep="0.005" gravity="0 0 0"/>
  <worldbody>
    <body name="slider">
      <joint name="joint_x" type="slide" axis="1 0 0"
             damping="2" limited="true" range="-2 2"/>
      <joint name="joint_y" type="slide" axis="0 1 0"
             damping="2" limited="true" range="-2 2"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="act_x" joint="joint_x" ctrllimited="true" ctrlrange="-50 50"/>
    <motor name="act_y" joint="joint_y" ctrllimited="true" ctrlrange="-50 50"/>
  </actuator>
</mujoco>
"""


def _norm(values) -> float:
    return sqrt(sum(float(value) ** 2 for value in values))


def _swap2(values):
    return (float(values[1]), float(values[0]))


def run_case(
    *,
    producer_swap_fault: bool,
    dispatch_swap_fault: bool,
    steps: int = 800,
) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    target = (0.8, -0.4)
    kp = 20.0
    kd = 6.0
    path_length = 0.0
    previous = tuple(float(value) for value in data.qpos)
    trajectory = [previous]

    for _ in range(steps):
        qpos = tuple(float(value) for value in data.qpos)
        qvel = tuple(float(value) for value in data.qvel)

        canonical = tuple(
            kp * (goal - position) - kd * velocity
            for goal, position, velocity in zip(target, qpos, qvel, strict=True)
        )

        # Fault A: producer emits a vector in the wrong joint order while
        # preserving shape, dtype, range, and unit.
        model_action = _swap2(canonical) if producer_swap_fault else canonical

        # Fault B: dispatch interprets model slots using the opposite actuator
        # order.  This is a second semantic boundary, not a numerical error.
        controller_action = (
            _swap2(model_action) if dispatch_swap_fault else model_action
        )

        data.ctrl[:] = controller_action
        mujoco.mj_step(model, data)

        current = tuple(float(value) for value in data.qpos)
        path_length += _norm(
            tuple(
                right - left
                for left, right in zip(previous, current, strict=True)
            )
        )
        previous = current
        trajectory.append(current)

    final = tuple(float(value) for value in data.qpos)
    final_error = _norm(
        tuple(goal - value for goal, value in zip(target, final, strict=True))
    )
    return {
        "producer_swap_fault": producer_swap_fault,
        "dispatch_swap_fault": dispatch_swap_fault,
        "final_qpos": final,
        "target": target,
        "final_error": final_error,
        "path_length": path_length,
        "mujoco_version": mujoco.__version__,
        "trajectory_final": trajectory[-1],
    }


def build_report() -> dict[str, object]:
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    static = analyze_transport_cancellation(
        (
            SemanticTransportFactor(
                "producer-order",
                swap,
                "controlled-fault/producer-order",
            ),
            SemanticTransportFactor(
                "dispatch-order",
                swap,
                "controlled-fault/dispatch-order",
            ),
        )
    )

    # Baseline under investigation: two independent semantic faults coexist.
    both_faults = run_case(
        producer_swap_fault=True,
        dispatch_swap_fault=True,
    )

    # Repair A only: producer is corrected, dispatch remains wrong.
    repair_producer_only = run_case(
        producer_swap_fault=False,
        dispatch_swap_fault=True,
    )

    # Repair B only: dispatch is corrected, producer remains wrong.
    repair_dispatch_only = run_case(
        producer_swap_fault=True,
        dispatch_swap_fault=False,
    )

    # Both semantic boundaries repaired.
    repair_both = run_case(
        producer_swap_fault=False,
        dispatch_swap_fault=False,
    )

    baseline_error = float(both_faults["final_error"])
    producer_only_error = float(repair_producer_only["final_error"])
    dispatch_only_error = float(repair_dispatch_only["final_error"])
    repaired_error = float(repair_both["final_error"])

    return {
        "schema_version": 1,
        "integration": "native-mujoco-compensating-semantic-faults",
        "faults": {
            "A": "producer joint-order packing swap",
            "B": "actuator dispatch-order swap",
        },
        "static_prediction": {
            "structurally_masked": static.structurally_masked,
            "net_is_identity": static.net.is_identity(),
            "repair_one_unmasks": static.repair_one_unmasks,
            "evidence_digest": static.digest,
        },
        "factorial_cells": {
            "fault_A+fault_B": both_faults,
            "repair_A_only": repair_producer_only,
            "repair_B_only": repair_dispatch_only,
            "repair_A+repair_B": repair_both,
        },
        "repair_lattice": {
            "baseline_both_faults": baseline_error,
            "producer_repair_only": producer_only_error,
            "dispatch_repair_only": dispatch_only_error,
            "both_repairs": repaired_error,
            "single_repair_paradox": (
                producer_only_error > baseline_error + 0.25
                and dispatch_only_error > baseline_error + 0.25
            ),
            "behavioral_masking": (
                baseline_error < 0.02
                and repaired_error < 0.02
                and producer_only_error > 0.25
                and dispatch_only_error > 0.25
            ),
        },
        "claim_boundary": (
            "controlled native-MuJoCo semantic-fault interaction assay; "
            "demonstrates behavioral masking under paired representation/order "
            "faults; not an upstream MuJoCo defect, not prospective I2, and "
            "not external adoption"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        lattice = report["repair_lattice"]
        print(
            "final error "
            f"both-faults={lattice['baseline_both_faults']:.6g}, "
            f"repair-A-only={lattice['producer_repair_only']:.6g}, "
            f"repair-B-only={lattice['dispatch_repair_only']:.6g}, "
            f"both-repairs={lattice['both_repairs']:.6g}"
        )

    static = report["static_prediction"]
    if not static["structurally_masked"] or not static["net_is_identity"]:
        raise SystemExit("static transport algebra did not predict cancellation")
    if tuple(static["repair_one_unmasks"]) != (
        "producer-order",
        "dispatch-order",
    ):
        raise SystemExit("static pre-screen did not identify both unmasking repairs")

    lattice = report["repair_lattice"]
    if not lattice["single_repair_paradox"]:
        raise SystemExit("single-repair paradox was not reproduced")
    if not lattice["behavioral_masking"]:
        raise SystemExit("paired semantic faults did not remain behaviorally masked")


if __name__ == "__main__":
    main()
