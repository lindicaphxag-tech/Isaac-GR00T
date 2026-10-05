"""Native MuJoCo closed-loop validation for SemRepair joint-order repair.

This integration uses a real MuJoCo dynamics model without rendering or policy
training. The same PD policy is exposed through a hidden policy joint ordering
that disagrees with the actuator ordering. SemRepair compiles the ordering
contract and executes the explicit permutation adapter before control.

The case is a controlled semantic-corruption experiment, not an upstream MuJoCo
bug or external adoption claim.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt

import mujoco

from research.semantic_invariants.embodied_runtime_adapters import (
    RuntimeAdapter,
    execute_adapter_plan,
)
from research.semantic_invariants.embodied_semantic_types import (
    SemanticAdapter,
    SemanticTensorType,
    synthesize_unique_adapter_plan,
)


MODEL_XML = """
<mujoco model="semrepair_joint_order">
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


def _reorder_transform(source_order, target_order):
    source_position = {label: index for index, label in enumerate(source_order)}

    def transform(values):
        return tuple(float(values[source_position[label]]) for label in target_order)

    return transform


def run_case(*, repaired: bool, steps: int = 800) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    target = (0.8, -0.4)
    policy_order = ("joint_y", "joint_x")
    controller_order = ("joint_x", "joint_y")

    source_type = SemanticTensorType(
        role="action",
        entity="joint_torque",
        representation="vector",
        unit="N*m",
        ordering="policy-order",
        provenance="requested",
        embodiment="mujoco-two-slide",
    )
    target_type = source_type.updated(ordering="controller-order")
    semantic_adapter = SemanticAdapter(
        "policy-to-controller-order",
        requires={"ordering": "policy-order", "embodiment": "mujoco-two-slide"},
        produces={"ordering": "controller-order"},
        effects=("reorder",),
    )
    plan = synthesize_unique_adapter_plan(
        source_type,
        target_type,
        (semantic_adapter,),
    )
    runtime_adapter = RuntimeAdapter(
        name="policy-to-controller-order",
        transform=_reorder_transform(policy_order, controller_order),
        certification="explicit-bijection:joint_y,joint_x->joint_x,joint_y",
    )

    kp = 20.0
    kd = 6.0
    path_length = 0.0
    previous = tuple(float(value) for value in data.qpos)

    for _ in range(steps):
        qpos = tuple(float(value) for value in data.qpos)
        qvel = tuple(float(value) for value in data.qvel)
        canonical = tuple(
            kp * (goal - position) - kd * velocity
            for goal, position, velocity in zip(target, qpos, qvel, strict=True)
        )
        policy_action = tuple(
            canonical[controller_order.index(label)] for label in policy_order
        )

        if repaired:
            execution = execute_adapter_plan(
                policy_action,
                plan,
                {runtime_adapter.name: runtime_adapter},
            )
            controller_action = execution.value
        else:
            controller_action = policy_action

        data.ctrl[:] = controller_action
        mujoco.mj_step(model, data)

        current = tuple(float(value) for value in data.qpos)
        path_length += _norm(
            tuple(right - left for left, right in zip(previous, current, strict=True))
        )
        previous = current

    final = tuple(float(value) for value in data.qpos)
    error = _norm(tuple(goal - value for goal, value in zip(target, final, strict=True)))
    return {
        "mode": "semrepair" if repaired else "unrepaired",
        "final_qpos": final,
        "target": target,
        "final_error": error,
        "path_length": path_length,
        "adapter": (
            ["policy-to-controller-order"] if repaired else []
        ),
        "mujoco_version": mujoco.__version__,
    }


def build_report() -> dict[str, object]:
    broken = run_case(repaired=False)
    repaired = run_case(repaired=True)
    return {
        "schema_version": 1,
        "integration": "native-mujoco-joint-order-closed-loop",
        "broken": broken,
        "repaired": repaired,
        "improvement": float(broken["final_error"]) - float(repaired["final_error"]),
        "claim_boundary": (
            "real MuJoCo closed-loop semantic-corruption assay; not an upstream "
            "MuJoCo defect and not external adoption"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native MuJoCo SemRepair assay")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"error: {report['broken']['final_error']:.6g} -> "
            f"{report['repaired']['final_error']:.6g}"
        )

    # CI gate: repaired controller must converge while the corrupted interface
    # remains materially wrong on the same physical model.
    if float(report["repaired"]["final_error"]) >= 0.02:
        raise SystemExit("SemRepair-controlled MuJoCo rollout did not converge")
    if float(report["broken"]["final_error"]) <= 0.25:
        raise SystemExit("joint-order corruption did not create a material closed-loop error")


if __name__ == "__main__":
    main()