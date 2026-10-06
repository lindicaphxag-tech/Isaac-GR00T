"""Native MuJoCo witness for nuisance-confounded semantic identification.

Semantic factor:
    upstream action scale in {1x, 2x}

Persistent nuisance factor:
    actuator transmission gear in {1x, 2x}

At the external joint trajectory, only the product scale*gear is visible. Hence
(scale=1, gear=2) and (scale=2, gear=1) are structurally confounded under every
linear probe in this experiment language.  Reading data.ctrl before the actuator
transmission breaks the confounding class and restores semantic identifiability.
"""

from __future__ import annotations

import json

import mujoco

from nuisance_semantic import (
    RobustSemanticExperiment,
    SemanticWorld,
    synthesize_nuisance_robust_plan,
)


WORLDS = (
    SemanticWorld("scale_1x", "gear_1x"),
    SemanticWorld("scale_1x", "gear_2x"),
    SemanticWorld("scale_2x", "gear_1x"),
    SemanticWorld("scale_2x", "gear_2x"),
)


def make_model(gear: float) -> mujoco.MjModel:
    xml = f"""
    <mujoco model="semantic_nuisance">
      <option timestep="0.002" gravity="0 0 0"/>
      <worldbody>
        <body name="slider">
          <joint name="j" type="slide" axis="1 0 0" damping="0"/>
          <geom type="sphere" size="0.02" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="m" joint="j" gear="{gear}"/>
      </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def run_world(scale: float, gear: float, probe: float, steps: int = 40) -> tuple[float, float]:
    model = make_model(gear)
    data = mujoco.MjData(model)
    data.ctrl[0] = scale * probe
    ctrl_before_transmission = float(data.ctrl[0])
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return float(data.qpos[0]), ctrl_before_transmission


def token(value: float) -> str:
    return f"{value:.12f}"


def world_key(scale: float, gear: float) -> str:
    return f"scale_{int(scale)}x::gear_{int(gear)}x"


def external_experiment(name: str, probe: float) -> RobustSemanticExperiment:
    outcomes = {}
    for scale in (1.0, 2.0):
        for gear in (1.0, 2.0):
            qpos, _ = run_world(scale, gear, probe)
            outcomes[world_key(scale, gear)] = token(qpos)
    return RobustSemanticExperiment(
        name=name,
        probe=f"raw_action={probe}",
        tap="joint_qpos_after_40_mj_step",
        outcomes=outcomes,
        probe_cost=0.1,
        evidence_for=("physical_response",),
    )


def ctrl_tap_experiment(probe: float) -> RobustSemanticExperiment:
    outcomes = {}
    for scale in (1.0, 2.0):
        for gear in (1.0, 2.0):
            _, ctrl = run_world(scale, gear, probe, steps=1)
            outcomes[world_key(scale, gear)] = token(ctrl)
    return RobustSemanticExperiment(
        name="pre_transmission_ctrl_tap",
        probe=f"raw_action={probe}",
        tap="mujoco_data.ctrl_before_actuator_transmission",
        outcomes=outcomes,
        probe_cost=0.1,
        tap_cost=0.25,
        evidence_for=("action_semantics",),
    )


def main() -> None:
    external = [
        external_experiment("external_positive_probe", 0.10),
        external_experiment("external_negative_probe", -0.07),
    ]

    # Native-dynamics equality: the two worlds have identical actuator product.
    for exp in external:
        assert (
            exp.outcomes["scale_1x::gear_2x"]
            == exp.outcomes["scale_2x::gear_1x"]
        )

    external_plan = synthesize_nuisance_robust_plan(WORLDS, external)
    assert external_plan.status == "nuisance_confounded"

    tap = ctrl_tap_experiment(0.10)
    semantic_plan = synthesize_nuisance_robust_plan(
        WORLDS,
        [*external, tap],
        required_evidence="action_semantics",
    )
    assert semantic_plan.status == "identified"
    assert semantic_plan.authorized_experiments == ("pre_transmission_ctrl_tap",)

    report = {
        "mujoco_version": mujoco.__version__,
        "nuisance_model": "persistent actuator transmission gear",
        "semantic_factor": "upstream action scale",
        "confounded_worlds": [
            "scale_1x::gear_2x",
            "scale_2x::gear_1x",
        ],
        "external_positive_qpos": external[0].outcomes,
        "external_negative_qpos": external[1].outcomes,
        "external_only_status": external_plan.status,
        "internal_tap_status": semantic_plan.status,
        "selected_semantic_experiment": semantic_plan.root.experiment,
        "claim_boundary": (
            "self-authored native MuJoCo mechanism evidence; not external adoption "
            "or prospective I2"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
