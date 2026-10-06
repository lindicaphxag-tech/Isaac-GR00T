"""Machine-readable behavioral evidence for SemRepair prototype assays."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from .embodied_closed_loop import run_joint_order_case
from .embodied_closed_loop_semantics import run_composed_rotation_assay
from .embodied_freshness_closed_loop import run_freshness_feedback


@dataclass(frozen=True)
class BehavioralComparison:
    family: str
    broken_metric: float
    repaired_metric: float
    improvement: float
    auxiliary: dict[str, float | int | str]


def build_behavioral_evidence() -> dict[str, object]:
    joint_broken = run_joint_order_case(repaired=False, steps=3)
    joint_repaired = run_joint_order_case(repaired=True, steps=3)

    rotation = run_composed_rotation_assay()
    rot_broken = rotation["none"]
    rot_repaired = rotation["compiler_mediated"]

    stale = run_freshness_feedback(guarded=False, horizon=6)
    fresh = run_freshness_feedback(guarded=True, horizon=6)

    comparisons = (
        BehavioralComparison(
            family="joint-order",
            broken_metric=joint_broken.final_error,
            repaired_metric=joint_repaired.final_error,
            improvement=joint_broken.final_error - joint_repaired.final_error,
            auxiliary={
                "broken_steps": len(joint_broken.trajectory) - 1,
                "repaired_steps": len(joint_repaired.trajectory) - 1,
            },
        ),
        BehavioralComparison(
            family="rotation-representation-plus-sign",
            broken_metric=rot_broken.final_error_degrees,
            repaired_metric=rot_repaired.final_error_degrees,
            improvement=rot_broken.final_error_degrees
            - rot_repaired.final_error_degrees,
            auxiliary={
                "broken_path_deg": rot_broken.path_length_degrees,
                "repaired_path_deg": rot_repaired.path_length_degrees,
                "repaired_steps_to_tolerance": (
                    rot_repaired.steps_to_tolerance
                    if rot_repaired.steps_to_tolerance is not None
                    else -1
                ),
            },
        ),
        BehavioralComparison(
            family="sensor-freshness",
            broken_metric=stale.final_error,
            repaired_metric=fresh.final_error,
            improvement=stale.final_error - fresh.final_error,
            auxiliary={
                "broken_path": stale.path_length,
                "repaired_path": fresh.path_length,
                "owner_resamples": fresh.resamples,
            },
        ),
    )

    return {
        "schema_version": 1,
        "claim_boundary": (
            "deterministic prototype assays; not a robotics benchmark and not "
            "evidence of real-stack closed-loop task success"
        ),
        "comparisons": [asdict(item) for item in comparisons],
        "all_improve": all(item.improvement > 0 for item in comparisons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit deterministic SemRepair behavioral evidence"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_behavioral_evidence()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for row in report["comparisons"]:
            print(
                f"{row['family']}: {row['broken_metric']:.6g} -> "
                f"{row['repaired_metric']:.6g}"
            )


if __name__ == "__main__":
    main()
