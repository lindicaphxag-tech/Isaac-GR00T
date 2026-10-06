"""Real ManiSkill claim-relative evidence case for semantic experiment planning.

The numbers are frozen from public validation surfaces:

Task-level replay (exact-head serial replay, public run 37394500249):
    old/old            9/10
    converter-only     1/10
    controller-only    0/10
    composed           8/10

Direct semantic fidelity (public run 37401814045, composed-generated corpus,
call-weighted mean SO(3) converter-to-controller target error):
    old/old            0.023799697040482255 deg
    converter-only     2.9362813791952065 deg
    controller-only    2.933590862288818 deg
    composed           0.016667921590028514 deg

The task metric is intentionally *not* authorized to prove rotation semantics.
Its purpose here is to demonstrate a claim-relative oracle boundary rather than
to maximize classification accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

from planner import (
    JointSemanticExperiment,
    SafeSemanticExperimentPlan,
    SemanticHypothesis,
    render_plan,
    synthesize_safe_semantic_experiment_plan,
)


HYPOTHESES = (
    SemanticHypothesis("old_old"),
    SemanticHypothesis("converter_only"),
    SemanticHypothesis("controller_only"),
    SemanticHypothesis("composed"),
)


EXPERIMENTS = (
    JointSemanticExperiment(
        name="official_demo_task_replay",
        probe="first_10_official_PegInsertionSide_demos",
        tap="episode_success",
        outcomes={
            "old_old": "9/10",
            "converter_only": "1/10",
            "controller_only": "0/10",
            "composed": "8/10",
        },
        probe_cost=0.25,
        tap_cost=0.0,
        risk=0.0,
        evidence_for=("task_performance",),
    ),
    JointSemanticExperiment(
        name="paired_so3_controller_target_fidelity",
        probe="paired_official_demo_controller_requests",
        tap="controller_target_orientation",
        outcomes={
            "old_old": "0.023799697040482255deg",
            "converter_only": "2.9362813791952065deg",
            "controller_only": "2.933590862288818deg",
            "composed": "0.016667921590028514deg",
        },
        probe_cost=1.0,
        tap_cost=0.25,
        risk=0.0,
        evidence_for=("rotation_semantics",),
    ),
)


@dataclass(frozen=True)
class ManiSkillClaimAuthorityResult:
    rotation_plan: SafeSemanticExperimentPlan
    task_plan: SafeSemanticExperimentPlan


def build_maniskill_claim_authority_case() -> ManiSkillClaimAuthorityResult:
    rotation_plan = synthesize_safe_semantic_experiment_plan(
        HYPOTHESES,
        EXPERIMENTS,
        required_evidence="rotation_semantics",
    )
    task_plan = synthesize_safe_semantic_experiment_plan(
        HYPOTHESES,
        EXPERIMENTS,
        required_evidence="task_performance",
    )
    return ManiSkillClaimAuthorityResult(
        rotation_plan=rotation_plan,
        task_plan=task_plan,
    )


def main() -> int:
    result = build_maniskill_claim_authority_case()
    print("rotation_semantics")
    print(f"status={result.rotation_plan.status}")
    print(render_plan(result.rotation_plan.root))
    print()
    print("task_performance")
    print(f"status={result.task_plan.status}")
    print(render_plan(result.task_plan.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
