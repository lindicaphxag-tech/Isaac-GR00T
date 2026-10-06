"""Release gate for the proof-carrying embodied semantic runtime.

The gate exercises the actual trusted path instead of counting files/tests.
It is intentionally dependency-light so the standalone artifact can run it in
CI before any external simulator integration.
"""

from __future__ import annotations

import argparse
import json

from .embodied_compilation_certificate import issue_compilation_certificate
from .embodied_repair_synthesis import RepairExample, RepairPrimitive, RepairProgram, build_vector_repair_catalog, synthesize_minimal_repair
from .embodied_repair_verification import certificate_matches_program, verify_repair_against_heldout
from .embodied_semantic_compiler import compile_semantic_boundary
from .embodied_semantic_types import SemanticAdapter, SemanticTensorType
from .proof_carrying_runtime import ExecutionProofRejected, ProofCarryingSemanticRuntime
from .semantic_effect_certificate import EffectCommitPolicy, issue_effect_certificate
from .semantic_effect_commit import EffectClass, EffectStatus, RuntimeDecision, SemanticEffectIntent
from .semantic_effect_evidence import EffectObservation
from .semantic_effect_model_check import bounded_model_check


def _proof_fixture():
    source = SemanticTensorType(
        role="action",
        entity="ee_rotation_delta",
        representation="axis_angle",
        unit="rad",
        provenance="requested",
    )
    target = source.updated(representation="euler_xyz")
    adapter = SemanticAdapter(
        "axis-to-euler",
        requires={"representation": "axis_angle"},
        produces={"representation": "euler_xyz"},
        cost=1.0,
    )
    result = compile_semantic_boundary(source, target, adapters=(adapter,))
    compilation = issue_compilation_certificate(result, adapters=(adapter,))
    intent = SemanticEffectIntent(
        effect_id="release-gate:handoff",
        action_name="handoff",
        effect_class=EffectClass.IRREVERSIBLE,
        dependency_version=compilation.decision_digest,
        precondition=lambda state: state["holder"] == "robot",
        postcondition=lambda state: state["holder"] == "human",
    )
    effect = issue_effect_certificate(
        intent,
        policy=EffectCommitPolicy(min_commit_planes=2),
        semantic_contract_id="embodied/effect/handoff@0.1",
    )
    return adapter, compilation, intent, effect


def _repair_identity_check() -> bool:
    result = synthesize_minimal_repair(
        [
            RepairExample((-1.0, 2.0, -3.0), (1.0, 2.0, 3.0)),
            RepairExample((-4.0, 5.0, -6.0), (4.0, 5.0, 6.0)),
        ],
        build_vector_repair_catalog(3),
        max_depth=1,
        allowed_families={"sign"},
    )
    if result.program is None:
        return False

    cert = verify_repair_against_heldout(
        contract_id="embodied/release-gate@0.1",
        program=result.program,
        heldout=[RepairExample((-0.5, 0.25, -0.75), (0.5, 0.25, 0.75))],
        verifier_id="release-gate-heldout-v1",
    )
    original = result.program.operations[0]
    substituted = RepairProgram(
        (
            RepairPrimitive(
                name=original.name,
                family=original.family,
                cost=original.cost,
                apply_fn=original.apply_fn,
                implementation_id="substituted-v1",
            ),
        )
    )
    return (
        certificate_matches_program(
            cert,
            contract_id="embodied/release-gate@0.1",
            program=result.program,
        )
        and not certificate_matches_program(
            cert,
            contract_id="embodied/release-gate@0.1",
            program=substituted,
        )
    )


def evaluate_runtime_release_gate(*, model_depth: int = 4) -> dict[str, object]:
    checks: dict[str, bool] = {}

    model = bounded_model_check(max_depth=model_depth)
    checks["bounded_replay_model_safe"] = bool(model["safe"])

    adapter, compilation, intent, effect = _proof_fixture()
    runtime = ProofCarryingSemanticRuntime()
    auth, prepared = runtime.prepare(
        compilation,
        effect,
        intent,
        semantic_contract_id="embodied/effect/handoff@0.1",
        adapters=(adapter,),
        current_state={"holder": "robot"},
        current_dependency_version=compilation.decision_digest,
        model_depth=model_depth,
    )
    checks["proof_authorized_prepare"] = prepared.decision == RuntimeDecision.DISPATCH

    runtime.mark_dispatched(
        auth,
        compilation,
        effect,
        intent,
        adapters=(adapter,),
        model_depth=model_depth,
    )

    one_plane = runtime.resolve(
        auth,
        compilation,
        effect,
        intent,
        (EffectObservation("camera", "camera", {"holder": "human"}),),
        adapters=(adapter,),
        model_depth=model_depth,
    )
    checks["single_plane_fails_closed"] = (
        one_plane.outcome.status == EffectStatus.AMBIGUOUS
        and one_plane.outcome.decision == RuntimeDecision.BLOCK
    )

    # Use a fresh runtime/effect for the successful two-plane branch because
    # the previous branch deliberately transitioned that effect to AMBIGUOUS.
    adapter2, compilation2, intent2, effect2 = _proof_fixture()
    runtime2 = ProofCarryingSemanticRuntime()
    auth2, _ = runtime2.prepare(
        compilation2,
        effect2,
        intent2,
        semantic_contract_id="embodied/effect/handoff@0.1",
        adapters=(adapter2,),
        current_state={"holder": "robot"},
        current_dependency_version=compilation2.decision_digest,
        model_depth=model_depth,
    )
    runtime2.mark_dispatched(
        auth2,
        compilation2,
        effect2,
        intent2,
        adapters=(adapter2,),
        model_depth=model_depth,
    )
    committed = runtime2.resolve(
        auth2,
        compilation2,
        effect2,
        intent2,
        (
            EffectObservation("camera", "camera", {"holder": "human"}),
            EffectObservation("force", "proprioception", {"holder": "human"}),
        ),
        adapters=(adapter2,),
        model_depth=model_depth,
    )
    checks["independent_evidence_commits"] = committed.outcome.status == EffectStatus.COMMITTED

    drifted = (
        adapter,
        SemanticAdapter(
            "axis-to-euler-alias",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
    )
    try:
        ProofCarryingSemanticRuntime().prepare(
            compilation,
            effect,
            intent,
            semantic_contract_id="embodied/effect/handoff@0.1",
            adapters=drifted,
            current_state={"holder": "robot"},
            current_dependency_version=compilation.decision_digest,
            model_depth=model_depth,
        )
    except ExecutionProofRejected:
        checks["registry_drift_rejected"] = True
    else:
        checks["registry_drift_rejected"] = False

    try:
        ProofCarryingSemanticRuntime().prepare(
            compilation,
            effect,
            intent,
            semantic_contract_id="embodied/effect/handoff@0.1",
            adapters=(adapter,),
            current_state={"holder": "robot"},
            current_dependency_version="stale",
            model_depth=model_depth,
        )
    except ExecutionProofRejected:
        checks["stale_dependency_rejected"] = True
    else:
        checks["stale_dependency_rejected"] = False

    checks["repair_implementation_identity_bound"] = _repair_identity_check()

    return {
        "schema": "proof-carrying-semantic-runtime-release-gate/v0.1",
        "model_depth": model_depth,
        "checks": checks,
        "passed": all(checks.values()),
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "model_traces_checked": model["traces_checked"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit proof-carrying embodied runtime release gates")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = evaluate_runtime_release_gate(model_depth=args.depth)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"passed={report['passed']} checks={report['passed_checks']}/{report['total_checks']}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()