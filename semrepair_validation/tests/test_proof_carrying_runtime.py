from dataclasses import replace

import pytest

from research.semantic_invariants.embodied_compilation_certificate import (
    issue_compilation_certificate,
)
from research.semantic_invariants.embodied_semantic_compiler import (
    compile_semantic_boundary,
)
from research.semantic_invariants.embodied_semantic_types import (
    SemanticAdapter,
    SemanticTensorType,
)
from research.semantic_invariants.proof_carrying_runtime import (
    ExecutionProofRejected,
    ProofCarryingSemanticRuntime,
)
from research.semantic_invariants.semantic_effect_certificate import (
    EffectCommitPolicy,
    issue_effect_certificate,
)
from research.semantic_invariants.semantic_effect_commit import (
    EffectClass,
    EffectStatus,
    RuntimeDecision,
    SemanticEffectIntent,
)
from research.semantic_invariants.semantic_effect_evidence import EffectObservation


CONTRACT = "embodied/effect/handoff@0.1"


def _fixture():
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
        effect_id="handoff:proof-runtime",
        action_name="handoff",
        effect_class=EffectClass.IRREVERSIBLE,
        dependency_version=compilation.decision_digest,
        precondition=lambda state: state["holder"] == "robot",
        postcondition=lambda state: state["holder"] == "human",
    )
    effect = issue_effect_certificate(
        intent,
        policy=EffectCommitPolicy(min_commit_planes=2),
        semantic_contract_id=CONTRACT,
    )
    return adapter, compilation, intent, effect


def test_complete_proof_carrying_execution_commits_only_after_independent_evidence():
    adapter, compilation, intent, effect = _fixture()
    runtime = ProofCarryingSemanticRuntime()

    auth, prepared = runtime.prepare(
        compilation,
        effect,
        intent,
        semantic_contract_id=CONTRACT,
        adapters=(adapter,),
        current_state={"holder": "robot"},
        current_dependency_version=compilation.decision_digest,
        model_depth=3,
    )
    assert prepared.decision == RuntimeDecision.DISPATCH

    runtime.mark_dispatched(
        auth,
        compilation,
        effect,
        intent,
        adapters=(adapter,),
        model_depth=3,
    )

    one_plane = runtime.resolve(
        auth,
        compilation,
        effect,
        intent,
        (EffectObservation("camera", "camera", {"holder": "human"}),),
        adapters=(adapter,),
        model_depth=3,
    )
    assert one_plane.outcome.status == EffectStatus.AMBIGUOUS
    assert one_plane.outcome.decision == RuntimeDecision.BLOCK


def test_two_independent_planes_complete_proof_carrying_commit():
    adapter, compilation, intent, effect = _fixture()
    runtime = ProofCarryingSemanticRuntime()

    auth, _ = runtime.prepare(
        compilation,
        effect,
        intent,
        semantic_contract_id=CONTRACT,
        adapters=(adapter,),
        current_state={"holder": "robot"},
        current_dependency_version=compilation.decision_digest,
        model_depth=2,
    )
    runtime.mark_dispatched(
        auth, compilation, effect, intent, adapters=(adapter,), model_depth=2
    )

    resolved = runtime.resolve(
        auth,
        compilation,
        effect,
        intent,
        (
            EffectObservation("camera", "camera", {"holder": "human"}),
            EffectObservation("force", "proprioception", {"holder": "human"}),
        ),
        adapters=(adapter,),
        model_depth=2,
    )

    assert resolved.outcome.status == EffectStatus.COMMITTED
    assert resolved.outcome.dispatch_count == 1


def test_adapter_registry_drift_blocks_before_physical_dispatch():
    adapter, compilation, intent, effect = _fixture()
    runtime = ProofCarryingSemanticRuntime()

    auth, _ = runtime.prepare(
        compilation,
        effect,
        intent,
        semantic_contract_id=CONTRACT,
        adapters=(adapter,),
        current_state={"holder": "robot"},
        current_dependency_version=compilation.decision_digest,
        model_depth=2,
    )

    drifted = adapter, SemanticAdapter(
        "axis-to-euler-alias",
        requires={"representation": "axis_angle"},
        produces={"representation": "euler_xyz"},
        cost=1.0,
    )
    with pytest.raises(ExecutionProofRejected, match="registry"):
        runtime.mark_dispatched(
            auth,
            compilation,
            effect,
            intent,
            adapters=drifted,
            model_depth=2,
        )


def test_tampered_effect_policy_blocks_before_dispatch():
    adapter, compilation, intent, effect = _fixture()
    runtime = ProofCarryingSemanticRuntime()

    auth, _ = runtime.prepare(
        compilation,
        effect,
        intent,
        semantic_contract_id=CONTRACT,
        adapters=(adapter,),
        current_state={"holder": "robot"},
        current_dependency_version=compilation.decision_digest,
        model_depth=2,
    )
    tampered = replace(
        effect,
        policy={**effect.policy, "min_commit_planes": 1},
    )

    with pytest.raises(ExecutionProofRejected):
        runtime.mark_dispatched(
            auth,
            compilation,
            tampered,
            intent,
            adapters=(adapter,),
            model_depth=2,
        )


def test_stale_dependency_is_rejected_even_when_both_certificates_are_valid():
    adapter, compilation, intent, effect = _fixture()
    runtime = ProofCarryingSemanticRuntime()

    with pytest.raises(ExecutionProofRejected, match="stale"):
        runtime.prepare(
            compilation,
            effect,
            intent,
            semantic_contract_id=CONTRACT,
            adapters=(adapter,),
            current_state={"holder": "robot"},
            current_dependency_version="old-compilation",
            model_depth=2,
        )