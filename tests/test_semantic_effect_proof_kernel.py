from dataclasses import replace

from research.semantic_invariants.embodied_compilation_certificate import issue_compilation_certificate
from research.semantic_invariants.embodied_semantic_compiler import compile_semantic_boundary
from research.semantic_invariants.embodied_semantic_types import SemanticAdapter, SemanticTensorType
from research.semantic_invariants.semantic_effect_certificate import EffectCommitPolicy, issue_effect_certificate
from research.semantic_invariants.semantic_effect_commit import EffectClass, SemanticEffectIntent
from research.semantic_invariants.semantic_effect_proof_kernel import verify_embodied_execution_bundle


def _semantic_fixture():
    source = SemanticTensorType(
        role="action",
        entity="ee_rotation_delta",
        representation="axis_angle",
        unit="rad",
        provenance="requested",
    )
    target = source.updated(representation="euler_xyz")

    adapter = SemanticAdapter(
        name="axis-angle-to-euler",
        requires={"representation": "axis_angle"},
        produces={"representation": "euler_xyz"},
        cost=1.0,
        effects=("reencode",),
    )
    result = compile_semantic_boundary(source, target, adapters=(adapter,))
    cert = issue_compilation_certificate(result, adapters=(adapter,))
    return adapter, cert


def _intent(dependency_version):
    return SemanticEffectIntent(
        effect_id="handoff:proof-chain",
        action_name="handoff",
        effect_class=EffectClass.IRREVERSIBLE,
        dependency_version=dependency_version,
        precondition=lambda state: state["holder"] == "robot",
        postcondition=lambda state: state["holder"] == "human",
    )


def test_end_to_end_proof_binds_physical_effect_to_compilation_digest():
    adapter, compilation = _semantic_fixture()
    intent = _intent(compilation.decision_digest)
    effect = issue_effect_certificate(
        intent,
        policy=EffectCommitPolicy(min_commit_planes=2),
        semantic_contract_id="embodied/effect/handoff@0.1",
    )

    proof = verify_embodied_execution_bundle(
        compilation,
        effect,
        intent,
        adapters=(adapter,),
        model_depth=3,
    )

    assert proof.valid
    assert proof.dependency_bound_to_compilation
    assert proof.runtime_model_safe


def test_recompiled_or_stale_dependency_breaks_execution_proof():
    adapter, compilation = _semantic_fixture()
    intent = _intent("stale-compilation-digest")
    effect = issue_effect_certificate(
        intent,
        policy=EffectCommitPolicy(min_commit_planes=2),
        semantic_contract_id="embodied/effect/handoff@0.1",
    )

    proof = verify_embodied_execution_bundle(
        compilation,
        effect,
        intent,
        adapters=(adapter,),
        model_depth=2,
    )

    assert not proof.valid
    assert not proof.dependency_bound_to_compilation


def test_tampered_effect_certificate_breaks_end_to_end_proof():
    adapter, compilation = _semantic_fixture()
    intent = _intent(compilation.decision_digest)
    effect = issue_effect_certificate(
        intent,
        policy=EffectCommitPolicy(min_commit_planes=2),
        semantic_contract_id="embodied/effect/handoff@0.1",
    )
    tampered = replace(
        effect,
        policy={**effect.policy, "min_commit_planes": 1},
    )

    proof = verify_embodied_execution_bundle(
        compilation,
        tampered,
        intent,
        adapters=(adapter,),
        model_depth=2,
    )

    assert not proof.valid
    assert not proof.effect_certificate_valid