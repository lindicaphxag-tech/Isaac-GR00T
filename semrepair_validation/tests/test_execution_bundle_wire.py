from copy import deepcopy

from research.semantic_invariants.embodied_compilation_certificate import (
    issue_compilation_certificate,
)
from research.semantic_invariants.embodied_semantic_compiler import compile_semantic_boundary
from research.semantic_invariants.embodied_semantic_types import SemanticAdapter, SemanticTensorType
from research.semantic_invariants.execution_bundle_wire import (
    canonical_json_digest,
    export_execution_bundle,
    verify_execution_bundle,
)
from research.semantic_invariants.semantic_effect_certificate import (
    EffectCommitPolicy,
    issue_effect_certificate,
)
from research.semantic_invariants.semantic_effect_commit import (
    EffectClass,
    SemanticEffectIntent,
)


CONTRACT = "embodied/effect/handoff@0.1"


def _bundle():
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
        effect_id="handoff:wire",
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
    return export_execution_bundle(
        compilation,
        effect,
        intent,
        semantic_contract_id=CONTRACT,
        adapters=(adapter,),
        max_steps=4,
    )


def _rebind_outer_digest(bundle):
    clone = deepcopy(bundle)
    clone.pop("bundle_digest", None)
    clone["bundle_digest"] = canonical_json_digest(clone)
    return clone


def test_wire_bundle_verifies_without_compiler_search():
    report = verify_execution_bundle(_bundle())

    assert report["valid"]
    assert all(report["checks"].values())


def test_registry_tamper_fails_even_when_outer_digest_is_recomputed():
    bundle = _bundle()
    bundle["adapter_registry"][0]["cost"] = 0.5
    bundle = _rebind_outer_digest(bundle)

    report = verify_execution_bundle(bundle)

    assert not report["valid"]
    assert report["checks"]["registry_digest"] is False


def test_effect_policy_tamper_fails_even_when_outer_digest_is_recomputed():
    bundle = _bundle()
    bundle["effect_certificate"]["policy"]["min_commit_planes"] = 1
    bundle = _rebind_outer_digest(bundle)

    report = verify_execution_bundle(bundle)

    assert not report["valid"]
    assert report["checks"]["effect_digest"] is False


def test_dependency_rebinding_fails_even_with_recomputed_effect_and_outer_digests():
    bundle = _bundle()
    bundle["intent"]["dependency_version"] = "0" * 64
    bundle["effect_certificate"]["dependency_version"] = "0" * 64

    effect = bundle["effect_certificate"]
    payload = {key: value for key, value in effect.items() if key != "decision_digest"}
    effect["decision_digest"] = canonical_json_digest(payload)
    bundle = _rebind_outer_digest(bundle)

    report = verify_execution_bundle(bundle)

    assert not report["valid"]
    assert report["checks"]["effect_digest"] is True
    assert report["checks"]["effect_matches_intent"] is True
    assert report["checks"]["compilation_effect_dependency"] is False


def test_non_minimum_selected_path_is_rejected_after_consistent_redigest():
    bundle = _bundle()
    direct = bundle["adapter_registry"][0]
    bundle["adapter_registry"] = [
        {
            "name": "axis-to-mid",
            "requires": {"representation": "axis_angle"},
            "produces": {"representation": "quaternion"},
            "cost": 1.0,
            "effects": [],
            "required_evidence": [],
        },
        {
            "name": "mid-to-euler",
            "requires": {"representation": "quaternion"},
            "produces": {"representation": "euler_xyz"},
            "cost": 1.0,
            "effects": [],
            "required_evidence": [],
        },
        {**direct, "cost": 3.0},
    ]
    comp = bundle["compilation_certificate"]
    comp["registry_digest"] = canonical_json_digest(
        sorted(
            bundle["adapter_registry"],
            key=lambda item: (item["name"], canonical_json_digest(item)),
        )
    )
    comp["selected_cost"] = 3.0
    comp_payload = {key: value for key, value in comp.items() if key != "decision_digest"}
    comp["decision_digest"] = canonical_json_digest(comp_payload)
    bundle["intent"]["dependency_version"] = comp["decision_digest"]
    effect = bundle["effect_certificate"]
    effect["dependency_version"] = comp["decision_digest"]
    effect_payload = {key: value for key, value in effect.items() if key != "decision_digest"}
    effect["decision_digest"] = canonical_json_digest(effect_payload)
    bundle = _rebind_outer_digest(bundle)

    report = verify_execution_bundle(bundle)

    assert not report["valid"]
    assert report["checks"]["minimum_cost_unique"] is False


def test_unknown_bundle_schema_fails_closed():
    bundle = _bundle()
    bundle["schema"] = "semrepair-execution-proof-bundle/v9"
    bundle = _rebind_outer_digest(bundle)

    report = verify_execution_bundle(bundle)

    assert not report["valid"]
    assert report["checks"]["bundle_schema"] is False


def test_wire_adapter_cannot_forge_requested_to_executed_provenance():
    bundle = _bundle()
    bundle["adapter_registry"].append(
        {
            "name": "forge-executed",
            "requires": {"provenance": "requested"},
            "produces": {"provenance": "executed"},
            "cost": 0.0,
            "effects": [],
            "required_evidence": [],
        }
    )
    bundle = _rebind_outer_digest(bundle)

    report = verify_execution_bundle(bundle)

    assert not report["valid"]
    assert report["checks"]["adapter_semantics_valid"] is False


def test_wire_bundle_rejects_unverified_repair_candidate():
    bundle = _bundle()
    comp = bundle["compilation_certificate"]
    comp["repair_candidate"] = "candidate-only"
    comp_payload = {key: value for key, value in comp.items() if key != "decision_digest"}
    comp["decision_digest"] = canonical_json_digest(comp_payload)
    bundle["intent"]["dependency_version"] = comp["decision_digest"]
    effect = bundle["effect_certificate"]
    effect["dependency_version"] = comp["decision_digest"]
    effect_payload = {key: value for key, value in effect.items() if key != "decision_digest"}
    effect["decision_digest"] = canonical_json_digest(effect_payload)
    bundle = _rebind_outer_digest(bundle)

    report = verify_execution_bundle(bundle)

    assert not report["valid"]
    assert report["checks"]["installable_compilation"] is False