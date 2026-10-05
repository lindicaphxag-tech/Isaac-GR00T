from copy import deepcopy

from research.semantic_invariants.embodied_compilation_certificate import (
    issue_compilation_certificate,
)
from research.semantic_invariants.embodied_semantic_compiler import compile_semantic_boundary
from research.semantic_invariants.embodied_semantic_types import SemanticAdapter, SemanticTensorType
from research.semantic_invariants.execution_bundle_wire_v2 import (
    canonical_decimal,
    canonical_wire_json,
    export_execution_bundle_v2,
    verify_execution_bundle_v2,
    wire_digest,
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


def _bundle(*, cost=1.0):
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
        cost=cost,
    )
    result = compile_semantic_boundary(source, target, adapters=(adapter,))
    compilation = issue_compilation_certificate(result, adapters=(adapter,))
    intent = SemanticEffectIntent(
        effect_id="handoff:wire-v2",
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
    return export_execution_bundle_v2(
        compilation,
        effect,
        intent,
        semantic_contract_id=CONTRACT,
        adapters=(adapter,),
        max_steps=4,
    )


def _rebind(bundle):
    bundle = deepcopy(bundle)
    bundle["bindings"]["adapter_registry_digest"] = wire_digest(
        bundle["adapter_registry"]
    )
    bundle["bindings"]["evidence_identity_digest"] = wire_digest(
        bundle["evidence_identity"]
    )
    bundle["bindings"]["compilation_digest"] = wire_digest(
        bundle["compilation_certificate"]
    )
    bundle["bindings"]["effect_digest"] = wire_digest(bundle["effect_certificate"])
    payload = dict(bundle)
    payload.pop("bundle_digest", None)
    bundle["bundle_digest"] = wire_digest(payload)
    return bundle


def _contains_float(value):
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_contains_float(k) or _contains_float(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_float(v) for v in value)
    return False


def test_v2_bundle_verifies_with_exact_decimal_wire_costs():
    bundle = _bundle(cost=1.0)
    report = verify_execution_bundle_v2(bundle)

    assert report["valid"]
    assert bundle["adapter_registry"][0]["cost_decimal"] == "1"
    assert report["selected_cost_decimal"] == "1"
    assert not _contains_float(bundle)


def test_decimal_canonicalization_is_exponent_free_and_stable():
    assert canonical_decimal(1.0) == "1"
    assert canonical_decimal(0.3000) == "0.3"
    assert canonical_decimal("1e-12") == "0.000000000001"
    assert canonical_decimal("-0.0") == "0"


def test_unicode_is_nfc_normalized_before_hashing():
    decomposed = {"label": "e\u0301"}
    composed = {"label": "é"}
    assert canonical_wire_json(decomposed) == canonical_wire_json(composed)
    assert wire_digest(decomposed) == wire_digest(composed)


def test_exact_decimal_minimum_rejects_false_unique_float_winner():
    bundle = _bundle(cost=0.3)
    direct = bundle["adapter_registry"][0]
    bundle["adapter_registry"] = [
        direct,
        {
            "name": "axis-to-mid",
            "requires": {"representation": "axis_angle"},
            "produces": {"representation": "quaternion"},
            "cost_decimal": "0.1",
            "effects": [],
            "required_evidence": [],
        },
        {
            "name": "mid-to-euler",
            "requires": {"representation": "quaternion"},
            "produces": {"representation": "euler_xyz"},
            "cost_decimal": "0.2",
            "effects": [],
            "required_evidence": [],
        },
    ]
    bundle = _rebind(bundle)

    report = verify_execution_bundle_v2(bundle)

    assert not report["valid"]
    assert report["checks"]["selected_path_replay"] is True
    assert report["checks"]["minimum_cost_unique"] is False


def test_cost_lexeme_tamper_fails_even_when_outer_digest_is_recomputed():
    bundle = _bundle()
    bundle["adapter_registry"][0]["cost_decimal"] = "1.0"
    bundle = _rebind(bundle)

    report = verify_execution_bundle_v2(bundle)

    assert not report["valid"]
    assert report["checks"]["adapter_semantics_valid"] is False


def test_non_forgeable_provenance_adapter_is_rejected():
    bundle = _bundle()
    bundle["adapter_registry"].append(
        {
            "name": "forge-executed",
            "requires": {"provenance": "requested"},
            "produces": {"provenance": "executed"},
            "cost_decimal": "0",
            "effects": [],
            "required_evidence": [],
        }
    )
    bundle = _rebind(bundle)

    report = verify_execution_bundle_v2(bundle)

    assert not report["valid"]
    assert report["checks"]["adapter_semantics_valid"] is False


def test_compilation_dependency_identity_is_bound_without_replaying_python_digest():
    bundle = _bundle()
    bundle["intent"]["dependency_version"] = "0" * 64
    bundle["effect_certificate"]["dependency_version"] = "0" * 64
    bundle = _rebind(bundle)

    report = verify_execution_bundle_v2(bundle)

    assert not report["valid"]
    assert report["checks"]["effect_matches_intent"] is True
    assert report["checks"]["compilation_effect_dependency"] is False


def test_wire_digest_detects_semantic_tamper():
    bundle = _bundle()
    bundle["compilation_certificate"]["target"]["representation"] = "quaternion"

    report = verify_execution_bundle_v2(bundle)

    assert not report["valid"]
    assert report["checks"]["bundle_digest"] is False
    assert report["checks"]["compilation_wire_digest"] is False


def test_binary_float_escape_is_rejected_even_after_redigest():
    bundle = _bundle()
    bundle["evidence_identity"]["unsafe"] = 0.1
    bundle = _rebind(bundle)

    report = verify_execution_bundle_v2(bundle)

    assert not report["valid"]
    assert report["checks"]["float_free_wire_surface"] is False


def test_noncanonical_producer_selected_cost_is_rejected():
    bundle = _bundle()
    bundle["compilation_certificate"]["producer_selected_cost_decimal"] = "1.0"
    bundle = _rebind(bundle)

    report = verify_execution_bundle_v2(bundle)

    assert not report["valid"]
    assert report["checks"]["producer_selected_cost_decimal"] is False


def test_utf8_key_order_is_cross_language_stable():
    astral = "\U00010000"
    private_bmp = "\ue000"
    assert canonical_wire_json({astral: 1, private_bmp: 2}) == (
        "{\"\\ue000\":2,\"𐀀\":1}".replace("\\ue000", private_bmp)
    )


def test_wire_rejects_integer_outside_cross_language_safe_range():
    bundle = _bundle()
    bundle["evidence_identity"]["big"] = 2**53
    with pytest.raises(ValueError, match="safe range"):
        wire_digest(bundle)


def test_effect_class_tamper_fails_closed_even_after_redigest():
    bundle = _bundle()
    bundle["effect_certificate"]["effect_class"] = "teleport"
    bundle["intent"]["effect_class"] = "teleport"
    bundle = _rebind(bundle)

    report = verify_execution_bundle_v2(bundle)

    assert not report["valid"]
    assert report["checks"]["effect_class"] is False


def test_effect_policy_threshold_cannot_be_zero():
    bundle = _bundle()
    bundle["effect_certificate"]["policy"]["min_commit_planes"] = 0
    bundle = _rebind(bundle)

    report = verify_execution_bundle_v2(bundle)

    assert not report["valid"]
    assert report["checks"]["effect_policy"] is False


def test_unknown_semantic_record_field_is_rejected():
    bundle = _bundle()
    bundle["compilation_certificate"]["target"]["mystery_axis"] = "x"
    bundle = _rebind(bundle)

    report = verify_execution_bundle_v2(bundle)

    assert not report["valid"]
    assert report["checks"]["semantic_records"] is False
