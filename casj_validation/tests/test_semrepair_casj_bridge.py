from dataclasses import asdict

from casj.proof_certificate import issue_runtime_trust_certificate
from casj.trust_region import DirectionalTrustRegion
from semrepair_casj_verifier import verify_casj_runtime_certificate


def _trust():
    return DirectionalTrustRegion(
        accepted=True,
        radius=0.8,
        curvature_scale_stability=0.1,
        first_order_norm_per_unit=2.0,
        curvature_norm_per_unit2=0.5,
        reference_action_scale=2.0,
        radius_from_first_order=1.0,
        radius_from_reference_scale=0.8,
        limiting_constraint="reference_action_scale",
        reason="bounded",
    )


def _issue():
    return issue_runtime_trust_certificate(
        _trust(),
        policy_digest="policy-v1",
        action_contract_digest="action-v1",
        baseline_context_digest="context-v1",
        support_id="block",
        intervention_chart_digest="chart-v1",
        repair_implementation_digest="repair-v1",
        evidence_digest="evidence-v1",
        direction_digest="direction-v1",
        repair_mode="local_linear_repair",
        operational_radius_cap=0.6,
        threshold_profile={"max_scale_instability": 0.15},
    )


def _verify(payload, motion=0.5, **overrides):
    kwargs = {
        "requested_motion_multiplier": motion,
        "policy_digest": "policy-v1",
        "action_contract_digest": "action-v1",
        "baseline_context_digest": "context-v1",
        "support_id": "block",
        "intervention_chart_digest": "chart-v1",
        "repair_implementation_digest": "repair-v1",
        "evidence_digest": "evidence-v1",
        "direction_digest": "direction-v1",
        "repair_mode": "local_linear_repair",
        "expected_threshold_profile": {"max_scale_instability": 0.15},
    }
    kwargs.update(overrides)
    return verify_casj_runtime_certificate(payload, **kwargs)


def test_casj_issuer_is_accepted_by_independent_semrepair_verifier():
    payload = asdict(_issue())
    check = _verify(payload)
    assert check.valid
    assert check.integrity
    assert check.identity_bound
    assert check.within_authority


def test_semrepair_rejects_identity_drift_from_valid_casj_certificate():
    payload = asdict(_issue())
    for field, changed in (
        ("policy_digest", "policy-v2"),
        ("action_contract_digest", "action-v2"),
        ("baseline_context_digest", "context-v2"),
        ("support_id", "other-block"),
        ("intervention_chart_digest", "chart-v2"),
        ("repair_implementation_digest", "repair-v2"),
        ("evidence_digest", "evidence-v2"),
        ("direction_digest", "direction-v2"),
    ):
        assert not _verify(payload, **{field: changed}).valid, field


def test_semrepair_rejects_motion_beyond_casj_authority():
    payload = asdict(_issue())
    assert _verify(payload, motion=0.6).valid
    check = _verify(payload, motion=0.600001)
    assert not check.valid
    assert not check.within_authority
