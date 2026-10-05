from dataclasses import replace
from math import inf

import numpy as np
import pytest

from casj.proof_certificate import (
    authorize_runtime_repair,
    issue_runtime_trust_certificate,
    verify_runtime_trust_certificate,
)
from casj.trust_region import DirectionalTrustRegion


def _trust(radius=0.8):
    return DirectionalTrustRegion(
        accepted=True,
        radius=radius,
        curvature_scale_stability=0.1,
        first_order_norm_per_unit=2.0,
        curvature_norm_per_unit2=0.5,
        reference_action_scale=2.0,
        radius_from_first_order=1.0,
        radius_from_reference_scale=0.8,
        limiting_constraint="reference_action_scale",
        reason="bounded",
    )


def _issue(radius=0.8, cap=0.6):
    return issue_runtime_trust_certificate(
        _trust(radius),
        policy_digest="policy-v1",
        action_contract_digest="action-v1",
        baseline_context_digest="context-v1",
        support_id="block",
        intervention_chart_digest="chart-v1",
        repair_implementation_digest="repair-v1",
        evidence_digest="evidence-v1",
        direction_digest="direction-v1",
        repair_mode="local_linear_repair",
        operational_radius_cap=cap,
        threshold_profile={"max_scale_instability": 0.15},
    )


def _authorize(cert, motion=0.5, **overrides):
    args = {
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
    }
    args.update(overrides)
    return authorize_runtime_repair(cert, **args)


def test_certificate_binds_identity_and_finite_motion_authority():
    cert = _issue()
    assert verify_runtime_trust_certificate(cert)
    assert cert.certified_motion_multiplier == pytest.approx(0.6)
    assert _authorize(cert, 0.6).authorized
    assert not _authorize(cert, 0.6001).authorized
    assert not _authorize(cert, policy_digest="policy-v2").authorized
    assert not _authorize(cert, action_contract_digest="action-v2").authorized
    assert not _authorize(cert, baseline_context_digest="context-v2").authorized
    assert not _authorize(cert, intervention_chart_digest="chart-v2").authorized


def test_integrity_rejects_tampered_certificate():
    cert = _issue()
    tampered = replace(cert, certified_motion_multiplier=0.7)
    assert not verify_runtime_trust_certificate(tampered)
    assert not _authorize(tampered).authorized


def test_unbounded_math_still_requires_finite_operational_cap():
    cert = _issue(radius=inf, cap=0.4)
    assert cert.mathematical_trust_radius == "inf"
    assert cert.certified_motion_multiplier == pytest.approx(0.4)
    assert _authorize(cert, 0.4).authorized
    assert not _authorize(cert, 0.41).authorized


def test_rejected_trust_region_cannot_mint_runtime_authority():
    rejected = replace(_trust(), accepted=False, radius=0.0)
    with pytest.raises(ValueError):
        issue_runtime_trust_certificate(
            rejected,
            policy_digest="p", action_contract_digest="a",
            baseline_context_digest="c", support_id="s",
            intervention_chart_digest="i", repair_implementation_digest="r",
            evidence_digest="e", direction_digest="d", repair_mode="repair",
            operational_radius_cap=1.0,
        )
