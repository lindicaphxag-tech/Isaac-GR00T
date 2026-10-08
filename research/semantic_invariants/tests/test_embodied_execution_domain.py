import math

import pytest

from research.semantic_invariants.embodied_execution_domain import (
    L2BallExecutionDomain,
    ProjectionAuthorityRequired,
    authorize_execution_domain,
    issue_projection_certificate,
    verify_projection_certificate,
)


def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def test_projection_certificate_accepts_feasible_low_residual_action():
    domain=L2BallExecutionDomain("controller/normalized-rotation@v1",3,1.0)
    cert=issue_projection_certificate(
        contract_id="ee-rotation@v1",
        domain=domain,
        target=(0.05,-0.02,0.01),
        action=(0.5,-0.2,0.1),
        forward=lambda a: tuple(0.1*x for x in a),
        forward_model_id="controller-forward@sha256:aaa",
        distance=euclidean,
        max_residual=1.0e-9,
    )
    assert cert.accepted
    check=verify_projection_certificate(
        cert,
        contract_id="ee-rotation@v1",
        domain=domain,
        target=(0.05,-0.02,0.01),
        forward=lambda a: tuple(0.1*x for x in a),
        forward_model_id="controller-forward@sha256:aaa",
        distance=euclidean,
    )
    assert check.valid
    assert check.accepted


def test_projection_certificate_rejects_out_of_domain_action():
    domain=L2BallExecutionDomain("controller/normalized-rotation@v1",3,1.0)
    cert=issue_projection_certificate(
        contract_id="ee-rotation@v1",
        domain=domain,
        target=(0.2,0.0,0.0),
        action=(2.0,0.0,0.0),
        forward=lambda a: tuple(0.1*x for x in a),
        forward_model_id="controller-forward@sha256:aaa",
        distance=euclidean,
        max_residual=1.0,
    )
    assert not cert.feasible
    assert not cert.accepted


def test_projection_certificate_rejects_large_semantic_residual():
    domain=L2BallExecutionDomain("controller/normalized-rotation@v1",3,1.0)
    cert=issue_projection_certificate(
        contract_id="ee-rotation@v1",
        domain=domain,
        target=(1.0,0.0,0.0),
        action=(1.0,0.0,0.0),
        forward=lambda a: tuple(0.1*x for x in a),
        forward_model_id="controller-forward@sha256:aaa",
        distance=euclidean,
        max_residual=0.2,
    )
    assert cert.feasible
    assert not cert.accepted


def test_forward_semantics_drift_invalidates_projection_certificate():
    domain=L2BallExecutionDomain("controller/normalized-rotation@v1",3,1.0)
    cert=issue_projection_certificate(
        contract_id="ee-rotation@v1",
        domain=domain,
        target=(0.05,0.0,0.0),
        action=(0.5,0.0,0.0),
        forward=lambda a: tuple(0.1*x for x in a),
        forward_model_id="controller-forward@sha256:aaa",
        distance=euclidean,
        max_residual=1.0e-9,
    )
    check=verify_projection_certificate(
        cert,
        contract_id="ee-rotation@v1",
        domain=domain,
        target=(0.05,0.0,0.0),
        forward=lambda a: tuple(0.2*x for x in a),
        forward_model_id="controller-forward@sha256:bbb",
        distance=euclidean,
    )
    assert not check.valid
    assert any("forward-semantics identity" in reason for reason in check.reasons)


def test_target_or_domain_drift_invalidates_projection_certificate():
    domain=L2BallExecutionDomain("controller/normalized-rotation@v1",3,1.0)
    cert=issue_projection_certificate(
        contract_id="ee-rotation@v1",
        domain=domain,
        target=(0.05,0.0,0.0),
        action=(0.5,0.0,0.0),
        forward=lambda a: tuple(0.1*x for x in a),
        forward_model_id="controller-forward@sha256:aaa",
        distance=euclidean,
        max_residual=1.0e-9,
    )
    changed_domain=L2BallExecutionDomain(
        "controller/normalized-rotation@v2",3,0.8
    )
    check=verify_projection_certificate(
        cert,
        contract_id="ee-rotation@v1",
        domain=changed_domain,
        target=(0.06,0.0,0.0),
        forward=lambda a: tuple(0.1*x for x in a),
        forward_model_id="controller-forward@sha256:aaa",
        distance=euclidean,
    )
    assert not check.valid
    assert any("domain identity" in reason for reason in check.reasons)
    assert any("target identity" in reason for reason in check.reasons)


def test_in_domain_repair_keeps_exact_action_without_projection():
    domain=L2BallExecutionDomain("controller/action-ball@v1",2,1.0)
    auth=authorize_execution_domain(
        contract_id="action@v1",
        raw_action=(0.2,-0.3),
        semantic_target=(0.2,-0.3),
        domain=domain,
        forward=lambda a:a,
        forward_model_id="identity@v1",
        distance=euclidean,
    )
    assert auth.authorized_action == (0.2,-0.3)
    assert not auth.projection_used
    assert auth.residual == 0.0


def test_out_of_domain_repair_cannot_be_silently_clipped():
    domain=L2BallExecutionDomain("controller/action-ball@v1",2,1.0)
    with pytest.raises(ProjectionAuthorityRequired, match="silent clipping"):
        authorize_execution_domain(
            contract_id="action@v1",
            raw_action=(2.0,0.0),
            semantic_target=(1.0,0.0),
            domain=domain,
            forward=lambda a:a,
            forward_model_id="identity@v1",
            distance=euclidean,
        )


def test_verified_projection_can_authorize_out_of_domain_repair():
    domain=L2BallExecutionDomain("controller/action-ball@v1",2,1.0)
    cert=issue_projection_certificate(
        contract_id="action@v1",
        domain=domain,
        target=(1.2,0.0),
        action=(1.0,0.0),
        forward=lambda a:a,
        forward_model_id="identity@v1",
        distance=euclidean,
        max_residual=0.25,
    )
    assert cert.accepted

    auth=authorize_execution_domain(
        contract_id="action@v1",
        raw_action=(1.2,0.0),
        semantic_target=(1.2,0.0),
        domain=domain,
        forward=lambda a:a,
        forward_model_id="identity@v1",
        distance=euclidean,
        projection_certificate=cert,
    )
    assert auth.projection_used
    assert auth.authorized_action == (1.0,0.0)
    assert auth.projection_certificate_digest == cert.decision_digest


def test_valid_but_high_residual_projection_has_no_execution_authority():
    domain=L2BallExecutionDomain("controller/action-ball@v1",2,1.0)
    cert=issue_projection_certificate(
        contract_id="action@v1",
        domain=domain,
        target=(2.0,0.0),
        action=(1.0,0.0),
        forward=lambda a:a,
        forward_model_id="identity@v1",
        distance=euclidean,
        max_residual=0.1,
    )
    assert not cert.accepted

    with pytest.raises(ProjectionAuthorityRequired, match="exceeds"):
        authorize_execution_domain(
            contract_id="action@v1",
            raw_action=(2.0,0.0),
            semantic_target=(2.0,0.0),
            domain=domain,
            forward=lambda a:a,
            forward_model_id="identity@v1",
            distance=euclidean,
            projection_certificate=cert,
        )
