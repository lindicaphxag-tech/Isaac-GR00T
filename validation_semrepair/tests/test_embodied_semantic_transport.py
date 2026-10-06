import pytest

from validation_semrepair.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
    analyze_transport_cancellation,
)


def _factor(name, transport, evidence):
    return SemanticTransportFactor(name, transport, evidence)


def test_two_joint_order_faults_cancel_exactly_and_each_local_fix_unmasks():
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    cert = analyze_transport_cancellation(
        (
            _factor("producer-order", swap, "producer/layout.py"),
            _factor("dispatch-order", swap, "controller/dispatch.py"),
        )
    )

    assert cert.structurally_masked
    assert cert.net.is_identity()
    assert cert.nonidentity_factors == ("producer-order", "dispatch-order")
    assert cert.repair_one_unmasks == ("producer-order", "dispatch-order")
    assert cert.net.apply((0.8, -0.4)) == pytest.approx((0.8, -0.4))


def test_two_sign_faults_can_hide_each_other():
    flip = MonomialSemanticTransport((0, 1), (-1.0, -1.0))
    cert = analyze_transport_cancellation(
        (
            _factor("producer-sign", flip, "producer/sign"),
            _factor("controller-sign", flip, "controller/sign"),
        )
    )

    assert cert.structurally_masked
    assert cert.net.apply((0.3, -0.2)) == pytest.approx((0.3, -0.2))


def test_inverse_unit_scale_faults_can_be_structurally_masked():
    mm_as_m = MonomialSemanticTransport((0, 1), (1000.0, 1000.0))
    compensating_scale = MonomialSemanticTransport(
        (0, 1), (0.001, 0.001)
    )
    cert = analyze_transport_cancellation(
        (
            _factor("dataset-unit", mm_as_m, "dataset/unit"),
            _factor("runtime-scale", compensating_scale, "runtime/scale"),
        ),
        atol=1.0e-12,
    )

    assert cert.structurally_masked


def test_non_canceling_chain_is_not_reported_as_masked():
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    sign = MonomialSemanticTransport((0, 1), (-1.0, 1.0))
    cert = analyze_transport_cancellation(
        (
            _factor("order", swap, "order"),
            _factor("sign", sign, "sign"),
        )
    )

    assert not cert.structurally_masked
    assert not cert.net.is_identity()


def test_inverse_round_trip_is_identity():
    transform = MonomialSemanticTransport(
        (2, 0, 1),
        (-2.0, 0.5, 4.0),
    )
    identity = transform.then(transform.inverse())

    assert identity.is_identity(atol=1.0e-12)
    assert identity.apply((1.0, 2.0, 3.0)) == pytest.approx(
        (1.0, 2.0, 3.0)
    )


def test_digest_binds_source_evidence_even_when_transform_is_same():
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    a = analyze_transport_cancellation(
        (
            _factor("a", swap, "evidence-v1"),
            _factor("b", swap, "evidence-b"),
        )
    )
    b = analyze_transport_cancellation(
        (
            _factor("a", swap, "evidence-v2"),
            _factor("b", swap, "evidence-b"),
        )
    )

    assert a.digest != b.digest
