import pytest

from research.semantic_invariants.embodied_semantic_observability import (
    analyze_semantic_observability,
)
from research.semantic_invariants.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
)


def _factor(name, transport, evidence):
    return SemanticTransportFactor(name, transport, evidence)


def test_double_swap_is_black_box_unidentifiable_but_internal_tap_exposes_it():
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    cert = analyze_semantic_observability(
        (
            _factor("producer-order", swap, "producer/layout"),
            _factor("dispatch-order", swap, "controller/layout"),
        )
    )

    assert cert.black_box_impossibility
    assert cert.net_identity
    assert not cert.end_to_end_identifiable
    assert cert.black_box_basis_witnesses == ()
    assert cert.earliest_required_tap == "producer-order"

    first = cert.internal_boundary_witnesses[0]
    assert first.after_factor == "producer-order"
    assert first.basis_index == 0
    assert first.expected == (1.0, 0.0)
    assert first.observed == pytest.approx((0.0, 1.0))
    assert first.residual_linf == pytest.approx(1.0)


def test_nonidentity_net_has_constructive_black_box_basis_witness():
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    cert = analyze_semantic_observability(
        (_factor("order", swap, "order/evidence"),)
    )

    assert not cert.net_identity
    assert cert.end_to_end_identifiable
    assert not cert.black_box_impossibility
    assert cert.black_box_basis_witnesses
    witness = cert.black_box_basis_witnesses[0]
    assert witness.after_factor is None
    assert witness.is_distinguishing


def test_two_sign_flips_are_globally_hidden_for_every_external_input():
    flip = MonomialSemanticTransport((0, 1), (-1.0, -1.0))
    cert = analyze_semantic_observability(
        (
            _factor("producer-sign", flip, "producer/sign"),
            _factor("controller-sign", flip, "controller/sign"),
        )
    )

    assert cert.black_box_impossibility
    assert cert.earliest_required_tap == "producer-sign"


def test_inverse_scale_faults_require_internal_evidence():
    up = MonomialSemanticTransport((0,), (1000.0,))
    down = MonomialSemanticTransport((0,), (0.001,))
    cert = analyze_semantic_observability(
        (
            _factor("millimeter-as-meter", up, "dataset/unit"),
            _factor("compensating-runtime-scale", down, "runtime/unit"),
        ),
        atol=1e-12,
    )

    assert cert.black_box_impossibility
    first = cert.internal_boundary_witnesses[0]
    assert first.observed == pytest.approx((1000.0,))


def test_identity_only_chain_is_not_mislabeled_as_hidden_fault():
    identity = MonomialSemanticTransport.identity(3)
    cert = analyze_semantic_observability(
        (
            _factor("boundary-a", identity, "a"),
            _factor("boundary-b", identity, "b"),
        )
    )

    assert cert.net_identity
    assert cert.nonidentity_factors == ()
    assert not cert.black_box_impossibility
    assert cert.earliest_required_tap is None
    assert cert.internal_boundary_witnesses == ()


def test_evidence_identity_changes_observability_digest():
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    first = analyze_semantic_observability(
        (_factor("order", swap, "source-v1"),)
    )
    second = analyze_semantic_observability(
        (_factor("order", swap, "source-v2"),)
    )

    assert first.digest != second.digest
