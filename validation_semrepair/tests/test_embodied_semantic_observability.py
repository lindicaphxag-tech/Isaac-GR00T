import pytest

from validation_semrepair.embodied_semantic_observability import (
    SemanticDiagnosisHypothesis,
    analyze_semantic_observability,
    synthesize_minimal_diagnostic_taps,
)
from validation_semrepair.embodied_semantic_transport import (
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



def _hypothesis(name, factors):
    return SemanticDiagnosisHypothesis(name=name, factors=tuple(factors))


def test_minimal_tap_plan_separates_hidden_double_swap():
    identity = MonomialSemanticTransport.identity(2)
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))

    correct = _hypothesis(
        "correct",
        (
            _factor("producer", identity, "correct/producer"),
            _factor("controller", identity, "correct/controller"),
        ),
    )
    masked = _hypothesis(
        "double-swap",
        (
            _factor("producer", swap, "fault/producer-swap"),
            _factor("controller", swap, "fault/controller-swap"),
        ),
    )

    plan = synthesize_minimal_diagnostic_taps((correct, masked))

    assert plan.complete
    assert plan.externally_resolved_pairs == ()
    assert plan.selected_internal_taps == ("producer",)
    assert plan.total_internal_cost == pytest.approx(1.0)
    assert plan.internally_resolved_pairs == (("correct", "double-swap"),)
    witness = plan.pair_witnesses[0]
    assert witness.tap_after_factor == "producer"
    assert witness.basis_index == 0
    assert witness.left_observed == pytest.approx((1.0, 0.0))
    assert witness.right_observed == pytest.approx((0.0, 1.0))


def test_external_output_is_used_before_internal_taps():
    identity = MonomialSemanticTransport.identity(2)
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    flip = MonomialSemanticTransport((0, 1), (-1.0, 1.0))

    correct = _hypothesis(
        "correct",
        (
            _factor("producer", identity, "correct/p"),
            _factor("controller", identity, "correct/c"),
        ),
    )
    visible_swap = _hypothesis(
        "visible-swap",
        (
            _factor("producer", swap, "swap/p"),
            _factor("controller", identity, "swap/c"),
        ),
    )
    visible_flip = _hypothesis(
        "visible-flip",
        (
            _factor("producer", flip, "flip/p"),
            _factor("controller", identity, "flip/c"),
        ),
    )

    plan = synthesize_minimal_diagnostic_taps(
        (correct, visible_swap, visible_flip)
    )

    assert plan.complete
    assert plan.selected_internal_taps == ()
    assert set(plan.externally_resolved_pairs) == {
        ("correct", "visible-flip"),
        ("correct", "visible-swap"),
        ("visible-swap", "visible-flip"),
    }


def test_weighted_tap_cost_selects_cheaper_equivalent_boundary():
    identity = MonomialSemanticTransport.identity(1)
    scale_two = MonomialSemanticTransport((0,), (2.0,))
    scale_half = MonomialSemanticTransport((0,), (0.5,))

    correct = _hypothesis(
        "correct",
        (
            _factor("a", identity, "correct/a"),
            _factor("b", identity, "correct/b"),
            _factor("c", identity, "correct/c"),
        ),
    )
    masked = _hypothesis(
        "masked-scale",
        (
            _factor("a", scale_two, "fault/a"),
            _factor("b", identity, "fault/b"),
            _factor("c", scale_half, "fault/c"),
        ),
    )

    plan = synthesize_minimal_diagnostic_taps(
        (correct, masked),
        tap_costs={"a": 10.0, "b": 1.0},
    )

    assert plan.complete
    assert plan.selected_internal_taps == ("b",)
    assert plan.total_internal_cost == pytest.approx(1.0)


def test_identical_hypotheses_fail_closed_as_intrinsically_unidentifiable():
    identity = MonomialSemanticTransport.identity(2)

    left = _hypothesis(
        "left",
        (
            _factor("producer", identity, "left/p"),
            _factor("controller", identity, "left/c"),
        ),
    )
    right = _hypothesis(
        "right",
        (
            _factor("producer", identity, "right/p"),
            _factor("controller", identity, "right/c"),
        ),
    )

    plan = synthesize_minimal_diagnostic_taps((left, right))

    assert not plan.complete
    assert plan.selected_internal_taps == ()
    assert plan.intrinsically_unidentifiable_pairs == (("left", "right"),)


def test_diagnostic_plan_digest_binds_evidence_identity_and_cost_model():
    identity = MonomialSemanticTransport.identity(1)
    scale_two = MonomialSemanticTransport((0,), (2.0,))
    scale_half = MonomialSemanticTransport((0,), (0.5,))

    correct = _hypothesis(
        "correct",
        (
            _factor("a", identity, "correct/a"),
            _factor("b", identity, "correct/b"),
        ),
    )
    first_fault = _hypothesis(
        "masked",
        (
            _factor("a", scale_two, "fault/source-v1"),
            _factor("b", scale_half, "fault/b"),
        ),
    )
    second_fault = _hypothesis(
        "masked",
        (
            _factor("a", scale_two, "fault/source-v2"),
            _factor("b", scale_half, "fault/b"),
        ),
    )

    first = synthesize_minimal_diagnostic_taps((correct, first_fault))
    second = synthesize_minimal_diagnostic_taps((correct, second_fault))
    expensive = synthesize_minimal_diagnostic_taps(
        (correct, first_fault),
        tap_costs={"a": 3.0},
    )

    assert first.digest != second.digest
    assert first.digest != expensive.digest


def test_inconsistent_factor_names_are_rejected():
    identity = MonomialSemanticTransport.identity(1)
    left = _hypothesis(
        "left",
        (
            _factor("a", identity, "left/a"),
            _factor("b", identity, "left/b"),
        ),
    )
    right = _hypothesis(
        "right",
        (
            _factor("a", identity, "right/a"),
            _factor("different", identity, "right/b"),
        ),
    )

    with pytest.raises(ValueError, match="same ordered factor names"):
        synthesize_minimal_diagnostic_taps((left, right))



def test_final_boundary_can_be_selected_when_external_output_is_unavailable():
    identity = MonomialSemanticTransport.identity(1)
    scale_two = MonomialSemanticTransport((0,), (2.0,))

    correct = _hypothesis(
        "correct",
        (_factor("only-boundary", identity, "correct/evidence"),),
    )
    faulty = _hypothesis(
        "faulty",
        (_factor("only-boundary", scale_two, "fault/evidence"),),
    )

    plan = synthesize_minimal_diagnostic_taps(
        (correct, faulty),
        external_output_observed=False,
    )

    assert plan.complete
    assert plan.externally_resolved_pairs == ()
    assert plan.selected_internal_taps == ("only-boundary",)
    assert plan.internally_resolved_pairs == (("correct", "faulty"),)
