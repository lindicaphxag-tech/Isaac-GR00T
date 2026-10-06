import pytest

from research.semantic_invariants.embodied_repair_interactions import (
    QualifiedRepairAuthorizationPlane,
    RepairAuthorizationPlane,
    RepairOutcome,
    analyze_repair_lattice,
    authorize_repair_subset,
    authorize_repair_subset_across_planes,
    authorize_repair_subset_across_qualified_planes,
)

from research.semantic_invariants.embodied_measurement_qualification import (
    MeasurementWorld,
    qualify_measurement,
)


def _out(repairs, value, evidence):
    return RepairOutcome(frozenset(repairs), value, evidence)


def test_maniskill_converter_controller_pair_exposes_strict_compensation():
    # Independent non-commuting rotation witness for:
    # A = #1472 controller sign repair
    # B = #1495 converter representation repair
    cert = analyze_repair_lattice(
        subject="mani-skill delta-pose converter -> PDEEPoseController",
        metric="orientation_error_deg",
        objective="minimize",
        repairs=("pr1472-controller-sign", "pr1495-xyz-euler"),
        outcomes=(
            _out((), 5.076696288574988, "baseline/noncommuting-rotation"),
            _out(
                ("pr1472-controller-sign",),
                64.74733765443055,
                "factorial/pr1472-only",
            ),
            _out(
                ("pr1495-xyz-euler",),
                66.12799929304104,
                "factorial/pr1495-only",
            ),
            _out(
                ("pr1472-controller-sign", "pr1495-xyz-euler"),
                0.0,
                "factorial/both",
            ),
        ),
    )

    assert cert.has_repair_paradox
    assert len(cert.compensating_bundles) == 1
    bundle = cert.compensating_bundles[0]
    assert bundle.repairs == (
        "pr1472-controller-sign",
        "pr1495-xyz-euler",
    )
    assert bundle.closure_gain == pytest.approx(5.076696288574988)
    assert bundle.masking_mode == "partial"
    assert not bundle.behaviorally_masked

    # Both singleton repairs look catastrophically worse if reviewed only
    # against the old end-to-end baseline.
    regressions = {
        edge.added_repair: edge.regression
        for edge in cert.edge_violations
        if edge.before == ()
    }
    assert regressions["pr1472-controller-sign"] > 50
    assert regressions["pr1495-xyz-euler"] > 50

    # The pair interaction is strongly non-additive.
    assert cert.interaction_for(
        ("pr1472-controller-sign", "pr1495-xyz-euler")
    ) < -100


def test_independent_improvements_do_not_claim_compensating_bundle():
    cert = analyze_repair_lattice(
        subject="ordinary additive repair",
        metric="tracking_error",
        objective="minimize",
        repairs=("a", "b"),
        outcomes=(
            _out((), 10.0, "e0"),
            _out(("a",), 7.0, "ea"),
            _out(("b",), 8.0, "eb"),
            _out(("a", "b"), 5.0, "eab"),
        ),
    )

    assert not cert.has_repair_paradox
    assert cert.compensating_bundles == ()
    assert all(edge.before != () for edge in cert.edge_violations)


def test_maximize_metric_uses_same_semantics():
    cert = analyze_repair_lattice(
        subject="success metric",
        metric="task_success",
        objective="maximize",
        repairs=("a", "b"),
        outcomes=(
            _out((), 0.8, "e0"),
            _out(("a",), 0.2, "ea"),
            _out(("b",), 0.3, "eb"),
            _out(("a", "b"), 1.0, "eab"),
        ),
    )

    assert cert.has_repair_paradox
    assert cert.compensating_bundles[0].closure_gain == pytest.approx(0.2)


def test_incomplete_factorial_lattice_fails_closed():
    with pytest.raises(ValueError, match="complete"):
        analyze_repair_lattice(
            subject="missing-cell",
            metric="error",
            objective="minimize",
            repairs=("a", "b"),
            outcomes=(
                _out((), 1.0, "e0"),
                _out(("a",), 2.0, "ea"),
                _out(("a", "b"), 0.0, "eab"),
            ),
        )


def test_certificate_digest_binds_measurements_and_evidence_ids():
    common = dict(
        subject="digest",
        metric="error",
        objective="minimize",
        repairs=("a",),
    )
    first = analyze_repair_lattice(
        **common,
        outcomes=(
            _out((), 2.0, "baseline"),
            _out(("a",), 1.0, "repair-v1"),
        ),
    )
    second = analyze_repair_lattice(
        **common,
        outcomes=(
            _out((), 2.0, "baseline"),
            _out(("a",), 1.0, "repair-v2"),
        ),
    )

    assert first.digest != second.digest



def test_complete_behavioral_masking_is_detected_even_when_task_metric_passes():
    cert = analyze_repair_lattice(
        subject="paired joint-order faults",
        metric="final_tracking_error",
        objective="minimize",
        repairs=("producer-order", "dispatch-order"),
        outcomes=(
            _out((), 1.0e-7, "both-faults-but-cancelled"),
            _out(("producer-order",), 3.6, "producer-fixed-only"),
            _out(("dispatch-order",), 3.6, "dispatch-fixed-only"),
            _out(
                ("producer-order", "dispatch-order"),
                1.0e-7,
                "fully-correct",
            ),
        ),
        tolerance=1.0e-6,
    )

    assert cert.has_repair_paradox
    bundle = cert.compensating_bundles[0]
    assert bundle.masking_mode == "complete"
    assert bundle.behaviorally_masked
    assert abs(bundle.closure_gain) <= 1.0e-6
    assert len(
        [edge for edge in cert.edge_violations if edge.before == ()]
    ) == 2


def test_interaction_authorization_rejects_partial_compensating_repairs():
    cert = analyze_repair_lattice(
        subject="maniskill converter-controller interaction",
        metric="replay_success",
        objective="maximize",
        repairs=("controller-sign", "converter-representation"),
        outcomes=(
            _out((), 0.9, "main"),
            _out(("controller-sign",), 0.1, "controller-only"),
            _out(("converter-representation",), 0.1, "converter-only"),
            _out(
                ("controller-sign", "converter-representation"),
                0.9,
                "composed",
            ),
        ),
        tolerance=1.0e-9,
    )

    converter_only = authorize_repair_subset(
        cert,
        ("converter-representation",),
    )
    controller_only = authorize_repair_subset(
        cert,
        ("controller-sign",),
    )
    composed = authorize_repair_subset(
        cert,
        ("controller-sign", "converter-representation"),
    )

    assert not converter_only.authorized
    assert not controller_only.authorized
    assert composed.authorized
    assert converter_only.certificate_digest == cert.digest
    assert any("incomplete subset" in reason for reason in converter_only.reasons)
    assert any("regresses" in reason for reason in converter_only.reasons)


def test_interaction_authorization_rejects_unknown_repair():
    cert = analyze_repair_lattice(
        subject="known",
        metric="error",
        objective="minimize",
        repairs=("a",),
        outcomes=(
            _out((), 1.0, "baseline"),
            _out(("a",), 0.0, "fixed"),
        ),
    )

    with pytest.raises(ValueError, match="unknown repairs"):
        authorize_repair_subset(cert, ("not-in-certificate",))



def test_multi_plane_authorization_rejects_semantically_correct_execution_regression():
    semantic = analyze_repair_lattice(
        subject="maniskill converter-controller",
        metric="orientation_error_deg",
        objective="minimize",
        repairs=("controller-sign", "converter-representation"),
        outcomes=(
            _out((), 5.076696288574988, "semantic/main"),
            _out(("controller-sign",), 64.74733765443055, "semantic/controller"),
            _out(("converter-representation",), 66.12799929304104, "semantic/converter"),
            _out(
                ("controller-sign", "converter-representation"),
                0.0,
                "semantic/composed",
            ),
        ),
    )
    execution = analyze_repair_lattice(
        subject="maniskill converter-controller",
        metric="official_demo_replay_success",
        objective="maximize",
        repairs=("controller-sign", "converter-representation"),
        outcomes=(
            _out((), 0.9, "replay/main"),
            _out(("controller-sign",), 0.0, "replay/controller"),
            _out(("converter-representation",), 0.1, "replay/converter"),
            _out(
                ("controller-sign", "converter-representation"),
                0.8,
                "replay/composed",
            ),
        ),
    )

    semantic_only = authorize_repair_subset(
        semantic,
        ("controller-sign", "converter-representation"),
    )
    assert semantic_only.authorized

    joint = authorize_repair_subset_across_planes(
        (
            RepairAuthorizationPlane("semantic-fidelity", semantic),
            RepairAuthorizationPlane("execution-domain", execution),
        ),
        ("controller-sign", "converter-representation"),
    )

    assert not joint.authorized
    assert any(
        reason.startswith("execution-domain:") and "regresses" in reason
        for reason in joint.reasons
    )


def test_multi_plane_authorization_allows_bundle_only_when_every_plane_passes():
    semantic = analyze_repair_lattice(
        subject="pipeline",
        metric="semantic_error",
        objective="minimize",
        repairs=("a", "b"),
        outcomes=(
            _out((), 1.0, "semantic/base"),
            _out(("a",), 2.0, "semantic/a"),
            _out(("b",), 2.0, "semantic/b"),
            _out(("a", "b"), 0.0, "semantic/ab"),
        ),
    )
    execution = analyze_repair_lattice(
        subject="pipeline",
        metric="task_success",
        objective="maximize",
        repairs=("a", "b"),
        outcomes=(
            _out((), 0.9, "execution/base"),
            _out(("a",), 0.1, "execution/a"),
            _out(("b",), 0.2, "execution/b"),
            _out(("a", "b"), 0.9, "execution/ab"),
        ),
    )

    joint = authorize_repair_subset_across_planes(
        (
            RepairAuthorizationPlane("semantic-fidelity", semantic),
            RepairAuthorizationPlane("execution-domain", execution),
        ),
        ("a", "b"),
    )
    assert joint.authorized


def test_multi_plane_authorization_rejects_mismatched_repair_sets():
    left = analyze_repair_lattice(
        subject="left",
        metric="error",
        objective="minimize",
        repairs=("a",),
        outcomes=(
            _out((), 1.0, "l0"),
            _out(("a",), 0.0, "la"),
        ),
    )
    right = analyze_repair_lattice(
        subject="right",
        metric="error",
        objective="minimize",
        repairs=("a", "b"),
        outcomes=(
            _out((), 1.0, "r0"),
            _out(("a",), 0.8, "ra"),
            _out(("b",), 0.7, "rb"),
            _out(("a", "b"), 0.0, "rab"),
        ),
    )

    with pytest.raises(ValueError, match="same repair set"):
        authorize_repair_subset_across_planes(
            (
                RepairAuthorizationPlane("left", left),
                RepairAuthorizationPlane("right", right),
            ),
            ("a",),
        )



def test_qualified_planes_reject_repeatable_but_nonidentifying_evidence():
    interaction = analyze_repair_lattice(
        subject="lerobot relative-action mapping",
        metric="roundtrip_error",
        objective="minimize",
        repairs=("mapping-repair",),
        outcomes=(
            _out((), 0.0, "legacy-roundtrip"),
            _out(("mapping-repair",), 0.0, "repaired-roundtrip"),
        ),
    )
    measurement = qualify_measurement(
        measurement_id="lerobot/relative-action-roundtrip-v1",
        semantic_anchor_id=None,
        repeat_outcome_digests=("roundtrip:0",) * 32,
        worlds=(
            MeasurementWorld("correct", "roundtrip:0", "forward:correct"),
            MeasurementWorld("wrong-prefix", "roundtrip:0", "forward:wrong"),
        ),
    )

    result = authorize_repair_subset_across_qualified_planes(
        (
            QualifiedRepairAuthorizationPlane(
                "roundtrip-self-consistency",
                interaction,
                measurement,
            ),
        ),
        ("mapping-repair",),
    )

    assert not result.authorized
    assert result.plane_decisions == ()
    assert any("not qualified" in reason for reason in result.reasons)


def test_qualified_planes_preserve_execution_nonregression_gate():
    semantic = analyze_repair_lattice(
        subject="maniskill converter-controller",
        metric="orientation_error_deg",
        objective="minimize",
        repairs=("controller-sign", "converter-representation"),
        outcomes=(
            _out((), 0.02, "semantic/main"),
            _out(("controller-sign",), 2.7, "semantic/controller"),
            _out(("converter-representation",), 2.7, "semantic/converter"),
            _out(
                ("controller-sign", "converter-representation"),
                0.014,
                "semantic/composed",
            ),
        ),
    )
    execution = analyze_repair_lattice(
        subject="maniskill converter-controller",
        metric="official_demo_replay_success",
        objective="maximize",
        repairs=("controller-sign", "converter-representation"),
        outcomes=(
            _out((), 0.9, "replay/main"),
            _out(("controller-sign",), 0.0, "replay/controller"),
            _out(("converter-representation",), 0.1, "replay/converter"),
            _out(
                ("controller-sign", "converter-representation"),
                0.8,
                "replay/composed",
            ),
        ),
    )

    semantic_measurement = qualify_measurement(
        measurement_id="maniskill/paired-so3-fidelity-v2",
        semantic_anchor_id="controller-contract/desired-so3",
        repeat_outcome_digests=("paired-corpus-digest",) * 3,
    )
    execution_measurement = qualify_measurement(
        measurement_id="maniskill/official-demo-replay-v1",
        semantic_anchor_id="env/PegInsertionSide-v1/success",
        repeat_outcome_digests=("success-sets-exact",) * 5,
    )

    result = authorize_repair_subset_across_qualified_planes(
        (
            QualifiedRepairAuthorizationPlane(
                "semantic-fidelity",
                semantic,
                semantic_measurement,
            ),
            QualifiedRepairAuthorizationPlane(
                "execution-domain",
                execution,
                execution_measurement,
            ),
        ),
        ("controller-sign", "converter-representation"),
    )

    assert not result.authorized
    assert len(result.plane_decisions) == 2
    assert len(result.measurement_digests) == 2
    assert any(
        reason.startswith("execution-domain:") and "regresses" in reason
        for reason in result.reasons
    )
