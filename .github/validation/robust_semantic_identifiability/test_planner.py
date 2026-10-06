from planner import (
    DiagnosisDecision,
    DiagnosisLeaf,
    JointSemanticExperiment,
    SemanticHypothesis,
    collect_plan_experiments,
    compatible_hypotheses_for_observation,
    observational_equivalence_classes,
    synthesize_safe_semantic_experiment_plan,
)


def hs(*names):
    return [SemanticHypothesis(name) for name in names]


def test_internal_tap_breaks_compensating_black_box_equivalence():
    hypotheses = hs("healthy", "double_fault")
    experiments = [
        JointSemanticExperiment(
            name="external_noncommuting_rotation",
            probe="Rx(0.3)Ry(-0.2)",
            tap="external_output",
            outcomes={
                "healthy": "same_pose",
                "double_fault": "same_pose",
            },
            risk=0.0,
        ),
        JointSemanticExperiment(
            name="converter_boundary_rotation",
            probe="Rx(0.3)Ry(-0.2)",
            tap="after_converter",
            outcomes={
                "healthy": "xyz_euler",
                "double_fault": "axis_angle",
            },
            tap_cost=0.25,
            risk=0.0,
        ),
    ]

    plan = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        max_total_risk=0.0,
    )

    assert plan.status == "identified"
    assert isinstance(plan.root, DiagnosisDecision)
    assert plan.root.experiment == "converter_boundary_rotation"
    assert plan.metrics.worst_case_cost == 1.25
    assert plan.metrics.worst_case_risk == 0.0


def test_external_only_compensating_faults_return_impossibility_certificate():
    hypotheses = hs("healthy", "double_fault")
    experiments = [
        JointSemanticExperiment(
            name="external_probe_a",
            probe="basis_x",
            tap="external_output",
            outcomes={"healthy": "0", "double_fault": "0"},
        ),
        JointSemanticExperiment(
            name="external_probe_b",
            probe="basis_y",
            tap="external_output",
            outcomes={"healthy": "1", "double_fault": "1"},
        ),
    ]

    assert observational_equivalence_classes(hypotheses, experiments) == (
        ("double_fault", "healthy"),
    )
    plan = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        max_total_risk=0.0,
    )
    assert plan.status == "unidentifiable"
    assert isinstance(plan.root, DiagnosisLeaf)
    assert plan.root.reason == "intrinsic_observational_equivalence"


def test_risk_budget_selects_safe_two_stage_plan_over_risky_one_shot():
    hypotheses = hs("h0", "h1", "h2", "h3")
    experiments = [
        JointSemanticExperiment(
            name="risky_one_shot",
            probe="large_physical_excitation",
            tap="external_output",
            outcomes={"h0": "0", "h1": "1", "h2": "2", "h3": "3"},
            probe_cost=0.5,
            risk=0.8,
        ),
        JointSemanticExperiment(
            name="safe_partition",
            probe="small_basis_probe",
            tap="external_output",
            outcomes={"h0": "left", "h1": "left", "h2": "right", "h3": "right"},
            probe_cost=1.0,
            risk=0.1,
        ),
        JointSemanticExperiment(
            name="safe_left",
            probe="small_composed_probe",
            tap="after_boundary_a",
            outcomes={"h0": "0", "h1": "1", "h2": "x", "h3": "x"},
            probe_cost=1.0,
            risk=0.1,
        ),
        JointSemanticExperiment(
            name="safe_right",
            probe="small_composed_probe",
            tap="after_boundary_b",
            outcomes={"h0": "x", "h1": "x", "h2": "2", "h3": "3"},
            probe_cost=1.0,
            risk=0.1,
        ),
    ]

    safe = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        max_total_risk=0.2,
    )
    assert safe.status == "identified"
    assert safe.metrics.worst_case_risk == 0.2
    assert safe.metrics.worst_case_cost == 2.0
    assert "risky_one_shot" not in collect_plan_experiments(safe.root)

    too_tight = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        max_total_risk=0.1,
    )
    assert too_tight.status == "budget_limited"

    relaxed = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        max_total_risk=0.8,
    )
    assert relaxed.status == "identified"
    assert isinstance(relaxed.root, DiagnosisDecision)
    assert relaxed.root.experiment == "risky_one_shot"
    assert relaxed.metrics.worst_case_cost == 0.5


def test_adaptive_tree_uses_fewer_worst_case_measurements_than_static_cover():
    hypotheses = hs("h0", "h1", "h2", "h3")
    experiments = [
        JointSemanticExperiment(
            name="coarse",
            probe="coarse_probe",
            tap="output",
            outcomes={"h0": "L", "h1": "L", "h2": "R", "h3": "R"},
        ),
        JointSemanticExperiment(
            name="left_discriminator",
            probe="left_probe",
            tap="tap_left",
            outcomes={"h0": "0", "h1": "1", "h2": "x", "h3": "x"},
        ),
        JointSemanticExperiment(
            name="right_discriminator",
            probe="right_probe",
            tap="tap_right",
            outcomes={"h0": "x", "h1": "x", "h2": "2", "h3": "3"},
        ),
    ]
    plan = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        max_total_risk=0.0,
    )
    assert plan.status == "identified"
    assert plan.metrics.worst_case_depth == 2
    assert plan.metrics.worst_case_cost == 2.0
    # Any non-adaptive complete cover needs all three experiments.
    assert len(collect_plan_experiments(plan.root)) == 3


def test_joint_probe_tap_cost_can_change_optimal_root():
    hypotheses = hs("healthy", "frame_fault", "unit_fault")
    experiments = [
        JointSemanticExperiment(
            name="cheap_external",
            probe="basis_probe",
            tap="output",
            outcomes={
                "healthy": "ok",
                "frame_fault": "bad",
                "unit_fault": "bad",
            },
            probe_cost=0.5,
        ),
        JointSemanticExperiment(
            name="cheap_internal_after_external",
            probe="basis_probe",
            tap="after_scaler",
            outcomes={
                "healthy": "same",
                "frame_fault": "frame",
                "unit_fault": "unit",
            },
            probe_cost=0.5,
            tap_cost=0.25,
        ),
        JointSemanticExperiment(
            name="expensive_direct_internal",
            probe="identity",
            tap="after_scaler",
            outcomes={
                "healthy": "healthy",
                "frame_fault": "frame",
                "unit_fault": "unit",
            },
            probe_cost=0.1,
            tap_cost=2.0,
        ),
    ]
    plan = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        max_total_risk=0.0,
    )
    assert plan.status == "identified"
    assert isinstance(plan.root, DiagnosisDecision)
    assert plan.root.experiment == "cheap_internal_after_external"
    assert plan.metrics.worst_case_cost == 0.75


def test_digest_binds_experiment_semantics_and_risk_budget():
    hypotheses = hs("a", "b")
    base = [
        JointSemanticExperiment(
            name="probe",
            probe="u",
            tap="output",
            outcomes={"a": "0", "b": "1"},
            risk=0.1,
        )
    ]
    plan_a = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        base,
        max_total_risk=0.1,
    )
    plan_b = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        [
            JointSemanticExperiment(
                name="probe",
                probe="u",
                tap="output",
                outcomes={"a": "0", "b": "1"},
                risk=0.2,
            )
        ],
        max_total_risk=0.2,
    )
    assert plan_a.digest != plan_b.digest


def test_incomplete_outcome_domain_fails_closed():
    hypotheses = hs("a", "b")
    experiments = [
        JointSemanticExperiment(
            name="broken",
            probe="u",
            tap="output",
            outcomes={"a": "0"},
        )
    ]
    try:
        synthesize_safe_semantic_experiment_plan(hypotheses, experiments)
    except ValueError as exc:
        assert "outcome domain mismatch" in str(exc)
    else:
        raise AssertionError("missing hypothesis outcome must fail closed")


def test_claim_relative_evidence_blocks_informative_but_unauthorized_metric():
    hypotheses = hs("old_old", "converter_only", "controller_only", "composed")
    experiments = [
        JointSemanticExperiment(
            name="task_replay_success",
            probe="official_PegInsertionSide_replay",
            tap="task_success",
            outcomes={
                "old_old": "high",
                "converter_only": "low",
                "controller_only": "zero",
                "composed": "high",
            },
            probe_cost=0.1,
            evidence_for=("task_performance",),
        ),
        JointSemanticExperiment(
            name="paired_so3_controller_target_fidelity",
            probe="paired_official_demo_requests",
            tap="controller_target_orientation",
            outcomes={
                # Public ManiSkill run 37401814045, composed-generated corpus,
                # call-weighted mean rotation error (deg), rounded only for the
                # symbolic finite planner outcome.
                "old_old": "0.023800deg",
                "converter_only": "2.936281deg",
                "controller_only": "2.933591deg",
                "composed": "0.016668deg",
            },
            probe_cost=2.0,
            evidence_for=("rotation_semantics",),
        ),
    ]

    plan = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        max_total_risk=0.0,
        required_evidence="rotation_semantics",
    )

    assert plan.status == "identified"
    assert isinstance(plan.root, DiagnosisDecision)
    assert plan.root.experiment == "paired_so3_controller_target_fidelity"
    assert plan.authorized_experiments == (
        "paired_so3_controller_target_fidelity",
    )
    assert plan.required_evidence == "rotation_semantics"
    assert "task_replay_success" not in collect_plan_experiments(plan.root)


def test_claim_scope_limited_is_distinct_from_intrinsic_unidentifiability():
    hypotheses = hs("old_old", "composed")
    experiments = [
        JointSemanticExperiment(
            name="task_replay_success",
            probe="official_PegInsertionSide_replay",
            tap="task_success",
            outcomes={"old_old": "high", "composed": "high"},
            evidence_for=("task_performance",),
        ),
        JointSemanticExperiment(
            name="semantic_target_fidelity",
            probe="paired_demo_requests",
            tap="controller_target_orientation",
            outcomes={"old_old": "residual", "composed": "canonical"},
            evidence_for=("rotation_semantics",),
        ),
    ]

    unrestricted = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
    )
    assert unrestricted.status == "identified"

    wrong_claim = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        required_evidence="task_performance",
    )
    assert wrong_claim.status == "evidence_scope_limited"
    assert isinstance(wrong_claim.root, DiagnosisLeaf)
    assert wrong_claim.root.reason == "claim_evidence_scope_insufficient"

    right_claim = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        required_evidence="rotation_semantics",
    )
    assert right_claim.status == "identified"


def test_missing_authorized_evidence_fails_closed_as_scope_limited():
    hypotheses = hs("a", "b")
    experiments = [
        JointSemanticExperiment(
            name="informative_but_wrong_scope",
            probe="u",
            tap="output",
            outcomes={"a": "0", "b": "1"},
            evidence_for=("task_performance",),
        )
    ]
    plan = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        required_evidence="rotation_semantics",
    )
    assert plan.status == "evidence_scope_limited"
    assert plan.authorized_experiments == ()
    assert plan.authorized_equivalence_classes == (("a", "b"),)


def test_digest_binds_claim_authority():
    hypotheses = hs("a", "b")
    experiments = [
        JointSemanticExperiment(
            name="same_measurement",
            probe="u",
            tap="output",
            outcomes={"a": "0", "b": "1"},
            evidence_for=("task_performance", "rotation_semantics"),
        )
    ]
    task = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        required_evidence="task_performance",
    )
    rotation = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        required_evidence="rotation_semantics",
    )
    assert task.status == rotation.status == "identified"
    assert task.digest != rotation.digest


def test_real_maniskill_case_routes_each_claim_to_its_authorized_oracle():
    from maniskill_claim_authority_case import build_maniskill_claim_authority_case

    result = build_maniskill_claim_authority_case()

    assert result.rotation_plan.status == "identified"
    assert isinstance(result.rotation_plan.root, DiagnosisDecision)
    assert (
        result.rotation_plan.root.experiment
        == "paired_so3_controller_target_fidelity"
    )
    assert result.rotation_plan.authorized_experiments == (
        "paired_so3_controller_target_fidelity",
    )

    assert result.task_plan.status == "identified"
    assert isinstance(result.task_plan.root, DiagnosisDecision)
    assert result.task_plan.root.experiment == "official_demo_task_replay"
    assert result.task_plan.authorized_experiments == (
        "official_demo_task_replay",
    )


def test_overlapping_numeric_error_boxes_are_not_falsely_separated():
    hypotheses = hs("a", "b", "c")
    experiments = [
        JointSemanticExperiment(
            name="noisy_scalar",
            probe="u",
            tap="numeric_boundary",
            outcomes={"a": (0.0,), "b": (0.15,), "c": (1.0,)},
            observation_atol=0.1,
        )
    ]

    plan = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        max_total_risk=0.0,
    )

    assert plan.status == "unidentifiable"
    assert isinstance(plan.root, DiagnosisLeaf)
    assert plan.root.reason == "intrinsic_observational_equivalence"


def test_robustly_separated_numeric_boxes_remain_identifiable():
    hypotheses = hs("a", "b", "c")
    experiments = [
        JointSemanticExperiment(
            name="separated_scalar",
            probe="u",
            tap="numeric_boundary",
            outcomes={"a": (0.0,), "b": (1.0,), "c": (2.0,)},
            observation_atol=0.1,
        )
    ]

    plan = synthesize_safe_semantic_experiment_plan(
        hypotheses,
        experiments,
        max_total_risk=0.0,
    )

    assert plan.status == "identified"
    assert isinstance(plan.root, DiagnosisDecision)
    assert plan.root.experiment == "separated_scalar"


def test_runtime_observation_reports_zero_one_or_multiple_compatible_hypotheses():
    experiment = JointSemanticExperiment(
        name="bounded_vector",
        probe="u",
        tap="controller_target",
        outcomes={"a": (0.0, 0.0), "b": (0.15, 0.0), "c": (1.0, 1.0)},
        observation_atol=0.1,
    )

    # The overlap region is intentionally ambiguous.
    assert compatible_hypotheses_for_observation(
        experiment, (0.075, 0.0)
    ) == ("a", "b")

    # A point near c is unique.
    assert compatible_hypotheses_for_observation(
        experiment, (1.02, 0.98)
    ) == ("c",)

    # Out-of-model evidence fails closed instead of snapping to nearest.
    assert compatible_hypotheses_for_observation(
        experiment, (4.0, 4.0)
    ) == ()


def test_maniskill_429_production_float_witness_is_inside_declared_semantic_ball():
    experiment = JointSemanticExperiment(
        name="maniskill_429_destination_chart",
        probe="source_native_[0.5,-0.5]",
        tap="destination_controller_native_action",
        outcomes={
            "reencoded_target_chart": (0.05, -0.1),
            "unrelated_chart": (0.8, 0.7),
        },
        observation_atol=1.0e-6,
        evidence_for=("action_chart_semantics",),
    )

    observed = (0.050000000745, -0.10000000149)
    assert compatible_hypotheses_for_observation(
        experiment,
        observed,
    ) == ("reencoded_target_chart",)

    plan = synthesize_safe_semantic_experiment_plan(
        hs("reencoded_target_chart", "unrelated_chart"),
        [experiment],
        required_evidence="action_chart_semantics",
    )
    assert plan.status == "identified"


def test_observation_tolerance_is_certificate_bound():
    hypotheses = hs("a", "b")
    narrow = [
        JointSemanticExperiment(
            name="numeric",
            probe="u",
            tap="output",
            outcomes={"a": (0.0,), "b": (1.0,)},
            observation_atol=0.01,
        )
    ]
    wide = [
        JointSemanticExperiment(
            name="numeric",
            probe="u",
            tap="output",
            outcomes={"a": (0.0,), "b": (1.0,)},
            observation_atol=0.2,
        )
    ]

    narrow_plan = synthesize_safe_semantic_experiment_plan(
        hypotheses, narrow
    )
    wide_plan = synthesize_safe_semantic_experiment_plan(
        hypotheses, wide
    )

    assert narrow_plan.digest != wide_plan.digest


def test_mixed_numeric_and_symbolic_outcomes_fail_closed():
    try:
        JointSemanticExperiment(
            name="mixed",
            probe="u",
            tap="output",
            outcomes={"a": (0.0,), "b": "zero"},
            observation_atol=0.1,
        )
    except ValueError as exc:
        assert "cannot mix numeric and symbolic" in str(exc)
    else:
        raise AssertionError("mixed observation families must fail closed")
