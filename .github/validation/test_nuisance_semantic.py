from nuisance_semantic import (
    RobustDecision,
    RobustLeaf,
    RobustSemanticExperiment,
    SemanticWorld,
    synthesize_nuisance_robust_plan,
)


def worlds():
    return [
        SemanticWorld("correct", "bias_minus"),
        SemanticWorld("correct", "bias_plus"),
        SemanticWorld("sign_flip", "bias_minus"),
        SemanticWorld("sign_flip", "bias_plus"),
    ]


def test_persistent_nuisance_can_require_adaptive_second_probe():
    # y = s*u + b, with s in {+1,-1}, persistent b in {-1,+1}.
    # At u=+1, outcome 0 is semantically ambiguous:
    #   correct+bias_minus == sign_flip+bias_plus.
    # A second probe at u=-1 resolves the persistent nuisance branch.
    exps = [
        RobustSemanticExperiment(
            name="probe_pos",
            probe="u=+1",
            tap="external_output",
            outcomes={
                "correct::bias_minus": "0",
                "correct::bias_plus": "2",
                "sign_flip::bias_minus": "-2",
                "sign_flip::bias_plus": "0",
            },
            probe_cost=1.0,
        ),
        RobustSemanticExperiment(
            name="probe_neg",
            probe="u=-1",
            tap="external_output",
            outcomes={
                "correct::bias_minus": "-2",
                "correct::bias_plus": "0",
                "sign_flip::bias_minus": "0",
                "sign_flip::bias_plus": "2",
            },
            probe_cost=1.0,
        ),
    ]

    plan = synthesize_nuisance_robust_plan(worlds(), exps)

    assert plan.status == "identified"
    assert plan.nuisance_model == "persistent"
    assert plan.worst_case_depth == 2
    assert plan.worst_case_cost == 2.0
    assert isinstance(plan.root, RobustDecision)

    # Nuisance itself need not be identified. Some leaves can retain multiple
    # worlds so long as they all share one semantic identity.
    def leaves(node):
        if isinstance(node, RobustLeaf):
            return [node]
        out = []
        for branch in node.branches:
            out.extend(leaves(branch.child))
        return out

    assert all(len(leaf.semantics) == 1 for leaf in leaves(plan.root))


def test_unknown_gain_can_make_scale_semantics_intrinsically_unidentifiable():
    ws = [
        SemanticWorld("scale_1x", "gain_1"),
        SemanticWorld("scale_1x", "gain_2"),
        SemanticWorld("scale_2x", "gain_1"),
        SemanticWorld("scale_2x", "gain_2"),
    ]
    exps = [
        RobustSemanticExperiment(
            name="linear_probe_1",
            probe="u=1",
            tap="external_output",
            outcomes={
                "scale_1x::gain_1": "1",
                "scale_1x::gain_2": "2",
                "scale_2x::gain_1": "2",
                "scale_2x::gain_2": "4",
            },
        ),
        RobustSemanticExperiment(
            name="linear_probe_3",
            probe="u=3",
            tap="external_output",
            outcomes={
                "scale_1x::gain_1": "3",
                "scale_1x::gain_2": "6",
                "scale_2x::gain_1": "6",
                "scale_2x::gain_2": "12",
            },
        ),
    ]

    plan = synthesize_nuisance_robust_plan(ws, exps)

    assert plan.status == "nuisance_confounded"
    assert isinstance(plan.root, RobustLeaf)
    assert plan.root.reason == "persistent_nuisance_semantic_confounding"
    # scale_1x@gain_2 and scale_2x@gain_1 are exactly observationally
    # equivalent for every supplied external linear probe.
    assert (
        "scale_1x::gain_2",
        "scale_2x::gain_1",
    ) in plan.full_world_equivalence


def test_internal_semantic_tap_breaks_gain_confounding():
    ws = [
        SemanticWorld("scale_1x", "gain_1"),
        SemanticWorld("scale_1x", "gain_2"),
        SemanticWorld("scale_2x", "gain_1"),
        SemanticWorld("scale_2x", "gain_2"),
    ]
    exps = [
        RobustSemanticExperiment(
            name="external_linear_probe",
            probe="u=1",
            tap="external_output",
            outcomes={
                "scale_1x::gain_1": "1",
                "scale_1x::gain_2": "2",
                "scale_2x::gain_1": "2",
                "scale_2x::gain_2": "4",
            },
            probe_cost=0.1,
        ),
        RobustSemanticExperiment(
            name="pre_gain_semantic_tap",
            probe="u=1",
            tap="before_unknown_gain",
            outcomes={
                "scale_1x::gain_1": "1",
                "scale_1x::gain_2": "1",
                "scale_2x::gain_1": "2",
                "scale_2x::gain_2": "2",
            },
            probe_cost=0.1,
            tap_cost=0.5,
            evidence_for=("scale_semantics",),
        ),
    ]

    plan = synthesize_nuisance_robust_plan(
        ws,
        exps,
        required_evidence="scale_semantics",
    )

    assert plan.status == "identified"
    assert isinstance(plan.root, RobustDecision)
    assert plan.root.experiment == "pre_gain_semantic_tap"
    assert plan.authorized_experiments == ("pre_gain_semantic_tap",)


def test_claim_authority_remains_fail_closed_under_nuisance():
    ws = [
        SemanticWorld("correct", "easy_scene"),
        SemanticWorld("correct", "hard_scene"),
        SemanticWorld("wrong", "easy_scene"),
        SemanticWorld("wrong", "hard_scene"),
    ]
    exps = [
        RobustSemanticExperiment(
            name="task_success",
            probe="rollout",
            tap="episode_success",
            outcomes={
                "correct::easy_scene": "success",
                "correct::hard_scene": "fail",
                "wrong::easy_scene": "success",
                "wrong::hard_scene": "fail",
            },
            evidence_for=("task_performance",),
        ),
        RobustSemanticExperiment(
            name="semantic_boundary",
            probe="diagnostic_action",
            tap="controller_input",
            outcomes={
                "correct::easy_scene": "canonical",
                "correct::hard_scene": "canonical",
                "wrong::easy_scene": "noncanonical",
                "wrong::hard_scene": "noncanonical",
            },
            evidence_for=("action_semantics",),
        ),
    ]

    task = synthesize_nuisance_robust_plan(
        ws,
        exps,
        required_evidence="task_performance",
    )
    semantic = synthesize_nuisance_robust_plan(
        ws,
        exps,
        required_evidence="action_semantics",
    )

    assert task.status == "evidence_scope_limited"
    assert semantic.status == "identified"
    assert isinstance(semantic.root, RobustDecision)
    assert semantic.root.experiment == "semantic_boundary"


def test_digest_binds_persistent_nuisance_worlds():
    base_worlds = [
        SemanticWorld("a", "n0"),
        SemanticWorld("b", "n0"),
    ]
    exps = [
        RobustSemanticExperiment(
            name="e",
            probe="u",
            tap="out",
            outcomes={"a::n0": "0", "b::n0": "1"},
        )
    ]
    a = synthesize_nuisance_robust_plan(base_worlds, exps)

    expanded_worlds = [
        SemanticWorld("a", "n0"),
        SemanticWorld("a", "n1"),
        SemanticWorld("b", "n0"),
        SemanticWorld("b", "n1"),
    ]
    expanded_exps = [
        RobustSemanticExperiment(
            name="e",
            probe="u",
            tap="out",
            outcomes={
                "a::n0": "0",
                "a::n1": "0",
                "b::n0": "1",
                "b::n1": "1",
            },
        )
    ]
    b = synthesize_nuisance_robust_plan(expanded_worlds, expanded_exps)

    assert a.status == b.status == "identified"
    assert a.digest != b.digest
