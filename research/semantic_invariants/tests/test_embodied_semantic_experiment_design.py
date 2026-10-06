import pytest

from research.semantic_invariants.embodied_semantic_experiment_design import (
    SemanticExperiment,
    canonical_basis_experiments,
    synthesize_optimal_experiment_plan,
)
from research.semantic_invariants.embodied_semantic_observability import (
    SemanticDiagnosisHypothesis,
)
from research.semantic_invariants.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
)


def _factor(name, transport, evidence):
    return SemanticTransportFactor(name, transport, evidence)


def _hypothesis(name, factors):
    return SemanticDiagnosisHypothesis(name=name, factors=tuple(factors))


def test_joint_planner_uses_internal_tap_when_double_swap_is_black_box_hidden():
    identity = MonomialSemanticTransport.identity(2)
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))

    correct = _hypothesis(
        "correct",
        (
            _factor("producer", identity, "correct/p"),
            _factor("controller", identity, "correct/c"),
        ),
    )
    masked = _hypothesis(
        "double-swap",
        (
            _factor("producer", swap, "fault/p"),
            _factor("controller", swap, "fault/c"),
        ),
    )

    experiments = canonical_basis_experiments(
        dimension=2,
        taps=(None, "producer"),
        external_cost=1.0,
        internal_cost=3.0,
    )
    plan = synthesize_optimal_experiment_plan((correct, masked), experiments)

    assert plan.complete
    assert plan.root.experiment is not None
    assert plan.root.experiment.tap_after_factor == "producer"
    assert plan.root.experiment.probe in ((1.0, 0.0), (0.0, 1.0))
    assert plan.expected_total_cost == pytest.approx(3.0)


def test_adaptive_plan_uses_cheap_external_test_before_expensive_internal_tap():
    identity = MonomialSemanticTransport.identity(2)
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))

    correct = _hypothesis(
        "correct",
        (
            _factor("producer", identity, "correct/p"),
            _factor("controller", identity, "correct/c"),
        ),
    )
    visible = _hypothesis(
        "visible-swap",
        (
            _factor("producer", swap, "visible/p"),
            _factor("controller", identity, "visible/c"),
        ),
    )
    masked = _hypothesis(
        "masked-double-swap",
        (
            _factor("producer", swap, "masked/p"),
            _factor("controller", swap, "masked/c"),
        ),
    )

    experiments = (
        SemanticExperiment(
            name="cheap-output-e0",
            probe=(1.0, 0.0),
            tap_after_factor=None,
            cost=1.0,
        ),
        SemanticExperiment(
            name="expensive-producer-e0",
            probe=(1.0, 0.0),
            tap_after_factor="producer",
            cost=3.0,
        ),
    )
    plan = synthesize_optimal_experiment_plan(
        (correct, visible, masked),
        experiments,
    )

    assert plan.complete
    assert plan.root.experiment is not None
    assert plan.root.experiment.name == "cheap-output-e0"
    assert plan.expected_total_cost == pytest.approx(3.0)
    assert plan.worst_case_total_cost == pytest.approx(4.0)

    ambiguous_child = next(
        child for child in plan.root.children if len(child.hypotheses) == 2
    )
    assert ambiguous_child.experiment is not None
    assert ambiguous_child.experiment.name == "expensive-producer-e0"


def test_hard_risk_ceiling_removes_unsafe_but_cheaper_experiment():
    identity = MonomialSemanticTransport.identity(1)
    scale_two = MonomialSemanticTransport((0,), (2.0,))

    correct = _hypothesis(
        "correct",
        (_factor("boundary", identity, "correct"),),
    )
    faulty = _hypothesis(
        "scaled",
        (_factor("boundary", scale_two, "scaled"),),
    )

    unsafe = SemanticExperiment(
        name="large-external-probe",
        probe=(10.0,),
        tap_after_factor=None,
        cost=0.1,
        risk=10.0,
    )
    safe = SemanticExperiment(
        name="safe-external-probe",
        probe=(1.0,),
        tap_after_factor=None,
        cost=2.0,
        risk=0.1,
    )

    plan = synthesize_optimal_experiment_plan(
        (correct, faulty),
        (unsafe, safe),
        max_risk=1.0,
    )

    assert plan.complete
    assert plan.admissible_experiments == ("safe-external-probe",)
    assert plan.root.experiment is not None
    assert plan.root.experiment.name == "safe-external-probe"
    assert plan.expected_total_cost == pytest.approx(2.0)


def test_risk_penalty_changes_choice_even_when_both_experiments_are_admissible():
    identity = MonomialSemanticTransport.identity(1)
    scale_two = MonomialSemanticTransport((0,), (2.0,))

    correct = _hypothesis(
        "correct",
        (_factor("boundary", identity, "correct"),),
    )
    faulty = _hypothesis(
        "scaled",
        (_factor("boundary", scale_two, "scaled"),),
    )

    cheap_risky = SemanticExperiment(
        name="cheap-risky",
        probe=(1.0,),
        cost=0.1,
        risk=2.0,
    )
    costly_safe = SemanticExperiment(
        name="costly-safe",
        probe=(2.0,),
        cost=1.0,
        risk=0.0,
    )

    no_penalty = synthesize_optimal_experiment_plan(
        (correct, faulty),
        (cheap_risky, costly_safe),
        max_risk=2.0,
        risk_weight=0.0,
    )
    penalized = synthesize_optimal_experiment_plan(
        (correct, faulty),
        (cheap_risky, costly_safe),
        max_risk=2.0,
        risk_weight=1.0,
    )

    assert no_penalty.root.experiment is not None
    assert no_penalty.root.experiment.name == "cheap-risky"
    assert penalized.root.experiment is not None
    assert penalized.root.experiment.name == "costly-safe"


def test_intrinsically_equivalent_hypotheses_fail_closed():
    identity = MonomialSemanticTransport.identity(2)

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
            _factor("b", identity, "right/b"),
        ),
    )

    experiments = canonical_basis_experiments(
        dimension=2,
        taps=(None, "a"),
    )
    plan = synthesize_optimal_experiment_plan((left, right), experiments)

    assert not plan.complete
    assert plan.root.experiment is None
    assert plan.unresolved_equivalence_classes == (("left", "right"),)
    assert plan.expected_total_cost == pytest.approx(0.0)
    assert plan.worst_case_total_cost == pytest.approx(0.0)


def test_expected_and_worst_case_objectives_can_choose_different_roots():
    identity = MonomialSemanticTransport.identity(2)
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    flip0 = MonomialSemanticTransport((0, 1), (-1.0, 1.0))

    correct = _hypothesis(
        "correct",
        (
            _factor("p", identity, "c/p"),
            _factor("c", identity, "c/c"),
        ),
    )
    rare_visible = _hypothesis(
        "rare-visible",
        (
            _factor("p", flip0, "r/p"),
            _factor("c", identity, "r/c"),
        ),
    )
    common_masked = _hypothesis(
        "common-masked",
        (
            _factor("p", swap, "m/p"),
            _factor("c", swap, "m/c"),
        ),
    )

    experiments = (
        SemanticExperiment("external", (1.0, 0.0), None, cost=1.0),
        SemanticExperiment("internal", (1.0, 0.0), "p", cost=2.0),
        SemanticExperiment("external-e1", (0.0, 1.0), None, cost=1.0),
    )

    expected = synthesize_optimal_experiment_plan(
        (correct, rare_visible, common_masked),
        experiments,
        priors={"correct": 0.45, "rare-visible": 0.1, "common-masked": 0.45},
        objective="expected",
    )
    worst = synthesize_optimal_experiment_plan(
        (correct, rare_visible, common_masked),
        experiments,
        priors={"correct": 0.45, "rare-visible": 0.1, "common-masked": 0.45},
        objective="worst_case",
    )

    assert expected.complete
    assert worst.complete
    assert expected.root.experiment is not None
    assert worst.root.experiment is not None


def test_plan_digest_binds_prior_cost_risk_and_evidence_identity():
    identity = MonomialSemanticTransport.identity(1)
    scale_two = MonomialSemanticTransport((0,), (2.0,))

    correct = _hypothesis(
        "correct",
        (_factor("a", identity, "correct-v1"),),
    )
    faulty = _hypothesis(
        "faulty",
        (_factor("a", scale_two, "fault-v1"),),
    )
    changed_evidence = _hypothesis(
        "faulty",
        (_factor("a", scale_two, "fault-v2"),),
    )
    experiment = SemanticExperiment("probe", (1.0,), cost=1.0, risk=0.2)

    base = synthesize_optimal_experiment_plan(
        (correct, faulty),
        (experiment,),
    )
    changed = synthesize_optimal_experiment_plan(
        (correct, changed_evidence),
        (experiment,),
    )
    reweighted = synthesize_optimal_experiment_plan(
        (correct, faulty),
        (experiment,),
        priors={"correct": 9.0, "faulty": 1.0},
    )

    assert base.digest != changed.digest
    assert base.digest != reweighted.digest


def test_partially_identifiable_problem_recovers_exact_unresolved_class():
    identity = MonomialSemanticTransport.identity(1)
    scale_two = MonomialSemanticTransport((0,), (2.0,))

    correct_a = _hypothesis(
        "correct-a",
        (_factor("boundary", identity, "correct/a"),),
    )
    correct_b = _hypothesis(
        "correct-b",
        (_factor("boundary", identity, "correct/b"),),
    )
    scaled = _hypothesis(
        "scaled",
        (_factor("boundary", scale_two, "scaled"),),
    )

    plan = synthesize_optimal_experiment_plan(
        (correct_a, correct_b, scaled),
        (SemanticExperiment("output", (1.0,), cost=1.0),),
    )

    assert not plan.complete
    assert plan.root.experiment is not None
    assert plan.root.experiment.name == "output"
    assert plan.expected_total_cost == pytest.approx(1.0)
    assert plan.unresolved_equivalence_classes == (("correct-a", "correct-b"),)
    assert any(
        child.hypotheses == ("correct-a", "correct-b")
        and not child.complete
        and child.experiment is None
        for child in plan.root.children
    )
