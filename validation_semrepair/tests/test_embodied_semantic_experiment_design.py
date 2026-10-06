from dataclasses import replace

import pytest

from validation_semrepair.embodied_semantic_experiment_design import (
    DiagnosisDecision,
    SemanticExperiment,
    build_transport_experiments,
    solve_optimal_semantic_diagnosis,
    verify_optimal_semantic_diagnosis,
)
from validation_semrepair.embodied_semantic_observability import (
    SemanticDiagnosisHypothesis,
)
from validation_semrepair.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
)


def _experiment(name, outcomes, *, cost=1.0, risk=0.0):
    return SemanticExperiment(
        name=name,
        outcomes=tuple(outcomes.items()),
        cost=cost,
        risk=risk,
    )


def test_exact_expected_cost_prefers_adaptive_partition():
    hypotheses = ("h0", "h1", "h2", "h3")
    experiments = (
        _experiment(
            "broad-first",
            {"h0": "A", "h1": "A", "h2": "B", "h3": "C"},
        ),
        _experiment(
            "pair-refiner",
            {"h0": 0, "h1": 1, "h2": 0, "h3": 0},
        ),
    )

    result = solve_optimal_semantic_diagnosis(
        hypotheses,
        experiments,
        objective="uniform_expected",
    )

    assert result.complete
    assert result.optimal_cost == pytest.approx(1.5)
    assert isinstance(result.policy, DiagnosisDecision)
    assert result.policy.experiment == "broad-first"

    verification = verify_optimal_semantic_diagnosis(result, experiments)
    assert verification.valid
    assert verification.policy_cost == pytest.approx(1.5)
    assert verification.independently_optimal_cost == pytest.approx(1.5)


def test_worst_case_cost_is_exact_over_frozen_experiment_table():
    hypotheses = ("h0", "h1", "h2", "h3")
    experiments = (
        _experiment(
            "broad-first",
            {"h0": "A", "h1": "A", "h2": "B", "h3": "C"},
        ),
        _experiment(
            "pair-refiner",
            {"h0": 0, "h1": 1, "h2": 0, "h3": 0},
        ),
    )

    result = solve_optimal_semantic_diagnosis(
        hypotheses,
        experiments,
        objective="worst_case",
    )

    assert result.complete
    assert result.optimal_cost == pytest.approx(2.0)
    assert verify_optimal_semantic_diagnosis(result, experiments).valid


def test_risk_gate_can_make_problem_intrinsically_unidentifiable():
    hypotheses = ("safe", "fault")
    experiments = (
        _experiment(
            "safe-but-useless",
            {"safe": "same", "fault": "same"},
            risk=0.0,
        ),
        _experiment(
            "risky-discriminator",
            {"safe": "ok", "fault": "bad"},
            risk=1.0,
        ),
    )

    blocked = solve_optimal_semantic_diagnosis(
        hypotheses,
        experiments,
        max_risk=0.0,
    )
    allowed = solve_optimal_semantic_diagnosis(
        hypotheses,
        experiments,
        max_risk=1.0,
    )

    assert blocked.status == "unidentifiable"
    assert blocked.unresolved_groups == (("fault", "safe"),)
    assert verify_optimal_semantic_diagnosis(blocked, experiments).valid

    assert allowed.complete
    assert allowed.optimal_cost == pytest.approx(2.0)
    assert verify_optimal_semantic_diagnosis(allowed, experiments).valid


def test_digest_tampering_is_rejected():
    hypotheses = ("a", "b")
    experiments = (
        _experiment("separate", {"a": 0, "b": 1}),
    )
    result = solve_optimal_semantic_diagnosis(hypotheses, experiments)
    tampered = replace(result, digest="0" * 64)

    verification = verify_optimal_semantic_diagnosis(tampered, experiments)

    assert not verification.valid
    assert not verification.digest_matches


def _factor(name, transport, evidence):
    return SemanticTransportFactor(name, transport, evidence)


def _hypothesis(name, producer, controller):
    return SemanticDiagnosisHypothesis(
        name=name,
        factors=(
            _factor("producer", producer, f"{name}/producer"),
            _factor("controller", controller, f"{name}/controller"),
        ),
    )


def test_joint_probe_tap_policy_opens_internal_tap_only_for_masked_branch():
    identity = MonomialSemanticTransport.identity(2)
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    flip_first = MonomialSemanticTransport((0, 1), (-1.0, 1.0))

    hypotheses = (
        _hypothesis("correct", identity, identity),
        _hypothesis("masked-double-swap", swap, swap),
        _hypothesis("visible-swap", swap, identity),
        _hypothesis("visible-flip", flip_first, identity),
    )
    experiments = build_transport_experiments(
        hypotheses,
        probe_cost=1.0,
        tap_costs={"producer": 0.0},
    )

    result = solve_optimal_semantic_diagnosis(
        tuple(item.name for item in hypotheses),
        experiments,
        objective="uniform_expected",
    )

    assert result.complete
    assert result.optimal_cost == pytest.approx(1.5)
    assert isinstance(result.policy, DiagnosisDecision)
    assert result.policy.experiment == "external::basis[0]"

    ambiguous_branch = next(
        branch
        for branch in result.policy.branches
        if isinstance(branch.child, DiagnosisDecision)
    )
    assert set(ambiguous_branch.child.hypotheses) == {
        "correct",
        "masked-double-swap",
    }
    assert ambiguous_branch.child.experiment == "tap:producer::basis[0]"

    verification = verify_optimal_semantic_diagnosis(result, experiments)
    assert verification.valid


def test_identical_semantic_hypotheses_fail_closed():
    hypotheses = ("alias-a", "alias-b")
    experiments = (
        _experiment(
            "probe-0",
            {"alias-a": (1.0, 0.0), "alias-b": (1.0, 0.0)},
        ),
        _experiment(
            "probe-1",
            {"alias-a": (0.0, 1.0), "alias-b": (0.0, 1.0)},
        ),
    )

    result = solve_optimal_semantic_diagnosis(hypotheses, experiments)

    assert result.status == "unidentifiable"
    assert result.policy is None
    assert result.optimal_cost is None
    assert result.unresolved_groups == (("alias-a", "alias-b"),)
    assert verify_optimal_semantic_diagnosis(result, experiments).valid


def test_experiment_domain_mismatch_is_rejected():
    with pytest.raises(ValueError, match="hypothesis domain mismatch"):
        solve_optimal_semantic_diagnosis(
            ("a", "b"),
            (
                SemanticExperiment(
                    "broken",
                    outcomes=(("a", 0),),
                ),
            ),
        )


def test_greedy_information_gain_can_be_suboptimal_for_total_diagnosis_cost():
    hypotheses = ("h0", "h1", "h2", "h3", "h4")
    experiments = (
        _experiment(
            "e0",
            {"h0": 0, "h1": 1, "h2": 1, "h3": 1, "h4": 0},
            cost=3.0,
        ),
        _experiment(
            "e1",
            {"h0": 0, "h1": 1, "h2": 2, "h3": 2, "h4": 2},
            cost=3.0,
        ),
        _experiment(
            "e2",
            {"h0": 0, "h1": 1, "h2": 2, "h3": 1, "h4": 0},
            cost=3.0,
        ),
        _experiment(
            "e3",
            {"h0": 0, "h1": 1, "h2": 0, "h3": 1, "h4": 1},
            cost=3.0,
        ),
    )

    result = solve_optimal_semantic_diagnosis(
        hypotheses,
        experiments,
        objective="uniform_expected",
    )

    assert result.complete
    assert result.optimal_cost == pytest.approx(4.8)
    assert isinstance(result.policy, DiagnosisDecision)
    assert result.policy.experiment == "e1"

    # Greedy one-step information gain prefers e2:
    # IG(e2)=1.521928... bits > IG(e1)=1.370950... bits.
    # Following that greedy root yields expected total cost 5.4, so the
    # locally most informative probe is not globally cost-optimal.
    assert result.optimal_cost < 5.4
