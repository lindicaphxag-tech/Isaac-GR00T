from dataclasses import replace

from validation_semrepair.embodied_semantic_experiment_design import (
    SemanticExperiment,
    build_transport_experiments,
    solve_optimal_semantic_diagnosis,
)
from validation_semrepair.embodied_semantic_identification_repair import (
    authorize_factorwise_inverse_bundle,
    execute_diagnosis_policy,
    verify_factorwise_bundle,
)
from validation_semrepair.embodied_semantic_observability import (
    SemanticDiagnosisHypothesis,
)
from validation_semrepair.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
)


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


def _problem():
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
    return hypotheses, experiments, result


def test_masked_fault_identification_authorizes_both_local_repairs():
    hypotheses, experiments, result = _problem()
    hidden = "masked-double-swap"

    execution = execute_diagnosis_policy(
        result,
        experiments,
        observe=lambda experiment: experiment.outcome_for(hidden),
    )
    authorization = authorize_factorwise_inverse_bundle(execution, hypotheses)

    assert execution.status == "identified"
    assert execution.identified_hypothesis == hidden
    assert len(execution.steps) == 2
    assert execution.steps[0].experiment.startswith("external::")
    assert execution.steps[1].experiment.startswith("tap:producer::")

    assert authorization is not None
    assert authorization.status == "authorized"
    assert authorization.repair_count == 2
    assert {item.factor_name for item in authorization.repairs} == {
        "producer",
        "controller",
    }

    hypothesis = next(item for item in hypotheses if item.name == hidden)
    assert verify_factorwise_bundle(authorization, hypothesis)


def test_visible_single_fault_authorizes_only_its_local_inverse():
    hypotheses, experiments, result = _problem()
    hidden = "visible-swap"

    execution = execute_diagnosis_policy(
        result,
        experiments,
        observe=lambda experiment: experiment.outcome_for(hidden),
    )
    authorization = authorize_factorwise_inverse_bundle(execution, hypotheses)

    assert execution.status == "identified"
    assert len(execution.steps) == 1
    assert authorization is not None
    assert authorization.repair_count == 1
    assert authorization.repairs[0].factor_name == "producer"

    hypothesis = next(item for item in hypotheses if item.name == hidden)
    assert verify_factorwise_bundle(authorization, hypothesis)


def test_correct_hypothesis_authorizes_no_repair():
    hypotheses, experiments, result = _problem()

    execution = execute_diagnosis_policy(
        result,
        experiments,
        observe=lambda experiment: experiment.outcome_for("correct"),
    )
    authorization = authorize_factorwise_inverse_bundle(execution, hypotheses)

    assert execution.status == "identified"
    assert authorization is not None
    assert authorization.status == "no_repair_needed"
    assert authorization.repair_count == 0

    hypothesis = next(item for item in hypotheses if item.name == "correct")
    assert verify_factorwise_bundle(authorization, hypothesis)


def test_unseen_observation_fails_closed_before_repair_authority():
    hypotheses, experiments, result = _problem()

    execution = execute_diagnosis_policy(
        result,
        experiments,
        observe=lambda experiment: ("unseen", experiment.name),
    )
    authorization = authorize_factorwise_inverse_bundle(execution, hypotheses)

    assert execution.status == "refused"
    assert execution.identified_hypothesis is None
    assert "outside the declared error bound" in execution.reason
    assert authorization is None


def test_ambiguous_problem_never_authorizes_repair():
    experiments = (
        SemanticExperiment(
            "uninformative",
            outcomes=(("a", 0), ("b", 0)),
        ),
    )
    result = solve_optimal_semantic_diagnosis(("a", "b"), experiments)

    execution = execute_diagnosis_policy(
        result,
        experiments,
        observe=lambda experiment: 0,
    )

    identity = MonomialSemanticTransport.identity(1)
    hypotheses = (
        SemanticDiagnosisHypothesis(
            "a",
            (_factor("boundary", identity, "a/evidence"),),
        ),
        SemanticDiagnosisHypothesis(
            "b",
            (_factor("boundary", identity, "b/evidence"),),
        ),
    )

    assert result.status == "unidentifiable"
    assert execution.status == "refused"
    assert authorize_factorwise_inverse_bundle(execution, hypotheses) is None


def test_bundle_verification_binds_source_evidence_identity():
    hypotheses, experiments, result = _problem()
    hidden = "visible-swap"
    execution = execute_diagnosis_policy(
        result,
        experiments,
        observe=lambda experiment: experiment.outcome_for(hidden),
    )
    authorization = authorize_factorwise_inverse_bundle(execution, hypotheses)
    assert authorization is not None

    original = next(item for item in hypotheses if item.name == hidden)
    changed = SemanticDiagnosisHypothesis(
        name=original.name,
        factors=(
            replace(
                original.factors[0],
                evidence_id="different/source/evidence",
            ),
            original.factors[1],
        ),
    )

    assert verify_factorwise_bundle(authorization, original)
    assert not verify_factorwise_bundle(authorization, changed)


def test_numeric_observation_within_declared_error_bound_still_identifies():
    experiments = (
        SemanticExperiment(
            "numeric",
            outcomes=(("a", (0.05, -0.1)), ("b", (0.8, 0.7))),
            observation_atol=1.0e-6,
        ),
    )
    result = solve_optimal_semantic_diagnosis(("a", "b"), experiments)

    execution = execute_diagnosis_policy(
        result,
        experiments,
        observe=lambda experiment: (
            0.050000000745,
            -0.10000000149,
        ),
    )

    assert execution.status == "identified"
    assert execution.identified_hypothesis == "a"


def test_numeric_observation_outside_declared_bound_refuses_authority():
    experiments = (
        SemanticExperiment(
            "numeric",
            outcomes=(("a", (0.0,)), ("b", (1.0,))),
            observation_atol=0.1,
        ),
    )
    result = solve_optimal_semantic_diagnosis(("a", "b"), experiments)

    execution = execute_diagnosis_policy(
        result,
        experiments,
        observe=lambda experiment: (2.0,),
    )

    assert execution.status == "refused"
    assert execution.identified_hypothesis is None
    assert "outside the declared error bound" in execution.reason
