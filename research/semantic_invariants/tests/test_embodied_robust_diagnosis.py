from dataclasses import replace

import pytest

from research.semantic_invariants.embodied_robust_diagnosis import (
    RobustDiagnosisRejected,
    RobustObservation,
    RobustProbe,
    RobustResolution,
    _feasible_beliefs,
    advance_robust_diagnosis,
    synthesize_robust_experiment_plan,
)
from research.semantic_invariants.embodied_semantic_experiment_design import (
    SemanticExperiment,
    _signature,
)
from research.semantic_invariants.embodied_semantic_observability import (
    SemanticDiagnosisHypothesis,
)
from research.semantic_invariants.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
)


def _hypotheses(scales):
    return tuple(
        SemanticDiagnosisHypothesis(
            name, (SemanticTransportFactor(
                "chart",
                MonomialSemanticTransport(tuple(range(len(v))), v),
                "source/{}".format(name),
            ),)
        ) for name, v in scales
    )


def test_rounding_boundary_is_not_a_certified_sensor_noise_separator():
    # These predictions differ by 0.002, less than 2*epsilon=0.2.
    # Naive deterministic rounding yields *different* buckets anyway.
    assert _signature((1.049,), atol=0.1) != _signature((1.051,), atol=0.1)
    hs = _hypotheses((("low", (1.049,)), ("high", (1.051,))))
    e = (SemanticExperiment("physical-probe", (1.0,), risk=0.1),)
    unsafe = synthesize_robust_experiment_plan(
        hs, e, epsilon=0.1, risk_budget=0.1, max_probe_risk=0.1,
    )
    assert not unsafe.complete
    assert unsafe.root is None
    with pytest.raises(RobustDiagnosisRejected, match="not identifiable"):
        advance_robust_diagnosis(plan=unsafe, trusted_plan_digest=unsafe.digest)

    # With a *genuinely* smaller validated error bound the same experiment
    # separates every in-bound observation.
    safe = synthesize_robust_experiment_plan(
        hs, e, epsilon=0.0009, risk_budget=0.1, max_probe_risk=0.1,
    )
    assert safe.complete
    decision = advance_robust_diagnosis(
        plan=safe, trusted_plan_digest=safe.digest
    )
    assert isinstance(decision, RobustProbe)
    result = advance_robust_diagnosis(
        plan=safe, trusted_plan_digest=safe.digest,
        observations=(RobustObservation("physical-probe", (1.0495,)),)
    )
    assert isinstance(result, RobustResolution)
    assert result.identified_hypothesis == "low"


def test_minimax_adaptive_diagnosis_is_risk_bounded_under_sensor_noise():
    hs = _hypotheses((
        ("correct", (1.0, 1.0)),
        ("x-fault", (2.0, 1.0)),
        ("y-fault", (1.0, 2.0)),
    ))
    es = (
        SemanticExperiment("e0", (1.0, 0.0), risk=0.6),
        SemanticExperiment("e1", (0.0, 1.0), risk=0.6),
    )
    denied = synthesize_robust_experiment_plan(
        hs, es, epsilon=0.1, risk_budget=1.0, max_probe_risk=0.6,
    )
    assert not denied.complete
    admitted = synthesize_robust_experiment_plan(
        hs, es, epsilon=0.1, risk_budget=1.2, max_probe_risk=0.6,
    )
    assert admitted.complete
    assert admitted.root.experiment is not None
    assert admitted.worst_path_risk == pytest.approx(1.2)
    assert admitted.digest != denied.digest

    first = advance_robust_diagnosis(
        plan=admitted, trusted_plan_digest=admitted.digest,
    )
    assert isinstance(first, RobustProbe)
    # Choose the exact model-consistent output for the correct mapping.
    predicted = dict(dict(admitted.predictions)[first.experiment.name])
    first_observation = RobustObservation(
        first.experiment.name, predicted["correct"]
    )
    second = advance_robust_diagnosis(
        plan=admitted, trusted_plan_digest=admitted.digest,
        observations=(first_observation,),
    )
    assert isinstance(second, RobustProbe)
    second_observation = RobustObservation(
        second.experiment.name,
        dict(dict(admitted.predictions)[second.experiment.name])["correct"],
    )
    resolution = advance_robust_diagnosis(
        plan=admitted, trusted_plan_digest=admitted.digest,
        observations=(first_observation, second_observation),
    )
    assert isinstance(resolution, RobustResolution)
    assert resolution.identified_hypothesis == "correct"
    assert resolution.spent_risk == pytest.approx(1.2)


def test_set_valued_observations_cover_nontransitive_overlap():
    beliefs = _feasible_beliefs(
        {"a": (1.0,), "b": (1.15,), "c": (1.3,)},
        0.1, cell_limit=100,
    )
    assert ("a", "b") in beliefs
    assert ("b", "c") in beliefs
    assert ("a", "b", "c") not in beliefs
    # No amount of repeating one ambiguous probe guarantees identification.
    hs = _hypotheses((
        ("a", (1.0,)), ("b", (1.15,)), ("c", (1.3,)),
    ))
    e = (SemanticExperiment("same-probe", (1.0,), risk=0.1),)
    plan = synthesize_robust_experiment_plan(
        hs, e, epsilon=0.1, risk_budget=1.0, max_probe_risk=0.1
    )
    assert not plan.complete


def test_unpredicted_noise_or_modified_plan_is_rejected():
    hs = _hypotheses((("low", (1.0,)), ("high", (2.0,))))
    e = (SemanticExperiment("probe", (1.0,), risk=0.1),)
    plan = synthesize_robust_experiment_plan(
        hs, e, epsilon=0.05, risk_budget=0.1, max_probe_risk=0.1,
    )
    assert plan.complete
    with pytest.raises(RobustDiagnosisRejected, match="error bound"):
        advance_robust_diagnosis(
            plan=plan, trusted_plan_digest=plan.digest,
            observations=(RobustObservation("probe", (10.0,)),),
        )
    with pytest.raises(RobustDiagnosisRejected, match="out-of-order"):
        advance_robust_diagnosis(
            plan=plan, trusted_plan_digest=plan.digest,
            observations=(RobustObservation("wrong-probe", (1.0,)),),
        )
    with pytest.raises(RobustDiagnosisRejected, match="trusted identity"):
        advance_robust_diagnosis(
            plan=replace(plan, epsilon=100.0),
            trusted_plan_digest=plan.digest,
        )
    with pytest.raises(RobustDiagnosisRejected, match="trusted identity"):
        advance_robust_diagnosis(
            plan=plan, trusted_plan_digest="0"*64,
        )


def test_risk_validation_and_exhaustive_cell_limit_fail_closed():
    hs = _hypotheses((("low", (1.0, 1.0)), ("high", (2.0, 2.0))))
    e = (SemanticExperiment("probe", (1.0, 1.0), risk=0.5),)
    with pytest.raises(ValueError, match="cell limit"):
        synthesize_robust_experiment_plan(
            hs, e, epsilon=0.2, risk_budget=1.0, max_probe_risk=0.5,
            cell_limit=1,
        )
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="finite/nonnegative"):
            synthesize_robust_experiment_plan(
                hs, e, epsilon=bad, risk_budget=1.0, max_probe_risk=0.5,
            )
    assert not synthesize_robust_experiment_plan(
        hs, e, epsilon=0.0, risk_budget=0.49, max_probe_risk=0.5,
    ).complete
