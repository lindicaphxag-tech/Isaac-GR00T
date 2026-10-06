import pytest

from research.semantic_invariants.embodied_semantic_autorepair_bridge import (
    compile_diagnosis_to_repair_plan,
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


def test_ambiguous_diagnosis_never_triggers_repair_synthesis():
    identity = MonomialSemanticTransport.identity(2)
    left = _hypothesis(
        "left",
        (_factor("boundary", identity, "left/evidence"),),
    )
    right = _hypothesis(
        "right",
        (_factor("boundary", identity, "right/evidence"),),
    )

    plan = compile_diagnosis_to_repair_plan((left, right))

    assert plan.diagnosis_status == "ambiguous"
    assert plan.hypothesis_name is None
    assert plan.obligations == ()
    assert not plan.eligible_for_independent_verification
    assert "singleton semantic diagnosis" in plan.reasons[0]


def test_single_swap_synthesizes_unique_inverse_and_can_enter_verification():
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    hypothesis = _hypothesis(
        "swapped",
        (_factor("producer", swap, "producer/order"),),
    )

    plan = compile_diagnosis_to_repair_plan((hypothesis,))

    assert plan.diagnosis_status == "singleton"
    assert plan.nonidentity_boundaries == ("producer",)
    assert not plan.requires_interaction_assay
    assert plan.eligible_for_independent_verification

    obligation = plan.obligations[0]
    assert obligation.status == "unique"
    assert obligation.program_name == "permute(1, 0)"
    assert obligation.program_implementation_ids == (
        "semrepair/permute/(1, 0)@v1",
    )
    assert obligation.example_count == 2


def test_two_locally_unique_faults_still_require_factorial_interaction_assay():
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    hypothesis = _hypothesis(
        "double-swap",
        (
            _factor("producer", swap, "producer/order"),
            _factor("controller", swap, "controller/order"),
        ),
    )

    plan = compile_diagnosis_to_repair_plan((hypothesis,))

    assert plan.nonidentity_boundaries == ("producer", "controller")
    assert all(item.status == "unique" for item in plan.obligations)
    assert plan.requires_interaction_assay
    assert not plan.eligible_for_independent_verification
    assert any("factorial interaction assay" in reason for reason in plan.reasons)


def test_unsupported_scale_fails_closed_in_bounded_repair_dsl():
    scale_two = MonomialSemanticTransport((0,), (2.0,))
    hypothesis = _hypothesis(
        "scale-two",
        (_factor("unit", scale_two, "unit/evidence"),),
    )

    plan = compile_diagnosis_to_repair_plan(
        (hypothesis,),
        max_depth=1,
    )

    obligation = plan.obligations[0]
    assert obligation.status == "no_solution"
    assert obligation.program_name is None
    assert not plan.eligible_for_independent_verification


def test_identity_diagnosis_does_not_invent_a_repair():
    identity = MonomialSemanticTransport.identity(3)
    hypothesis = _hypothesis(
        "correct",
        (_factor("boundary", identity, "correct/evidence"),),
    )

    plan = compile_diagnosis_to_repair_plan((hypothesis,))

    assert plan.nonidentity_boundaries == ()
    assert plan.obligations[0].status == "identity"
    assert plan.obligations[0].program_name == "identity"
    assert not plan.eligible_for_independent_verification
    assert any("already semantically identity" in reason for reason in plan.reasons)


def test_bridge_digest_binds_evidence_identity():
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))
    first = _hypothesis(
        "swapped",
        (_factor("producer", swap, "source-v1"),),
    )
    second = _hypothesis(
        "swapped",
        (_factor("producer", swap, "source-v2"),),
    )

    first_plan = compile_diagnosis_to_repair_plan((first,))
    second_plan = compile_diagnosis_to_repair_plan((second,))

    assert first_plan.digest != second_plan.digest
