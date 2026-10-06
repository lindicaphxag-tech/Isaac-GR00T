from dataclasses import replace

import pytest

from validation_semrepair.embodied_semantic_experiment_design import (
    DiagnosisDecision,
    SemanticExperiment,
    solve_optimal_semantic_diagnosis,
)
from validation_semrepair.embodied_semantic_repair_aware_design import (
    RepairAuthority,
    RepairDecisionLeaf,
    RepairDecisionNode,
    solve_repair_aware_semantic_diagnosis,
    verify_repair_aware_semantic_diagnosis,
)


def _experiment(name, outcomes, *, cost=1.0, risk=0.0, atol=0.0):
    return SemanticExperiment(
        name=name,
        outcomes=tuple(outcomes.items()),
        cost=cost,
        risk=risk,
        observation_atol=atol,
    )


def _authorities(mapping):
    return tuple(
        RepairAuthority(hypothesis=name, authority_id=authority)
        for name, authority in mapping.items()
    )


def test_repair_aware_policy_stops_before_full_abi_identification():
    hypotheses = ("a0", "a1", "b", "c")
    experiments = (
        _experiment(
            "repair-class-test",
            {"a0": "A", "a1": "A", "b": "B", "c": "C"},
            cost=1.0,
        ),
        _experiment(
            "identity-refiner",
            {"a0": 0, "a1": 1, "b": 0, "c": 0},
            cost=5.0,
        ),
    )
    authorities = _authorities(
        {
            "a0": "bundle:A",
            "a1": "bundle:A",
            "b": "bundle:B",
            "c": "reject",
        }
    )

    repair_aware = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        experiments,
        authorities,
        objective="uniform_expected",
    )
    full_identity = solve_optimal_semantic_diagnosis(
        hypotheses,
        experiments,
        objective="uniform_expected",
    )

    assert repair_aware.complete
    assert repair_aware.optimal_cost == pytest.approx(1.0)
    assert isinstance(repair_aware.policy, RepairDecisionNode)
    assert repair_aware.policy.experiment == "repair-class-test"

    merged_leaf = next(
        branch.child
        for branch in repair_aware.policy.branches
        if isinstance(branch.child, RepairDecisionLeaf)
        and set(branch.child.hypotheses) == {"a0", "a1"}
    )
    assert merged_leaf.authority_id == "bundle:A"

    assert full_identity.complete
    assert isinstance(full_identity.policy, DiagnosisDecision)
    assert full_identity.optimal_cost == pytest.approx(3.5)
    assert repair_aware.optimal_cost < full_identity.optimal_cost
    assert verify_repair_aware_semantic_diagnosis(
        repair_aware, experiments
    ).valid


def test_observational_aliases_with_different_repairs_fail_closed():
    hypotheses = ("alias-a", "alias-b")
    experiments = (
        _experiment("external", {"alias-a": (0.0,), "alias-b": (0.0,)}),
    )
    authorities = _authorities(
        {"alias-a": "bundle:A", "alias-b": "bundle:B"}
    )

    result = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        experiments,
        authorities,
    )

    assert result.status == "unsafe_unidentifiable"
    assert result.policy is None
    assert result.unsafe_ambiguity_groups == (("alias-a", "alias-b"),)
    assert verify_repair_aware_semantic_diagnosis(result, experiments).valid


def test_semantic_aliases_with_same_exact_repair_need_no_probe():
    hypotheses = ("cause-left", "cause-right")
    authorities = _authorities(
        {
            "cause-left": "bundle:same-digest",
            "cause-right": "bundle:same-digest",
        }
    )

    result = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        (),
        authorities,
    )

    assert result.complete
    assert result.optimal_cost == 0.0
    assert isinstance(result.policy, RepairDecisionLeaf)
    assert set(result.policy.hypotheses) == set(hypotheses)
    assert result.policy.authority_id == "bundle:same-digest"
    assert verify_repair_aware_semantic_diagnosis(result, ()).valid


def test_compensating_fault_cannot_collapse_to_noop_authority():
    hypotheses = ("correct", "masked-double-swap")
    authorities = _authorities(
        {
            "correct": "noop",
            # Exact factorized bundle identity, not end-to-end net identity.
            "masked-double-swap": "bundle:producer+controller",
        }
    )
    external_only = (
        _experiment(
            "external-output",
            {
                "correct": (1.0, 0.0),
                "masked-double-swap": (1.0, 0.0),
            },
        ),
    )

    blocked = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        external_only,
        authorities,
    )
    assert blocked.status == "unsafe_unidentifiable"

    with_internal_tap = external_only + (
        _experiment(
            "tap:producer",
            {
                "correct": (1.0, 0.0),
                "masked-double-swap": (0.0, 1.0),
            },
            cost=2.0,
        ),
    )
    resolved = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        with_internal_tap,
        authorities,
    )

    assert resolved.complete
    assert resolved.optimal_cost == pytest.approx(2.0)
    assert isinstance(resolved.policy, RepairDecisionNode)
    assert resolved.policy.experiment == "tap:producer"
    assert verify_repair_aware_semantic_diagnosis(
        resolved, with_internal_tap
    ).valid


def test_bounded_error_blocks_repair_authority_when_balls_overlap():
    hypotheses = ("a", "b")
    experiments = (
        _experiment(
            "noisy-probe",
            {"a": (0.0,), "b": (0.15,)},
            atol=0.1,
        ),
    )
    authorities = _authorities({"a": "bundle:A", "b": "bundle:B"})

    result = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        experiments,
        authorities,
    )

    assert result.status == "unsafe_unidentifiable"
    assert verify_repair_aware_semantic_diagnosis(result, experiments).valid


def test_risk_gate_can_remove_only_repair_deciding_probe():
    hypotheses = ("a", "b")
    experiments = (
        _experiment(
            "safe-useless",
            {"a": "same", "b": "same"},
            risk=0.0,
        ),
        _experiment(
            "risky-separator",
            {"a": "left", "b": "right"},
            risk=1.0,
        ),
    )
    authorities = _authorities({"a": "bundle:A", "b": "bundle:B"})

    blocked = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        experiments,
        authorities,
        max_risk=0.0,
    )
    allowed = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        experiments,
        authorities,
        max_risk=1.0,
    )

    assert blocked.status == "unsafe_unidentifiable"
    assert allowed.complete
    assert allowed.optimal_cost == pytest.approx(2.0)
    assert verify_repair_aware_semantic_diagnosis(allowed, experiments).valid


def test_authority_or_result_digest_tampering_is_rejected():
    hypotheses = ("a", "b")
    experiments = (
        _experiment("separate", {"a": 0, "b": 1}),
    )
    authorities = _authorities({"a": "bundle:A", "b": "bundle:B"})
    result = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        experiments,
        authorities,
    )

    tampered_digest = replace(result, digest="0" * 64)
    assert not verify_repair_aware_semantic_diagnosis(
        tampered_digest, experiments
    ).valid

    tampered_authority = replace(
        result,
        authority_by_hypothesis=(("a", "bundle:A"), ("b", "bundle:A")),
    )
    assert not verify_repair_aware_semantic_diagnosis(
        tampered_authority, experiments
    ).valid
