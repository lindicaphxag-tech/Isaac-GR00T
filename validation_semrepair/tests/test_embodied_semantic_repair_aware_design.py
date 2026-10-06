from dataclasses import replace

import pytest

from validation_semrepair.embodied_semantic_experiment_design import (
    DiagnosisDecision,
    SemanticExperiment,
    solve_optimal_semantic_diagnosis,
)
from validation_semrepair.embodied_semantic_repair_aware_design import (
    BoundRepairAuthority,
    RepairAuthority,
    RepairDecisionLeaf,
    RepairDecisionNode,
    bind_repair_authority,
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


def test_repair_authority_quotient_never_costs_more_than_full_identification():
    hypotheses = ("h0", "h1", "h2", "h3")
    experiments = (
        _experiment(
            "coarse",
            {"h0": "A", "h1": "A", "h2": "B", "h3": "C"},
            cost=1.0,
        ),
        _experiment(
            "fine",
            {"h0": 0, "h1": 1, "h2": 2, "h3": 3},
            cost=4.0,
        ),
    )
    authorities = _authorities(
        {
            "h0": "bundle:A",
            "h1": "bundle:A",
            "h2": "bundle:B",
            "h3": "reject",
        }
    )

    full = solve_optimal_semantic_diagnosis(
        hypotheses,
        experiments,
        objective="uniform_expected",
    )
    repair = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        experiments,
        authorities,
        objective="uniform_expected",
    )

    assert full.complete and repair.complete
    assert repair.optimal_cost <= full.optimal_cost
    assert repair.optimal_cost == pytest.approx(1.0)
    assert full.optimal_cost == pytest.approx(3.0)


def test_unique_authority_per_hypothesis_reduces_to_full_identification():
    hypotheses = ("h0", "h1", "h2")
    experiments = (
        _experiment(
            "split",
            {"h0": 0, "h1": 1, "h2": 1},
            cost=1.0,
        ),
        _experiment(
            "refine",
            {"h0": 0, "h1": 0, "h2": 1},
            cost=2.0,
        ),
    )
    authorities = _authorities(
        {"h0": "A0", "h1": "A1", "h2": "A2"}
    )

    full = solve_optimal_semantic_diagnosis(
        hypotheses,
        experiments,
        objective="uniform_expected",
    )
    repair = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        experiments,
        authorities,
        objective="uniform_expected",
    )

    assert full.complete and repair.complete
    assert repair.optimal_cost == pytest.approx(full.optimal_cost)


def test_single_authority_class_has_zero_diagnostic_value_requirement():
    hypotheses = ("h0", "h1", "h2")
    authorities = _authorities(
        {"h0": "same", "h1": "same", "h2": "same"}
    )

    repair = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        (),
        authorities,
    )

    assert repair.complete
    assert repair.optimal_cost == 0.0
    assert isinstance(repair.policy, RepairDecisionLeaf)
    assert set(repair.policy.hypotheses) == set(hypotheses)
    assert repair.policy.authority_id == "same"


def _bound(bundle, impl, evidence="evidence:v1", deps="deps:v1"):
    return BoundRepairAuthority(
        repair_bundle_id=bundle,
        implementation_ids=(impl,),
        evidence_digest=evidence,
        dependency_digest=deps,
    )


def test_same_symbolic_repair_with_different_implementation_is_not_one_class():
    hypotheses = ("impl-a", "impl-b")
    left = _bound("swap-x-y", "binary:A")
    right = _bound("swap-x-y", "binary:B")
    assert left.repair_bundle_id == right.repair_bundle_id
    assert left.authority_id != right.authority_id

    authorities = (
        bind_repair_authority("impl-a", left),
        bind_repair_authority("impl-b", right),
    )
    external_only = (
        _experiment(
            "same-external-behavior",
            {"impl-a": (0.0,), "impl-b": (0.0,)},
        ),
    )

    result = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        external_only,
        authorities,
    )

    assert result.status == "unsafe_unidentifiable"
    assert result.policy is None


def test_dependency_or_evidence_drift_splits_repair_authority_class():
    base = _bound("permute-joints", "impl:v7")
    dependency_drift = _bound(
        "permute-joints",
        "impl:v7",
        deps="deps:v2",
    )
    evidence_drift = _bound(
        "permute-joints",
        "impl:v7",
        evidence="evidence:v2",
    )

    assert base.authority_id != dependency_drift.authority_id
    assert base.authority_id != evidence_drift.authority_id
    assert dependency_drift.authority_id != evidence_drift.authority_id


def test_identical_concrete_binding_can_share_zero_cost_authority_class():
    hypotheses = ("semantic-cause-a", "semantic-cause-b")
    binding = _bound("scale-mm-to-m", "impl:scale-v3")
    authorities = (
        bind_repair_authority("semantic-cause-a", binding),
        bind_repair_authority("semantic-cause-b", binding),
    )

    result = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        (),
        authorities,
    )

    assert result.complete
    assert result.optimal_cost == 0.0
    assert isinstance(result.policy, RepairDecisionLeaf)
    assert result.policy.authority_id == binding.authority_id
