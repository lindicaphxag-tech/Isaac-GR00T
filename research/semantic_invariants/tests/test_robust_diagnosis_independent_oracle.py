"""Independent rational-interval reference oracle for robust semantic diagnosis.

This implementation is intentionally separate from the shipped robust solver:
it never imports its DP, observation arrangement, tree or helper functions.
The oracle enumerates exact closed interval endpoints with fractions and
solves an exhaustive finite tree under a declared additive risk budget.

A random battery compares *the returned optimum*, not merely success status.
Designed for 1-dimensional non-negative monomial test transports only.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import product
import random

import pytest

from research.semantic_invariants.embodied_robust_diagnosis import (
    RobustDiagnosisNode,
    synthesize_robust_experiment_plan,
)
from research.semantic_invariants.embodied_semantic_experiment_design import SemanticExperiment
from research.semantic_invariants.embodied_semantic_observability import SemanticDiagnosisHypothesis
from research.semantic_invariants.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
)


def _exact_outcome_families(
    predictions: tuple[Fraction, ...], epsilon: Fraction
) -> tuple[tuple[int, ...], ...]:
    """Solve the arrangement of closed 1-D intervals with exact arithmetic."""
    ends = sorted({
        p + delta
        for p in predictions
        for delta in (-epsilon, epsilon)
    })
    points = list(ends)
    points.extend((a + b) / 2 for a, b in zip(ends, ends[1:]))
    return tuple(sorted({
        tuple(i for i, p in enumerate(predictions) if abs(y - p) <= epsilon)
        for y in points
    } - {()}))


def _brute_force_reference(
    predicted_by_exp: tuple[tuple[Fraction, ...], ...],
    *,
    epsilon: Fraction,
    costs: tuple[Fraction, ...],
    risks: tuple[int, ...],
    budget: int,
    authority_ids: tuple[str, ...],
) -> tuple[Fraction, int] | None:
    """A separate subset-and-budget minimax oracle; None means impossible."""
    families = tuple(
        _exact_outcome_families(pred, epsilon)
        for pred in predicted_by_exp
    )
    n = len(authority_ids)

    @lru_cache(maxsize=None)
    def solve(mask: int, remaining: int) -> tuple[Fraction, int] | None:
        members = [j for j in range(n) if mask & (1 << j)]
        if len({authority_ids[j] for j in members}) == 1:
            return (Fraction(0), 0)
        candidates: list[tuple[Fraction, int]] = []
        for exp_i, family in enumerate(families):
            if risks[exp_i] > remaining:
                continue
            possible = tuple(sorted({
                sum(1 << j for j in outcome if mask & (1 << j))
                for outcome in family
            } - {0}))
            if not possible or mask in possible:
                continue
            children = [
                solve(child, remaining-risks[exp_i])
                for child in possible
            ]
            if any(child is None for child in children):
                continue
            worst_cost = costs[exp_i] + max(child[0] for child in children)
            worst_risk = risks[exp_i] + max(child[1] for child in children)
            candidates.append((worst_cost, worst_risk))
        return min(candidates) if candidates else None

    return solve((1 << n) - 1, budget)


def _assert_witness_tree(
    node: RobustDiagnosisNode,
    *,
    epsilon: Fraction,
    predicted_by_name: dict[str, dict[str, Fraction]],
    authority: dict[str, str],
    risk_budget: Fraction,
) -> tuple[Fraction, Fraction]:
    """Verify *all* decision branches against the independently enumerated regions."""
    names = tuple(node.hypotheses)
    if node.experiment is None:
        assert len({authority[n] for n in names}) == 1
        assert node.worst_remaining_risk == pytest.approx(0)
        return Fraction(0), Fraction(0)
    exp = node.experiment
    observations = _exact_outcome_families(
        tuple(predicted_by_name[exp.name][name] for name in names),
        epsilon,
    )
    expected = {
        tuple(names[index] for index in outcome)
        for outcome in observations
    }
    branches = {branch.consistent_hypotheses: branch.child for branch in node.branches}
    assert set(branches) == expected
    assert set(branches) and names not in branches
    results = [
        _assert_witness_tree(
            child,
            epsilon=epsilon,
            predicted_by_name=predicted_by_name,
            authority=authority,
            risk_budget=risk_budget-Fraction(str(exp.risk)),
        )
        for child in branches.values()
    ]
    risk = Fraction(str(exp.risk)) + max(pair[1] for pair in results)
    cost = Fraction(str(exp.cost)) + max(pair[0] for pair in results)
    assert risk <= risk_budget
    assert node.worst_remaining_risk == pytest.approx(float(risk))
    assert node.worst_remaining_cost == pytest.approx(float(cost))
    return cost, risk


def test_robust_planner_matches_independent_exact_rational_minimax_oracle():
    # Frozen PRNG so the entire falsification bank is deterministic and
    # cannot be cherry-picked after observing failures.
    rng = random.Random(20261008)
    accepted = 0
    refused = 0
    for case in range(192):
        n = rng.randrange(2, 5)
        n_probes = rng.randrange(1, 5)
        epsilon_q = rng.choice([0, 1, 2, 3])
        epsilon = Fraction(epsilon_q, 4)
        scales = [rng.randrange(2, 14) for _ in range(n)]
        probes = [rng.choice([1, 2, 3]) for _ in range(n_probes)]
        # Half-unit prices and integer risk to avoid nonlinear float ambiguity.
        costs_q = [rng.randrange(1, 6) for _ in range(n_probes)]
        risks = [rng.choice([1, 2, 3]) for _ in range(n_probes)]
        budget = rng.randrange(0, 8)
        authority_ids = tuple(
            "repair/{}".format(rng.randrange(1, min(n, 4)))
            for _ in range(n)
        )
        hs = tuple(
            SemanticDiagnosisHypothesis(
                name=f"world-{i}",
                factors=(SemanticTransportFactor(
                    "controller-chart",
                    MonomialSemanticTransport((0,), (float(scale) / 4,)),
                    f"frozen-source/case-{case}/world-{i}",
                ),),
            )
            for i, scale in enumerate(scales)
        )
        exps = tuple(
            SemanticExperiment(
                f"probe-{j}", (float(probe),),
                cost=cost/2, risk=float(risk),
            )
            for j, (probe, cost, risk) in enumerate(
                zip(probes, costs_q, risks, strict=True)
            )
        )
        predicted = tuple(
            tuple(Fraction(scale * probe, 4) for scale in scales)
            for probe in probes
        )
        reference = _brute_force_reference(
            predicted, epsilon=epsilon,
            costs=tuple(Fraction(c, 2) for c in costs_q),
            risks=tuple(risks), budget=budget,
            authority_ids=authority_ids,
        )
        plan = synthesize_robust_experiment_plan(
            hs, exps, epsilon=float(epsilon),
            risk_budget=float(budget),
            max_probe_risk=float(max(risks)),
            authorities={
                f"world-{i}": authority_ids[i] for i in range(n)
            },
            cell_limit=10000,
        )
        assert plan.complete == (reference is not None), {
            "case": case, "scales": scales, "probes": probes,
            "costs_q": costs_q, "risks": risks, "budget": budget,
            "epsilon_q": epsilon_q, "authority": authority_ids,
        }
        if reference is None:
            refused += 1
            continue
        accepted += 1
        assert plan.worst_cost == pytest.approx(float(reference[0]))
        assert plan.worst_path_risk == pytest.approx(float(reference[1]))
        table = {
            f"probe-{j}": {
                f"world-{i}": Fraction(scale * probes[j], 4)
                for i, scale in enumerate(scales)
            }
            for j in range(n_probes)
        }
        certified = _assert_witness_tree(
            plan.root,
            epsilon=epsilon,
            predicted_by_name=table,
            authority={f"world-{i}": authority_ids[i] for i in range(n)},
            risk_budget=Fraction(budget),
        )
        assert certified == reference
    assert accepted > 0 and refused > 0
