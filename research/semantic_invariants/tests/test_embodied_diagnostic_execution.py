from dataclasses import replace

import pytest

from research.semantic_invariants.embodied_diagnostic_execution import (
    DiagnosticExecutionRejected,
    DiagnosticObservedStep,
    DiagnosticProbeAuthority,
    DiagnosticResolution,
    authorize_next_diagnostic_probe,
    diagnostic_tree_commitment,
)
from research.semantic_invariants.embodied_semantic_experiment_design import (
    SemanticExperiment,
    synthesize_optimal_experiment_plan,
)
from research.semantic_invariants.embodied_semantic_observability import (
    SemanticDiagnosisHypothesis,
)
from research.semantic_invariants.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
)


def _case(*, budget=1.2):
    identity = MonomialSemanticTransport.identity(2)
    x_scaled = MonomialSemanticTransport((0, 1), (2.0, 1.0))
    y_scaled = MonomialSemanticTransport((0, 1), (1.0, 2.0))
    hs = tuple(
        SemanticDiagnosisHypothesis(
            name=name,
            factors=(SemanticTransportFactor("chart", t, name),),
        )
        for name, t in (
            ("identity", identity), ("x-scaled", x_scaled), ("y-scaled", y_scaled)
        )
    )
    exps = (
        SemanticExperiment("probe-x", (1.0, 0.0), risk=0.6),
        SemanticExperiment("probe-y", (0.0, 1.0), risk=0.6),
    )
    return synthesize_optimal_experiment_plan(
        hs, exps, max_risk=0.6, total_risk_budget=budget
    )


def _start(plan, trace=()):
    return authorize_next_diagnostic_probe(
        plan=plan,
        trusted_problem_digest=plan.digest,
        trusted_tree_commitment=diagnostic_tree_commitment(plan),
        trace=trace,
    )


def _identity_step(node, step_idx):
    cl = next(c for c in node.observation_classes if "identity" in c.hypotheses)
    return DiagnosticObservedStep(
        node.experiment.name, tuple(cl.signature), f"sensor/{step_idx}"
    )


def test_online_gate_replays_actual_path_and_enforces_episode_risk():
    plan = _case()
    first = _start(plan)
    assert isinstance(first, DiagnosticProbeAuthority)
    assert first.risk_after_probe == pytest.approx(0.6)
    assert first.step_index == 0

    first_obs = _identity_step(plan.root, 0)
    second = _start(plan, (first_obs,))
    assert isinstance(second, DiagnosticProbeAuthority)
    assert second.step_index == 1
    assert second.risk_consumed == pytest.approx(0.6)
    assert second.risk_after_probe == pytest.approx(1.2)

    cl = next(c for c in plan.root.observation_classes
              if "identity" in c.hypotheses)
    child = plan.root.children[plan.root.observation_classes.index(cl)]
    final = _start(plan, (first_obs, _identity_step(child, 1)))
    assert isinstance(final, DiagnosticResolution)
    assert final.hypotheses == ("identity",)
    assert final.identified
    assert final.risk_consumed == pytest.approx(1.2)


def test_overspending_or_incomplete_plans_do_not_dispatch():
    for budget in (0.0, 0.5, 0.6, 1.0):
        p = _case(budget=budget)
        assert not p.complete
        with pytest.raises(DiagnosticExecutionRejected, match="complete"):
            _start(p)

    p = _case(budget=float("inf"))
    assert p.complete
    with pytest.raises(DiagnosticExecutionRejected, match="finite-risk-budget"):
        _start(p)


def test_unauthorized_observations_abort_without_advancing():
    p = _case()
    forged = (
        DiagnosticObservedStep(p.root.experiment.name, (123.0, 321.0), "evidence/1"),
        DiagnosticObservedStep("wrong-probe", (1.0, 0.0), "evidence/2"),
        DiagnosticObservedStep(p.root.experiment.name, (float("nan"),), "evidence/3"),
        DiagnosticObservedStep(p.root.experiment.name, (1.0, 0.0), ""),
    )
    for bad in forged:
        with pytest.raises(DiagnosticExecutionRejected):
            _start(p, (bad,))


def test_entire_tree_binding_catches_forged_risk_and_branch_lists():
    p = _case()
    trusted = diagnostic_tree_commitment(p)
    lower_risk_exp = replace(p.root.experiment, risk=0.0)
    forged_node = replace(p.root, experiment=lower_risk_exp)
    forged = replace(p, root=forged_node)
    assert forged.digest == p.digest  # original problem digest alone is insufficient
    assert diagnostic_tree_commitment(forged) != trusted
    with pytest.raises(DiagnosticExecutionRejected, match="untrusted"):
        authorize_next_diagnostic_probe(
            plan=forged,
            trusted_problem_digest=p.digest,
            trusted_tree_commitment=trusted,
        )

    amputated = replace(p, root=replace(p.root, children=p.root.children[:-1]))
    with pytest.raises(DiagnosticExecutionRejected, match="partition"):
        _start(amputated)

    wrong_budget = replace(p, max_path_risk=0.0)
    with pytest.raises(DiagnosticExecutionRejected, match="maximum path risk"):
        _start(wrong_budget)


def test_plan_freeze_cannot_be_replayed_as_different_problem():
    p = _case()
    with pytest.raises(DiagnosticExecutionRejected, match="untrusted"):
        authorize_next_diagnostic_probe(
            plan=p,
            trusted_problem_digest="0" * 64,
            trusted_tree_commitment=diagnostic_tree_commitment(p),
        )
    with pytest.raises(DiagnosticExecutionRejected, match="untrusted"):
        authorize_next_diagnostic_probe(
            plan=p,
            trusted_problem_digest=p.digest,
            trusted_tree_commitment="0" * 64,
        )
