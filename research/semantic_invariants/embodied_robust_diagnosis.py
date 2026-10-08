"""Minimax semantic diagnosis with bounded adversarial sensor error.

Unlike deterministic quantization, an observation within epsilon of two
semantic hypotheses remains ambiguous. The planner enumerates *every*
possible consistency-set outcome of each admissible physical probe, then
finds a minimum worst-case-cost adaptive tree satisfying a cumulative
episode-risk budget. It refuses instead of promising identification when
an adversarial but in-bound sensor outcome can preserve ambiguity.

Exact for the supplied finite hypothesis library, finite probe set and
axis-aligned componentwise deterministic error intervals. Exponential in
hypothesis count and dimension; not stochastic Bayesian inference or a
certification of actual sensor error bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from math import isfinite
from typing import Mapping, Sequence

from .embodied_semantic_experiment_design import SemanticExperiment
from .embodied_semantic_observability import SemanticDiagnosisHypothesis
from .embodied_semantic_transport import MonomialSemanticTransport


class RobustDiagnosisRejected(RuntimeError):
    """Insufficient evidence or valid worst-case diagnostic authority."""


@dataclass(frozen=True)
class RobustObservation:
    experiment_name: str
    measured: tuple[float, ...]


@dataclass(frozen=True)
class RobustBranch:
    consistent_hypotheses: tuple[str, ...]
    child: "RobustDiagnosisNode"


@dataclass(frozen=True)
class RobustDiagnosisNode:
    hypotheses: tuple[str, ...]
    experiment: SemanticExperiment | None
    branches: tuple[RobustBranch, ...]
    worst_remaining_cost: float
    worst_remaining_risk: float


@dataclass(frozen=True)
class RobustDiagnosisPlan:
    hypothesis_names: tuple[str, ...]
    # Exact execution-authority equivalence classes, not only hidden ABI labels.
    authority_by_hypothesis: tuple[tuple[str, str], ...]
    experiments: tuple[SemanticExperiment, ...]
    # Semantic transport identities and evidence IDs are part of the
    # trusted diagnosis context, not just modeled observed values.
    hypothesis_sources: tuple[tuple[str, tuple[tuple[str, str, tuple[int, ...], tuple[float, ...]], ...]], ...]
    # Sorted (experiment_name, ((hypothesis_name, predicted_vector), ...))
    predictions: tuple[tuple[str, tuple[tuple[str, tuple[float, ...]], ...]], ...]
    epsilon: float
    risk_budget: float
    max_probe_risk: float
    risk_weight: float
    complete: bool
    root: RobustDiagnosisNode | None
    worst_cost: float
    worst_path_risk: float
    digest: str


@dataclass(frozen=True)
class RobustProbe:
    experiment: SemanticExperiment
    remaining_hypotheses: tuple[str, ...]
    spent_risk: float
    remaining_risk: float


@dataclass(frozen=True)
class RobustResolution:
    authority_id: str
    consistent_hypotheses: tuple[str, ...]
    spent_risk: float

    @property
    def identified_hypothesis(self) -> str | None:
        return (
            self.consistent_hypotheses[0]
            if len(self.consistent_hypotheses) == 1 else None
        )


def _prediction(
    hypothesis: SemanticDiagnosisHypothesis, experiment: SemanticExperiment
) -> tuple[float, ...]:
    dim = hypothesis.factors[0].transport.dimension
    prefix = MonomialSemanticTransport.identity(dim)
    found = None
    for f in hypothesis.factors:
        prefix = prefix.then(f.transport)
        if f.name == experiment.tap_after_factor:
            found = prefix
    transport = prefix if experiment.tap_after_factor is None else found
    if transport is None:
        raise ValueError("probe references an unknown semantic tap")
    out = tuple(float(x) for x in transport.apply(experiment.probe))
    if not all(isfinite(x) for x in out):
        raise ValueError("non-finite modeled sensor prediction")
    return out


def _feasible_beliefs(
    predictions: dict[str, tuple[float, ...]],
    epsilon: float,
    *,
    cell_limit: int,
) -> tuple[tuple[str, ...], ...]:
    """Exact arrangement enumeration of axis-aligned epsilon-boxes.

    On any open cell between consecutive interval endpoints, box membership
    is constant; endpoint points themselves are included to cover closed
    uncertainty intervals. This handles non-transitive ambiguity, e.g.
    A overlaps B and B overlaps C but A does not overlap C.
    """
    names = tuple(sorted(predictions))
    dim = len(next(iter(predictions.values())))
    samples: list[tuple[float, ...]] = []
    grid_size = 1
    for axis in range(dim):
        endpoints = sorted({
            boundary
            for name in names
            for boundary in (
                predictions[name][axis] - epsilon,
                predictions[name][axis] + epsilon,
            )
        })
        axis_samples = list(endpoints)
        axis_samples += [
            (lo + hi) / 2.0
            for lo, hi in zip(endpoints, endpoints[1:])
        ]
        # Any point outside every interval would yield the empty set, which
        # is excluded as impossible under the assumed in-bound noise model.
        unique = tuple(sorted(set(axis_samples)))
        grid_size *= len(unique)
        if grid_size > cell_limit:
            raise ValueError(
                "robust observation arrangement exceeds explicit cell limit"
            )
        samples.append(unique)

    beliefs: set[tuple[str, ...]] = set()
    for observation in product(*samples):
        consistent = tuple(
            name for name in names
            if all(
                abs(predictions[name][idx] - observation[idx]) <= epsilon
                for idx in range(dim)
            )
        )
        if consistent:
            beliefs.add(consistent)
    return tuple(sorted(beliefs))


def _node_data(node: RobustDiagnosisNode | None) -> object:
    if node is None:
        return None
    return {
        "hypotheses": node.hypotheses,
        "experiment": (
            None if node.experiment is None else (
                node.experiment.name, node.experiment.probe,
                node.experiment.tap_after_factor,
                node.experiment.cost, node.experiment.risk,
            )
        ),
        "worst_remaining_cost": node.worst_remaining_cost,
        "worst_remaining_risk": node.worst_remaining_risk,
        "branches": [
            [branch.consistent_hypotheses, _node_data(branch.child)]
            for branch in node.branches
        ],
    }


def _digest(plan: RobustDiagnosisPlan) -> str:
    payload = {
        "schema": "semrepair-robust-diagnosis-v1",
        "hypothesis_names": plan.hypothesis_names,
        "authority_by_hypothesis": plan.authority_by_hypothesis,
        "hypothesis_sources": plan.hypothesis_sources,
        "experiments": [
            (x.name, x.probe, x.tap_after_factor, x.cost, x.risk)
            for x in plan.experiments
        ],
        "predictions": plan.predictions,
        "epsilon": plan.epsilon,
        "risk_budget": plan.risk_budget,
        "max_probe_risk": plan.max_probe_risk,
        "risk_weight": plan.risk_weight,
        "complete": plan.complete,
        "root": _node_data(plan.root),
        "worst_cost": plan.worst_cost,
        "worst_path_risk": plan.worst_path_risk,
    }
    return sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def synthesize_robust_experiment_plan(
    hypotheses: Sequence[SemanticDiagnosisHypothesis],
    experiments: Sequence[SemanticExperiment],
    *,
    epsilon: float,
    risk_budget: float,
    max_probe_risk: float,
    authorities: Mapping[str, str] | None = None,
    risk_weight: float = 0.0,
    cell_limit: int = 100_000,
) -> RobustDiagnosisPlan:
    """Solve finite minimax diagnosis; refuse if any sensor outcome stays ambiguous."""
    if (
        not isfinite(epsilon) or epsilon < 0
        or not isfinite(risk_budget) or risk_budget < 0
        or not isfinite(max_probe_risk) or max_probe_risk < 0
        or not isfinite(risk_weight) or risk_weight < 0
    ):
        raise ValueError("epsilon, risk and cost parameters must be finite/nonnegative")
    if not isinstance(cell_limit, int) or cell_limit <= 0:
        raise ValueError("cell_limit must be a positive integer")
    hs, es = tuple(hypotheses), tuple(experiments)
    if len(hs) < 2 or not es:
        raise ValueError("at least two hypotheses and one experiment are required")
    names = tuple(h.name for h in hs)
    if len(set(names)) != len(names) or any(not x for x in names):
        raise ValueError("unique nonempty hypothesis names are required")
    if authorities is None:
        # Default preserves the existing *full-identification* objective.
        authority = {name: name for name in names}
    else:
        if set(authorities) != set(names) or any(
            not isinstance(value, str) or not value.strip()
            for value in authorities.values()
        ):
            raise ValueError("authorities must exactly bind all hypotheses")
        authority = dict(authorities)
    if len(set(x.name for x in es)) != len(es):
        raise ValueError("experiment names must be unique")
    factor_names = tuple(x.name for x in hs[0].factors)
    if not factor_names or len(set(factor_names)) != len(factor_names):
        raise ValueError("hypotheses require unique named transport factors")
    dim = hs[0].factors[0].transport.dimension
    for h in hs:
        if tuple(x.name for x in h.factors) != factor_names:
            raise ValueError("hypotheses have inconsistent factor names")
        if any(x.transport.dimension != dim for x in h.factors):
            raise ValueError("hypotheses have inconsistent dimension")
    for e in es:
        if len(e.probe) != dim:
            raise ValueError("probe and semantic dimension differ")
        if e.tap_after_factor is not None and e.tap_after_factor not in factor_names:
            raise ValueError("probe requests unavailable semantic tap")
        if not all(isfinite(float(x)) for x in e.probe):
            raise ValueError("nonfinite experiment probe")
    admissible = tuple(
        sorted((e for e in es if e.risk <= max_probe_risk), key=lambda x: x.name)
    )
    if not admissible:
        raise ValueError("no experiments under the per-probe risk limit")
    modeled = {
        e.name: {h.name: _prediction(h, e) for h in hs}
        for e in admissible
    }
    responses = {
        name: _feasible_beliefs(p, epsilon, cell_limit=cell_limit)
        for name, p in modeled.items()
    }
    all_names = tuple(sorted(names))

    @lru_cache(maxsize=None)
    def solve(state: tuple[str, ...], remaining: float):
        # Same exact implementation/evidence-bound authority can be accepted
        # even if its hidden semantic ABI remains unidentifiable.
        if len({authority[name] for name in state}) == 1:
            return RobustDiagnosisNode(state, None, (), 0.0, 0.0)
        best: RobustDiagnosisNode | None = None
        for exp in admissible:
            if exp.risk > remaining:
                continue
            outcomes = tuple(sorted(set(
                tuple(x for x in belief if x in state)
                for belief in responses[exp.name]
            ) - {()}))
            # Any observation that retains all candidates prevents guaranteed
            # identification under an adversarial, repeatable noise budget.
            if not outcomes or state in outcomes:
                continue
            branches: list[RobustBranch] = []
            for next_state in outcomes:
                child = solve(next_state, max(0.0, remaining-exp.risk))
                if child is None:
                    break
                branches.append(RobustBranch(next_state, child))
            else:
                cost = exp.cost + risk_weight*exp.risk + max(
                    branch.child.worst_remaining_cost for branch in branches
                )
                risk = exp.risk + max(
                    branch.child.worst_remaining_risk for branch in branches
                )
                candidate = RobustDiagnosisNode(
                    state, exp, tuple(branches), cost, risk
                )
                if best is None or (
                    candidate.worst_remaining_cost,
                    candidate.worst_remaining_risk,
                    candidate.experiment.name,
                ) < (
                    best.worst_remaining_cost,
                    best.worst_remaining_risk,
                    best.experiment.name,
                ):
                    best = candidate
        return best

    root = solve(all_names, float(risk_budget))
    predicted = tuple(
        (e.name, tuple((name, modeled[e.name][name]) for name in all_names))
        for e in admissible
    )
    by_name = {h.name: h for h in hs}
    sources = tuple(
        (
            name,
            tuple(
                (
                    f.name, f.evidence_id,
                    f.transport.source_for_output, f.transport.scale,
                )
                for f in by_name[name].factors
            ),
        )
        for name in all_names
    )
    plan = RobustDiagnosisPlan(
        hypothesis_names=all_names,
        authority_by_hypothesis=tuple(
            (name, authority[name]) for name in all_names
        ),
        experiments=admissible,
        hypothesis_sources=sources, predictions=predicted,
        epsilon=float(epsilon),
        risk_budget=float(risk_budget), max_probe_risk=float(max_probe_risk),
        risk_weight=float(risk_weight), complete=root is not None,
        root=root, worst_cost=root.worst_remaining_cost if root else 0.0,
        worst_path_risk=root.worst_remaining_risk if root else 0.0,
        digest="",
    )
    return RobustDiagnosisPlan(**{
        **plan.__dict__, "digest": _digest(plan),
    })


def advance_robust_diagnosis(
    *,
    plan: RobustDiagnosisPlan,
    trusted_plan_digest: str,
    observations: Sequence[RobustObservation] = (),
) -> RobustProbe | RobustResolution:
    """Return an unambiguous *proposed* probe or resolution, never physical authority."""
    if not trusted_plan_digest or _digest(plan) != plan.digest or plan.digest != trusted_plan_digest:
        raise RobustDiagnosisRejected("robust diagnostic plan has no trusted identity")
    if not plan.complete or plan.root is None:
        raise RobustDiagnosisRejected("bounded-noise problem not identifiable")
    if plan.root.hypotheses != plan.hypothesis_names:
        raise RobustDiagnosisRejected("robust diagnosis root changed")
    node = plan.root
    spent = 0.0
    modeled = {n: dict(p) for n, p in plan.predictions}
    for observation in observations:
        if node.experiment is None or observation.experiment_name != node.experiment.name:
            raise RobustDiagnosisRejected("stale or out-of-order semantic probe observation")
        if (
            len(observation.measured) != len(node.experiment.probe)
            or not all(isfinite(float(x)) for x in observation.measured)
        ):
            raise RobustDiagnosisRejected("invalid or nonfinite semantic measurement")
        next_state = tuple(
            name for name in node.hypotheses
            if all(
                abs(y - z) <= plan.epsilon
                for y, z in zip(
                    modeled[node.experiment.name][name],
                    observation.measured, strict=True,
                )
            )
        )
        if not next_state:
            raise RobustDiagnosisRejected("observation violates modeled sensor-error bound")
        matches = [
            branch.child for branch in node.branches
            if branch.consistent_hypotheses == next_state
        ]
        if len(matches) != 1:
            raise RobustDiagnosisRejected("unmodeled ambiguity class or changed decision tree")
        spent += node.experiment.risk
        node = matches[0]

    if node.experiment is None:
        authority_by_name = dict(plan.authority_by_hypothesis)
        terminal = {authority_by_name[name] for name in node.hypotheses}
        if len(terminal) != 1 or spent > plan.risk_budget:
            raise RobustDiagnosisRejected("uncertified or conflicting repair authority")
        return RobustResolution(
            next(iter(terminal)), node.hypotheses, spent,
        )
    if spent + node.experiment.risk > plan.risk_budget:
        raise RobustDiagnosisRejected("probe would exceed cumulative risk")
    return RobustProbe(
        node.experiment, node.hypotheses, spent,
        plan.risk_budget-spent,
    )
