"""Joint active experiment design for hidden embodied interface semantics.

This module plans where to observe and what to probe together.

The existing semantic-observability kernel answers whether a frozen transport
chain is externally identifiable and which internal taps can separate a finite
hypothesis set. This module adds an exact adaptive planner over a finite,
pre-declared experiment library. Each experiment binds a concrete input probe,
an external output or named internal semantic tap, execution cost, and physical
or operational risk.

For the supported deterministic finite hypothesis model, dynamic programming
returns an optimal decision tree under either expected-cost or worst-case-cost
objective, subject to a hard per-experiment risk ceiling. If the admissible
experiment library cannot separate some hypotheses, the planner reports the
remaining observational equivalence classes instead of inventing a diagnosis.

Optimality is only over the supplied finite hypothesis and experiment sets.
This is not a general solution to nonlinear robot system identification or
physical safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from math import isfinite
from typing import Literal, Mapping, Sequence

from .embodied_semantic_observability import SemanticDiagnosisHypothesis
from .embodied_semantic_transport import MonomialSemanticTransport


Objective = Literal["expected", "worst_case"]


@dataclass(frozen=True)
class SemanticExperiment:
    """One admissible intervention/observation pair."""

    name: str
    probe: tuple[float, ...]
    tap_after_factor: str | None = None
    cost: float = 1.0
    risk: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name must be non-empty")
        if not self.probe:
            raise ValueError("experiment probe must be non-empty")
        if not isfinite(float(self.cost)) or self.cost < 0:
            raise ValueError("experiment cost must be finite and non-negative")
        if not isfinite(float(self.risk)) or self.risk < 0:
            raise ValueError("experiment risk must be finite and non-negative")


@dataclass(frozen=True)
class ObservationClass:
    signature: tuple[int | float, ...]
    hypotheses: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentDecisionNode:
    hypotheses: tuple[str, ...]
    experiment: SemanticExperiment | None
    observation_classes: tuple[ObservationClass, ...]
    children: tuple["ExperimentDecisionNode", ...]
    complete: bool
    expected_remaining_cost: float
    worst_case_remaining_cost: float


@dataclass(frozen=True)
class SemanticExperimentPlan:
    hypothesis_names: tuple[str, ...]
    admissible_experiments: tuple[str, ...]
    objective: Objective
    max_risk: float
    risk_weight: float
    complete: bool
    expected_total_cost: float
    worst_case_total_cost: float
    root: ExperimentDecisionNode
    unresolved_equivalence_classes: tuple[tuple[str, ...], ...]
    digest: str


def canonical_basis_experiments(
    *,
    dimension: int,
    taps: Sequence[str | None],
    external_cost: float = 1.0,
    internal_cost: float = 2.0,
    risk: float = 0.0,
) -> tuple[SemanticExperiment, ...]:
    """Generate canonical basis probes at external and selected internal taps."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    experiments: list[SemanticExperiment] = []
    for tap in taps:
        cost = external_cost if tap is None else internal_cost
        label = "output" if tap is None else tap
        for index in range(dimension):
            probe = tuple(1.0 if i == index else 0.0 for i in range(dimension))
            experiments.append(
                SemanticExperiment(
                    name=f"{label}/e{index}",
                    probe=probe,
                    tap_after_factor=tap,
                    cost=cost,
                    risk=risk,
                )
            )
    return tuple(experiments)


def _prefix_transports(
    hypothesis: SemanticDiagnosisHypothesis,
) -> tuple[MonomialSemanticTransport, ...]:
    dimension = hypothesis.factors[0].transport.dimension
    prefix = MonomialSemanticTransport.identity(dimension)
    result: list[MonomialSemanticTransport] = []
    for factor in hypothesis.factors:
        prefix = prefix.then(factor.transport)
        result.append(prefix)
    return tuple(result)


def _validate_problem(
    hypotheses: Sequence[SemanticDiagnosisHypothesis],
    experiments: Sequence[SemanticExperiment],
) -> tuple[
    tuple[SemanticDiagnosisHypothesis, ...],
    tuple[SemanticExperiment, ...],
    tuple[str, ...],
    int,
]:
    hs = tuple(hypotheses)
    es = tuple(experiments)
    if len(hs) < 2:
        raise ValueError("at least two hypotheses are required")
    if not es:
        raise ValueError("at least one experiment is required")

    names = tuple(item.name for item in hs)
    if len(set(names)) != len(names):
        raise ValueError("hypothesis names must be unique")
    experiment_names = tuple(item.name for item in es)
    if len(set(experiment_names)) != len(experiment_names):
        raise ValueError("experiment names must be unique")

    factor_names = tuple(f.name for f in hs[0].factors)
    if not factor_names:
        raise ValueError("hypotheses require at least one factor")
    dimension = hs[0].factors[0].transport.dimension

    for item in hs:
        if tuple(f.name for f in item.factors) != factor_names:
            raise ValueError("all hypotheses must use the same ordered factor names")
        if any(f.transport.dimension != dimension for f in item.factors):
            raise ValueError("all hypothesis transports must share one dimension")

    known_taps = set(factor_names)
    for experiment in es:
        if len(experiment.probe) != dimension:
            raise ValueError(
                f"experiment {experiment.name!r} probe dimension mismatch"
            )
        if (
            experiment.tap_after_factor is not None
            and experiment.tap_after_factor not in known_taps
        ):
            raise ValueError(
                f"experiment {experiment.name!r} references unknown tap "
                f"{experiment.tap_after_factor!r}"
            )

    return hs, es, factor_names, dimension


def _observation(
    prefixes: tuple[MonomialSemanticTransport, ...],
    factor_names: tuple[str, ...],
    experiment: SemanticExperiment,
) -> tuple[float, ...]:
    if experiment.tap_after_factor is None:
        transport = prefixes[-1]
    else:
        index = factor_names.index(experiment.tap_after_factor)
        transport = prefixes[index]
    return transport.apply(experiment.probe)


def _signature(
    value: Sequence[float],
    *,
    atol: float,
) -> tuple[int | float, ...]:
    if atol < 0:
        raise ValueError("atol must be non-negative")
    if atol == 0:
        return tuple(float(x) for x in value)
    return tuple(int(round(float(x) / atol)) for x in value)


def _problem_digest(
    hypotheses: Sequence[SemanticDiagnosisHypothesis],
    experiments: Sequence[SemanticExperiment],
    *,
    priors: Mapping[str, float],
    objective: Objective,
    max_risk: float,
    risk_weight: float,
    atol: float,
) -> str:
    payload = {
        "objective": objective,
        "max_risk": max_risk,
        "risk_weight": risk_weight,
        "atol": atol,
        "priors": dict(sorted((k, float(v)) for k, v in priors.items())),
        "hypotheses": [
            {
                "name": hypothesis.name,
                "factors": [
                    {
                        "name": factor.name,
                        "source_for_output": list(
                            factor.transport.source_for_output
                        ),
                        "scale": list(factor.transport.scale),
                        "evidence_id": factor.evidence_id,
                    }
                    for factor in hypothesis.factors
                ],
            }
            for hypothesis in hypotheses
        ],
        "experiments": [
            {
                "name": experiment.name,
                "probe": list(experiment.probe),
                "tap_after_factor": experiment.tap_after_factor,
                "cost": experiment.cost,
                "risk": experiment.risk,
            }
            for experiment in experiments
        ],
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def synthesize_optimal_experiment_plan(
    hypotheses: Sequence[SemanticDiagnosisHypothesis],
    experiments: Sequence[SemanticExperiment],
    *,
    priors: Mapping[str, float] | None = None,
    objective: Objective = "expected",
    max_risk: float = float("inf"),
    risk_weight: float = 0.0,
    atol: float = 0.0,
) -> SemanticExperimentPlan:
    """Synthesize an exact adaptive experiment tree over a finite problem."""
    hs, es, factor_names, _ = _validate_problem(hypotheses, experiments)
    if objective not in ("expected", "worst_case"):
        raise ValueError("objective must be 'expected' or 'worst_case'")
    if max_risk < 0 or (not isfinite(max_risk) and max_risk != float("inf")):
        raise ValueError("max_risk must be non-negative or infinity")
    if risk_weight < 0 or not isfinite(risk_weight):
        raise ValueError("risk_weight must be finite and non-negative")
    if atol < 0 or not isfinite(atol):
        raise ValueError("atol must be finite and non-negative")

    by_name = {item.name: item for item in hs}
    prefixes = {item.name: _prefix_transports(item) for item in hs}

    if priors is None:
        weight = {item.name: 1.0 for item in hs}
    else:
        unknown = set(priors) - set(by_name)
        missing = set(by_name) - set(priors)
        if unknown or missing:
            raise ValueError(
                f"priors must cover hypotheses exactly; "
                f"unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        weight = {}
        for name, value in priors.items():
            numeric = float(value)
            if numeric <= 0 or not isfinite(numeric):
                raise ValueError("all prior weights must be finite and positive")
            weight[name] = numeric

    admissible = tuple(item for item in es if item.risk <= max_risk)
    if not admissible:
        raise ValueError("no experiments satisfy the risk ceiling")

    signatures: dict[str, dict[str, tuple[int | float, ...]]] = {}
    for experiment in admissible:
        signatures[experiment.name] = {
            hypothesis.name: _signature(
                _observation(
                    prefixes[hypothesis.name],
                    factor_names,
                    experiment,
                ),
                atol=atol,
            )
            for hypothesis in hs
        }
    experiment_by_name = {item.name: item for item in admissible}

    # Observational equivalence is defined over the complete admissible
    # experiment library. If two hypotheses have the same joint signature,
    # no adaptive ordering of these experiments can ever separate them.
    joint_signature = {
        name: tuple(
            signatures[experiment.name][name]
            for experiment in admissible
        )
        for name in by_name
    }
    equivalence_groups: dict[
        tuple[tuple[int | float, ...], ...], list[str]
    ] = {}
    for name, signature in joint_signature.items():
        equivalence_groups.setdefault(signature, []).append(name)
    intrinsic_classes = tuple(
        sorted(
            (
                tuple(sorted(names))
                for names in equivalence_groups.values()
                if len(names) > 1
            ),
            key=lambda item: (len(item), item),
        )
    )

    def partition_state(
        state: tuple[str, ...],
        experiment: SemanticExperiment,
    ) -> tuple[tuple[tuple[int | float, ...], tuple[str, ...]], ...]:
        groups: dict[tuple[int | float, ...], list[str]] = {}
        for name in state:
            sig = signatures[experiment.name][name]
            groups.setdefault(sig, []).append(name)
        return tuple(
            sorted(
                (sig, tuple(sorted(members)))
                for sig, members in groups.items()
            )
        )

    def state_mass(state: tuple[str, ...]) -> float:
        return sum(weight[name] for name in state)

    @lru_cache(maxsize=None)
    def solve(
        state: tuple[str, ...],
    ) -> tuple[
        bool,
        float,
        float,
        str | None,
        tuple[tuple[tuple[int | float, ...], tuple[str, ...]], ...],
    ]:
        if len(state) <= 1:
            return True, 0.0, 0.0, None, ()

        # A state whose hypotheses share one joint signature across every
        # admissible experiment is an intrinsic observational equivalence
        # class. Stop without spending more intervention budget.
        first_signature = joint_signature[state[0]]
        if all(joint_signature[name] == first_signature for name in state[1:]):
            return False, 0.0, 0.0, None, ()

        candidates = []
        total_mass = state_mass(state)
        for experiment in admissible:
            groups = partition_state(state, experiment)
            if len(groups) <= 1:
                continue
            child_results = [solve(members) for _, members in groups]

            immediate = experiment.cost + risk_weight * experiment.risk
            expected = immediate + sum(
                (state_mass(members) / total_mass) * child[1]
                for (_, members), child in zip(
                    groups, child_results, strict=True
                )
            )
            worst = immediate + max(child[2] for child in child_results)
            complete = all(item[0] for item in child_results)
            primary = expected if objective == "expected" else worst
            secondary = worst if objective == "expected" else expected
            candidates.append(
                (
                    primary,
                    secondary,
                    immediate,
                    experiment.name,
                    groups,
                    expected,
                    worst,
                    complete,
                )
            )

        if not candidates:
            raise RuntimeError(
                "state has distinct joint signatures but no informative experiment"
            )

        best = min(
            candidates,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
            ),
        )
        return best[7], best[5], best[6], best[3], best[4]

    unresolved: set[tuple[str, ...]] = set()

    def build(state: tuple[str, ...]) -> ExperimentDecisionNode:
        complete, expected, worst, experiment_name, groups = solve(state)
        if len(state) <= 1:
            return ExperimentDecisionNode(
                hypotheses=state,
                experiment=None,
                observation_classes=(),
                children=(),
                complete=True,
                expected_remaining_cost=0.0,
                worst_case_remaining_cost=0.0,
            )
        if not complete or experiment_name is None:
            unresolved.add(state)
            return ExperimentDecisionNode(
                hypotheses=state,
                experiment=None,
                observation_classes=(),
                children=(),
                complete=False,
                expected_remaining_cost=float("inf"),
                worst_case_remaining_cost=float("inf"),
            )

        experiment = experiment_by_name[experiment_name]
        children = tuple(build(members) for _, members in groups)
        return ExperimentDecisionNode(
            hypotheses=state,
            experiment=experiment,
            observation_classes=tuple(
                ObservationClass(signature=sig, hypotheses=members)
                for sig, members in groups
            ),
            children=children,
            complete=all(child.complete for child in children),
            expected_remaining_cost=expected,
            worst_case_remaining_cost=worst,
        )

    initial = tuple(sorted(by_name))
    root = build(initial)
    complete, expected, worst, _, _ = solve(initial)
    return SemanticExperimentPlan(
        hypothesis_names=initial,
        admissible_experiments=tuple(item.name for item in admissible),
        objective=objective,
        max_risk=float(max_risk),
        risk_weight=float(risk_weight),
        complete=complete and root.complete,
        expected_total_cost=expected,
        worst_case_total_cost=worst,
        root=root,
        unresolved_equivalence_classes=intrinsic_classes,
        digest=_problem_digest(
            hs,
            admissible,
            priors=weight,
            objective=objective,
            max_risk=float(max_risk),
            risk_weight=float(risk_weight),
            atol=atol,
        ),
    )
