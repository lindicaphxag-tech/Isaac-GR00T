"""Active inference of hidden embodied semantic contracts.

The module selects safe diagnostic probes by expected information gain over a
finite hypothesis set.  It is deliberately generic: action-ABI probing is one
backend, but the same mechanism can distinguish clock scope, ordering,
representation, or execution-path semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Callable, Hashable, Sequence


Observation = Hashable
ProbePayload = tuple[float, ...] | str


@dataclass(frozen=True)
class SemanticHypothesis:
    name: str
    predict: Callable[[ProbePayload], Observation]


@dataclass(frozen=True)
class SemanticProbe:
    name: str
    payload: ProbePayload
    cost: float = 1.0
    risk: float = 0.0

    def __post_init__(self) -> None:
        if self.cost <= 0:
            raise ValueError("probe cost must be positive")
        if self.risk < 0:
            raise ValueError("probe risk must be non-negative")


@dataclass(frozen=True)
class ProbeScore:
    probe: SemanticProbe
    information_gain_bits: float
    expected_posterior_entropy_bits: float
    utility: float


@dataclass(frozen=True)
class InferenceStep:
    probe: str
    observed: Observation
    survivors: tuple[str, ...]


@dataclass(frozen=True)
class InferenceResult:
    status: str
    survivors: tuple[str, ...]
    steps: tuple[InferenceStep, ...]


def _entropy_uniform(count: int) -> float:
    if count <= 1:
        return 0.0
    return log2(count)


def score_probe(
    hypotheses: Sequence[SemanticHypothesis],
    probe: SemanticProbe,
    *,
    risk_weight: float = 1.0,
) -> ProbeScore:
    """Score one probe under a uniform prior over surviving hypotheses."""

    if not hypotheses:
        raise ValueError("at least one hypothesis is required")
    groups: dict[Observation, int] = {}
    for hypothesis in hypotheses:
        outcome = hypothesis.predict(probe.payload)
        groups[outcome] = groups.get(outcome, 0) + 1

    n = len(hypotheses)
    prior = _entropy_uniform(n)
    expected = 0.0
    for size in groups.values():
        probability = size / n
        expected += probability * _entropy_uniform(size)

    gain = prior - expected
    utility = gain / (probe.cost + risk_weight * probe.risk)
    return ProbeScore(
        probe=probe,
        information_gain_bits=gain,
        expected_posterior_entropy_bits=expected,
        utility=utility,
    )


def choose_probe(
    hypotheses: Sequence[SemanticHypothesis],
    probes: Sequence[SemanticProbe],
    *,
    max_risk: float = 0.0,
    risk_weight: float = 1.0,
) -> ProbeScore | None:
    """Choose the highest-utility admissible probe.

    Ties are deterministic by lower risk, then lower cost, then probe name.
    """

    admissible = [probe for probe in probes if probe.risk <= max_risk]
    if not admissible:
        return None
    scores = [
        score_probe(hypotheses, probe, risk_weight=risk_weight)
        for probe in admissible
    ]
    return max(
        scores,
        key=lambda item: (
            item.utility,
            item.information_gain_bits,
            -item.probe.risk,
            -item.probe.cost,
            item.probe.name,
        ),
    )


def active_semantic_inference(
    hypotheses: Sequence[SemanticHypothesis],
    probes: Sequence[SemanticProbe],
    *,
    observe: Callable[[SemanticProbe], Observation],
    max_risk: float = 0.0,
    risk_weight: float = 1.0,
    max_rounds: int = 8,
) -> InferenceResult:
    """Actively identify hidden semantics using safe discriminating probes."""

    survivors = list(hypotheses)
    unused = list(probes)
    steps: list[InferenceStep] = []

    if not survivors:
        raise ValueError("at least one hypothesis is required")

    for _ in range(max_rounds):
        if len(survivors) <= 1:
            break

        choice = choose_probe(
            survivors,
            unused,
            max_risk=max_risk,
            risk_weight=risk_weight,
        )
        if choice is None or choice.information_gain_bits <= 0:
            break

        probe = choice.probe
        observed = observe(probe)
        survivors = [
            hypothesis
            for hypothesis in survivors
            if hypothesis.predict(probe.payload) == observed
        ]
        unused = [candidate for candidate in unused if candidate.name != probe.name]
        steps.append(
            InferenceStep(
                probe=probe.name,
                observed=observed,
                survivors=tuple(item.name for item in survivors),
            )
        )

        if not survivors:
            return InferenceResult(status="no_match", survivors=(), steps=tuple(steps))

    status = "identified" if len(survivors) == 1 else "ambiguous"
    return InferenceResult(
        status=status,
        survivors=tuple(item.name for item in survivors),
        steps=tuple(steps),
    )
