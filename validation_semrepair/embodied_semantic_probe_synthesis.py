"""Exact bounded synthesis of distinguishing embodied-semantic probes.

The adaptive experiment-design layer can optimize over a frozen experiment
table. This module removes one remaining manual step for the monomial semantic
transport class: constructing candidate physical probes.

Given a finite hypothesis set, an observation surface, a bounded discrete probe
alphabet, and an observation-error bound, the synthesizer searches the complete
finite probe space and returns the minimum-cost single probe whose predicted
outcomes remain pairwise separated after accounting for bounded error.

The optimization order is:

1. number of actuated dimensions (sparsity),
2. maximum absolute probe amplitude,
3. L1 amplitude,
4. lexicographic probe value.

Optimality is only claimed over the supplied finite alphabet and surface.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import combinations, product
import json
from math import isfinite
from typing import Sequence

from .embodied_semantic_observability import SemanticDiagnosisHypothesis
from .embodied_semantic_transport import MonomialSemanticTransport


@dataclass(frozen=True)
class ProbePairSeparation:
    left: str
    right: str
    distance_linf: float


@dataclass(frozen=True)
class SemanticProbeSynthesisResult:
    status: str
    surface: str
    probe: tuple[float, ...] | None
    objective: tuple[int, float, float, tuple[float, ...]] | None
    minimum_pairwise_separation: float | None
    pairwise_separations: tuple[ProbePairSeparation, ...]
    alphabet: tuple[float, ...]
    observation_atol: float
    minimum_margin: float
    candidates_evaluated: int
    hypothesis_names: tuple[str, ...]
    unresolved_pairs: tuple[tuple[str, str], ...]
    digest: str

    @property
    def complete(self) -> bool:
        return self.status == "optimal" and self.probe is not None


@dataclass(frozen=True)
class ProbeSynthesisVerification:
    valid: bool
    independently_optimal_probe: tuple[float, ...] | None
    digest_matches: bool
    message: str


def _prefix_transport(
    hypothesis: SemanticDiagnosisHypothesis,
    surface: str,
) -> MonomialSemanticTransport:
    dimension = hypothesis.factors[0].transport.dimension
    transport = MonomialSemanticTransport.identity(dimension)

    if surface == "external":
        for factor in hypothesis.factors:
            transport = transport.then(factor.transport)
        return transport

    prefix = "tap:"
    if not surface.startswith(prefix):
        raise ValueError("surface must be 'external' or 'tap:<factor-name>'")
    target = surface[len(prefix):]
    for factor in hypothesis.factors:
        transport = transport.then(factor.transport)
        if factor.name == target:
            return transport
    raise ValueError(f"unknown tap factor {target!r}")


def _linf(left: Sequence[float], right: Sequence[float]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def _probe_objective(
    probe: tuple[float, ...],
) -> tuple[int, float, float, tuple[float, ...]]:
    nonzero = tuple(value for value in probe if value != 0.0)
    support = len(nonzero)
    max_abs = max((abs(value) for value in nonzero), default=0.0)
    l1 = sum(abs(value) for value in nonzero)
    return support, max_abs, l1, probe


def _candidate_probes(
    dimension: int,
    alphabet: Sequence[float],
):
    nonzero_levels = tuple(sorted({float(x) for x in alphabet if float(x) != 0.0}))
    if not nonzero_levels:
        return
    candidates: list[tuple[tuple[int, float, float, tuple[float, ...]], tuple[float, ...]]] = []
    for support_size in range(1, dimension + 1):
        for indices in combinations(range(dimension), support_size):
            for values in product(nonzero_levels, repeat=support_size):
                probe = [0.0] * dimension
                for index, value in zip(indices, values, strict=True):
                    probe[index] = value
                item = tuple(probe)
                candidates.append((_probe_objective(item), item))
    for _, probe in sorted(candidates, key=lambda item: item[0]):
        yield probe


def _pairwise_separations(
    hypothesis_names: Sequence[str],
    transports: dict[str, MonomialSemanticTransport],
    probe: tuple[float, ...],
) -> tuple[ProbePairSeparation, ...]:
    outputs = {
        name: transports[name].apply(probe)
        for name in hypothesis_names
    }
    result: list[ProbePairSeparation] = []
    for i, left in enumerate(hypothesis_names):
        for right in hypothesis_names[i + 1:]:
            result.append(
                ProbePairSeparation(
                    left=left,
                    right=right,
                    distance_linf=_linf(outputs[left], outputs[right]),
                )
            )
    return tuple(result)


def _is_robustly_distinguishing(
    separations: Sequence[ProbePairSeparation],
    *,
    observation_atol: float,
    minimum_margin: float,
) -> bool:
    threshold = 2.0 * observation_atol + minimum_margin
    return all(item.distance_linf > threshold for item in separations)


def _unresolved_pairs_over_space(
    hypothesis_names: Sequence[str],
    transports: dict[str, MonomialSemanticTransport],
    candidates: Sequence[tuple[float, ...]],
    *,
    observation_atol: float,
    minimum_margin: float,
) -> tuple[tuple[str, str], ...]:
    threshold = 2.0 * observation_atol + minimum_margin
    unresolved: list[tuple[str, str]] = []
    for i, left in enumerate(hypothesis_names):
        for right in hypothesis_names[i + 1:]:
            distinguishable = False
            for probe in candidates:
                lhs = transports[left].apply(probe)
                rhs = transports[right].apply(probe)
                if _linf(lhs, rhs) > threshold:
                    distinguishable = True
                    break
            if not distinguishable:
                unresolved.append((left, right))
    return tuple(unresolved)


def _digest_payload(result: SemanticProbeSynthesisResult) -> str:
    payload = {
        "status": result.status,
        "surface": result.surface,
        "probe": result.probe,
        "objective": result.objective,
        "minimum_pairwise_separation": result.minimum_pairwise_separation,
        "pairwise_separations": [
            {
                "left": item.left,
                "right": item.right,
                "distance_linf": item.distance_linf,
            }
            for item in result.pairwise_separations
        ],
        "alphabet": result.alphabet,
        "observation_atol": result.observation_atol,
        "minimum_margin": result.minimum_margin,
        "candidates_evaluated": result.candidates_evaluated,
        "hypothesis_names": result.hypothesis_names,
        "unresolved_pairs": result.unresolved_pairs,
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def synthesize_bounded_semantic_probe(
    hypotheses: Sequence[SemanticDiagnosisHypothesis],
    *,
    surface: str = "external",
    alphabet: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    observation_atol: float = 0.0,
    minimum_margin: float = 0.0,
    max_dimension: int = 8,
    max_candidates: int = 200_000,
) -> SemanticProbeSynthesisResult:
    """Return the exact minimum-cost robust single probe over a finite alphabet."""

    items = tuple(hypotheses)
    if len(items) < 2:
        raise ValueError("at least two hypotheses are required")
    names = tuple(item.name for item in items)
    if len(set(names)) != len(names):
        raise ValueError("hypothesis names must be unique")
    if not isfinite(observation_atol) or observation_atol < 0:
        raise ValueError("observation_atol must be finite and non-negative")
    if not isfinite(minimum_margin) or minimum_margin < 0:
        raise ValueError("minimum_margin must be finite and non-negative")

    levels = tuple(sorted({float(value) for value in alphabet}))
    if not levels or any(not isfinite(value) for value in levels):
        raise ValueError("alphabet must contain finite values")
    if all(value == 0.0 for value in levels):
        raise ValueError("alphabet must contain a non-zero value")

    dimension = items[0].factors[0].transport.dimension
    if dimension > max_dimension:
        raise ValueError(
            f"exact probe synthesis is capped at dimension {max_dimension}"
        )
    factor_names = tuple(factor.name for factor in items[0].factors)
    for item in items:
        if tuple(factor.name for factor in item.factors) != factor_names:
            raise ValueError("hypothesis factor layouts differ")
        if any(
            factor.transport.dimension != dimension
            for factor in item.factors
        ):
            raise ValueError("hypothesis dimensions differ")

    transports = {
        item.name: _prefix_transport(item, surface)
        for item in items
    }
    candidate_list = list(_candidate_probes(dimension, levels))
    if len(candidate_list) > max_candidates:
        raise ValueError(
            f"candidate space {len(candidate_list)} exceeds cap {max_candidates}"
        )

    evaluated = 0
    for probe in candidate_list:
        evaluated += 1
        separations = _pairwise_separations(names, transports, probe)
        if not _is_robustly_distinguishing(
            separations,
            observation_atol=observation_atol,
            minimum_margin=minimum_margin,
        ):
            continue
        minimum = min(item.distance_linf for item in separations)
        result = SemanticProbeSynthesisResult(
            status="optimal",
            surface=surface,
            probe=probe,
            objective=_probe_objective(probe),
            minimum_pairwise_separation=minimum,
            pairwise_separations=separations,
            alphabet=levels,
            observation_atol=observation_atol,
            minimum_margin=minimum_margin,
            candidates_evaluated=evaluated,
            hypothesis_names=names,
            unresolved_pairs=(),
            digest="",
        )
        return replace(result, digest=_digest_payload(result))

    unresolved = _unresolved_pairs_over_space(
        names,
        transports,
        candidate_list,
        observation_atol=observation_atol,
        minimum_margin=minimum_margin,
    )
    result = SemanticProbeSynthesisResult(
        status=(
            "unidentifiable"
            if unresolved
            else "no_single_probe"
        ),
        surface=surface,
        probe=None,
        objective=None,
        minimum_pairwise_separation=None,
        pairwise_separations=(),
        alphabet=levels,
        observation_atol=observation_atol,
        minimum_margin=minimum_margin,
        candidates_evaluated=len(candidate_list),
        hypothesis_names=names,
        unresolved_pairs=unresolved,
        digest="",
    )
    return replace(result, digest=_digest_payload(result))


def verify_bounded_semantic_probe(
    result: SemanticProbeSynthesisResult,
    hypotheses: Sequence[SemanticDiagnosisHypothesis],
    *,
    max_dimension: int = 8,
    max_candidates: int = 200_000,
) -> ProbeSynthesisVerification:
    """Independently rerun finite synthesis and compare optimum + digest."""

    expected_digest = _digest_payload(replace(result, digest=""))
    digest_matches = expected_digest == result.digest

    independent = synthesize_bounded_semantic_probe(
        hypotheses,
        surface=result.surface,
        alphabet=result.alphabet,
        observation_atol=result.observation_atol,
        minimum_margin=result.minimum_margin,
        max_dimension=max_dimension,
        max_candidates=max_candidates,
    )
    valid = (
        digest_matches
        and independent.status == result.status
        and independent.probe == result.probe
        and independent.objective == result.objective
        and independent.unresolved_pairs == result.unresolved_pairs
    )
    return ProbeSynthesisVerification(
        valid=valid,
        independently_optimal_probe=independent.probe,
        digest_matches=digest_matches,
        message=(
            "bounded probe optimum verified"
            if valid
            else "probe optimum, status, unresolved pairs, or digest mismatch"
        ),
    )
