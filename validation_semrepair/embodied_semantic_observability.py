"""Observability analysis for semantic transport faults.

End-to-end task success is not a complete semantic oracle.  If non-identity
boundary transports compose to identity, the faulty chain is observationally
equivalent to a correct identity chain for every possible input at the external
I/O boundary. No black-box input probe can distinguish them.

For a non-identity monomial net transport, however, a canonical basis vector is
always sufficient to expose at least one wrong ordering/sign/scale component.
For an exactly canceling chain, this module instead synthesizes the earliest
internal boundary tap and basis probe that exposes the hidden fault.

This turns semantic masking into an explicit observability problem rather than
a heuristic test-coverage problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping, Sequence

from .embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
    compose_transport_chain,
)


@dataclass(frozen=True)
class BoundaryProbe:
    after_factor: str | None
    basis_index: int
    expected: tuple[float, ...]
    observed: tuple[float, ...]
    residual_linf: float

    @property
    def is_distinguishing(self) -> bool:
        return self.residual_linf > 0.0


@dataclass(frozen=True)
class SemanticObservabilityCertificate:
    factor_names: tuple[str, ...]
    nonidentity_factors: tuple[str, ...]
    net_identity: bool
    end_to_end_identifiable: bool
    black_box_basis_witnesses: tuple[BoundaryProbe, ...]
    internal_boundary_witnesses: tuple[BoundaryProbe, ...]
    earliest_required_tap: str | None
    digest: str

    @property
    def black_box_impossibility(self) -> bool:
        return (
            self.net_identity
            and bool(self.nonidentity_factors)
            and not self.end_to_end_identifiable
        )


def _basis(dimension: int, index: int) -> tuple[float, ...]:
    return tuple(1.0 if i == index else 0.0 for i in range(dimension))


def _residual_linf(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def _distinguishing_basis_probes(
    transport: MonomialSemanticTransport,
    *,
    after_factor: str | None,
    atol: float,
) -> tuple[BoundaryProbe, ...]:
    probes: list[BoundaryProbe] = []
    for index in range(transport.dimension):
        expected = _basis(transport.dimension, index)
        observed = transport.apply(expected)
        residual = _residual_linf(expected, observed)
        if residual > atol:
            probes.append(
                BoundaryProbe(
                    after_factor=after_factor,
                    basis_index=index,
                    expected=expected,
                    observed=observed,
                    residual_linf=residual,
                )
            )
    return tuple(probes)


def _digest_payload(
    factors: Sequence[SemanticTransportFactor],
    *,
    atol: float,
) -> str:
    payload = {
        "atol": atol,
        "factors": [
            {
                "name": factor.name,
                "source_for_output": list(factor.transport.source_for_output),
                "scale": list(factor.transport.scale),
                "evidence_id": factor.evidence_id,
            }
            for factor in factors
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def analyze_semantic_observability(
    factors: Sequence[SemanticTransportFactor],
    *,
    atol: float = 0.0,
) -> SemanticObservabilityCertificate:
    """Synthesize black-box or internal witnesses for a transport chain.

    For the supported monomial class, a non-identity net transport must differ
    from identity on at least one canonical basis vector.  If the net transport
    is identity but at least one factor is non-identity, external I/O behavior
    cannot expose the hidden semantic faults at all; the earliest non-identity
    prefix becomes the required evidence tap.
    """

    if atol < 0:
        raise ValueError("atol must be non-negative")
    items = tuple(factors)
    if not items:
        raise ValueError("at least one factor is required")

    dimension = items[0].transport.dimension
    if any(item.transport.dimension != dimension for item in items):
        raise ValueError("all factors must have the same dimension")

    nonidentity = tuple(
        item.name
        for item in items
        if not item.transport.is_identity(atol=atol)
    )
    net = compose_transport_chain(items)
    net_identity = net.is_identity(atol=atol)

    black_box = (
        ()
        if net_identity
        else _distinguishing_basis_probes(
            net,
            after_factor=None,
            atol=atol,
        )
    )

    internal: list[BoundaryProbe] = []
    earliest_tap: str | None = None
    prefix = MonomialSemanticTransport.identity(dimension)
    for item in items:
        prefix = prefix.then(item.transport)
        probes = _distinguishing_basis_probes(
            prefix,
            after_factor=item.name,
            atol=atol,
        )
        if probes:
            if earliest_tap is None:
                earliest_tap = item.name
            # One minimal witness per non-identity prefix is enough to prove
            # that the semantic state at that boundary differs from identity.
            internal.append(probes[0])

    return SemanticObservabilityCertificate(
        factor_names=tuple(item.name for item in items),
        nonidentity_factors=nonidentity,
        net_identity=net_identity,
        end_to_end_identifiable=bool(black_box),
        black_box_basis_witnesses=black_box,
        internal_boundary_witnesses=tuple(internal),
        earliest_required_tap=earliest_tap if net_identity and nonidentity else None,
        digest=_digest_payload(items, atol=atol),
    )


@dataclass(frozen=True)
class SemanticDiagnosisHypothesis:
    """One candidate semantic transport chain for diagnosis."""

    name: str
    factors: tuple[SemanticTransportFactor, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("hypothesis name must be non-empty")
        if not self.factors:
            raise ValueError("hypothesis requires at least one factor")


@dataclass(frozen=True)
class PairwiseDiagnosticWitness:
    left: str
    right: str
    tap_after_factor: str | None
    basis_index: int
    left_observed: tuple[float, ...]
    right_observed: tuple[float, ...]
    residual_linf: float


@dataclass(frozen=True)
class SemanticDiagnosisPlan:
    """Minimum-cost internal tap plan for a finite frozen hypothesis set."""

    hypothesis_names: tuple[str, ...]
    factor_names: tuple[str, ...]
    external_output_observed: bool
    selected_internal_taps: tuple[str, ...]
    total_internal_cost: float
    pair_witnesses: tuple[PairwiseDiagnosticWitness, ...]
    externally_resolved_pairs: tuple[tuple[str, str], ...]
    internally_resolved_pairs: tuple[tuple[str, str], ...]
    intrinsically_unidentifiable_pairs: tuple[tuple[str, str], ...]
    digest: str

    @property
    def complete(self) -> bool:
        return not self.intrinsically_unidentifiable_pairs


def _prefix_transports(
    factors: Sequence[SemanticTransportFactor],
) -> tuple[MonomialSemanticTransport, ...]:
    dimension = factors[0].transport.dimension
    prefix = MonomialSemanticTransport.identity(dimension)
    result: list[MonomialSemanticTransport] = []
    for factor in factors:
        prefix = prefix.then(factor.transport)
        result.append(prefix)
    return tuple(result)


def _transport_pair_witness(
    left: MonomialSemanticTransport,
    right: MonomialSemanticTransport,
    *,
    left_name: str,
    right_name: str,
    tap_after_factor: str | None,
    atol: float,
) -> PairwiseDiagnosticWitness | None:
    if left.dimension != right.dimension:
        raise ValueError("hypothesis transport dimensions differ")
    for index in range(left.dimension):
        basis = _basis(left.dimension, index)
        lhs = left.apply(basis)
        rhs = right.apply(basis)
        residual = _residual_linf(lhs, rhs)
        if residual > atol:
            return PairwiseDiagnosticWitness(
                left=left_name,
                right=right_name,
                tap_after_factor=tap_after_factor,
                basis_index=index,
                left_observed=lhs,
                right_observed=rhs,
                residual_linf=residual,
            )
    return None


def _diagnosis_digest(
    hypotheses: Sequence[SemanticDiagnosisHypothesis],
    *,
    external_output_observed: bool,
    tap_costs: Mapping[str, float],
    atol: float,
) -> str:
    payload = {
        "external_output_observed": external_output_observed,
        "atol": atol,
        "tap_costs": dict(sorted(tap_costs.items())),
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
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def synthesize_minimal_diagnostic_taps(
    hypotheses: Sequence[SemanticDiagnosisHypothesis],
    *,
    tap_costs: Mapping[str, float] | None = None,
    external_output_observed: bool = True,
    atol: float = 0.0,
) -> SemanticDiagnosisPlan:
    """Find a minimum-cost set of internal taps that separates hypotheses.

    The solver is exact over the frozen finite hypothesis set. External output
    is treated as an already-available observation when requested. Remaining
    ambiguous hypothesis pairs are covered by internal boundary taps.

    For monomial semantic transports, canonical basis probes are complete for
    pairwise transport discrimination. If two hypotheses remain identical at
    every admissible boundary, the plan reports them as intrinsically
    unidentifiable instead of inventing a diagnosis.
    """

    from itertools import combinations

    if atol < 0:
        raise ValueError("atol must be non-negative")

    items = tuple(hypotheses)
    if len(items) < 2:
        raise ValueError("at least two hypotheses are required")
    names = tuple(item.name for item in items)
    if len(set(names)) != len(names):
        raise ValueError("hypothesis names must be unique")

    factor_names = tuple(factor.name for factor in items[0].factors)
    if len(set(factor_names)) != len(factor_names):
        raise ValueError("factor names must be unique within a hypothesis")
    dimension = items[0].factors[0].transport.dimension

    for item in items:
        if tuple(factor.name for factor in item.factors) != factor_names:
            raise ValueError(
                "all hypotheses must use the same ordered factor names"
            )
        if any(
            factor.transport.dimension != dimension
            for factor in item.factors
        ):
            raise ValueError(
                "all hypothesis transports must share one dimension"
            )

    candidate_tap_names = (
        factor_names[:-1] if external_output_observed else factor_names
    )
    costs = {name: 1.0 for name in candidate_tap_names}
    if tap_costs is not None:
        unknown_costs = set(tap_costs) - set(costs)
        if unknown_costs:
            raise ValueError(
                "tap_costs references non-internal boundaries: "
                f"{sorted(unknown_costs)}"
            )
        for name, value in tap_costs.items():
            numeric = float(value)
            if (not isfinite(numeric)) or numeric < 0:
                raise ValueError(
                    "tap costs must be finite and non-negative"
                )
            costs[name] = numeric

    prefixes = {
        item.name: _prefix_transports(item.factors)
        for item in items
    }
    pairs = tuple(
        (items[i].name, items[j].name)
        for i in range(len(items))
        for j in range(i + 1, len(items))
    )

    external_witness: dict[
        tuple[str, str], PairwiseDiagnosticWitness
    ] = {}
    internal_witnesses: dict[
        str, dict[tuple[str, str], PairwiseDiagnosticWitness]
    ] = {name: {} for name in costs}

    for left_name, right_name in pairs:
        left_prefixes = prefixes[left_name]
        right_prefixes = prefixes[right_name]

        if external_output_observed:
            witness = _transport_pair_witness(
                left_prefixes[-1],
                right_prefixes[-1],
                left_name=left_name,
                right_name=right_name,
                tap_after_factor=None,
                atol=atol,
            )
            if witness is not None:
                external_witness[(left_name, right_name)] = witness

        for index, tap_name in enumerate(factor_names):
            if tap_name not in internal_witnesses:
                continue
            witness = _transport_pair_witness(
                left_prefixes[index],
                right_prefixes[index],
                left_name=left_name,
                right_name=right_name,
                tap_after_factor=tap_name,
                atol=atol,
            )
            if witness is not None:
                internal_witnesses[tap_name][
                    (left_name, right_name)
                ] = witness

    unresolved = set(pairs) - set(external_witness)
    observable_by_any_internal: set[tuple[str, str]] = set()
    for mapping in internal_witnesses.values():
        observable_by_any_internal.update(mapping)
    intrinsically_unidentifiable = tuple(
        sorted(unresolved - observable_by_any_internal)
    )
    cover_target = unresolved - set(intrinsically_unidentifiable)

    candidate_taps = tuple(sorted(costs))
    best: tuple[float, int, tuple[str, ...]] | None = None
    for size in range(len(candidate_taps) + 1):
        for subset in combinations(candidate_taps, size):
            covered: set[tuple[str, str]] = set()
            for tap in subset:
                covered.update(internal_witnesses[tap])
            if not cover_target <= covered:
                continue
            score = (
                sum(costs[tap] for tap in subset),
                len(subset),
                subset,
            )
            if best is None or score < best:
                best = score

    if best is None:
        raise RuntimeError("no diagnostic tap cover found")

    selected = best[2]
    witnesses: list[PairwiseDiagnosticWitness] = list(
        external_witness.values()
    )
    internally_resolved: list[tuple[str, str]] = []
    for pair in sorted(cover_target):
        candidates = [
            internal_witnesses[tap][pair]
            for tap in selected
            if pair in internal_witnesses[tap]
        ]
        if not candidates:
            raise RuntimeError(
                f"selected taps do not cover pair {pair!r}"
            )
        witness = min(
            candidates,
            key=lambda item: (
                costs[item.tap_after_factor or ""],
                item.tap_after_factor or "",
                item.basis_index,
            ),
        )
        witnesses.append(witness)
        internally_resolved.append(pair)

    return SemanticDiagnosisPlan(
        hypothesis_names=names,
        factor_names=factor_names,
        external_output_observed=external_output_observed,
        selected_internal_taps=selected,
        total_internal_cost=best[0],
        pair_witnesses=tuple(
            sorted(
                witnesses,
                key=lambda item: (
                    item.left,
                    item.right,
                    item.tap_after_factor or "~output",
                    item.basis_index,
                ),
            )
        ),
        externally_resolved_pairs=tuple(sorted(external_witness)),
        internally_resolved_pairs=tuple(internally_resolved),
        intrinsically_unidentifiable_pairs=intrinsically_unidentifiable,
        digest=_diagnosis_digest(
            items,
            external_output_observed=external_output_observed,
            tap_costs=costs,
            atol=atol,
        ),
    )
