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
from typing import Sequence

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
