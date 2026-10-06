"""Algebraic pre-screen for canceling embodied semantic transports.

Many silent robot-interface faults are monomial transforms: joint/feature
permutations, sign conventions, and per-axis unit/scale changes. Individual
faults can be wrong while their composition is exactly (or numerically) the
identity, making an end-to-end task test unable to distinguish the faulty chain
from a correct one.

This module provides a dependency-light algebraic pre-screen. It does not
replace physical factorial validation; it identifies high-risk chains where
factorial repair analysis should be mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Sequence


Vector = tuple[float, ...]


@dataclass(frozen=True)
class MonomialSemanticTransport:
    """y[i] = scale[i] * x[source_for_output[i]]."""

    source_for_output: tuple[int, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.source_for_output)
        if n == 0:
            raise ValueError("transport dimension must be positive")
        if len(self.scale) != n:
            raise ValueError("scale/permutation dimensions differ")
        if set(self.source_for_output) != set(range(n)):
            raise ValueError("source_for_output must be a permutation")
        if any((not isfinite(value)) or value == 0.0 for value in self.scale):
            raise ValueError("all transport scales must be finite and non-zero")

    @classmethod
    def identity(cls, dimension: int) -> "MonomialSemanticTransport":
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        return cls(tuple(range(dimension)), (1.0,) * dimension)

    @property
    def dimension(self) -> int:
        return len(self.source_for_output)

    def apply(self, value: Sequence[float]) -> Vector:
        if len(value) != self.dimension:
            raise ValueError("value/transport dimensions differ")
        return tuple(
            self.scale[i] * float(value[self.source_for_output[i]])
            for i in range(self.dimension)
        )

    def then(
        self,
        other: "MonomialSemanticTransport",
    ) -> "MonomialSemanticTransport":
        """Compose transports in execution order: other(self(x))."""

        if self.dimension != other.dimension:
            raise ValueError("transport dimensions differ")
        source = tuple(
            self.source_for_output[other.source_for_output[i]]
            for i in range(self.dimension)
        )
        scale = tuple(
            other.scale[i] * self.scale[other.source_for_output[i]]
            for i in range(self.dimension)
        )
        return MonomialSemanticTransport(source, scale)

    def inverse(self) -> "MonomialSemanticTransport":
        source = [0] * self.dimension
        scale = [0.0] * self.dimension
        for output_index, source_index in enumerate(self.source_for_output):
            source[source_index] = output_index
            scale[source_index] = 1.0 / self.scale[output_index]
        return MonomialSemanticTransport(tuple(source), tuple(scale))

    def is_identity(self, *, atol: float = 0.0) -> bool:
        if self.source_for_output != tuple(range(self.dimension)):
            return False
        return all(abs(value - 1.0) <= atol for value in self.scale)


@dataclass(frozen=True)
class SemanticTransportFactor:
    name: str
    transport: MonomialSemanticTransport
    evidence_id: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("transport factor name must be non-empty")
        if not self.evidence_id:
            raise ValueError("transport factor requires evidence_id")


@dataclass(frozen=True)
class TransportCancellationCertificate:
    factors: tuple[SemanticTransportFactor, ...]
    net: MonomialSemanticTransport
    exact_or_tolerant_identity: bool
    nonidentity_factors: tuple[str, ...]
    repair_one_unmasks: tuple[str, ...]
    digest: str

    @property
    def structurally_masked(self) -> bool:
        return (
            self.exact_or_tolerant_identity
            and len(self.nonidentity_factors) >= 2
            and bool(self.repair_one_unmasks)
        )


def compose_transport_chain(
    factors: Sequence[SemanticTransportFactor],
) -> MonomialSemanticTransport:
    if not factors:
        raise ValueError("at least one transport factor is required")
    dimension = factors[0].transport.dimension
    net = MonomialSemanticTransport.identity(dimension)
    for factor in factors:
        if factor.transport.dimension != dimension:
            raise ValueError("all transport factors must have the same dimension")
        net = net.then(factor.transport)
    return net


def _digest(
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


def analyze_transport_cancellation(
    factors: Sequence[SemanticTransportFactor],
    *,
    atol: float = 0.0,
) -> TransportCancellationCertificate:
    """Detect a structurally masked transform chain.

    The repair-one-unmasks field lists locally faulty factors whose replacement
    with an identity transform makes the net chain non-identity. Those factors
    are high-priority candidates for factorial repair analysis and atomic
    deployment gating.
    """

    if atol < 0:
        raise ValueError("atol must be non-negative")
    items = tuple(factors)
    net = compose_transport_chain(items)
    nonidentity = tuple(
        factor.name
        for factor in items
        if not factor.transport.is_identity(atol=atol)
    )

    repair_one_unmasks: list[str] = []
    if net.is_identity(atol=atol):
        for index, factor in enumerate(items):
            if factor.transport.is_identity(atol=atol):
                continue
            replacement = SemanticTransportFactor(
                name=f"{factor.name}/repaired",
                transport=MonomialSemanticTransport.identity(
                    factor.transport.dimension
                ),
                evidence_id=f"{factor.evidence_id}/identity-repair",
            )
            counterfactual = (
                items[:index] + (replacement,) + items[index + 1 :]
            )
            if not compose_transport_chain(counterfactual).is_identity(atol=atol):
                repair_one_unmasks.append(factor.name)

    return TransportCancellationCertificate(
        factors=items,
        net=net,
        exact_or_tolerant_identity=net.is_identity(atol=atol),
        nonidentity_factors=nonidentity,
        repair_one_unmasks=tuple(repair_one_unmasks),
        digest=_digest(items, atol=atol),
    )
