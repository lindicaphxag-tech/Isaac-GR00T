from dataclasses import replace

import pytest

from validation_semrepair.embodied_semantic_observability import (
    SemanticDiagnosisHypothesis,
)
from validation_semrepair.embodied_semantic_probe_synthesis import (
    synthesize_bounded_semantic_probe,
    verify_bounded_semantic_probe,
)
from validation_semrepair.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
)


def _factor(name, transport, evidence):
    return SemanticTransportFactor(name, transport, evidence)


def _one_factor(name, transport):
    return SemanticDiagnosisHypothesis(
        name=name,
        factors=(_factor("boundary", transport, f"{name}/evidence"),),
    )


def test_synthesizes_minimum_sparse_probe_over_frozen_alphabet():
    identity = MonomialSemanticTransport.identity(2)
    double = MonomialSemanticTransport((0, 1), (2.0, 2.0))
    negative = MonomialSemanticTransport((0, 1), (-1.0, -1.0))

    hypotheses = (
        _one_factor("identity", identity),
        _one_factor("double", double),
        _one_factor("negative", negative),
    )
    result = synthesize_bounded_semantic_probe(
        hypotheses,
        alphabet=(-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0),
    )

    assert result.complete
    assert result.objective is not None
    assert result.objective[0] == 1
    assert result.objective[1] == pytest.approx(0.25)
    assert sum(value != 0.0 for value in result.probe) == 1
    assert verify_bounded_semantic_probe(result, hypotheses).valid


def test_noise_margin_can_force_larger_probe_amplitude():
    identity = MonomialSemanticTransport.identity(1)
    scaled = MonomialSemanticTransport((0,), (1.2,))
    hypotheses = (
        _one_factor("identity", identity),
        _one_factor("scaled", scaled),
    )

    result = synthesize_bounded_semantic_probe(
        hypotheses,
        alphabet=(-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0),
        observation_atol=0.02,
        minimum_margin=0.02,
    )

    assert result.complete
    assert result.objective is not None
    assert result.objective[1] == pytest.approx(0.5)
    assert result.minimum_pairwise_separation == pytest.approx(0.1)
    assert verify_bounded_semantic_probe(result, hypotheses).valid


def test_compensating_fault_is_unidentifiable_externally_but_tap_probe_exists():
    identity = MonomialSemanticTransport.identity(2)
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))

    correct = SemanticDiagnosisHypothesis(
        name="correct",
        factors=(
            _factor("producer", identity, "correct/producer"),
            _factor("controller", identity, "correct/controller"),
        ),
    )
    masked = SemanticDiagnosisHypothesis(
        name="double-swap",
        factors=(
            _factor("producer", swap, "masked/producer"),
            _factor("controller", swap, "masked/controller"),
        ),
    )
    hypotheses = (correct, masked)

    external = synthesize_bounded_semantic_probe(
        hypotheses,
        surface="external",
        alphabet=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    internal = synthesize_bounded_semantic_probe(
        hypotheses,
        surface="tap:producer",
        alphabet=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )

    assert external.status == "unidentifiable"
    assert external.unresolved_pairs == (("correct", "double-swap"),)
    assert internal.complete
    assert internal.objective is not None
    assert internal.objective[0] == 1
    assert verify_bounded_semantic_probe(external, hypotheses).valid
    assert verify_bounded_semantic_probe(internal, hypotheses).valid


def test_identical_hypotheses_report_unidentifiable_pair():
    identity = MonomialSemanticTransport.identity(2)
    hypotheses = (
        _one_factor("a", identity),
        _one_factor("b", identity),
    )

    result = synthesize_bounded_semantic_probe(
        hypotheses,
        alphabet=(-1.0, 0.0, 1.0),
    )

    assert result.status == "unidentifiable"
    assert result.probe is None
    assert result.unresolved_pairs == (("a", "b"),)
    assert verify_bounded_semantic_probe(result, hypotheses).valid


def test_probe_certificate_digest_tampering_is_rejected():
    identity = MonomialSemanticTransport.identity(1)
    scaled = MonomialSemanticTransport((0,), (2.0,))
    hypotheses = (
        _one_factor("identity", identity),
        _one_factor("scaled", scaled),
    )
    result = synthesize_bounded_semantic_probe(
        hypotheses,
        alphabet=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    tampered = replace(result, digest="0" * 64)

    verification = verify_bounded_semantic_probe(tampered, hypotheses)

    assert not verification.valid
    assert not verification.digest_matches


def test_dimension_cap_fails_closed():
    identity = MonomialSemanticTransport.identity(3)
    scaled = MonomialSemanticTransport((0, 1, 2), (2.0, 1.0, 1.0))
    hypotheses = (
        _one_factor("identity", identity),
        _one_factor("scaled", scaled),
    )

    with pytest.raises(ValueError, match="capped at dimension"):
        synthesize_bounded_semantic_probe(
            hypotheses,
            max_dimension=2,
        )
