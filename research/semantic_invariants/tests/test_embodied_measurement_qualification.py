import pytest

from research.semantic_invariants.embodied_measurement_qualification import (
    MeasurementNotQualified,
    MeasurementWorld,
    qualify_measurement,
    require_qualified_measurement,
)


def test_external_task_anchor_plus_exact_repeats_qualifies_measurement():
    cert = qualify_measurement(
        measurement_id="maniskill/peg-insertion/task-success-v1",
        semantic_anchor_id="env/PegInsertionSide-v1/success",
        repeat_outcome_digests=("success-set:013456789",) * 5,
        worlds=(
            MeasurementWorld(
                evidence_id="main-repeatability",
                measurement_digest="success-set:013456789",
                semantic_target_digest="task-outcome:success/failure",
            ),
        ),
    )

    assert cert.repeatable
    assert cert.identifiable
    assert cert.qualified
    assert cert.decision == "measurement_qualified"
    require_qualified_measurement(cert)


def test_perfect_roundtrip_can_be_non_identifying():
    cert = qualify_measurement(
        measurement_id="lerobot/relative-action-roundtrip-v1",
        semantic_anchor_id=None,
        repeat_outcome_digests=("roundtrip-error:0",) * 32,
        worlds=(
            MeasurementWorld(
                evidence_id="correct-anchor-world",
                measurement_digest="roundtrip-error:0",
                semantic_target_digest="forward-target:correct",
            ),
            MeasurementWorld(
                evidence_id="wrong-prefix-anchor-world",
                measurement_digest="roundtrip-error:0",
                semantic_target_digest="forward-target:wrong",
            ),
        ),
    )

    assert cert.repeatable
    assert not cert.identifiable
    assert not cert.qualified
    assert cert.decision == "non_identifying_measurement"

    with pytest.raises(MeasurementNotQualified, match="not qualified"):
        require_qualified_measurement(cert)


def test_external_anchor_does_not_rescue_non_repeatable_measurement():
    cert = qualify_measurement(
        measurement_id="unstable/task-success",
        semantic_anchor_id="env/task/success",
        repeat_outcome_digests=("set:A", "set:B", "set:A"),
    )

    assert not cert.repeatable
    assert cert.identifiable
    assert not cert.qualified
    assert cert.decision == "non_repeatable_measurement"


def test_collision_blocks_identifiability_even_with_named_anchor():
    cert = qualify_measurement(
        measurement_id="ambiguous-channel",
        semantic_anchor_id="nominal/external-anchor",
        repeat_outcome_digests=("same-observation",) * 3,
        worlds=(
            MeasurementWorld("w1", "same-observation", "semantic:A"),
            MeasurementWorld("w2", "same-observation", "semantic:B"),
        ),
    )

    assert not cert.identifiable
    assert any("non-identifying" in reason for reason in cert.reasons)


def test_self_attested_measurement_qualification_is_not_authority():
    """A caller may not change the bool, digest, or anchor after issuance."""
    from dataclasses import replace

    nonidentifying = qualify_measurement(
        measurement_id="untrusted-roundtrip",
        semantic_anchor_id=None,
        repeat_outcome_digests=("same", "same", "same"),
    )
    assert not nonidentifying.qualified
    for forged in (
        replace(nonidentifying, qualified=True),
        replace(nonidentifying, qualified=True, decision="measurement_qualified"),
        replace(nonidentifying, semantic_anchor_id="invented/oracle"),
    ):
        with pytest.raises(MeasurementNotQualified, match="integrity"):
            require_qualified_measurement(forged)

    valid = qualify_measurement(
        measurement_id="qualified-source",
        semantic_anchor_id="oracle/origin",
        repeat_outcome_digests=("source-digest",) * 3,
    )
    require_qualified_measurement(valid)
    for forged in (
        replace(valid, repeat_outcome_digests=("fake",) * 3),
        replace(valid, digest="0" * 64),
        replace(valid, minimum_repeats=10),
    ):
        with pytest.raises(MeasurementNotQualified, match="integrity"):
            require_qualified_measurement(forged)
