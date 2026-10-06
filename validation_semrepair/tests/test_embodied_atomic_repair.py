import pytest

from validation_semrepair.embodied_atomic_repair import (
    AtomicRepairEvidenceMismatch,
    AtomicRepairRequired,
    authorize_repair_deployment,
    certify_atomic_repair_bundle,
    required_atomic_closure,
)
from validation_semrepair.embodied_repair_interactions import (
    RepairOutcome,
    analyze_repair_lattice,
)


PR1472 = "eed9be164797d41540421bda8adb3840377d7087"
PR1495 = "cdd6db713ffe7edc3e0df3abfab51ea5320c1c0b"


def _maniskill_interaction():
    return analyze_repair_lattice(
        subject="mani-skill delta-pose converter -> PDEEPoseController",
        metric="orientation_error_deg",
        objective="minimize",
        repairs=("pr1472-controller-sign", "pr1495-xyz-euler"),
        outcomes=(
            RepairOutcome(
                frozenset(),
                5.076696288574988,
                "baseline/noncommuting-rotation",
            ),
            RepairOutcome(
                frozenset(("pr1472-controller-sign",)),
                64.74733765443055,
                "factorial/pr1472-only",
            ),
            RepairOutcome(
                frozenset(("pr1495-xyz-euler",)),
                66.12799929304104,
                "factorial/pr1495-only",
            ),
            RepairOutcome(
                frozenset(
                    ("pr1472-controller-sign", "pr1495-xyz-euler")
                ),
                0.0,
                "factorial/both",
            ),
        ),
    )


def _certificate(interaction):
    return certify_atomic_repair_bundle(
        interaction=interaction,
        repairs=("pr1472-controller-sign", "pr1495-xyz-euler"),
        implementations={
            "pr1472-controller-sign": PR1472,
            "pr1495-xyz-euler": PR1495,
        },
    )


def test_maniskill_single_hotfix_is_rejected_as_nonmonotone_intermediate_state():
    interaction = _maniskill_interaction()
    certificate = _certificate(interaction)

    with pytest.raises(AtomicRepairRequired, match="atomically"):
        authorize_repair_deployment(
            interaction=interaction,
            certificate=certificate,
            requested_implementations={
                "pr1495-xyz-euler": PR1495,
            },
        )


def test_exact_two_repair_bundle_is_authorized():
    interaction = _maniskill_interaction()
    certificate = _certificate(interaction)

    authorization = authorize_repair_deployment(
        interaction=interaction,
        certificate=certificate,
        requested_implementations={
            "pr1472-controller-sign": PR1472,
            "pr1495-xyz-euler": PR1495,
        },
    )

    assert authorization.repairs == (
        "pr1472-controller-sign",
        "pr1495-xyz-euler",
    )
    assert authorization.interaction_digest == interaction.digest
    assert authorization.certificate_digest == certificate.digest


def test_commit_drift_in_one_repair_fails_closed():
    interaction = _maniskill_interaction()
    certificate = _certificate(interaction)

    with pytest.raises(AtomicRepairEvidenceMismatch, match="drift"):
        authorize_repair_deployment(
            interaction=interaction,
            certificate=certificate,
            requested_implementations={
                "pr1472-controller-sign": "different-commit",
                "pr1495-xyz-euler": PR1495,
            },
        )


def test_changed_factorial_evidence_invalidates_old_atomic_certificate():
    interaction = _maniskill_interaction()
    certificate = _certificate(interaction)

    changed = analyze_repair_lattice(
        subject=interaction.subject,
        metric=interaction.metric,
        objective=interaction.objective,
        repairs=interaction.repairs,
        outcomes=(
            RepairOutcome(frozenset(), 5.1, "new-baseline"),
            RepairOutcome(
                frozenset(("pr1472-controller-sign",)),
                64.74733765443055,
                "factorial/pr1472-only",
            ),
            RepairOutcome(
                frozenset(("pr1495-xyz-euler",)),
                66.12799929304104,
                "factorial/pr1495-only",
            ),
            RepairOutcome(
                frozenset(
                    ("pr1472-controller-sign", "pr1495-xyz-euler")
                ),
                0.0,
                "factorial/both",
            ),
        ),
    )

    with pytest.raises(AtomicRepairEvidenceMismatch, match="digest changed"):
        authorize_repair_deployment(
            interaction=changed,
            certificate=certificate,
            requested_implementations={
                "pr1472-controller-sign": PR1472,
                "pr1495-xyz-euler": PR1495,
            },
        )



def test_requested_single_repair_expands_to_minimal_atomic_closure():
    interaction = _maniskill_interaction()

    assert required_atomic_closure(
        interaction=interaction,
        requested_repairs=("pr1495-xyz-euler",),
    ) == (
        "pr1472-controller-sign",
        "pr1495-xyz-euler",
    )

    assert required_atomic_closure(
        interaction=interaction,
        requested_repairs=(
            "pr1472-controller-sign",
            "pr1495-xyz-euler",
        ),
    ) == (
        "pr1472-controller-sign",
        "pr1495-xyz-euler",
    )



def test_extra_uncertified_repair_is_rejected():
    interaction = _maniskill_interaction()
    certificate = _certificate(interaction)

    with pytest.raises(
        AtomicRepairEvidenceMismatch,
        match="does not authorize extra repairs",
    ):
        authorize_repair_deployment(
            interaction=interaction,
            certificate=certificate,
            requested_implementations={
                "pr1472-controller-sign": PR1472,
                "pr1495-xyz-euler": PR1495,
                "unreviewed-extra-change": "deadbeef",
            },
        )
