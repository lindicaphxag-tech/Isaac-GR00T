import math

import pytest

from research.semantic_invariants.embodied_atomic_repair import (
    AtomicRepairRequired,
    certify_atomic_repair_bundle,
)
from research.semantic_invariants.embodied_authority_kernel import (
    LocalRepairProof,
    PreDispatchAuthorityRejected,
    authorize_pre_dispatch,
)
from research.semantic_invariants.embodied_execution_domain import (
    L2BallExecutionDomain,
    ProjectionAuthorityRequired,
    issue_projection_certificate,
)
from research.semantic_invariants.embodied_repair_interactions import (
    RepairOutcome,
    analyze_repair_lattice,
)
from research.semantic_invariants.embodied_repair_synthesis import (
    RepairExample,
    RepairPrimitive,
    RepairProgram,
)
from research.semantic_invariants.embodied_repair_verification import (
    verify_repair_against_heldout,
)


def _identity_a(value, context):
    return tuple(value)


def _identity_b(value, context):
    return tuple(value)


def _reverse(value, context):
    return tuple(reversed(value))


def _euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _proof(name, fn):
    program = RepairProgram(
        (
            RepairPrimitive(
                name=name,
                family="test",
                cost=1,
                apply_fn=fn,
                implementation_id=f"{name}@v1",
            ),
        )
    )
    contract = f"contract/{name}@v1"
    certificate = verify_repair_against_heldout(
        contract_id=contract,
        program=program,
        heldout=(RepairExample((0.25,), (0.25,)),),
        verifier_id="authority-kernel-test-bank",
    )
    return LocalRepairProof(
        repair_name=name,
        contract_id=contract,
        program=program,
        certificate=certificate,
    )


def _interaction():
    return analyze_repair_lattice(
        subject="compensating-two-boundary-test",
        metric="semantic_error",
        objective="minimize",
        repairs=("repair-a", "repair-b"),
        outcomes=(
            RepairOutcome(frozenset(), 1.0, "baseline"),
            RepairOutcome(frozenset(("repair-a",)), 2.0, "a-only"),
            RepairOutcome(frozenset(("repair-b",)), 2.5, "b-only"),
            RepairOutcome(
                frozenset(("repair-a", "repair-b")),
                0.1,
                "both",
            ),
        ),
    )


def _atomic(interaction, a, b):
    return certify_atomic_repair_bundle(
        interaction=interaction,
        repairs=("repair-a", "repair-b"),
        implementations={
            "repair-a": a.certificate.program_fingerprint,
            "repair-b": b.certificate.program_fingerprint,
        },
    )


def test_single_local_repair_inside_domain_receives_authority():
    a = _proof("repair-a", _identity_a)
    domain = L2BallExecutionDomain("action-ball@v1", 1, 1.0)

    authority = authorize_pre_dispatch(
        repairs=(a,),
        raw_action=(0.25,),
        semantic_target=(0.25,),
        domain=domain,
        forward=lambda x: x,
        forward_model_id="identity-forward@v1",
        distance=_euclidean,
    )

    assert authority.authorized_action == (0.25,)
    assert not authority.projection_used
    assert authority.atomic_certificate_digest is None


def test_partial_compensating_bundle_is_rejected_before_dispatch():
    a = _proof("repair-a", _identity_a)
    interaction = _interaction()
    domain = L2BallExecutionDomain("action-ball@v1", 1, 1.0)

    with pytest.raises(AtomicRepairRequired, match="closed"):
        authorize_pre_dispatch(
            repairs=(a,),
            raw_action=(0.25,),
            semantic_target=(0.25,),
            domain=domain,
            forward=lambda x: x,
            forward_model_id="identity-forward@v1",
            distance=_euclidean,
            interaction=interaction,
        )


def test_full_implementation_bound_bundle_can_advance():
    a = _proof("repair-a", _identity_a)
    b = _proof("repair-b", _identity_b)
    interaction = _interaction()
    atomic = _atomic(interaction, a, b)
    domain = L2BallExecutionDomain("action-ball@v1", 1, 1.0)

    authority = authorize_pre_dispatch(
        repairs=(a, b),
        raw_action=(0.25,),
        semantic_target=(0.25,),
        domain=domain,
        forward=lambda x: x,
        forward_model_id="identity-forward@v1",
        distance=_euclidean,
        interaction=interaction,
        atomic_certificate=atomic,
    )

    assert authority.atomic_certificate_digest == atomic.digest
    assert authority.interaction_digest == interaction.digest


def test_out_of_domain_bundle_cannot_bypass_projection_gate():
    a = _proof("repair-a", _identity_a)
    b = _proof("repair-b", _identity_b)
    interaction = _interaction()
    atomic = _atomic(interaction, a, b)
    domain = L2BallExecutionDomain("action-ball@v1", 1, 1.0)

    with pytest.raises(ProjectionAuthorityRequired, match="silent clipping"):
        authorize_pre_dispatch(
            repairs=(a, b),
            raw_action=(1.2,),
            semantic_target=(1.2,),
            domain=domain,
            forward=lambda x: x,
            forward_model_id="identity-forward@v1",
            distance=_euclidean,
            interaction=interaction,
            atomic_certificate=atomic,
        )


def test_valid_projection_completes_pre_dispatch_authority():
    a = _proof("repair-a", _identity_a)
    b = _proof("repair-b", _identity_b)
    interaction = _interaction()
    atomic = _atomic(interaction, a, b)
    domain = L2BallExecutionDomain("action-ball@v1", 1, 1.0)
    contract = "+".join(sorted((a.contract_id, b.contract_id)))
    projection = issue_projection_certificate(
        contract_id=contract,
        domain=domain,
        target=(1.2,),
        action=(1.0,),
        forward=lambda x: x,
        forward_model_id="identity-forward@v1",
        distance=_euclidean,
        max_residual=0.25,
    )

    authority = authorize_pre_dispatch(
        repairs=(a, b),
        raw_action=(1.2,),
        semantic_target=(1.2,),
        domain=domain,
        forward=lambda x: x,
        forward_model_id="identity-forward@v1",
        distance=_euclidean,
        interaction=interaction,
        atomic_certificate=atomic,
        projection_certificate=projection,
    )

    assert authority.projection_used
    assert authority.authorized_action == (1.0,)
    assert authority.projection_certificate_digest == projection.decision_digest


def test_drifted_local_program_cannot_enter_composed_authority():
    a = _proof("repair-a", _identity_a)
    drifted = RepairProgram(
        (
            RepairPrimitive(
                name="repair-a",
                family="test",
                cost=1,
                apply_fn=_reverse,
                implementation_id="repair-a@v1",
            ),
        )
    )
    invalid = LocalRepairProof(
        repair_name=a.repair_name,
        contract_id=a.contract_id,
        program=drifted,
        certificate=a.certificate,
    )
    domain = L2BallExecutionDomain("action-ball@v1", 1, 1.0)

    with pytest.raises(PreDispatchAuthorityRejected, match="does not match"):
        authorize_pre_dispatch(
            repairs=(invalid,),
            raw_action=(0.25,),
            semantic_target=(0.25,),
            domain=domain,
            forward=lambda x: x,
            forward_model_id="identity-forward@v1",
            distance=_euclidean,
        )
