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


def test_exhaustive_predispatch_bypass_matrix_matches_policy():
    """Every combination must agree with the explicit authority formula.

    The matrix is intentionally small and exhaustive over the trust-boundary
    booleans that matter here:
      local certificate valid,
      compensating interaction present,
      complete atomic bundle requested,
      raw action inside execution domain,
      valid projection supplied when outside.
    """

    base_a = _proof("repair-a", _identity_a)
    base_b = _proof("repair-b", _identity_b)
    interaction = _interaction()
    atomic = _atomic(interaction, base_a, base_b)
    domain = L2BallExecutionDomain("action-ball@v1", 1, 1.0)

    cases = 0
    for local_valid in (False, True):
        for interaction_required in (False, True):
            for full_bundle in (False, True):
                for in_domain in (False, True):
                    for projection_valid in (False, True):
                        cases += 1

                        a = base_a
                        if not local_valid:
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
                            a = LocalRepairProof(
                                repair_name=base_a.repair_name,
                                contract_id=base_a.contract_id,
                                program=drifted,
                                certificate=base_a.certificate,
                            )

                        if interaction_required and full_bundle:
                            repairs = (a, base_b)
                            active_interaction = interaction
                            active_atomic = atomic
                        elif interaction_required:
                            repairs = (a,)
                            active_interaction = interaction
                            active_atomic = None
                        else:
                            repairs = (a,)
                            active_interaction = None
                            active_atomic = None

                        raw = (0.25,) if in_domain else (1.2,)
                        target = raw
                        projection = None
                        if (not in_domain) and projection_valid:
                            projection = issue_projection_certificate(
                                contract_id="+".join(
                                    sorted(proof.contract_id for proof in repairs)
                                ),
                                domain=domain,
                                target=target,
                                action=(1.0,),
                                forward=lambda x: x,
                                forward_model_id="identity-forward@v1",
                                distance=_euclidean,
                                max_residual=0.25,
                            )

                        expected = bool(
                            local_valid
                            and (
                                (not interaction_required)
                                or full_bundle
                            )
                            and (
                                in_domain
                                or projection_valid
                            )
                        )

                        accepted = True
                        try:
                            authorize_pre_dispatch(
                                repairs=repairs,
                                raw_action=raw,
                                semantic_target=target,
                                domain=domain,
                                forward=lambda x: x,
                                forward_model_id="identity-forward@v1",
                                distance=_euclidean,
                                interaction=active_interaction,
                                atomic_certificate=active_atomic,
                                projection_certificate=projection,
                            )
                        except RuntimeError:
                            accepted = False

                        assert accepted is expected, {
                            "local_valid": local_valid,
                            "interaction_required": interaction_required,
                            "full_bundle": full_bundle,
                            "in_domain": in_domain,
                            "projection_valid": projection_valid,
                        }

    assert cases == 32
