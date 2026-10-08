from dataclasses import replace

import pytest

from research.semantic_invariants.embodied_repair_interactions import (
    RepairOutcome,
    analyze_repair_lattice,
)
from research.semantic_invariants.embodied_version_bound_authority import (
    ExecutionContextMismatch,
    authorize_version_bound_repair_subset,
    freeze_version_bound_context,
    verify_live_version_bound_context,
)


def _frozen_case():
    interaction = analyze_repair_lattice(
        subject="maniskill/example-pinned-version",
        metric="orientation_error_deg",
        objective="minimize",
        repairs=("converter", "controller"),
        outcomes=(
            RepairOutcome(frozenset(), 5.0, "baseline"),
            RepairOutcome(frozenset(("converter",)), 50.0, "converter"),
            RepairOutcome(frozenset(("controller",)), 51.0, "controller"),
            RepairOutcome(frozenset(("converter", "controller")), 0.001, "joint"),
        ),
    )
    sources = {
        "mani_skill/trajectory/utils/actions/conversion.py": b"converter-old-blob",
        "mani_skill/agents/controllers/pd_ee_pose.py": b"controller-old-blob",
    }
    config = b"rot_lower=-0.1;rot_upper=+0.1"
    protocol = b"peg-insertion:seed=42;metric=orientation-error-v1"
    frozen = freeze_version_bound_context(
        interaction=interaction,
        source_files=sources,
        configuration=config,
        protocol=protocol,
    )
    return interaction, sources, config, protocol, frozen


def _decision(interaction, sources, config, protocol, frozen, repairs):
    return authorize_version_bound_repair_subset(
        frozen=frozen,
        interaction=interaction,
        trusted_seal_digest=frozen.seal_digest,
        live_source_files=sources,
        live_configuration=config,
        live_protocol=protocol,
        requested_repairs=repairs,
    )


def test_same_exact_stack_rejects_singleton_but_authorizes_bundle():
    interaction, sources, config, protocol, frozen = _frozen_case()
    assert not _decision(
        interaction, sources, config, protocol, frozen, ("converter",)
    ).decision.authorized
    joint = _decision(
        interaction, sources, config, protocol, frozen, ("converter", "controller")
    )
    assert joint.decision.authorized
    assert joint.context_seal == frozen.seal_digest


@pytest.mark.parametrize("drift_target", ["converter", "controller", "config", "protocol"])
def test_stale_or_changed_stack_fails_closed(drift_target):
    interaction, sources, config, protocol, frozen = _frozen_case()
    sources = dict(sources)
    if drift_target == "converter":
        sources["mani_skill/trajectory/utils/actions/conversion.py"] = b"new-mapper-aware-converter"
    if drift_target == "controller":
        sources["mani_skill/agents/controllers/pd_ee_pose.py"] = b"new-controller"
    if drift_target == "config":
        config = b"rot_lower=-0.2;rot_upper=+0.2"
    if drift_target == "protocol":
        protocol = b"different-protocol"

    with pytest.raises(ExecutionContextMismatch, match="drift"):
        _decision(
            interaction, sources, config, protocol, frozen, ("converter", "controller")
        )


def test_swapping_source_labels_and_omitting_files_fails_closed():
    interaction, sources, config, protocol, frozen = _frozen_case()
    swapped = {name: sources[other] for name, other in zip(
        tuple(sources), tuple(reversed(sources)), strict=True
    )}
    with pytest.raises(ExecutionContextMismatch, match="drift"):
        _decision(interaction, swapped, config, protocol, frozen, ("converter", "controller"))
    reduced = {next(iter(sources)): next(iter(sources.values()))}
    with pytest.raises(ExecutionContextMismatch, match="drift"):
        _decision(interaction, reduced, config, protocol, frozen, ("converter", "controller"))


def test_self_resealed_new_version_is_not_authorized_under_old_trust_root():
    interaction, sources, config, protocol, frozen = _frozen_case()
    mutated = dict(sources)
    mutated["mani_skill/trajectory/utils/actions/conversion.py"] = b"changed"
    forged_new = freeze_version_bound_context(
        interaction=interaction,
        source_files=mutated,
        configuration=config,
        protocol=protocol,
    )
    # The malicious caller can compute fresh unkeyed SHA-256 hashes but cannot
    # match a trusted release root pinned separately from the caller.
    with pytest.raises(ExecutionContextMismatch, match="not trusted"):
        verify_live_version_bound_context(
            frozen=forged_new,
            interaction=interaction,
            trusted_seal_digest=frozen.seal_digest,
            live_source_files=mutated,
            live_configuration=config,
            live_protocol=protocol,
        )


def test_interaction_reanalysis_or_certificate_splice_cannot_reuse_old_context():
    interaction, sources, config, protocol, frozen = _frozen_case()
    changed_outcomes = tuple(
        replace(outcome, evidence_id=outcome.evidence_id + "-new")
        for outcome in interaction.outcomes
    )
    updated = analyze_repair_lattice(
        subject=interaction.subject,
        metric=interaction.metric,
        objective=interaction.objective,
        repairs=interaction.repairs,
        outcomes=changed_outcomes,
    )
    with pytest.raises(ExecutionContextMismatch, match="drift"):
        _decision(updated, sources, config, protocol, frozen, ("converter", "controller"))


def test_missing_trust_root_and_invalid_input_bytes_are_denied():
    interaction, sources, config, protocol, frozen = _frozen_case()
    with pytest.raises(ExecutionContextMismatch, match="trusted context seal"):
        verify_live_version_bound_context(
            frozen=frozen,
            interaction=interaction,
            trusted_seal_digest="",
            live_source_files=sources,
            live_configuration=config,
            live_protocol=protocol,
        )
    malformed = dict(sources)
    malformed[next(iter(sources))] = "not-raw-bytes"
    with pytest.raises(ExecutionContextMismatch, match="invalid live"):
        verify_live_version_bound_context(
            frozen=frozen,
            interaction=interaction,
            trusted_seal_digest=frozen.seal_digest,
            live_source_files=malformed,
            live_configuration=config,
            live_protocol=protocol,
        )


def test_version_bound_context_is_enforced_by_full_predispatch_runtime_gate():
    from research.semantic_invariants.embodied_authority_kernel import LocalRepairProof
    from research.semantic_invariants.embodied_execution_domain import L2BallExecutionDomain
    from research.semantic_invariants.embodied_measurement_qualification import (
        qualify_measurement,
    )
    from research.semantic_invariants.embodied_repair_interactions import (
        QualifiedRepairAuthorizationPlane,
    )
    from research.semantic_invariants.embodied_repair_synthesis import (
        RepairExample,
        RepairPrimitive,
        RepairProgram,
    )
    from research.semantic_invariants.embodied_repair_verification import (
        verify_repair_against_heldout,
    )
    from research.semantic_invariants.embodied_version_bound_authority import (
        authorize_version_bound_pre_dispatch,
    )

    def identity(value, context):
        return tuple(value)

    program = RepairProgram(
        (RepairPrimitive(
            name="repair",
            family="example",
            cost=1,
            apply_fn=identity,
            implementation_id="identity@v1",
        ),)
    )
    certificate = verify_repair_against_heldout(
        contract_id="contract/example-v1",
        program=program,
        heldout=(RepairExample((0.25,), (0.25,)),),
        verifier_id="heldout-bank@v1",
    )
    measurement = qualify_measurement(
        measurement_id="source/task-signal",
        semantic_anchor_id="external/task-oracle",
        repeat_outcome_digests=("identical",) * 3,
    )
    proof = LocalRepairProof(
        repair_name="repair",
        contract_id="contract/example-v1",
        program=program,
        certificate=certificate,
        measurement=measurement,
    )
    semantic = analyze_repair_lattice(
        subject="robot-A@pinned",
        metric="semantic_error",
        objective="minimize",
        repairs=("repair",),
        outcomes=(
            RepairOutcome(frozenset(), 2.0, "sem-before"),
            RepairOutcome(frozenset(("repair",)), 1.0, "sem-after"),
        ),
    )
    execution = analyze_repair_lattice(
        subject="robot-A@pinned",
        metric="task_success",
        objective="maximize",
        repairs=("repair",),
        outcomes=(
            RepairOutcome(frozenset(), 0.8, "task-before"),
            RepairOutcome(frozenset(("repair",)), 0.9, "task-after"),
        ),
    )
    planes = (
        QualifiedRepairAuthorizationPlane("semantic-fidelity", semantic, measurement),
        QualifiedRepairAuthorizationPlane("execution-effect", execution, measurement),
    )
    sources = {"controller.py": b"unchanged-source"}
    frozen = freeze_version_bound_context(
        interaction=semantic,
        source_files=sources,
        configuration=b"controller-config@v1",
        protocol=b"fixed-protocol@v1",
        qualified_planes=planes,
    )
    kwargs = dict(
        frozen=frozen,
        trusted_seal_digest=frozen.seal_digest,
        live_source_files=sources,
        live_configuration=b"controller-config@v1",
        live_protocol=b"fixed-protocol@v1",
        interaction=semantic,
        repairs=(proof,),
        raw_action=(0.25,),
        semantic_target=(0.25,),
        domain=L2BallExecutionDomain("action-ball@v1", 1, 1.0),
        forward=lambda x: x,
        forward_model_id="forward@v1",
        distance=lambda a, b: abs(a[0] - b[0]),
        qualified_planes=planes,
    )
    receipt = authorize_version_bound_pre_dispatch(**kwargs)
    assert receipt.authorized_action == (0.25,)
    assert receipt.qualified_plane_digests

    with pytest.raises(ExecutionContextMismatch, match="drift"):
        authorize_version_bound_pre_dispatch(
            **{**kwargs, "live_source_files": {"controller.py": b"new-code"}},
        )

    # Even with the same source, config, subject and primary interaction,
    # evidence from another run/protocol must not be silently substituted.
    changed_measurement = qualify_measurement(
        measurement_id="source/task-signal",
        semantic_anchor_id="external/task-oracle-v2",
        repeat_outcome_digests=("identical",) * 3,
    )
    spliced_measurement = (
        QualifiedRepairAuthorizationPlane("semantic-fidelity", semantic, changed_measurement),
        planes[1],
    )
    with pytest.raises(ExecutionContextMismatch, match="drift"):
        authorize_version_bound_pre_dispatch(
            **{**kwargs, "qualified_planes": spliced_measurement},
        )

    changed_semantic = analyze_repair_lattice(
        subject=semantic.subject,
        metric=semantic.metric,
        objective=semantic.objective,
        repairs=semantic.repairs,
        outcomes=(
            RepairOutcome(frozenset(), 2.0, "sem-before"),
            RepairOutcome(frozenset(("repair",)), 0.5, "sem-after-new-run"),
        ),
    )
    spliced_outcomes = (
        QualifiedRepairAuthorizationPlane("semantic-fidelity", changed_semantic, measurement),
        planes[1],
    )
    with pytest.raises(ExecutionContextMismatch, match="drift"):
        authorize_version_bound_pre_dispatch(
            **{**kwargs, "qualified_planes": spliced_outcomes},
        )

    # Merely renaming or dropping a required plane cannot inherit authority.
    with pytest.raises(ExecutionContextMismatch, match="drift"):
        authorize_version_bound_pre_dispatch(
            **{**kwargs, "qualified_planes": planes[:1]},
        )

    # Reorder alone does not alter evidence identity: commitments are sorted.
    reversed_authority = authorize_version_bound_pre_dispatch(
        **{**kwargs, "qualified_planes": planes[::-1]},
    )
    assert reversed_authority.authorized_action == (0.25,)

    unsealed = freeze_version_bound_context(
        interaction=semantic,
        source_files=sources,
        configuration=b"controller-config@v1",
        protocol=b"fixed-protocol@v1",
    )
    with pytest.raises(ExecutionContextMismatch, match="evidence-plane commitments"):
        authorize_version_bound_pre_dispatch(
            **{**kwargs, "frozen": unsealed, "trusted_seal_digest": unsealed.seal_digest},
        )
