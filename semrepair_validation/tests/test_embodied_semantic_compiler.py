import pytest

from research.semantic_invariants.embodied_refinement import (
    SemanticEvidence,
    refine_executed_action,
)
from research.semantic_invariants.embodied_repair_runtime import (
    RepairNotVerifiedError,
    SemanticRepairMediator,
)
from research.semantic_invariants.embodied_repair_synthesis import RepairExample
from research.semantic_invariants.embodied_rotation_chain import (
    compact_axis_angle_from_matrix,
    euler_xyz_matrix,
)
from research.semantic_invariants.embodied_semantic_compiler import (
    FieldInferenceSpec,
    SemanticCompilationError,
    SemanticProofObligationError,
    compile_semantic_boundary,
)
from research.semantic_invariants.embodied_semantic_inference import (
    SemanticHypothesis,
    SemanticProbe,
)
from research.semantic_invariants.embodied_semantic_types import (
    SemanticAdapter,
    SemanticTensorType,
)


def test_compiler_infers_hidden_representation_then_uses_typed_repair():
    source = SemanticTensorType(
        role="action",
        entity="ee_rotation_delta",
        representation=None,
        unit="rad",
        provenance="requested",
    )
    target = source.updated(representation="euler_xyz")

    hypotheses = {
        "euler_xyz": SemanticHypothesis(
            "euler_xyz",
            lambda payload: "same" if payload == "single" else "euler",
        ),
        "axis_angle": SemanticHypothesis(
            "axis_angle",
            lambda payload: "same" if payload == "single" else "axis",
        ),
    }
    inference = FieldInferenceSpec(
        field="representation",
        hypotheses=hypotheses,
        probes=(
            SemanticProbe("single-axis", "single"),
            SemanticProbe("three-axis", "multi"),
        ),
        observe=lambda probe: "axis" if probe.payload == "multi" else "same",
    )
    adapters = [
        SemanticAdapter(
            "axis-angle-to-quaternion",
            requires={"representation": "axis_angle"},
            produces={"representation": "quaternion"},
            effects=("reencode",),
        ),
        SemanticAdapter(
            "quaternion-to-euler",
            requires={"representation": "quaternion"},
            produces={"representation": "euler_xyz"},
            effects=("reencode",),
        ),
    ]

    result = compile_semantic_boundary(
        source,
        target,
        adapters=adapters,
        inference_specs=(inference,),
    )

    assert result.status == "repaired"
    assert result.inferred_source.representation == "axis_angle"
    assert result.adapter_plan is not None
    assert [a.name for a in result.adapter_plan.adapters] == [
        "axis-angle-to-quaternion",
        "quaternion-to-euler",
    ]
    assert result.repair_candidate is None


def test_compiler_requires_runtime_refinement_for_requested_to_executed():
    source = SemanticTensorType(
        role="action",
        entity="joint_command",
        representation="joint_position",
        provenance="requested",
    )
    target = source.updated(provenance="executed")

    with pytest.raises(SemanticCompilationError, match="owner boundary"):
        compile_semantic_boundary(source, target)

    evidence = SemanticEvidence(
        kind="execution_receipt",
        subject_id="action-1",
        claims={"executed": True, "controller_id": "arm"},
        issuer="controller",
    )
    refinement = refine_executed_action(source, evidence)

    result = compile_semantic_boundary(
        source,
        target,
        refinements=(refinement,),
    )
    assert result.status == "repaired"
    assert result.refined_source.provenance == "executed"
    assert result.effects == ("execute",)


def test_compiler_rejects_ambiguous_inference_instead_of_guessing():
    source = SemanticTensorType(
        role="state",
        entity="joint_position",
        ordering=None,
    )
    target = source.updated(ordering="policy-order")
    hypotheses = {
        "urdf-order": SemanticHypothesis("urdf-order", lambda payload: "same"),
        "policy-order": SemanticHypothesis("policy-order", lambda payload: "same"),
    }
    inference = FieldInferenceSpec(
        field="ordering",
        hypotheses=hypotheses,
        probes=(SemanticProbe("uninformative", "x"),),
        observe=lambda probe: "same",
    )

    with pytest.raises(SemanticCompilationError, match="ambiguous"):
        compile_semantic_boundary(
            source,
            target,
            inference_specs=(inference,),
        )


def test_refinement_receipt_cannot_be_reused_after_type_changes():
    source = SemanticTensorType(
        role="action",
        entity="joint_command",
        representation="joint_position",
        provenance="requested",
    )
    evidence = SemanticEvidence(
        kind="execution_receipt",
        subject_id="action-1",
        claims={"executed": True, "controller_id": "arm"},
        issuer="controller",
    )
    refinement = refine_executed_action(source, evidence)
    changed = source.updated(representation="joint_velocity")
    target = changed.updated(provenance="executed")

    with pytest.raises(SemanticCompilationError, match="different semantic value"):
        compile_semantic_boundary(
            changed,
            target,
            refinements=(refinement,),
        )


def test_context_adapter_surfaces_proof_obligation_and_boolean_cannot_bypass_it():
    source = SemanticTensorType(
        role="observation",
        entity="camera_frame",
        clock="dataset_global",
        scope="dataset",
        provenance="observed",
    )
    target = source.updated(clock="episode_local", scope="episode")
    retime = SemanticAdapter(
        "retime-with-episode-origin",
        requires={"clock": "dataset_global", "scope": "dataset"},
        produces={"clock": "episode_local", "scope": "episode"},
        witness_keys=("episode_origin",),
        effects=("retime",),
    )

    with pytest.raises(SemanticProofObligationError) as blocked:
        compile_semantic_boundary(
            source,
            target,
            adapters=(retime,),
            allow_proof_required=True,
        )
    assert blocked.value.obligations == ("episode_origin",)

    result = compile_semantic_boundary(
        source,
        target,
        adapters=(retime,),
        adapter_evidence={"episode_origin": 120.0},
    )
    assert result.status == "repaired"
    assert result.adapter_plan is not None
    assert result.adapter_plan.evidence_used == ("episode_origin",)


def test_maniskill_style_mismatch_yields_unique_unverified_repair_candidate():
    source = SemanticTensorType(
        role="action",
        entity="ee_rotation_delta",
        frame="root_aligned_body",
        representation="axis_angle",
        unit="rad",
        provenance="requested",
    )
    target = source.updated(representation="euler_xyz")
    targets = ((0.04, 0.04, 0.04), (0.03, -0.04, 0.05))
    examples = tuple(
        RepairExample(
            compact_axis_angle_from_matrix(euler_xyz_matrix(euler)),
            euler,
            label=f"witness-{index}",
        )
        for index, euler in enumerate(targets)
    )

    result = compile_semantic_boundary(
        source,
        target,
        repair_examples=examples,
        max_repair_depth=1,
        repair_atol=1e-9,
    )

    assert result.status == "repair_candidate"
    assert result.adapter_plan is None
    assert result.repair_candidate is not None
    assert result.repair_candidate.name == "axis-angle->euler-xyz"
    assert result.repair_families == ("rotation-representation",)

    # Compilation may propose a repair, but runtime installation remains gated
    # on an independent verifier.
    with pytest.raises(RepairNotVerifiedError):
        SemanticRepairMediator(
            contract_id="embodied/representation/controller-roundtrip@0.2",
            program=result.repair_candidate,
            certificate=None,  # type: ignore[arg-type]
        )


def test_numeric_examples_cannot_convert_event_owned_provenance():
    source = SemanticTensorType(
        role="action",
        entity="joint_command",
        representation="joint_position",
        provenance="requested",
    )
    target = source.updated(provenance="executed")

    with pytest.raises(SemanticCompilationError, match="owner boundary"):
        compile_semantic_boundary(
            source,
            target,
            repair_examples=(
                RepairExample((0.1, 0.2), (0.1, 0.2)),
            ),
        )


def test_known_numeric_family_without_witness_fails_closed():
    source = SemanticTensorType(
        role="state",
        entity="joint_position",
        representation="vector",
        ordering="urdf-order",
    )
    target = source.updated(ordering="policy-order")

    with pytest.raises(SemanticCompilationError):
        compile_semantic_boundary(source, target)

def test_compiler_rejects_equal_cost_explicit_adapter_paths():
    source = SemanticTensorType(
        role="action",
        entity="ee_rotation_delta",
        representation="axis_angle",
        unit="rad",
        provenance="requested",
    )
    target = source.updated(representation="euler_xyz")
    adapters = (
        SemanticAdapter(
            "direct-a",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
        SemanticAdapter(
            "direct-b",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
    )

    with pytest.raises(SemanticCompilationError, match="ambiguous"):
        compile_semantic_boundary(source, target, adapters=adapters)

def test_compiler_synthesizes_composed_representation_and_sign_repair():
    source = SemanticTensorType(
        role="action",
        entity="ee_rotation_delta",
        frame="root_aligned_body",
        representation="axis_angle",
        convention="positive_action_negative_rotation",
        unit="rad",
        provenance="requested",
    )
    target = source.updated(
        representation="euler_xyz",
        convention="positive_action_positive_rotation",
    )

    targets = ((0.08, 0.08, 0.08), (0.04, -0.07, 0.06))
    examples = []
    for index, euler in enumerate(targets):
        axis_angle = compact_axis_angle_from_matrix(euler_xyz_matrix(euler))
        observed = tuple(-value for value in axis_angle)
        examples.append(
            RepairExample(
                observed,
                euler,
                label=f"composed-witness-{index}",
            )
        )

    result = compile_semantic_boundary(
        source,
        target,
        repair_examples=tuple(examples),
        max_repair_depth=2,
        repair_atol=1e-9,
    )

    assert result.status == "repair_candidate"
    assert result.repair_candidate is not None
    assert set(result.repair_families) == {"rotation-representation", "sign"}
    assert "axis-angle->euler-xyz" in result.repair_candidate.name
    assert "sign" in result.repair_candidate.name