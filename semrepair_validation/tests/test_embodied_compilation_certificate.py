import pytest

from research.semantic_invariants.embodied_compilation_certificate import (
    adapter_registry_digest,
    certificate_matches_environment,
    issue_compilation_certificate,
    verify_compilation_certificate,
)
from research.semantic_invariants.embodied_semantic_compiler import (
    SemanticCompilationError,
    compile_semantic_boundary,
)
from research.semantic_invariants.embodied_semantic_types import (
    SemanticAdapter,
    SemanticTensorType,
)


def _rotation(rep):
    return SemanticTensorType(
        role="action",
        entity="ee_rotation_delta",
        representation=rep,
        unit="rad",
        provenance="requested",
    )


def test_certificate_binds_unique_repair_to_registry_snapshot():
    source = _rotation("axis_angle")
    target = _rotation("euler_xyz")
    adapters = (
        SemanticAdapter(
            "axis-to-euler",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
    )
    result = compile_semantic_boundary(source, target, adapters=adapters)
    cert = issue_compilation_certificate(result, adapters=adapters)

    assert cert.installable
    assert cert.selected_adapter_path == ("axis-to-euler",)
    assert verify_compilation_certificate(cert)
    assert certificate_matches_environment(cert, adapters=adapters)


def test_adding_equal_cost_adapter_invalidates_old_certificate_and_recompile_abstains():
    source = _rotation("axis_angle")
    target = _rotation("euler_xyz")
    original = (
        SemanticAdapter(
            "axis-to-euler-a",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
    )
    result = compile_semantic_boundary(source, target, adapters=original)
    cert = issue_compilation_certificate(result, adapters=original)

    expanded = original + (
        SemanticAdapter(
            "axis-to-euler-b",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
    )

    assert adapter_registry_digest(expanded) != cert.registry_digest
    assert not certificate_matches_environment(cert, adapters=expanded)
    with pytest.raises(SemanticCompilationError, match="ambiguous"):
        compile_semantic_boundary(source, target, adapters=expanded)


def test_context_evidence_identity_drift_invalidates_certificate():
    source = SemanticTensorType(
        role="state",
        entity="eef_position",
        frame="camera",
        representation="xyz",
        unit="m",
    )
    target = source.updated(frame="base")
    adapters = (
        SemanticAdapter(
            "camera-to-base",
            requires={"frame": "camera"},
            produces={"frame": "base"},
            witness_keys=("transform:camera->base",),
        ),
    )
    result = compile_semantic_boundary(
        source,
        target,
        adapters=adapters,
        adapter_evidence={"transform:camera->base": object()},
    )
    cert = issue_compilation_certificate(
        result,
        adapters=adapters,
        evidence_identity={"transform:camera->base": "sha256:transform-v1"},
    )

    assert certificate_matches_environment(
        cert,
        adapters=adapters,
        evidence_identity={"transform:camera->base": "sha256:transform-v1"},
    )
    assert not certificate_matches_environment(
        cert,
        adapters=adapters,
        evidence_identity={"transform:camera->base": "sha256:transform-v2"},
    )


def test_certificate_refuses_unidentified_consumed_evidence():
    source = SemanticTensorType(
        role="state",
        entity="eef_position",
        frame="camera",
    )
    target = source.updated(frame="base")
    adapters = (
        SemanticAdapter(
            "camera-to-base",
            requires={"frame": "camera"},
            produces={"frame": "base"},
            witness_keys=("transform:camera->base",),
        ),
    )
    result = compile_semantic_boundary(
        source,
        target,
        adapters=adapters,
        adapter_evidence={"transform:camera->base": object()},
    )

    with pytest.raises(ValueError, match="stable identities"):
        issue_compilation_certificate(result, adapters=adapters)
