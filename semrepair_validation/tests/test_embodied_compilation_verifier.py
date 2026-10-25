from research.semantic_invariants.embodied_compilation_certificate import (
    issue_compilation_certificate,
)
from research.semantic_invariants.embodied_compilation_verifier import (
    verify_pure_adapter_certificate,
)
from research.semantic_invariants.embodied_semantic_compiler import (
    CompilationResult,
    compile_semantic_boundary,
)
from research.semantic_invariants.embodied_semantic_types import (
    AdapterPlan,
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


def _manual_result(source, target, plan):
    return CompilationResult(
        source=source,
        inferred_source=source,
        refined_source=source,
        target=target,
        inference=(),
        refinements=(),
        adapter_plan=plan,
        repair_candidate=None,
        repair_families=(),
        effects=plan.effects,
    )


def test_independent_checker_accepts_valid_unique_minimum_certificate():
    source = _rotation("axis_angle")
    target = _rotation("euler_xyz")
    adapters = (
        SemanticAdapter(
            "direct",
            requires={"representation": "axis_angle"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
            effects=("reencode",),
        ),
        SemanticAdapter(
            "axis-to-quat",
            requires={"representation": "axis_angle"},
            produces={"representation": "quaternion"},
            cost=1.0,
        ),
        SemanticAdapter(
            "quat-to-euler",
            requires={"representation": "quaternion"},
            produces={"representation": "euler_xyz"},
            cost=1.0,
        ),
    )

    result = compile_semantic_boundary(source, target, adapters=adapters)
    cert = issue_compilation_certificate(result, adapters=adapters)
    check = verify_pure_adapter_certificate(cert, adapters=adapters)

    assert check.valid
    assert check.reasons == ()
    assert check.cheaper_paths == ()
    assert check.equal_cost_alternatives == ()


def test_independent_checker_rejects_certificate_for_nonminimum_path_even_if_hash_is_valid():
    source = _rotation("axis_angle")
    target = _rotation("euler_xyz")
    cheap = SemanticAdapter(
        "direct",
        requires={"representation": "axis_angle"},
        produces={"representation": "euler_xyz"},
        cost=1.0,
    )
    first = SemanticAdapter(
        "axis-to-quat",
        requires={"representation": "axis_angle"},
        produces={"representation": "quaternion"},
        cost=1.0,
    )
    second = SemanticAdapter(
        "quat-to-euler",
        requires={"representation": "quaternion"},
        produces={"representation": "euler_xyz"},
        cost=1.0,
    )
    adapters = (cheap, first, second)

    expensive_plan = AdapterPlan(
        source=source,
        target=target,
        adapters=(first, second),
        result=target,
        total_cost=2.0,
        effects=(),
        evidence_used=(),
    )
    result = _manual_result(source, target, expensive_plan)
    cert = issue_compilation_certificate(result, adapters=adapters)

    check = verify_pure_adapter_certificate(cert, adapters=adapters)

    assert not check.valid
    assert ("direct",) in check.cheaper_paths
    assert any("not minimum cost" in reason for reason in check.reasons)


def test_independent_checker_rejects_equal_cost_second_semantic_explanation():
    source = _rotation("axis_angle")
    target = _rotation("euler_xyz")
    a = SemanticAdapter(
        "direct-a",
        requires={"representation": "axis_angle"},
        produces={"representation": "euler_xyz"},
        cost=1.0,
    )
    b = SemanticAdapter(
        "direct-b",
        requires={"representation": "axis_angle"},
        produces={"representation": "euler_xyz"},
        cost=1.0,
    )
    adapters = (a, b)

    selected_plan = AdapterPlan(
        source=source,
        target=target,
        adapters=(a,),
        result=target,
        total_cost=1.0,
        effects=(),
        evidence_used=(),
    )
    result = _manual_result(source, target, selected_plan)
    cert = issue_compilation_certificate(result, adapters=adapters)

    check = verify_pure_adapter_certificate(cert, adapters=adapters)

    assert not check.valid
    assert ("direct-b",) in check.equal_cost_alternatives
    assert any("not uniquely minimum" in reason for reason in check.reasons)


def test_independent_checker_requires_current_context_evidence_identity():
    source = SemanticTensorType(
        role="state",
        entity="eef_position",
        frame="camera",
        unit="m",
    )
    target = source.updated(frame="base")
    adapter = SemanticAdapter(
        "camera-to-base",
        requires={"frame": "camera"},
        produces={"frame": "base"},
        witness_keys=("transform:camera->base",),
    )
    adapters = (adapter,)
    result = compile_semantic_boundary(
        source,
        target,
        adapters=adapters,
        adapter_evidence={"transform:camera->base": object()},
    )
    cert = issue_compilation_certificate(
        result,
        adapters=adapters,
        evidence_identity={"transform:camera->base": "sha256:v1"},
    )

    valid = verify_pure_adapter_certificate(
        cert,
        adapters=adapters,
        evidence_identity={"transform:camera->base": "sha256:v1"},
    )
    stale = verify_pure_adapter_certificate(
        cert,
        adapters=adapters,
        evidence_identity={"transform:camera->base": "sha256:v2"},
    )

    assert valid.valid
    assert not stale.valid
    assert any("evidence identity digest has drifted" in reason for reason in stale.reasons)

def test_independent_checker_quotients_out_zero_cost_type_cycles():
    source = _rotation("axis_angle")
    target = _rotation("euler_xyz")
    cycle_out = SemanticAdapter(
        "axis-to-quat-free",
        requires={"representation": "axis_angle"},
        produces={"representation": "quaternion"},
        cost=0.0,
    )
    cycle_back = SemanticAdapter(
        "quat-to-axis-free",
        requires={"representation": "quaternion"},
        produces={"representation": "axis_angle"},
        cost=0.0,
    )
    direct = SemanticAdapter(
        "direct",
        requires={"representation": "axis_angle"},
        produces={"representation": "euler_xyz"},
        cost=1.0,
    )
    adapters = (cycle_out, cycle_back, direct)

    result = compile_semantic_boundary(source, target, adapters=adapters)
    cert = issue_compilation_certificate(result, adapters=adapters)
    check = verify_pure_adapter_certificate(cert, adapters=adapters)

    assert check.valid
    assert check.equal_cost_alternatives == ()
