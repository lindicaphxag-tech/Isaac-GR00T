from pathlib import Path

from research.semantic_invariants.embodied_mode_closure_evidence import (
    infer_triangulated_mode_closure,
)


ROOT = Path(__file__).resolve().parents[2]


def test_groot_real_source_action_representation_closure():
    enum_source = (ROOT / "gr00t/data/types.py").read_text(encoding="utf-8")
    runtime_source = (
        ROOT / "gr00t/data/state_action/state_action_processor.py"
    ).read_text(encoding="utf-8")
    usage_sources = [
        (ROOT / "gr00t/configs/data/embodiment_configs.py").read_text(
            encoding="utf-8"
        ),
        (ROOT / "examples/SO100/so100_config.py").read_text(encoding="utf-8"),
    ]

    # Ground the intentional ABSOLUTE fall-through in two independent source
    # planes before treating it as a baseline rather than an uncovered mode.
    assert "use_relative_action: bool = False" in runtime_source
    assert "ActionRepresentation.ABSOLUTE" in usage_sources[0]

    result = infer_triangulated_mode_closure(
        enum_source=enum_source,
        runtime_source=runtime_source,
        usage_sources=usage_sources,
        enum_name="ActionRepresentation",
        baseline_evidence={
            "ABSOLUTE": (
                "runtime-default-no-relative-conversion",
                "production-config-use",
            ),
        },
        minimum_default_planes=2,
    )

    assert result.closure.declared == frozenset(
        {"RELATIVE", "DELTA", "ABSOLUTE"}
    )
    assert result.closure.handled == frozenset({"RELATIVE"})
    assert result.closure.explicit_default == frozenset({"ABSOLUTE"})
    assert result.closure.unresolved == frozenset({"DELTA"})

    counts = dict(result.usage_counts)
    assert counts["RELATIVE"] > 0
    assert counts["ABSOLUTE"] > 0
    assert counts["DELTA"] == 0


def test_groot_delta_is_reported_unresolved_not_auto_labeled_bug():
    enum_source = (ROOT / "gr00t/data/types.py").read_text(encoding="utf-8")
    runtime_source = (
        ROOT / "gr00t/data/state_action/state_action_processor.py"
    ).read_text(encoding="utf-8")

    result = infer_triangulated_mode_closure(
        enum_source=enum_source,
        runtime_source=runtime_source,
        usage_sources=[],
        enum_name="ActionRepresentation",
    )

    # Without independent baseline evidence both fall-through modes remain
    # unresolved. SemRepair does not equate "unhandled" with "bug".
    assert "ABSOLUTE" in result.closure.unresolved
    assert "DELTA" in result.closure.unresolved
