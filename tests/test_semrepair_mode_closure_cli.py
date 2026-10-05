from research.semantic_invariants.semrepair_mode_closure_cli import (
    run_mode_closure,
)


def test_mode_closure_cli_reports_unresolved_delta(tmp_path):
    enum_source = tmp_path / "types.py"
    runtime_source = tmp_path / "runtime.py"

    enum_source.write_text(
        """
from enum import Enum
class ActionRepresentation(Enum):
    RELATIVE = "relative"
    DELTA = "delta"
    ABSOLUTE = "absolute"
""",
        encoding="utf-8",
    )
    runtime_source.write_text(
        """
def process(config, action):
    if config.rep == ActionRepresentation.RELATIVE:
        return to_relative(action)
    return action
""",
        encoding="utf-8",
    )

    report = run_mode_closure(
        enum_source=enum_source,
        runtime_source=runtime_source,
        enum_name="ActionRepresentation",
        explicit_default=("ABSOLUTE",),
    )

    assert report["declared"] == ["ABSOLUTE", "DELTA", "RELATIVE"]
    assert report["handled"] == ["RELATIVE"]
    assert report["explicit_default"] == ["ABSOLUTE"]
    assert report["unresolved"] == ["DELTA"]
    assert not report["closed"]
    assert "independent semantic evidence" in report["claim_boundary"]