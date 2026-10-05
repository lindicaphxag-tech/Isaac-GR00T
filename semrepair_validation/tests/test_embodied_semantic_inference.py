from research.semantic_invariants.embodied_semantic_inference import (
    SemanticHypothesis,
    SemanticProbe,
    active_semantic_inference,
    choose_probe,
)
from research.semantic_invariants.embodied_rotation_chain import (
    compact_axis_angle_from_matrix,
    euler_xyz_matrix,
)


def test_active_inference_prefers_discriminating_three_axis_probe():
    hypotheses = [
        SemanticHypothesis(
            "euler_xyz",
            lambda payload: "same" if payload == (0.5, 0.0, 0.0) else "euler-result",
        ),
        SemanticHypothesis(
            "axis_angle",
            lambda payload: "same" if payload == (0.5, 0.0, 0.0) else "axis-result",
        ),
    ]
    probes = [
        SemanticProbe("single-axis", (0.5, 0.0, 0.0), cost=1.0),
        SemanticProbe("three-axis", (0.5, 0.5, 0.5), cost=1.0),
    ]

    choice = choose_probe(hypotheses, probes)
    assert choice is not None
    assert choice.probe.name == "three-axis"
    assert choice.information_gain_bits == 1.0


def test_active_inference_identifies_hidden_semantics():
    hypotheses = [
        SemanticHypothesis("xyz", lambda payload: {"p0": 0, "p1": 1}[payload]),
        SemanticHypothesis("zyx", lambda payload: {"p0": 1, "p1": 0}[payload]),
        SemanticHypothesis("identity", lambda payload: 0),
    ]
    probes = [
        SemanticProbe("probe-0", "p0", cost=1.0),
        SemanticProbe("probe-1", "p1", cost=1.0),
    ]

    result = active_semantic_inference(
        hypotheses,
        probes,
        observe=lambda probe: {"p0": 1, "p1": 0}[probe.payload],
    )

    assert result.status == "identified"
    assert result.survivors == ("zyx",)


def test_unsafe_probe_is_not_selected_without_budget():
    hypotheses = [
        SemanticHypothesis("a", lambda payload: 0 if payload == "safe" else 0),
        SemanticHypothesis("b", lambda payload: 0 if payload == "safe" else 1),
    ]
    probes = [
        SemanticProbe("safe-but-useless", "safe", risk=0.0),
        SemanticProbe("informative-but-risky", "risky", risk=2.0),
    ]

    choice = choose_probe(hypotheses, probes, max_risk=0.0)
    assert choice is not None
    assert choice.probe.name == "safe-but-useless"
    assert choice.information_gain_bits == 0.0


def test_inference_reports_ambiguity_when_no_probe_can_separate_hypotheses():
    hypotheses = [
        SemanticHypothesis("a", lambda payload: "same"),
        SemanticHypothesis("b", lambda payload: "same"),
    ]
    result = active_semantic_inference(
        hypotheses,
        [SemanticProbe("probe", "x")],
        observe=lambda probe: "same",
    )

    assert result.status == "ambiguous"
    assert set(result.survivors) == {"a", "b"}


def _round3(values):
    return tuple(round(float(value), 6) for value in values)


def test_real_rotation_geometry_selects_noncommuting_probe():
    """Single-axis motion is ambiguous; 3-axis motion separates the encodings."""

    def euler_prediction(payload):
        return _round3(payload)

    def axis_angle_prediction(payload):
        return _round3(compact_axis_angle_from_matrix(euler_xyz_matrix(payload)))

    hypotheses = [
        SemanticHypothesis("euler_xyz", euler_prediction),
        SemanticHypothesis("axis_angle", axis_angle_prediction),
    ]
    probes = [
        SemanticProbe("single-axis", (0.5, 0.0, 0.0), cost=1.0, risk=0.0),
        SemanticProbe("three-axis", (0.5, 0.5, 0.5), cost=1.0, risk=0.0),
    ]

    choice = choose_probe(hypotheses, probes)

    assert choice is not None
    assert choice.probe.name == "three-axis"
    assert choice.information_gain_bits == 1.0

    result = active_semantic_inference(
        hypotheses,
        probes,
        observe=lambda probe: axis_angle_prediction(probe.payload),
        max_rounds=1,
    )

    assert result.status == "identified"
    assert result.survivors == ("axis_angle",)


def test_joint_order_one_hot_probe_identifies_permutation():
    """One-hot joint excitation identifies a hidden two-joint ordering swap."""

    def identity(payload):
        index = int(payload)
        return (1.0, 0.0) if index == 0 else (0.0, 1.0)

    def swapped(payload):
        index = int(payload)
        return (0.0, 1.0) if index == 0 else (1.0, 0.0)

    hypotheses = [
        SemanticHypothesis("identity-order", identity),
        SemanticHypothesis("swap-01", swapped),
    ]
    probes = [
        SemanticProbe("excite-joint-0", "0", cost=1.0, risk=0.0),
        SemanticProbe("excite-joint-1", "1", cost=1.0, risk=0.0),
    ]

    result = active_semantic_inference(
        hypotheses,
        probes,
        observe=lambda probe: swapped(probe.payload),
        max_rounds=1,
    )

    assert result.status == "identified"
    assert result.survivors == ("swap-01",)
