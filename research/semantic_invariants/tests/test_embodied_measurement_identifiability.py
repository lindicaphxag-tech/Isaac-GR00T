from research.semantic_invariants.embodied_measurement_identifiability import (
    CounterfactualSemanticCase,
    audit_measurement_identifiability,
)


def _abs_distance(a, b):
    return abs(float(a) - float(b))


def test_shared_forward_inverse_error_creates_identifiability_collision():
    # Latent semantic anchor b changes, but the observable is a round-trip:
    # G_b(F_b(a)) = a for every b. The measurement is therefore blind to b.
    action = 7.0
    cases = (
        CounterfactualSemanticCase("correct-anchor", 2.0),
        CounterfactualSemanticCase("wrong-anchor-1", 11.0),
        CounterfactualSemanticCase("wrong-anchor-2", -5.0),
    )

    def roundtrip(anchor):
        relative = action - anchor
        return relative + anchor

    audit = audit_measurement_identifiability(
        channel_id="forward-inverse-roundtrip",
        cases=cases,
        observe=roundtrip,
        semantic_distance=_abs_distance,
        observable_distance=_abs_distance,
    )

    assert audit.disproved
    assert audit.status == "fail"
    assert len(audit.collisions) == 3
    assert all(item.observable_distance == 0.0 for item in audit.collisions)


def test_external_one_way_anchor_breaks_the_collision():
    action = 7.0
    correct_anchor = 2.0
    cases = (
        CounterfactualSemanticCase("correct-anchor", 2.0),
        CounterfactualSemanticCase("wrong-anchor", 11.0),
    )

    # Compare the produced one-way semantic target against an independently
    # specified correct target, rather than composing it with its own inverse.
    correct_relative = action - correct_anchor

    def anchored_forward_error(anchor):
        produced = action - anchor
        return abs(produced - correct_relative)

    audit = audit_measurement_identifiability(
        channel_id="one-way-external-anchor",
        cases=cases,
        observe=anchored_forward_error,
        semantic_distance=_abs_distance,
        observable_distance=_abs_distance,
    )

    assert not audit.disproved
    # Finite collision search never upgrades itself to a global pass.
    assert audit.status == "undetermined"


def test_no_collision_is_not_misrepresented_as_identifiability_proof():
    cases = (
        CounterfactualSemanticCase("a", 0.0),
        CounterfactualSemanticCase("b", 1.0),
        CounterfactualSemanticCase("c", 2.0),
    )
    audit = audit_measurement_identifiability(
        channel_id="identity-observable",
        cases=cases,
        observe=lambda x: x,
        semantic_distance=_abs_distance,
        observable_distance=_abs_distance,
    )

    assert not audit.disproved
    assert audit.status == "undetermined"
