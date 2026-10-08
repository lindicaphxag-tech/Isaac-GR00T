"""Source-pinned *mechanism abstraction* for ManiSkill #1495 / #1472.

Not a native simulator replay: this fixes actual source commit identities and
models only a single-axis, unsaturated signed rotational action. Multi-axis
XYZ Euler composition, controller dynamics, IK, rollout success, and physical
noise calibration are explicitly outside the abstraction.
"""

import pytest

from research.semantic_invariants.embodied_robust_diagnosis import (
    RobustObservation,
    RobustResolution,
    RobustProbe,
    advance_robust_diagnosis,
    synthesize_robust_experiment_plan,
)
from research.semantic_invariants.embodied_semantic_experiment_design import (
    SemanticExperiment,
)
from research.semantic_invariants.embodied_semantic_observability import (
    SemanticDiagnosisHypothesis,
)
from research.semantic_invariants.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
)

# Source reviews: baseline controller multiplies by rot_lower; #1472
# proposes replacing it with rot_upper. #1495 inspects the signed controller
# mapper instead of assuming positive rotational action scaling.
MANISKILL_BASE = "62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3"
CONVERSION_HEAD = "69facfaafaa0ef233d36ef19e6cd9a0f03532ee0"
CONTROLLER_OVERLAY = "eed9be164797d41540421bda8adb3840377d7087"


def _four_worlds():
    sign_worlds = (
        ("canonical", +1, +1),
        ("converter-inverted", -1, +1),
        ("controller-inverted", +1, -1),
        ("compensating-double-inversion", -1, -1),
    )
    return tuple(
        SemanticDiagnosisHypothesis(
            name,
            (
                SemanticTransportFactor(
                    "conversion-output",
                    MonomialSemanticTransport((0, 1, 2), (1., 1., float(enc))),
                    f"model-from-mani-pr1495:{CONVERSION_HEAD};sign={enc}",
                ),
                SemanticTransportFactor(
                    "controller-output",
                    MonomialSemanticTransport((0, 1, 2), (1., 1., float(ctrl))),
                    f"model-from-mani-pr1472:{CONTROLLER_OVERLAY};sign={ctrl}",
                ),
            ),
        )
        for name, enc, ctrl in sign_worlds
    )


def _experiments(*, include_tap=True):
    physical = SemanticExperiment(
        "eef-yaw-after-controller", (0., 0., 0.3),
        tap_after_factor="controller-output", cost=1.0, risk=0.05,
    )
    tap = SemanticExperiment(
        "normalized-action-after-converter", (0., 0., 0.3),
        tap_after_factor="conversion-output", cost=0.2, risk=0.0,
    )
    return (physical, tap) if include_tap else (physical,)


def _identities():
    return {h.name: f"sha256:fixture-repair-for-{h.name}" for h in _four_worlds()}


def test_end_effector_alone_cannot_see_compensating_action_sign_defects():
    world_names = tuple(h.name for h in _four_worlds())
    without_tap = synthesize_robust_experiment_plan(
        _four_worlds(), _experiments(include_tap=False),
        epsilon=0.01, risk_budget=0.05, max_probe_risk=0.05,
        authorities=_identities(),
    )
    assert not without_tap.complete
    assert without_tap.root is None

    # Even a robot that produces the expected positive yaw is compatible
    # with two incompatible internal repair mechanisms.
    physical_pairs = {
        "canonical": +0.3,
        "converter-inverted": -0.3,
        "controller-inverted": -0.3,
        "compensating-double-inversion": +0.3,
    }
    assert physical_pairs["canonical"] == physical_pairs["compensating-double-inversion"]
    assert physical_pairs["converter-inverted"] == physical_pairs["controller-inverted"]
    assert set(world_names) == set(physical_pairs)


def test_internal_semantic_tap_resolves_all_four_authority_worlds_within_risk():
    hs = _four_worlds()
    plan = synthesize_robust_experiment_plan(
        hs, _experiments(),
        epsilon=0.01, risk_budget=0.05, max_probe_risk=0.05,
        authorities=_identities(),
    )
    assert plan.complete
    assert plan.worst_path_risk == pytest.approx(0.05)
    assert plan.worst_cost == pytest.approx(1.2)

    for actual in hs:
        observations = []
        decision = advance_robust_diagnosis(
            plan=plan, trusted_plan_digest=plan.digest,
        )
        for _ in range(2):
            assert isinstance(decision, RobustProbe)
            exp = decision.experiment
            # This is the declared model's exact expected observation,
            # NOT an actual SAPIEN or end-effector measurement.
            predictions = dict(dict(plan.predictions)[exp.name])
            observations.append(RobustObservation(exp.name, predictions[actual.name]))
            decision = advance_robust_diagnosis(
                plan=plan, trusted_plan_digest=plan.digest,
                observations=tuple(observations),
            )
        assert isinstance(decision, RobustResolution)
        assert decision.authority_id == _identities()[actual.name]
        assert decision.consistent_hypotheses == (actual.name,)
        assert decision.spent_risk == pytest.approx(0.05)


def test_insufficient_actuation_risk_budget_fail_closes_even_with_free_internal_tap():
    denied = synthesize_robust_experiment_plan(
        _four_worlds(), _experiments(),
        epsilon=0.01, risk_budget=0.04, max_probe_risk=0.05,
        authorities=_identities(),
    )
    assert not denied.complete

    # If all matched end-effector outcomes also require the same
    # concrete repair identity, the internal tap need not be opened.
    equal_ee_authorities = {
        "canonical": "verified/positive-yaw",
        "compensating-double-inversion": "verified/positive-yaw",
        "converter-inverted": "verified/negative-yaw",
        "controller-inverted": "verified/negative-yaw",
    }
    plan = synthesize_robust_experiment_plan(
        _four_worlds(), _experiments(include_tap=False),
        epsilon=0.01, risk_budget=0.05, max_probe_risk=0.05,
        authorities=equal_ee_authorities,
    )
    assert plan.complete
    assert plan.worst_path_risk == pytest.approx(0.05)


def test_source_identity_and_internal_tap_identity_are_bound_into_digest():
    plain = synthesize_robust_experiment_plan(
        _four_worlds(), _experiments(),
        epsilon=0.01, risk_budget=0.05, max_probe_risk=0.05,
        authorities=_identities(),
    )
    worlds = list(_four_worlds())
    modified = list(worlds[0].factors)
    modified[0] = SemanticTransportFactor(
        modified[0].name, modified[0].transport,
        "untrusted:changed-converter-commit",
    )
    from dataclasses import replace
    worlds[0] = replace(worlds[0], factors=tuple(modified))
    changed = synthesize_robust_experiment_plan(
        worlds, _experiments(),
        epsilon=0.01, risk_budget=0.05, max_probe_risk=0.05,
        authorities=_identities(),
    )
    assert changed.digest != plain.digest
