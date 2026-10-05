from math import isinf

import numpy as np
import pytest

from casj.trust_region import certify_directional_trust_region, trust_region_accepts


def test_linear_direction_has_unbounded_second_order_radius():
    cert = certify_directional_trust_region(
        fine_second_per_unit2=np.zeros(2),
        coarse_second_per_unit2=np.zeros(2),
        first_order_per_unit=np.array([2.0, 0.0]),
        reference_action_scale=1.0,
    )
    assert cert.accepted
    assert isinf(cert.radius)
    assert trust_region_accepts(cert, 100.0)


def test_radius_is_analytic_minimum_of_frozen_remainder_bounds():
    # ||H||=2, ||J||=4, reference=10.
    # first-order bound: 2*0.25*4/2 = 1
    # reference bound: sqrt(2*0.05*10/2) = sqrt(0.5)
    cert = certify_directional_trust_region(
        fine_second_per_unit2=np.array([2.0, 0.0]),
        coarse_second_per_unit2=np.array([2.0, 0.0]),
        first_order_per_unit=np.array([4.0, 0.0]),
        reference_action_scale=10.0,
    )
    assert cert.radius == pytest.approx(np.sqrt(0.5))
    assert cert.limiting_constraint == "reference_action_scale"
    assert trust_region_accepts(cert, 0.7)
    assert not trust_region_accepts(cert, 0.8)


def test_unstable_curvature_collapses_trust_region_instead_of_relaxing_threshold():
    cert = certify_directional_trust_region(
        fine_second_per_unit2=np.array([1.0, 0.0]),
        coarse_second_per_unit2=np.array([2.0, 0.0]),
        first_order_per_unit=np.array([1.0, 0.0]),
        reference_action_scale=1.0,
    )
    assert not cert.accepted
    assert cert.radius == 0.0
    assert cert.limiting_constraint == "curvature_scale_stability"
    assert not trust_region_accepts(cert, 0.01)


def test_zero_first_order_uses_absolute_reference_bound():
    cert = certify_directional_trust_region(
        fine_second_per_unit2=np.array([0.5]),
        coarse_second_per_unit2=np.array([0.5]),
        first_order_per_unit=np.array([0.0]),
        reference_action_scale=2.0,
    )
    assert isinf(cert.radius_from_first_order)
    assert cert.radius == pytest.approx(np.sqrt(0.4))


@pytest.mark.parametrize("value", [-1.0, -0.1])
def test_negative_requested_multiplier_is_rejected(value):
    cert = certify_directional_trust_region(
        fine_second_per_unit2=np.zeros(1),
        coarse_second_per_unit2=np.zeros(1),
        first_order_per_unit=np.ones(1),
        reference_action_scale=1.0,
    )
    with pytest.raises(ValueError):
        trust_region_accepts(cert, value)
