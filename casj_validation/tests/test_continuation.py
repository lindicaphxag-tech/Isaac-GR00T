import numpy as np
import pytest

from casj.continuation import (
    PathDerivativeSample,
    certify_path_integrated_transport,
)


def _stable_sample(fn):
    def sample(tau: float) -> PathDerivativeSample:
        value = np.asarray(fn(tau), dtype=float)
        return PathDerivativeSample(fine=value, coarse=value.copy())
    return sample


def test_constant_field_integrates_exactly():
    cert = certify_path_integrated_transport(
        _stable_sample(lambda tau: np.array([[2.0, -1.0]])),
        reference_action_scale=10.0,
    )
    assert cert.accepted
    np.testing.assert_allclose(cert.correction, [[2.0, -1.0]], atol=1e-12)
    assert cert.richardson_error_norm == pytest.approx(0.0)


def test_cubic_directional_field_is_exact_under_simpson():
    cert = certify_path_integrated_transport(
        _stable_sample(
            lambda t: np.array([1.0 + 2.0*t + 3.0*t**2 + 4.0*t**3])
        ),
        reference_action_scale=4.0,
        coarse_intervals=4,
    )
    assert cert.accepted
    np.testing.assert_allclose(cert.correction, [4.0], atol=1e-12)


def test_scale_unstable_local_derivative_fails_closed():
    def sample(tau: float) -> PathDerivativeSample:
        return PathDerivativeSample(
            fine=np.array([1.0 + tau]),
            coarse=np.array([2.0 + tau]),
        )

    cert = certify_path_integrated_transport(sample, reference_action_scale=2.0)
    assert not cert.accepted
    assert cert.max_local_scale_instability > 0.15
    assert "unstable" in cert.reason


def test_high_quadrature_defect_is_rejected_without_threshold_relaxation():
    cert = certify_path_integrated_transport(
        _stable_sample(lambda t: np.array([np.exp(8.0*t)])),
        reference_action_scale=1.0,
        coarse_intervals=2,
    )
    assert not cert.accepted
    assert cert.error_to_reference_scale > 0.05
    assert "action scale" in cert.reason


def test_two_scale_noise_can_be_small_and_still_integrate():
    def sample(tau: float) -> PathDerivativeSample:
        fine = np.array([1.0 + tau, 2.0 - 0.5*tau])
        coarse = fine * 1.01
        return PathDerivativeSample(fine=fine, coarse=coarse)

    cert = certify_path_integrated_transport(
        sample,
        reference_action_scale=5.0,
        coarse_intervals=4,
    )
    assert cert.accepted
    np.testing.assert_allclose(cert.correction, [1.5, 1.75], atol=1e-12)


@pytest.mark.parametrize("intervals", [0, 1, 3, 5])
def test_requires_even_nested_simpson_partition(intervals):
    with pytest.raises(ValueError):
        certify_path_integrated_transport(
            _stable_sample(lambda tau: np.ones(1)),
            reference_action_scale=1.0,
            coarse_intervals=intervals,
        )
