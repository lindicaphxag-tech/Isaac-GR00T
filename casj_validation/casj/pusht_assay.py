"""Pure helpers for physical-direction PushT intervention assays."""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import directional_second_derivative
from .continuation import PathDerivativeSample
from .pusht_state import PushTSnapshot


def estimate_directional_curvature_via_query(
    *,
    policy_query: Any,
    history: list[PushTSnapshot],
    baseline: np.ndarray,
    randomness: Any,
    support_delta: np.ndarray,
    step_fraction: float,
) -> np.ndarray:
    """Estimate second action derivative along a full x/y/theta pose path.

    ``step_fraction`` parameterizes a dimensionless path: fraction ``s``
    applies ``s * support_delta`` in the heterogeneous pixel/radian chart.
    Central differences along this path include mixed terms such as
    ``d²a/(dx dtheta)`` that coordinate-wise Hessian probes miss.
    """
    support_delta = np.asarray(support_delta, dtype=np.float64)
    if support_delta.shape != (3,):
        raise ValueError("support_delta must contain x, y, and angle")
    if step_fraction <= 0:
        raise ValueError("step_fraction must be positive")
    if not np.any(support_delta):
        return np.zeros_like(baseline, dtype=np.float64)

    plus_history = [
        state.shifted_block(
            step_fraction * support_delta[:2],
            step_fraction * support_delta[2],
        )
        for state in history
    ]
    minus_history = [
        state.shifted_block(
            -step_fraction * support_delta[:2],
            -step_fraction * support_delta[2],
        )
        for state in history
    ]
    plus = policy_query.query(plus_history, randomness=randomness)
    minus = policy_query.query(minus_history, randomness=randomness)
    return directional_second_derivative(
        plus,
        minus,
        baseline,
        step=step_fraction,
    )

def estimate_pathwise_directional_derivative_via_query(
    *,
    policy_query: Any,
    history: list[PushTSnapshot],
    randomness: Any,
    support_delta: np.ndarray,
    path_fraction: float,
    fine_step_fraction: float,
    coarse_step_fraction: float,
) -> PathDerivativeSample:
    """Estimate d(action_chunk)/d(tau) along a physical support-motion path.

    Only the support pose is changed. The policy, observation history outside
    that support intervention, and stochastic replay token remain fixed.
    Probes are restricted to the certified finite path [0, 1]. Central
    differences are used in the interior and one-sided differences at path
    boundaries, so no out-of-path support state is introduced by the assay.
    """
    support_delta = np.asarray(support_delta, dtype=np.float64)
    if support_delta.shape != (3,):
        raise ValueError("support_delta must contain x, y, and angle")
    if not 0.0 <= float(path_fraction) <= 1.0:
        raise ValueError("path_fraction must lie in [0, 1]")
    if fine_step_fraction <= 0 or coarse_step_fraction <= 0:
        raise ValueError("path derivative probe scales must be positive")
    if coarse_step_fraction <= fine_step_fraction:
        raise ValueError("coarse_step_fraction must exceed fine_step_fraction")

    if not np.any(support_delta):
        baseline = policy_query.query(history, randomness=randomness)
        zeros = np.zeros_like(baseline, dtype=np.float64)
        return PathDerivativeSample(fine=zeros, coarse=zeros.copy())

    def query_at(fraction: float) -> np.ndarray:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("path probe fraction must remain in [0, 1]")
        shifted = [
            state.shifted_block(
                fraction * support_delta[:2],
                fraction * support_delta[2],
            )
            for state in history
        ]
        return policy_query.query(shifted, randomness=randomness)

    def bounded_derivative(step: float) -> np.ndarray:
        # Central differences are preferred in the path interior.  At tau=0
        # and tau=1 (or within one probe step of an endpoint), use a one-sided
        # stencil so the assay never invents support states outside the finite
        # runtime displacement being certified.
        lower = path_fraction - step
        upper = path_fraction + step
        if lower >= 0.0 and upper <= 1.0:
            plus = query_at(upper)
            minus = query_at(lower)
            return (plus - minus) / (2.0 * step)
        center = query_at(path_fraction)
        if upper <= 1.0:
            plus = query_at(upper)
            return (plus - center) / step
        if lower >= 0.0:
            minus = query_at(lower)
            return (center - minus) / step
        raise ValueError("path derivative probe step is too large for [0, 1]")

    fine = bounded_derivative(fine_step_fraction)
    coarse = bounded_derivative(coarse_step_fraction)
    return PathDerivativeSample(fine=fine, coarse=coarse)
