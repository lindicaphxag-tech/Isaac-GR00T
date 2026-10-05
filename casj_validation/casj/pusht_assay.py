"""Pure helpers for physical-direction PushT intervention assays."""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import directional_second_derivative
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