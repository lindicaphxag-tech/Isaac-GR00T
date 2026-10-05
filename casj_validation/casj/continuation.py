from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class PathDerivativeSample:
    """Two-scale pathwise action derivative at one continuation coordinate.

    fine and coarse are independent finite-difference estimates of
    d(action_chunk) / d(tau) at the same physical path coordinate tau.
    They must use the same frozen policy state and, for stochastic policies,
    the same complete replay randomness.
    """

    fine: np.ndarray
    coarse: np.ndarray


@dataclass(frozen=True)
class PathTransportCertificate:
    """A posteriori certificate for path-integrated CASJ transport.

    The correction is the composite-Simpson integral of the fine pathwise
    derivative over tau in [0, 1]. The same derivative samples are integrated
    on a nested coarse grid, and their difference supplies a Richardson
    quadrature-defect estimate. This is a numerical certificate under sampled
    smoothness assumptions, not a formal supremum bound on unseen curvature.
    """

    accepted: bool
    correction: np.ndarray
    coarse_correction: np.ndarray
    richardson_error: np.ndarray
    richardson_error_norm: float
    error_to_correction: float
    error_to_reference_scale: float
    max_local_scale_instability: float
    coarse_intervals: int
    fine_intervals: int
    derivative_queries: int
    reason: str


def _relative_difference(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    return float(
        np.linalg.norm(a - b)
        / max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), eps)
    )


def _composite_simpson(values: np.ndarray, *, interval_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim < 2:
        raise ValueError("values must have shape [N + 1, ...]")
    if interval_count < 2 or interval_count % 2:
        raise ValueError("Simpson interval_count must be even and >= 2")
    if values.shape[0] != interval_count + 1:
        raise ValueError("values first dimension must equal interval_count + 1")

    weighted = values[0] + values[-1]
    weighted = weighted + 4.0 * np.sum(values[1:-1:2], axis=0)
    if interval_count > 2:
        weighted = weighted + 2.0 * np.sum(values[2:-1:2], axis=0)
    return weighted / (3.0 * interval_count)


def certify_path_integrated_transport(
    sample_derivative: Callable[[float], PathDerivativeSample],
    *,
    reference_action_scale: float,
    coarse_intervals: int = 4,
    max_scale_instability: float = 0.15,
    max_remainder_to_correction: float = 0.25,
    max_remainder_to_reference_scale: float = 0.05,
    eps: float = 1e-12,
) -> PathTransportCertificate:
    """Integrate a CASJ directional field along a finite support-motion path.

    The method is intentionally different from extrapolating one local
    Jacobian to a large displacement. It samples the pathwise derivative at a
    nested Simpson grid, checks two-scale derivative stability at every node,
    then integrates the locally valid tangent responses.

    The acceptance limits deliberately reuse the frozen CASJ local certificate
    ratios (0.15 scale stability, 0.25 effect-relative remainder, 0.05
    reference-relative remainder) instead of tuning new thresholds on public
    outcomes.

    sample_derivative(tau) must return fine/coarse estimates of
    d(action_chunk)/d(tau) at tau in [0, 1]. For stochastic policies the two
    scales and every path node must be coupled to the same replay randomness.
    """
    if coarse_intervals < 2 or coarse_intervals % 2:
        raise ValueError("coarse_intervals must be even and >= 2")
    if not isfinite(reference_action_scale) or reference_action_scale <= 0:
        raise ValueError("reference_action_scale must be finite and positive")
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    for value, name in (
        (max_scale_instability, "max_scale_instability"),
        (max_remainder_to_correction, "max_remainder_to_correction"),
        (max_remainder_to_reference_scale, "max_remainder_to_reference_scale"),
    ):
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    fine_intervals = 2 * coarse_intervals
    nodes = np.linspace(0.0, 1.0, fine_intervals + 1)
    fine_values: list[np.ndarray] = []
    instabilities: list[float] = []
    expected_shape: tuple[int, ...] | None = None

    for tau in nodes:
        sample = sample_derivative(float(tau))
        fine = np.asarray(sample.fine, dtype=float)
        coarse = np.asarray(sample.coarse, dtype=float)
        if fine.size == 0 or coarse.size == 0:
            raise ValueError("path derivative samples must be non-empty")
        if fine.shape != coarse.shape:
            raise ValueError("fine and coarse path derivatives must match shape")
        if expected_shape is None:
            expected_shape = fine.shape
        elif fine.shape != expected_shape:
            raise ValueError("all path derivative samples must share one shape")
        if not np.all(np.isfinite(fine)) or not np.all(np.isfinite(coarse)):
            raise ValueError("path derivative samples must be finite")

        fine_values.append(fine)
        instabilities.append(_relative_difference(fine, coarse, eps))

    stacked = np.stack(fine_values, axis=0)
    fine_integral = _composite_simpson(stacked, interval_count=fine_intervals)
    coarse_integral = _composite_simpson(
        stacked[::2], interval_count=coarse_intervals
    )

    # Composite Simpson is fourth order. For nested h and h/2 grids, the
    # leading error estimate for the fine integral is |S_h/2 - S_h| / 15.
    richardson = (fine_integral - coarse_integral) / 15.0
    error_norm = float(np.linalg.norm(richardson))
    correction_norm = float(np.linalg.norm(fine_integral))
    error_to_correction = error_norm / max(correction_norm, eps)
    error_to_reference = error_norm / float(reference_action_scale)
    max_instability = max(instabilities)

    failures = [
        (
            max_instability > max_scale_instability,
            "a local path derivative is unstable across probe scales",
        ),
        (
            error_to_correction > max_remainder_to_correction,
            "nested path integration has excessive defect relative to correction",
        ),
        (
            error_to_reference > max_remainder_to_reference_scale,
            "nested path integration has excessive defect relative to action scale",
        ),
    ]
    for failed, reason in failures:
        if failed:
            return PathTransportCertificate(
                accepted=False,
                correction=fine_integral,
                coarse_correction=coarse_integral,
                richardson_error=richardson,
                richardson_error_norm=error_norm,
                error_to_correction=error_to_correction,
                error_to_reference_scale=error_to_reference,
                max_local_scale_instability=max_instability,
                coarse_intervals=coarse_intervals,
                fine_intervals=fine_intervals,
                derivative_queries=len(nodes),
                reason=reason,
            )

    return PathTransportCertificate(
        accepted=True,
        correction=fine_integral,
        coarse_correction=coarse_integral,
        richardson_error=richardson,
        richardson_error_norm=error_norm,
        error_to_correction=error_to_correction,
        error_to_reference_scale=error_to_reference,
        max_local_scale_instability=max_instability,
        coarse_intervals=coarse_intervals,
        fine_intervals=fine_intervals,
        derivative_queries=len(nodes),
        reason=(
            "all local derivatives are scale-stable and nested Simpson transport "
            "passes the frozen CASJ remainder ratios"
        ),
    )
