from __future__ import annotations

from dataclasses import dataclass
from math import inf, sqrt

import numpy as np


@dataclass(frozen=True)
class DirectionalTrustRegion:
    """Certified radius for one physical support-motion direction.

    The radius is expressed as a multiplier of the direction used to estimate
    first- and second-order action response. A radius >= 1 therefore certifies
    the full requested disturbance when that disturbance is the unit direction.
    """

    accepted: bool
    radius: float
    curvature_scale_stability: float
    first_order_norm_per_unit: float
    curvature_norm_per_unit2: float
    reference_action_scale: float
    radius_from_first_order: float
    radius_from_reference_scale: float
    limiting_constraint: str
    reason: str


def certify_directional_trust_region(
    *,
    fine_second_per_unit2: np.ndarray,
    coarse_second_per_unit2: np.ndarray,
    first_order_per_unit: np.ndarray,
    reference_action_scale: float,
    max_curvature_scale_instability: float = 0.25,
    max_remainder_to_first_order: float = 0.25,
    max_remainder_to_reference_scale: float = 0.05,
    eps: float = 1e-12,
) -> DirectionalTrustRegion:
    """Invert the existing Taylor-remainder limits into a trust radius.

    For displacement multiplier r along one fixed physical direction, the
    second-order remainder bound is 0.5 * r^2 * ||H_d||. The corresponding
    first-order effect is r * ||J_d||. The two existing certificate limits
    therefore imply analytic upper bounds on r. No threshold is learned or
    tuned from observed repair outcomes.
    """
    fine = np.asarray(fine_second_per_unit2, dtype=float)
    coarse = np.asarray(coarse_second_per_unit2, dtype=float)
    first = np.asarray(first_order_per_unit, dtype=float)
    if fine.shape != coarse.shape:
        raise ValueError("fine and coarse directional curvature must have identical shapes")
    if reference_action_scale <= 0:
        raise ValueError("reference_action_scale must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    for value, name in (
        (max_curvature_scale_instability, "max_curvature_scale_instability"),
        (max_remainder_to_first_order, "max_remainder_to_first_order"),
        (max_remainder_to_reference_scale, "max_remainder_to_reference_scale"),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    fine_norm = float(np.linalg.norm(fine))
    coarse_norm = float(np.linalg.norm(coarse))
    stability = float(
        np.linalg.norm(fine - coarse) / max(fine_norm, coarse_norm, eps)
    )
    first_norm = float(np.linalg.norm(first))

    if stability > max_curvature_scale_instability:
        return DirectionalTrustRegion(
            accepted=False,
            radius=0.0,
            curvature_scale_stability=stability,
            first_order_norm_per_unit=first_norm,
            curvature_norm_per_unit2=fine_norm,
            reference_action_scale=float(reference_action_scale),
            radius_from_first_order=0.0,
            radius_from_reference_scale=0.0,
            limiting_constraint="curvature_scale_stability",
            reason="directional curvature is unstable across probe scales",
        )

    if fine_norm <= eps:
        return DirectionalTrustRegion(
            accepted=True,
            radius=inf,
            curvature_scale_stability=stability,
            first_order_norm_per_unit=first_norm,
            curvature_norm_per_unit2=fine_norm,
            reference_action_scale=float(reference_action_scale),
            radius_from_first_order=inf,
            radius_from_reference_scale=inf,
            limiting_constraint="none",
            reason="direction is locally linear through the measured second order",
        )

    radius_reference = sqrt(
        2.0 * max_remainder_to_reference_scale * reference_action_scale / fine_norm
    )
    radius_first = (
        2.0 * max_remainder_to_first_order * first_norm / fine_norm
        if first_norm > eps
        else inf
    )
    radius = min(radius_reference, radius_first)
    if radius_reference <= radius_first:
        limiting = "reference_action_scale"
    else:
        limiting = "first_order_effect"

    return DirectionalTrustRegion(
        accepted=bool(radius > 0.0),
        radius=float(radius),
        curvature_scale_stability=stability,
        first_order_norm_per_unit=first_norm,
        curvature_norm_per_unit2=fine_norm,
        reference_action_scale=float(reference_action_scale),
        radius_from_first_order=float(radius_first),
        radius_from_reference_scale=float(radius_reference),
        limiting_constraint=limiting,
        reason=(
            "bounded directional Taylor model; requested motion is certified "
            "iff its direction multiplier does not exceed radius"
        ),
    )


def trust_region_accepts(
    certificate: DirectionalTrustRegion, requested_multiplier: float = 1.0
) -> bool:
    """Return whether a requested motion lies inside the frozen trust region."""
    if requested_multiplier < 0:
        raise ValueError("requested_multiplier must be non-negative")
    return bool(certificate.accepted and requested_multiplier <= certificate.radius)
