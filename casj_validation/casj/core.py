from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class RelationType(str, Enum):
    NO_LOCAL_SUPPORT = "no_local_support"
    RIGID_FOLLOW_FINGERPRINT = "rigid_follow_fingerprint"
    SINGLE_SUPPORT_LOCAL = "single_support_local"
    COMMON_MOTION_FINGERPRINT = "common_motion_fingerprint"
    RELATIVE_INVARIANT_FINGERPRINT = "relative_invariant_fingerprint"
    GENERAL_RELATIONAL = "general_relational"


class RepairMode(str, Enum):
    KEEP = "keep"
    EXACT_TRANSPORT = "exact_transport"
    LOCAL_LINEAR_REPAIR = "local_linear_repair"
    REPLAN = "replan"


@dataclass(frozen=True)
class CASJEstimate:
    """Counterfactual Action-Support Jacobian.

    jacobian[t, j] maps a local perturbation of candidate support j into the
    decoded task-space action response at action position t.

    Shape: [H, K, D_action, D_support].
    """

    jacobian: np.ndarray
    fit_residual: np.ndarray
    codes: np.ndarray

    @property
    def support_norm(self) -> np.ndarray:
        return np.linalg.norm(self.jacobian, axis=(2, 3))


@dataclass(frozen=True)
class RelationGeometry:
    relation_type: RelationType
    active_supports: tuple[int, ...]
    support_norms: np.ndarray
    sum_block: np.ndarray
    sum_to_identity_residual: float
    sum_to_zero_residual: float
    single_identity_residual: float | None


@dataclass(frozen=True)
class CASJCertificate:
    accepted: bool
    repair_mode: RepairMode
    relation: RelationGeometry
    scale_stability_error: float
    fit_residual: float
    active_code_condition: float
    reason: str


def gaussian_codes(num_probes: int, num_supports: int, *, seed: int = 0) -> np.ndarray:
    """Deterministic normalized Gaussian support codes."""
    if num_probes <= 0 or num_supports <= 0:
        raise ValueError("num_probes and num_supports must be positive")
    rng = np.random.default_rng(seed)
    codes = rng.normal(size=(num_probes, num_supports))
    return codes / np.maximum(np.linalg.norm(codes, axis=0, keepdims=True), 1e-12)


def _group_omp(codes: np.ndarray, response: np.ndarray, *, sparsity: int) -> tuple[np.ndarray, float]:
    """Recover row-sparse coefficients in response ~= codes @ coefficients."""
    codes = np.asarray(codes, dtype=float)
    response = np.asarray(response, dtype=float)
    if codes.ndim != 2:
        raise ValueError("codes must have shape [M, K]")
    if response.ndim != 2 or response.shape[0] != codes.shape[0]:
        raise ValueError("response must have shape [M, D_action]")
    if not 1 <= sparsity <= codes.shape[1]:
        raise ValueError("sparsity must lie inside [1, K]")

    active: list[int] = []
    residual = response.copy()
    solution = np.zeros((codes.shape[1], response.shape[1]), dtype=float)
    local = np.empty((0, response.shape[1]), dtype=float)

    for _ in range(sparsity):
        scores = np.linalg.norm(codes.T @ residual, axis=1)
        if active:
            scores[active] = -np.inf
        active.append(int(np.argmax(scores)))
        local, *_ = np.linalg.lstsq(codes[:, active], response, rcond=None)
        residual = response - codes[:, active] @ local

    solution[active] = local
    fit = float(np.linalg.norm(residual) / max(np.linalg.norm(response), 1e-12))
    return solution, fit


def recover_casj(
    responses: np.ndarray,
    *,
    codes: np.ndarray,
    sparsity: int,
) -> CASJEstimate:
    """Recover a sparse CASJ from simultaneous coded interventions.

    responses has shape [Q, M, H, D_action]:
      Q: support perturbation basis directions
      M: simultaneous coded probes
      H: action-chunk horizon

    For direction q and action position t:

        responses[q, :, t, :] ~= codes @ J[t, :, :, q].
    """
    responses = np.asarray(responses, dtype=float)
    codes = np.asarray(codes, dtype=float)
    if responses.ndim != 4:
        raise ValueError("responses must have shape [Q, M, H, D_action]")
    q_dim, probes, horizon, action_dim = responses.shape
    if codes.ndim != 2 or codes.shape[0] != probes:
        raise ValueError("codes must have shape [M, K] and match responses")

    num_supports = codes.shape[1]
    jacobian = np.zeros((horizon, num_supports, action_dim, q_dim), dtype=float)
    residual = np.zeros((horizon, q_dim), dtype=float)

    for q in range(q_dim):
        for t in range(horizon):
            block, fit = _group_omp(codes, responses[q, :, t, :], sparsity=sparsity)
            jacobian[t, :, :, q] = block
            residual[t, q] = fit

    return CASJEstimate(jacobian=jacobian, fit_residual=residual, codes=codes)


def apply_casj(estimate: CASJEstimate, support_delta: np.ndarray) -> np.ndarray:
    """Predict a local decoded-action correction: delta_a ~= sum_j J_j delta_s_j."""
    support_delta = np.asarray(support_delta, dtype=float)
    j = estimate.jacobian
    expected = (j.shape[1], j.shape[3])
    if support_delta.shape != expected:
        raise ValueError(f"support_delta must have shape {expected}")
    return np.einsum("hkdo,ko->hd", j, support_delta)


def _relative_frobenius(error: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(error) / max(float(np.linalg.norm(reference)), 1e-12))


def classify_relation(
    blocks: np.ndarray,
    *,
    active_threshold: float = 0.10,
    rigid_tolerance: float = 0.20,
    relation_tolerance: float = 0.20,
) -> RelationGeometry:
    """Infer local relation geometry from one action position's CASJ blocks.

    For square local task coordinates these are local fingerprints, not
    globally unique semantic identifications:
      all J_j ~= 0                  -> no detectable local support response
      J_A ~= I                     -> rigid-follow-compatible fingerprint
      sum_j J_j ~= I               -> common-motion-compatible fingerprint
      sum_j J_j ~= 0               -> relative-invariant-compatible fingerprint

    The converses do not generally hold. A matching first-order fingerprint
    only licenses a repair after scale, curvature, conditioning, and physical
    validity checks also pass.
    """
    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
        raise ValueError("blocks must have shape [K, D, D]")

    norms = np.linalg.norm(blocks, axis=(1, 2))
    active = tuple(np.flatnonzero(norms > active_threshold).tolist())
    identity = np.eye(blocks.shape[1])
    sum_block = blocks[list(active)].sum(axis=0) if active else np.zeros_like(identity)

    sum_to_identity = _relative_frobenius(sum_block - identity, identity)
    sum_to_zero = float(np.linalg.norm(sum_block)) / max(
        float(np.sum(norms[list(active)])) if active else 0.0,
        1e-12,
    )
    single_identity = (
        _relative_frobenius(blocks[active[0]] - identity, identity)
        if len(active) == 1
        else None
    )

    if not active:
        kind = RelationType.NO_LOCAL_SUPPORT
    elif len(active) == 1 and single_identity is not None and single_identity <= rigid_tolerance:
        kind = RelationType.RIGID_FOLLOW_FINGERPRINT
    elif len(active) == 1:
        kind = RelationType.SINGLE_SUPPORT_LOCAL
    elif sum_to_identity <= relation_tolerance:
        kind = RelationType.COMMON_MOTION_FINGERPRINT
    elif sum_to_zero <= relation_tolerance:
        kind = RelationType.RELATIVE_INVARIANT_FINGERPRINT
    else:
        kind = RelationType.GENERAL_RELATIONAL

    return RelationGeometry(
        relation_type=kind,
        active_supports=active,
        support_norms=norms,
        sum_block=sum_block,
        sum_to_identity_residual=sum_to_identity,
        sum_to_zero_residual=sum_to_zero,
        single_identity_residual=single_identity,
    )


def _scale_error(fine: np.ndarray, coarse: np.ndarray) -> float:
    return float(
        np.linalg.norm(fine - coarse)
        / max(np.linalg.norm(fine), np.linalg.norm(coarse), 1e-12)
    )


def certify_repair(
    *,
    fine_blocks: np.ndarray,
    coarse_blocks: np.ndarray,
    fit_residual: float,
    active_code_condition: float,
    active_threshold: float = 0.10,
    rigid_tolerance: float = 0.20,
    relation_tolerance: float = 0.20,
    max_scale_instability: float = 0.15,
    max_fit_residual: float = 0.15,
    max_code_condition: float = 10.0,
    max_active_supports: int = 3,
) -> CASJCertificate:
    """Conservatively map a local CASJ to KEEP / TRANSPORT / REPAIR / REPLAN."""
    fine = classify_relation(
        fine_blocks,
        active_threshold=active_threshold,
        rigid_tolerance=rigid_tolerance,
        relation_tolerance=relation_tolerance,
    )
    coarse = classify_relation(
        coarse_blocks,
        active_threshold=active_threshold,
        rigid_tolerance=rigid_tolerance,
        relation_tolerance=relation_tolerance,
    )

    # A near-zero derivative is itself a valid semantic result; relative error
    # is ill-conditioned around zero.
    if fine.relation_type is RelationType.NO_LOCAL_SUPPORT and coarse.relation_type is RelationType.NO_LOCAL_SUPPORT:
        return CASJCertificate(
            True, RepairMode.KEEP, fine, 0.0, fit_residual, active_code_condition,
            "both intervention scales indicate no significant support dependence",
        )

    if fine.active_supports != coarse.active_supports or fine.relation_type is not coarse.relation_type:
        return CASJCertificate(
            False, RepairMode.REPLAN, fine, float("inf"), fit_residual, active_code_condition,
            "support set or relation geometry changes across perturbation scales",
        )

    stability = _scale_error(fine_blocks, coarse_blocks)
    failures = [
        (stability > max_scale_instability, "CASJ is not stable across perturbation scales"),
        (fit_residual > max_fit_residual, "coded probes are not explained by the sparse local model"),
        (active_code_condition > max_code_condition, "active support code subproblem is ill-conditioned"),
        (len(fine.active_supports) > max_active_supports, "too many active supports for trusted local repair"),
    ]
    for failed, reason in failures:
        if failed:
            return CASJCertificate(
                False, RepairMode.REPLAN, fine, stability, fit_residual, active_code_condition, reason
            )

    if fine.relation_type is RelationType.RIGID_FOLLOW_FINGERPRINT:
        mode, reason = (
            RepairMode.EXACT_TRANSPORT,
            "single support block matches rigid equivariant geometry",
        )
    else:
        mode, reason = (
            RepairMode.LOCAL_LINEAR_REPAIR,
            "stable sparse non-rigid/relational support geometry",
        )
    return CASJCertificate(
        True, mode, fine, stability, fit_residual, active_code_condition, reason
    )



@dataclass(frozen=True)
class ProbeDesignCertificate:
    mutual_coherence: float
    sparsity: int
    uniqueness_margin: float
    certified: bool


@dataclass(frozen=True)
class CertifiedCASJCertificate:
    accepted: bool
    repair_mode: RepairMode
    relation: RelationGeometry
    probe_design: ProbeDesignCertificate
    base_certificate: CASJCertificate | None
    curvature_effect: float
    curvature_ratio: float
    reason: str


@dataclass(frozen=True)
class CASJChunkPlan:
    decisions: tuple[CertifiedCASJCertificate, ...]
    modes: tuple[RepairMode, ...]
    local_chunk: np.ndarray
    local_correction: np.ndarray
    exact_supports: tuple[int | None, ...]
    requires_replan: bool
    reason: str


@dataclass(frozen=True)
class DirectionalRemainderWitness:
    """Finite-displacement Taylor remainder witness along the actual support motion."""

    accepted: bool
    fine_curvature_norm: float
    curvature_scale_stability: float
    predicted_remainder_norm: float
    remainder_to_first_order: float
    remainder_to_reference_scale: float
    reason: str


def directional_second_derivative(
    plus: np.ndarray,
    minus: np.ndarray,
    baseline: np.ndarray,
    *,
    step: float,
) -> np.ndarray:
    """Central second derivative along one normalized support-motion direction."""
    if step <= 0:
        raise ValueError("step must be positive")
    plus = np.asarray(plus, dtype=float)
    minus = np.asarray(minus, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if plus.shape != minus.shape or plus.shape != baseline.shape:
        raise ValueError("plus, minus, and baseline must have identical shapes")
    return (plus + minus - 2.0 * baseline) / (step * step)


def certify_directional_remainder(
    *,
    fine_second: np.ndarray,
    coarse_second: np.ndarray,
    displacement_norm: float,
    first_order_effect: np.ndarray,
    reference_action_scale: float,
    max_curvature_scale_instability: float = 0.25,
    max_remainder_to_first_order: float = 0.25,
    max_remainder_to_reference_scale: float = 0.05,
) -> DirectionalRemainderWitness:
    """Certify a finite displacement from directional second-order evidence.

    fine_second and coarse_second estimate the second derivative of the decoded
    action along the actual runtime support-motion direction at two probe
    scales. This includes mixed Hessian terms along that direction.

    The predicted second-order remainder is one half times displacement norm
    squared times the directional second-derivative norm.
    """
    fine = np.asarray(fine_second, dtype=float)
    coarse = np.asarray(coarse_second, dtype=float)
    first = np.asarray(first_order_effect, dtype=float)
    if fine.shape != coarse.shape:
        raise ValueError("fine_second and coarse_second must have identical shapes")
    if displacement_norm < 0:
        raise ValueError("displacement_norm must be non-negative")
    if reference_action_scale <= 0:
        raise ValueError("reference_action_scale must be positive")

    fine_norm = float(np.linalg.norm(fine))
    coarse_norm = float(np.linalg.norm(coarse))
    curvature_stability = float(
        np.linalg.norm(fine - coarse)
        / max(fine_norm, coarse_norm, 1e-12)
    )
    predicted = 0.5 * displacement_norm * displacement_norm * fine_norm
    first_norm = float(np.linalg.norm(first))
    ratio_first = predicted / max(first_norm, 1e-12)
    ratio_reference = predicted / reference_action_scale

    failures = [
        (
            curvature_stability > max_curvature_scale_instability,
            "directional curvature is unstable across probe scales",
        ),
        (
            ratio_reference > max_remainder_to_reference_scale,
            "predicted finite-displacement remainder is too large",
        ),
        (
            first_norm > 1e-12 and ratio_first > max_remainder_to_first_order,
            "predicted remainder is too large relative to the first-order effect",
        ),
    ]
    for failed, reason in failures:
        if failed:
            return DirectionalRemainderWitness(
                False,
                fine_norm,
                curvature_stability,
                predicted,
                ratio_first,
                ratio_reference,
                reason,
            )

    return DirectionalRemainderWitness(
        True,
        fine_norm,
        curvature_stability,
        predicted,
        ratio_first,
        ratio_reference,
        "directional Taylor remainder is bounded at the requested support displacement",
    )


def mutual_coherence(codes: np.ndarray) -> float:
    """Maximum absolute inner product between normalized code columns."""
    codes = np.asarray(codes, dtype=float)
    if codes.ndim != 2:
        raise ValueError("codes must have shape [M, K]")
    norms = np.linalg.norm(codes, axis=0, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("every code column must be nonzero")
    normalized = codes / norms
    gram = np.abs(normalized.T @ normalized)
    np.fill_diagonal(gram, 0.0)
    return float(np.max(gram)) if gram.size else 0.0


def certify_probe_design(
    codes: np.ndarray,
    *,
    sparsity: int,
) -> ProbeDesignCertificate:
    """Conservative coherence certificate for sparse support identification."""
    if sparsity <= 0:
        raise ValueError("sparsity must be positive")
    mu = mutual_coherence(codes)
    threshold = 1.0 / (2 * sparsity - 1)
    margin = threshold - mu
    return ProbeDesignCertificate(
        mutual_coherence=mu,
        sparsity=sparsity,
        uniqueness_margin=margin,
        certified=bool(margin > 1e-12),
    )


def _sylvester_hadamard(order: int) -> np.ndarray:
    if order <= 0 or order & (order - 1):
        raise ValueError("order must be a positive power of two")
    h = np.array([[1.0]])
    while h.shape[0] < order:
        h = np.block([[h, h], [h, -h]])
    return h


def certificate_codes(num_supports: int, *, sparsity: int) -> np.ndarray:
    """Small deterministic partial-Hadamard design passing the coherence gate."""
    if num_supports <= 0:
        raise ValueError("num_supports must be positive")
    order = 1 << math.ceil(math.log2(num_supports))
    full = _sylvester_hadamard(order)[:, :num_supports]

    for rows_to_keep in range(1, order + 1):
        for start in range(order):
            rows = [(start + offset) % order for offset in range(rows_to_keep)]
            candidate = full[rows]
            if certify_probe_design(candidate, sparsity=sparsity).certified:
                return candidate
    return full


def central_coded_response(
    plus: np.ndarray,
    minus: np.ndarray,
    baseline: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Central first derivative plus a symmetric second-order witness."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    plus = np.asarray(plus, dtype=float)
    minus = np.asarray(minus, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if plus.shape != minus.shape or plus.ndim != 3:
        raise ValueError("plus and minus must have shape [M, H, D_action]")
    if baseline.shape != plus.shape[1:]:
        raise ValueError("baseline must have shape [H, D_action]")
    first = (plus - minus) / (2.0 * epsilon)
    curvature = (plus + minus - 2.0 * baseline[None]) / (epsilon * epsilon)
    return first, curvature


def certify_runtime(
    *,
    fine_blocks: np.ndarray,
    coarse_blocks: np.ndarray,
    codes: np.ndarray,
    sparsity: int,
    fit_residual: float,
    active_code_condition: float,
    curvature_effect: float,
    curvature_ratio: float,
    active_threshold: float = 0.10,
    rigid_tolerance: float = 0.20,
    relation_tolerance: float = 0.20,
    max_scale_instability: float = 0.15,
    max_fit_residual: float = 0.15,
    max_code_condition: float = 10.0,
    max_active_supports: int = 3,
    max_curvature_effect: float = 0.05,
    max_curvature_ratio: float = 0.25,
) -> CertifiedCASJCertificate:
    """Strict certificate used by the paper-core runtime.

    A near-zero first derivative does not by itself imply KEEP. The finite
    second-order effect must also be small; otherwise a locally quadratic
    response could be silently misclassified as support-independent.

    curvature_effect is expected to be dimensionless: normalize the estimated
    second-order action displacement by a task-relevant action/support scale
    before calling this function. curvature_ratio compares that same second-
    order displacement with the predicted first-order action displacement.
    """
    if curvature_effect < 0 or curvature_ratio < 0:
        raise ValueError("curvature witnesses must be non-negative")

    design = certify_probe_design(codes, sparsity=sparsity)
    fine_relation = classify_relation(
        fine_blocks,
        active_threshold=active_threshold,
        rigid_tolerance=rigid_tolerance,
        relation_tolerance=relation_tolerance,
    )
    coarse_relation = classify_relation(
        coarse_blocks,
        active_threshold=active_threshold,
        rigid_tolerance=rigid_tolerance,
        relation_tolerance=relation_tolerance,
    )

    if not design.certified:
        return CertifiedCASJCertificate(
            False,
            RepairMode.REPLAN,
            fine_relation,
            design,
            None,
            float(curvature_effect),
            float(curvature_ratio),
            "coded intervention design is not certified for the claimed sparsity",
        )

    if (
        fine_relation.relation_type is RelationType.NO_LOCAL_SUPPORT
        and coarse_relation.relation_type is RelationType.NO_LOCAL_SUPPORT
    ):
        if curvature_effect > max_curvature_effect:
            return CertifiedCASJCertificate(
                False,
                RepairMode.REPLAN,
                fine_relation,
                design,
                None,
                float(curvature_effect),
                float(curvature_ratio),
                "first-order response is near zero but second-order effect is too large",
            )
        return CertifiedCASJCertificate(
            True,
            RepairMode.KEEP,
            fine_relation,
            design,
            None,
            float(curvature_effect),
            float(curvature_ratio),
            "support response is negligible through second order",
        )

    base = certify_repair(
        fine_blocks=fine_blocks,
        coarse_blocks=coarse_blocks,
        fit_residual=fit_residual,
        active_code_condition=active_code_condition,
        active_threshold=active_threshold,
        rigid_tolerance=rigid_tolerance,
        relation_tolerance=relation_tolerance,
        max_scale_instability=max_scale_instability,
        max_fit_residual=max_fit_residual,
        max_code_condition=max_code_condition,
        max_active_supports=max_active_supports,
    )
    if not base.accepted:
        return CertifiedCASJCertificate(
            False,
            RepairMode.REPLAN,
            base.relation,
            design,
            base,
            float(curvature_effect),
            float(curvature_ratio),
            base.reason,
        )

    if curvature_effect > max_curvature_effect or curvature_ratio > max_curvature_ratio:
        return CertifiedCASJCertificate(
            False,
            RepairMode.REPLAN,
            base.relation,
            design,
            base,
            float(curvature_effect),
            float(curvature_ratio),
            "local response is too nonlinear for certified repair",
        )

    return CertifiedCASJCertificate(
        True,
        base.repair_mode,
        base.relation,
        design,
        base,
        float(curvature_effect),
        float(curvature_ratio),
        base.reason,
    )


def _step_values(value: float | np.ndarray, horizon: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(horizon, float(array), dtype=float)
    if array.shape != (horizon,):
        raise ValueError(f"{name} must be scalar or have shape {(horizon,)}")
    return array


def plan_chunk(
    baseline_chunk: np.ndarray,
    *,
    fine_jacobian: np.ndarray,
    coarse_jacobian: np.ndarray,
    support_delta: np.ndarray,
    codes: np.ndarray,
    sparsity: int,
    fit_residual: float | np.ndarray,
    active_code_condition: float | np.ndarray,
    curvature_effect: float | np.ndarray,
    curvature_ratio: float | np.ndarray,
    executed_prefix: int = 0,
    **certificate_kwargs,
) -> CASJChunkPlan:
    """Certify the whole suffix before exposing any repair.

    local_correction stores the tangent-space correction licensed at each
    action position. local_chunk remains the Euclidean convenience
    materialization baseline + local_correction for backwards compatibility;
    manifold-aware runtimes must use local_correction with ActionChart.retract.

    Exact rigid positions have zero local correction and expose the unique
    support id in exact_supports for the representation-correct finite
    transport layer.
    """
    baseline = np.asarray(baseline_chunk, dtype=float)
    fine = np.asarray(fine_jacobian, dtype=float)
    coarse = np.asarray(coarse_jacobian, dtype=float)
    delta = np.asarray(support_delta, dtype=float)

    if baseline.ndim != 2:
        raise ValueError("baseline_chunk must have shape [H, D]")
    horizon, dim = baseline.shape
    if (
        fine.shape != coarse.shape
        or fine.ndim != 4
        or fine.shape[0] != horizon
        or fine.shape[2:] != (dim, dim)
    ):
        raise ValueError("CASJ estimates must have shape [H, K, D, D]")
    if delta.shape != (fine.shape[1], dim):
        raise ValueError(f"support_delta must have shape {(fine.shape[1], dim)}")
    if not 0 <= executed_prefix <= horizon:
        raise ValueError("executed_prefix must lie inside [0, H]")

    residual = _step_values(fit_residual, horizon, "fit_residual")
    condition = _step_values(active_code_condition, horizon, "active_code_condition")
    curvature_e = _step_values(curvature_effect, horizon, "curvature_effect")
    curvature_r = _step_values(curvature_ratio, horizon, "curvature_ratio")

    decisions = tuple(
        certify_runtime(
            fine_blocks=fine[t],
            coarse_blocks=coarse[t],
            codes=codes,
            sparsity=sparsity,
            fit_residual=float(residual[t]),
            active_code_condition=float(condition[t]),
            curvature_effect=float(curvature_e[t]),
            curvature_ratio=float(curvature_r[t]),
            **certificate_kwargs,
        )
        for t in range(horizon)
    )
    modes = tuple(
        RepairMode.KEEP if t < executed_prefix else decisions[t].repair_mode
        for t in range(horizon)
    )

    failed = next(
        (t for t in range(executed_prefix, horizon) if modes[t] is RepairMode.REPLAN),
        None,
    )
    if failed is not None:
        return CASJChunkPlan(
            decisions,
            modes,
            baseline.copy(),
            np.zeros_like(baseline),
            tuple([None] * horizon),
            True,
            f"action position {failed} is uncertified: {decisions[failed].reason}",
        )

    local = baseline.copy()
    local_correction = np.zeros_like(baseline)
    exact_supports: list[int | None] = [None] * horizon
    for t in range(executed_prefix, horizon):
        if modes[t] is RepairMode.LOCAL_LINEAR_REPAIR:
            local_correction[t] = np.einsum("kdo,ko->d", fine[t], delta)
            local[t] = baseline[t] + local_correction[t]
        elif modes[t] is RepairMode.EXACT_TRANSPORT:
            active = decisions[t].relation.active_supports
            if len(active) != 1:
                fallback = list(modes)
                fallback[t] = RepairMode.REPLAN
                return CASJChunkPlan(
                    decisions,
                    tuple(fallback),
                    baseline.copy(),
                    np.zeros_like(baseline),
                    tuple([None] * horizon),
                    True,
                    f"exact transport at position {t} has no unique support",
                )
            exact_supports[t] = active[0]

    return CASJChunkPlan(
        decisions,
        modes,
        local,
        local_correction,
        tuple(exact_supports),
        False,
        "entire unexecuted suffix passed the CASJ runtime certificate",
    )



def plan_tangent_chunk(
    *,
    fine_jacobian: np.ndarray,
    coarse_jacobian: np.ndarray,
    support_delta: np.ndarray,
    codes: np.ndarray,
    sparsity: int,
    fit_residual: float | np.ndarray,
    active_code_condition: float | np.ndarray,
    curvature_effect: float | np.ndarray,
    curvature_ratio: float | np.ndarray,
    executed_prefix: int = 0,
    **certificate_kwargs,
) -> CASJChunkPlan:
    """Plan CASJ repairs entirely in the decoded action tangent space.

    This is the manifold-safe entry point. It returns local_correction vectors
    but does not assume that the decoded action itself can be added to them.
    ActionChart.retract and ActionChart.encode perform the later commit.

    The current generic relation classifier requires square action/support
    local coordinates. Non-square fingerprints such as a 3D point action under
    a 6-DoF support intervention use their dedicated geometry certificate
    before selecting a finite transport.
    """
    fine = np.asarray(fine_jacobian, dtype=float)
    coarse = np.asarray(coarse_jacobian, dtype=float)
    if fine.shape != coarse.shape or fine.ndim != 4:
        raise ValueError(
            "fine/coarse CASJ must share shape [H, K, D_action, D_support]"
        )
    horizon, _, action_dim, support_dim = fine.shape
    if action_dim != support_dim:
        raise ValueError(
            "generic tangent planning currently requires D_action == D_support; "
            "use a dedicated non-square geometry fingerprint otherwise"
        )

    zero_chart_origin = np.zeros((horizon, action_dim), dtype=float)
    return plan_chunk(
        zero_chart_origin,
        fine_jacobian=fine,
        coarse_jacobian=coarse,
        support_delta=support_delta,
        codes=codes,
        sparsity=sparsity,
        fit_residual=fit_residual,
        active_code_condition=active_code_condition,
        curvature_effect=curvature_effect,
        curvature_ratio=curvature_ratio,
        executed_prefix=executed_prefix,
        **certificate_kwargs,
    )