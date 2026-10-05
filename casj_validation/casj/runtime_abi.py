from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .core import CASJEstimate, recover_casj


class PairingContractError(RuntimeError):
    """Raised when a policy cannot replay the same stochastic query."""


@runtime_checkable
class FrozenChunkPolicy(Protocol):
    """Framework-neutral frozen action-chunk policy interface.

    The randomness token is a replay token, not a mutable RNG handle. Reusing
    the same token must reproduce the same complete sampling trajectory.
    """

    def query(self, observation: Any, *, randomness: Any) -> np.ndarray:
        """Return one raw action chunk for the observation."""


@runtime_checkable
class ActionChart(Protocol):
    """Decode native actions and measure local action-manifold responses.

    CASJ never assumes that decoded actions live in a Euclidean vector space.
    A chart may decode to [H, D], [H, 4, 4], joint configurations, or another
    host-owned physical representation. local_response must map a changed
    decoded chunk into a common tangent vector at the baseline, shape [H, D_a].
    """

    def decode(self, raw_chunk: np.ndarray, *, observation: Any) -> Any:
        ...

    def local_response(self, changed: Any, baseline: Any) -> np.ndarray:
        """Return local coordinates of changed relative to baseline."""

    def pairing_error(self, first: Any, second: Any) -> float:
        """Return a chart-appropriate replay error for identical queries."""


@runtime_checkable
class SupportInterventionChart(Protocol):
    """Compile coded support interventions into a policy observation."""

    @property
    def num_supports(self) -> int:
        ...

    @property
    def support_dim(self) -> int:
        ...

    def intervene(
        self,
        observation: Any,
        *,
        direction: int,
        coefficients: np.ndarray,
        magnitude: float,
    ) -> Any:
        """Apply a simultaneous signed perturbation to candidate supports.

        coefficients has shape [K]. For a support basis direction q, support j
        receives magnitude * coefficients[j] in that physical intervention
        coordinate.
        """


@runtime_checkable
class FiniteTransportOperator(Protocol):
    """Materialize exact finite geometry in decoded task space."""

    def transport(
        self,
        decoded_action: Any,
        *,
        action_index: int,
        support_index: int,
    ) -> Any:
        """Return one finite transported decoded task-space action."""


@dataclass(frozen=True)
class CASJProbeBatch:
    """Paired two-scale interventional responses.

    first-derivative tensors have shape [Q, M, H, D_action].
    second-derivative tensors use the same tangent coordinates and retain coded
    probe rows so downstream logic can inspect directional curvature.
    """

    baseline: Any
    fine_first: np.ndarray
    coarse_first: np.ndarray
    fine_second: np.ndarray
    coarse_second: np.ndarray
    codes: np.ndarray
    fine_epsilon: np.ndarray
    coarse_epsilon: np.ndarray
    pairing_error: float

    @property
    def query_count(self) -> int:
        # baseline + repeatability replay + +/- at two scales
        q = self.fine_first.shape[0]
        m = self.fine_first.shape[1]
        return 2 + 4 * q * m


@dataclass(frozen=True)
class TwoScaleCASJ:
    fine: CASJEstimate
    coarse: CASJEstimate
    probes: CASJProbeBatch


def _decode(
    policy: FrozenChunkPolicy,
    action_chart: ActionChart,
    observation: Any,
    *,
    randomness: Any,
) -> Any:
    raw = policy.query(observation, randomness=randomness)
    return action_chart.decode(np.asarray(raw), observation=observation)


def _response(
    action_chart: ActionChart,
    changed: Any,
    baseline: Any,
) -> np.ndarray:
    response = np.asarray(
        action_chart.local_response(changed, baseline),
        dtype=float,
    )
    if response.ndim != 2:
        raise ValueError(
            "action_chart.local_response must return shape [H, D_action], "
            f"got {response.shape}"
        )
    return response


def paired_repeatability_error(
    policy: FrozenChunkPolicy,
    action_chart: ActionChart,
    observation: Any,
    *,
    randomness: Any,
) -> tuple[Any, float]:
    """Replay one stochastic query and measure the pairing-contract error.

    This catches adapters that reuse a mutable generator object whose internal
    state advances between calls. A valid paired counterfactual adapter should
    reconstruct the full sampler randomness from an immutable/replayable token.
    """
    first = _decode(
        policy,
        action_chart,
        observation,
        randomness=randomness,
    )
    second = _decode(
        policy,
        action_chart,
        observation,
        randomness=randomness,
    )
    error = float(action_chart.pairing_error(first, second))
    if not np.isfinite(error) or error < 0:
        raise PairingContractError(
            "action_chart returned an invalid paired replay error"
        )
    return first, error


def _collect_scale(
    *,
    policy: FrozenChunkPolicy,
    action_chart: ActionChart,
    support_chart: SupportInterventionChart,
    observation: Any,
    randomness: Any,
    baseline: Any,
    codes: np.ndarray,
    epsilon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_dim = int(support_chart.support_dim)
    epsilon = np.asarray(epsilon, dtype=float)
    if epsilon.shape != (q_dim,) or np.any(epsilon <= 0):
        raise ValueError(
            f"probe epsilon must be positive with shape {(q_dim,)}"
        )

    first_by_direction: list[np.ndarray] = []
    second_by_direction: list[np.ndarray] = []

    for q in range(q_dim):
        step = float(epsilon[q])
        first_by_probe: list[np.ndarray] = []
        second_by_probe: list[np.ndarray] = []

        for code in codes:
            plus_observation = support_chart.intervene(
                observation,
                direction=q,
                coefficients=code,
                magnitude=step,
            )
            minus_observation = support_chart.intervene(
                observation,
                direction=q,
                coefficients=code,
                magnitude=-step,
            )
            plus = _decode(
                policy,
                action_chart,
                plus_observation,
                randomness=randomness,
            )
            minus = _decode(
                policy,
                action_chart,
                minus_observation,
                randomness=randomness,
            )

            plus_response = _response(action_chart, plus, baseline)
            minus_response = _response(action_chart, minus, baseline)
            if plus_response.shape != minus_response.shape:
                raise ValueError(
                    "positive/negative probes returned different local-response shapes"
                )

            first_by_probe.append(
                (plus_response - minus_response) / (2.0 * step)
            )
            # Both local responses already live in the baseline tangent chart,
            # whose baseline response is exactly zero.
            second_by_probe.append(
                (plus_response + minus_response) / (step * step)
            )

        first_by_direction.append(np.stack(first_by_probe, axis=0))
        second_by_direction.append(np.stack(second_by_probe, axis=0))

    first = np.stack(first_by_direction, axis=0)
    second = np.stack(second_by_direction, axis=0)
    return first, second


def collect_two_scale_probes(
    *,
    policy: FrozenChunkPolicy,
    action_chart: ActionChart,
    support_chart: SupportInterventionChart,
    observation: Any,
    randomness: Any,
    codes: np.ndarray,
    fine_epsilon: float | np.ndarray,
    coarse_epsilon: float | np.ndarray,
    max_pairing_error: float = 1e-6,
) -> CASJProbeBatch:
    """Compile and execute paired central-difference CASJ probes.

    The exact same replayable randomness token is supplied to the baseline and
    every +/- counterfactual. Action subtraction is delegated to ActionChart,
    so Euclidean, SE(3), and other manifolds use the correct local response.
    """
    codes = np.asarray(codes, dtype=float)
    if codes.ndim != 2 or codes.shape[0] == 0:
        raise ValueError("codes must have non-empty shape [M, K]")
    if codes.shape[1] != int(support_chart.num_supports):
        raise ValueError(
            "code columns must match support_chart.num_supports"
        )

    q_dim = int(support_chart.support_dim)
    fine_steps = np.broadcast_to(
        np.asarray(fine_epsilon, dtype=float), (q_dim,)
    ).copy()
    coarse_steps = np.broadcast_to(
        np.asarray(coarse_epsilon, dtype=float), (q_dim,)
    ).copy()
    if np.any(fine_steps <= 0) or np.any(coarse_steps <= fine_steps):
        raise ValueError(
            "require positive per-direction fine epsilon and coarse > fine"
        )

    baseline, pairing_error = paired_repeatability_error(
        policy,
        action_chart,
        observation,
        randomness=randomness,
    )
    if pairing_error > max_pairing_error:
        raise PairingContractError(
            "same observation/randomness token is not replayable: "
            f"chart replay error={pairing_error:.6g}, "
            f"limit={max_pairing_error:.6g}"
        )

    fine_first, fine_second = _collect_scale(
        policy=policy,
        action_chart=action_chart,
        support_chart=support_chart,
        observation=observation,
        randomness=randomness,
        baseline=baseline,
        codes=codes,
        epsilon=fine_steps,
    )
    coarse_first, coarse_second = _collect_scale(
        policy=policy,
        action_chart=action_chart,
        support_chart=support_chart,
        observation=observation,
        randomness=randomness,
        baseline=baseline,
        codes=codes,
        epsilon=coarse_steps,
    )

    return CASJProbeBatch(
        baseline=baseline,
        fine_first=fine_first,
        coarse_first=coarse_first,
        fine_second=fine_second,
        coarse_second=coarse_second,
        codes=codes,
        fine_epsilon=fine_steps,
        coarse_epsilon=coarse_steps,
        pairing_error=pairing_error,
    )


def recover_two_scale_casj(
    probes: CASJProbeBatch,
    *,
    sparsity: int,
) -> TwoScaleCASJ:
    """Recover fine/coarse CASJ estimates from one paired probe batch."""
    fine = recover_casj(
        probes.fine_first,
        codes=probes.codes,
        sparsity=sparsity,
    )
    coarse = recover_casj(
        probes.coarse_first,
        codes=probes.codes,
        sparsity=sparsity,
    )
    return TwoScaleCASJ(fine=fine, coarse=coarse, probes=probes)


def materialize_certified_chunk(
    decoded_baseline: Any,
    plan: Any,
    *,
    action_chart: ActionChart,
    exact_transport: FiniteTransportOperator,
    observation: Any,
) -> np.ndarray:
    """Commit one certified plan on the action manifold, then re-encode it.

    The order is deliberate:

    1. apply local-linear CASJ corrections by ActionChart.retract in decoded
       task space;
    2. apply exact finite support transports in decoded task space;
    3. re-encode the **whole repaired chunk** into the native policy/controller
       action representation.

    Whole-chunk re-encoding is required for body/spatial deltas and joint
    trajectories because their native actions depend on sequence context,
    runtime anchor state, or IK continuity.
    """
    if getattr(plan, "requires_replan", True):
        raise PairingContractError(
            "cannot materialize a CASJ plan that requires replan"
        )

    exact_supports = tuple(plan.exact_supports)
    modes = tuple(plan.modes)
    correction = np.asarray(plan.local_correction, dtype=float)
    horizon = len(exact_supports)
    if len(modes) != horizon or correction.shape[0] != horizon:
        raise ValueError("CASJ plan fields disagree on action horizon")

    # Retract every tangent correction at once. KEEP and EXACT_TRANSPORT
    # positions carry zero tangent correction.
    decoded = action_chart.retract(decoded_baseline, correction)
    if len(decoded) != horizon:
        raise ValueError("decoded baseline/plan horizons do not match")

    # Exact transports are licensed position-wise by the certificate, but they
    # operate on decoded physical actions. Native re-encoding is deferred until
    # the entire decoded chunk is complete.
    decoded_list = [np.asarray(item, dtype=float).copy() for item in decoded]
    for t, support in enumerate(exact_supports):
        if support is None:
            continue
        decoded_list[t] = np.asarray(
            exact_transport.transport(
                decoded_list[t],
                action_index=t,
                support_index=int(support),
            ),
            dtype=float,
        )

    decoded_repaired = np.stack(decoded_list, axis=0)
    return np.asarray(
        action_chart.encode(decoded_repaired, observation=observation),
        dtype=float,
    )