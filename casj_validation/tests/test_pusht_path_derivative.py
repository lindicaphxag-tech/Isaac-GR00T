import numpy as np

from casj.pusht_assay import estimate_pathwise_directional_derivative_via_query
from casj.pusht_state import PushTSnapshot


def _snapshot(block_x: float = 0.0) -> PushTSnapshot:
    return PushTSnapshot(
        agent_position=np.zeros(2),
        agent_velocity=np.zeros(2),
        block_position=np.array([block_x, 0.0], dtype=float),
        block_angle=0.0,
        block_velocity=np.zeros(2),
        block_angular_velocity=0.0,
    )


class _LinearPathPolicy:
    def query(self, history, *, randomness):
        del randomness
        # The test support path is x in [0, 10].  Any request outside it means
        # the finite-path assay has leaked beyond the certified intervention.
        x = float(history[-1].block_position[0])
        assert 0.0 <= x <= 10.0
        return np.array([[x / 10.0]], dtype=float)


def test_path_derivative_uses_forward_stencil_at_start_boundary():
    result = estimate_pathwise_directional_derivative_via_query(
        policy_query=_LinearPathPolicy(),
        history=[_snapshot()],
        randomness=0,
        support_delta=np.array([10.0, 0.0, 0.0]),
        path_fraction=0.0,
        fine_step_fraction=0.1,
        coarse_step_fraction=0.2,
    )
    np.testing.assert_allclose(result.fine, [[1.0]], atol=1e-12)
    np.testing.assert_allclose(result.coarse, [[1.0]], atol=1e-12)


def test_path_derivative_uses_backward_stencil_at_end_boundary():
    result = estimate_pathwise_directional_derivative_via_query(
        policy_query=_LinearPathPolicy(),
        history=[_snapshot()],
        randomness=0,
        support_delta=np.array([10.0, 0.0, 0.0]),
        path_fraction=1.0,
        fine_step_fraction=0.1,
        coarse_step_fraction=0.2,
    )
    np.testing.assert_allclose(result.fine, [[1.0]], atol=1e-12)
    np.testing.assert_allclose(result.coarse, [[1.0]], atol=1e-12)


def test_path_derivative_keeps_central_stencil_in_interior():
    result = estimate_pathwise_directional_derivative_via_query(
        policy_query=_LinearPathPolicy(),
        history=[_snapshot()],
        randomness=0,
        support_delta=np.array([10.0, 0.0, 0.0]),
        path_fraction=0.5,
        fine_step_fraction=0.1,
        coarse_step_fraction=0.2,
    )
    np.testing.assert_allclose(result.fine, [[1.0]], atol=1e-12)
    np.testing.assert_allclose(result.coarse, [[1.0]], atol=1e-12)
