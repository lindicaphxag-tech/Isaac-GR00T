import numpy as np

from gr00t.data.state_action.state_action_processor import StateActionProcessor


EMBODIMENT = "cst_test"


def _stats(low, high, dim=2):
    low = [float(low)] * dim
    high = [float(high)] * dim
    mean = [0.0] * dim
    std = [1.0] * dim
    return {
        "min": low,
        "max": high,
        "mean": mean,
        "std": std,
        "q01": low,
        "q99": high,
    }


def _processor():
    modality_configs = {
        EMBODIMENT: {
            "state": {
                "delta_indices": [0],
                "modality_keys": ["joint"],
                "sin_cos_embedding_keys": None,
                "mean_std_embedding_keys": None,
                "action_configs": None,
            },
            "action": {
                "delta_indices": [0, 1, 2],
                "modality_keys": ["joint"],
                "sin_cos_embedding_keys": None,
                "mean_std_embedding_keys": None,
                "action_configs": [
                    {
                        "rep": "RELATIVE",
                        "type": "NON_EEF",
                        "format": "DEFAULT",
                        "state_key": "joint",
                    }
                ],
            },
        }
    }
    statistics = {
        EMBODIMENT: {
            "state": {"joint": _stats(-1.0, 1.0)},
            # Absolute stats are deliberately different. When relative action
            # processing is enabled, GR00T must normalize in the relative chart.
            "action": {"joint": _stats(-2.0, 2.0)},
            "relative_action": {"joint": _stats(-0.25, 0.25)},
        }
    }
    return StateActionProcessor(
        modality_configs=modality_configs,
        statistics=statistics,
        use_percentiles=False,
        clip_outliers=True,
        use_relative_action=True,
    )


def test_relative_processor_random_exact_roundtrip_is_semantically_stable():
    proc = _processor()
    rng = np.random.default_rng(20261006)

    for _ in range(500):
        reference = rng.uniform(-0.5, 0.5, size=2)
        relative = rng.uniform(-0.24, 0.24, size=(3, 2))
        absolute = reference[None, :] + relative
        state = {"joint": reference[None, :]}

        normalized = proc.apply_action(
            {"joint": absolute.copy()},
            EMBODIMENT,
            state=state,
        )
        recovered = proc.unapply_action(
            normalized,
            EMBODIMENT,
            state=state,
        )
        np.testing.assert_allclose(
            recovered["joint"],
            absolute,
            atol=1e-12,
            rtol=0.0,
        )


def test_batched_unapply_preserves_each_batch_reference_state():
    proc = _processor()
    references = np.array([[0.2, -0.1], [-0.3, 0.4]], dtype=float)
    relatives = np.array(
        [
            [[0.10, -0.05], [-0.20, 0.15], [0.00, 0.20]],
            [[-0.10, 0.10], [0.05, -0.20], [0.20, 0.00]],
        ],
        dtype=float,
    )
    absolutes = references[:, None, :] + relatives

    normalized_batches = []
    for batch in range(2):
        normalized = proc.apply_action(
            {"joint": absolutes[batch]},
            EMBODIMENT,
            state={"joint": references[batch][None, :]},
        )
        normalized_batches.append(normalized["joint"])

    recovered = proc.unapply_action(
        {"joint": np.stack(normalized_batches, axis=0)},
        EMBODIMENT,
        state={"joint": references[:, None, :]},
    )
    np.testing.assert_allclose(
        recovered["joint"],
        absolutes,
        atol=1e-12,
        rtol=0.0,
    )


def test_clipped_relative_action_is_intentionally_lossy_not_false_roundtrip():
    proc = _processor()
    reference = np.array([0.2, -0.1], dtype=float)
    # First coordinate requests +0.50 although relative stats authorize only
    # [-0.25, +0.25]. clip_outliers=True must saturate it.
    absolute = np.array(
        [
            [0.70, -0.10],
            [0.20, -0.10],
            [0.20, -0.10],
        ],
        dtype=float,
    )
    state = {"joint": reference[None, :]}

    normalized = proc.apply_action(
        {"joint": absolute.copy()},
        EMBODIMENT,
        state=state,
    )
    recovered = proc.unapply_action(
        normalized,
        EMBODIMENT,
        state=state,
    )

    assert normalized["joint"][0, 0] == 1.0
    np.testing.assert_allclose(recovered["joint"][0, 0], 0.45, atol=1e-12)
    assert not np.allclose(recovered["joint"], absolute)


def test_relative_stats_not_absolute_stats_define_representability():
    proc = _processor()
    # A +0.24 relative command is close to the relative chart boundary. If the
    # processor accidentally used the much wider absolute stats [-2,2], its
    # normalized magnitude would be only 0.12 instead of 0.96.
    reference = np.array([0.0, 0.0], dtype=float)
    absolute = np.array([[0.24, -0.24]] * 3, dtype=float)
    normalized = proc.apply_action(
        {"joint": absolute},
        EMBODIMENT,
        state={"joint": reference[None, :]},
    )
    np.testing.assert_allclose(
        normalized["joint"],
        np.array([[0.96, -0.96]] * 3),
        atol=1e-12,
        rtol=0.0,
    )
