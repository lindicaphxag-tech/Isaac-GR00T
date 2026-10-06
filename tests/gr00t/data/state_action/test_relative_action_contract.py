from gr00t.data.state_action.state_action_processor import StateActionProcessor
import numpy as np


EMBODIMENT = "relative_contract_test"


def _stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "min": values.tolist(),
        "max": values.tolist(),
        "mean": np.zeros_like(values).tolist(),
        "std": np.ones_like(values).tolist(),
        "q01": values.tolist(),
        "q99": values.tolist(),
    }


def _processor():
    modality_configs = {
        EMBODIMENT: {
            "state": {
                "delta_indices": [0],
                "modality_keys": ["arm"],
                "sin_cos_embedding_keys": None,
                "mean_std_embedding_keys": None,
                "action_configs": None,
            },
            "action": {
                "delta_indices": [0, 1],
                "modality_keys": ["arm"],
                "sin_cos_embedding_keys": None,
                "mean_std_embedding_keys": None,
                "action_configs": [
                    {
                        "rep": "RELATIVE",
                        "type": "NON_EEF",
                        "format": "DEFAULT",
                        "state_key": "arm",
                    }
                ],
            },
        }
    }
    statistics = {
        EMBODIMENT: {
            "state": {
                "arm": {
                    "min": [-10.0, -10.0],
                    "max": [10.0, 10.0],
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                    "q01": [-10.0, -10.0],
                    "q99": [10.0, 10.0],
                }
            },
            "action": {
                "arm": {
                    "min": [-10.0, -10.0],
                    "max": [10.0, 10.0],
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                    "q01": [-10.0, -10.0],
                    "q99": [10.0, 10.0],
                }
            },
            "relative_action": {
                "arm": {
                    "min": [-2.0, -2.0],
                    "max": [2.0, 2.0],
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                    "q01": [-2.0, -2.0],
                    "q99": [2.0, 2.0],
                }
            },
        }
    }
    return StateActionProcessor(
        modality_configs=modality_configs,
        statistics=statistics,
        use_percentiles=False,
        clip_outliers=False,
        use_relative_action=True,
    )


def test_relative_training_representation_roundtrips_to_absolute_inference():
    proc = _processor()
    state = {"arm": np.array([[0.2, -0.1], [0.5, 0.4]], dtype=np.float64)}
    absolute = {
        "arm": np.array(
            [[0.7, 0.1], [0.9, 0.5]],
            dtype=np.float64,
        )
    }

    processed = proc.apply_action(absolute, EMBODIMENT, state=state)
    restored = proc.unapply_action(processed, EMBODIMENT, state=state)

    np.testing.assert_allclose(restored["arm"], absolute["arm"], atol=1e-10)


def test_same_model_space_relative_chunk_maps_to_different_absolute_actions_if_anchor_changes():
    proc = _processor()

    # This is the model-space action after denormalization: the same relative
    # command is interpreted against the execution-time reference state.
    relative_raw = np.array([[0.2, -0.3], [0.4, 0.1]], dtype=np.float64)
    params = proc.norm_params[EMBODIMENT]["action"]["arm"]
    processed = {
        "arm": 2.0 * (relative_raw - params["min"]) / (params["max"] - params["min"]) - 1.0
    }

    state_a = {"arm": np.array([[0.0, 0.0], [0.5, 0.4]], dtype=np.float64)}
    state_b = {"arm": np.array([[0.0, 0.0], [-0.2, 0.8]], dtype=np.float64)}

    absolute_a = proc.unapply_action(processed, EMBODIMENT, state=state_a)["arm"]
    absolute_b = proc.unapply_action(processed, EMBODIMENT, state=state_b)["arm"]

    np.testing.assert_allclose(
        absolute_b - absolute_a,
        np.broadcast_to(state_b["arm"][-1] - state_a["arm"][-1], relative_raw.shape),
        atol=1e-10,
    )
    assert not np.allclose(absolute_a, absolute_b)


def test_relative_mode_is_two_part_contract_not_output_format_switch():
    proc = _processor()
    assert proc.use_relative_action is True

    # Per-group RELATIVE decides the learned representation; unapply_action()
    # deliberately returns controller-facing absolute values.
    cfg = proc.modality_configs[EMBODIMENT]["action"].action_configs[0]
    assert cfg.rep.value.lower() == "relative"
