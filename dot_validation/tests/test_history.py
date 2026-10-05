import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "prepare_dot_natural_history.py"
SPEC = importlib.util.spec_from_file_location("prepare_dot_natural_history", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_select_history_uses_dot_inference_offsets_not_consecutive_frames():
    rows = []
    for frame in range(21):
        rows.append(
            {
                "episode_index": 0,
                "frame_index": frame,
                "observation.environment_state": [float(frame)] * 16,
                "observation.state": [float(frame), -float(frame)],
            }
        )

    env, agent = MODULE.select_history(
        rows,
        episode_index=0,
        target_frame=20,
        offsets=(-10, -1, 0),
    )

    np.testing.assert_array_equal(env[:, 0], [10.0, 19.0, 20.0])
    np.testing.assert_array_equal(agent[:, 0], [10.0, 19.0, 20.0])


def test_history_hash_changes_with_current_support_state():
    env = np.zeros((3, 16), dtype=np.float32)
    agent = np.zeros((3, 2), dtype=np.float32)
    before = MODULE.array_sha256(env, agent)
    env[-1, 0] = 1.0
    after = MODULE.array_sha256(env, agent)
    assert before != after
