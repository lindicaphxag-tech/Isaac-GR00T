from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import EmbodimentTag, ModalityConfig


def _episode(length: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state.joint_position": [np.array([float(i)]) for i in range(length)],
            "language.annotation.human.action.task_description": [f"step {i}" for i in range(length)],
        }
    )


def _configs(delta_indices: list[int]) -> dict[str, ModalityConfig]:
    return {
        "state": ModalityConfig(delta_indices=delta_indices, modality_keys=["joint_position"]),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    }


@pytest.mark.parametrize(
    ("step_index", "expected_history"),
    [
        (0, 0.0),
        (1, 0.0),
        (2, 0.0),
        (3, 1.0),
    ],
)
def test_negative_history_indices_pad_at_episode_start(step_index: int, expected_history: float) -> None:
    data = extract_step_data(
        _episode(),
        step_index,
        _configs([-2, 0]),
        EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT,
        allow_padding=False,
    )

    np.testing.assert_array_equal(data.states["joint_position"][0], [expected_history])


def test_positive_out_of_range_indices_raise_instead_of_wrapping() -> None:
    with pytest.raises(IndexError, match="outside the episode"):
        extract_step_data(
            _episode(),
            3,
            _configs([0, 1]),
            EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT,
            allow_padding=False,
        )


def test_allow_padding_clamps_both_sides() -> None:
    data = extract_step_data(
        _episode(),
        0,
        _configs([-3, 10]),
        EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT,
        allow_padding=True,
    )

    np.testing.assert_array_equal(data.states["joint_position"], [[0.0], [3.0]])
