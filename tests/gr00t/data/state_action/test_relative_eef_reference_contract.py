# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-path reference-frame contract tests for relative EEF actions."""

from gr00t.data.state_action.action_chunking import EndEffectorActionChunk
from gr00t.data.state_action.pose import EndEffectorPose
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
import numpy as np


EMBODIMENT = "relative_eef_contract"
KEY = "eef"


def _processor() -> StateActionProcessor:
    modality_configs = {
        EMBODIMENT: {
            "state": ModalityConfig(
                delta_indices=[-1, 0],
                modality_keys=[KEY],
            ),
            "action": ModalityConfig(
                delta_indices=[0, 1],
                modality_keys=[KEY],
                action_configs=[
                    ActionConfig(
                        rep=ActionRepresentation.RELATIVE,
                        type=ActionType.EEF,
                        format=ActionFormat.XYZ_ROTVEC,
                        state_key=KEY,
                    )
                ],
            ),
        }
    }

    # Relative-action normalization is deliberately identity on [-1, 1]. This
    # lets the test inspect the public apply_action() output directly rather
    # than passing only through an absolute->relative->absolute roundtrip, which
    # could hide a shared wrong reference frame.
    unit_stats = {
        "min": [-1.0] * 6,
        "max": [1.0] * 6,
        "q01": [-1.0] * 6,
        "q99": [1.0] * 6,
        "mean": [0.0] * 6,
        "std": [1.0] * 6,
    }
    state_stats = {
        "min": [-10.0] * 6,
        "max": [10.0] * 6,
        "q01": [-10.0] * 6,
        "q99": [10.0] * 6,
        "mean": [0.0] * 6,
        "std": [1.0] * 6,
    }
    statistics = {
        EMBODIMENT: {
            "state": {KEY: state_stats},
            "action": {KEY: state_stats},
            "relative_action": {KEY: unit_stats},
        }
    }

    return StateActionProcessor(
        modality_configs=modality_configs,
        statistics=statistics,
        clip_outliers=False,
        use_relative_action=True,
    )


def _pose(values: list[float]) -> EndEffectorPose:
    return EndEffectorPose.from_action_format(
        np.asarray(values, dtype=np.float64),
        ActionFormat.XYZ_ROTVEC,
    )


def test_relative_eef_training_and_inference_anchor_on_latest_state():
    processor = _processor()

    old = np.array(
        [-0.7, 0.4, 0.2, -0.35, 0.10, 0.20],
        dtype=np.float64,
    )
    current = np.array(
        [0.8, -0.5, 0.6, 0.25, -0.20, 0.30],
        dtype=np.float64,
    )
    state_history = np.stack([old, current])

    expected_relative = np.array(
        [
            [0.12, -0.03, 0.04, 0.04, -0.02, 0.08],
            [0.18, 0.06, -0.02, -0.03, 0.05, 0.12],
        ],
        dtype=np.float64,
    )

    current_pose = _pose(current)
    relative_chunk = EndEffectorActionChunk.from_array(
        expected_relative,
        ActionFormat.XYZ_ROTVEC,
    )
    absolute_chunk = relative_chunk.to_absolute_chunking(
        reference_frame=current_pose
    ).to(ActionFormat.XYZ_ROTVEC)

    # Training/preprocessing contract: relative actions are defined from the
    # newest state (delta 0), i.e. the final state-history element.
    processed = processor.apply_action(
        {KEY: absolute_chunk},
        EMBODIMENT,
        state={KEY: state_history},
    )[KEY]
    np.testing.assert_allclose(
        processed,
        expected_relative,
        atol=1e-6,
    )

    # Explicit negative control: using the oldest history state produces a
    # materially different relative chunk, so this test cannot pass merely
    # because forward and inverse paths share the same wrong anchor.
    wrong_old_relative = EndEffectorActionChunk.from_array(
        absolute_chunk,
        ActionFormat.XYZ_ROTVEC,
    ).relative_chunking(reference_frame=_pose(old)).to(ActionFormat.XYZ_ROTVEC)
    assert not np.allclose(
        processed,
        wrong_old_relative,
        atol=1e-3,
    )

    # Deployment contract: batched [B, T_state, D] histories must use the same
    # newest reference when decoded back to absolute EEF actions.
    decoded = processor.unapply_action(
        {KEY: expected_relative[None]},
        EMBODIMENT,
        state={KEY: state_history[None]},
    )[KEY]
    np.testing.assert_allclose(
        decoded,
        absolute_chunk[None],
        atol=1e-6,
    )
