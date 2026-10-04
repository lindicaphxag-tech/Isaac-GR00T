# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from textwrap import dedent

import pytest

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


def _mismatched_action_modality() -> ModalityConfig:
    return ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ],
    )


def test_action_config_cardinality_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="must match number of modality keys"):
        _mismatched_action_modality()


def test_action_config_cardinality_validation_survives_python_optimization():
    code = dedent(
        """
        from gr00t.data.types import (
            ActionConfig,
            ActionFormat,
            ActionRepresentation,
            ActionType,
            ModalityConfig,
        )

        try:
            ModalityConfig(
                delta_indices=[0],
                modality_keys=["arm", "gripper"],
                action_configs=[
                    ActionConfig(
                        rep=ActionRepresentation.RELATIVE,
                        type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT,
                    )
                ],
            )
        except ValueError:
            raise SystemExit(0)

        raise SystemExit("invalid action config was accepted under python -O")
        """
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
