# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


def _config(rep):
    return ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm"],
        action_configs=[
            ActionConfig(
                rep=rep,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ],
    )


def test_delta_action_representation_fails_closed():
    with pytest.raises(ValueError, match="DELTA is not implemented"):
        _config(ActionRepresentation.DELTA)


@pytest.mark.parametrize(
    "representation",
    [ActionRepresentation.RELATIVE, ActionRepresentation.ABSOLUTE],
)
def test_supported_action_representations_remain_valid(representation):
    config = _config(representation)
    assert config.action_configs[0].rep is representation


def test_delta_dict_config_also_fails_closed_after_parsing():
    with pytest.raises(ValueError, match="DELTA is not implemented"):
        ModalityConfig(
            delta_indices=[0],
            modality_keys=["arm"],
            action_configs=[
                {
                    "rep": "DELTA",
                    "type": "NON_EEF",
                    "format": "DEFAULT",
                }
            ],
        )
