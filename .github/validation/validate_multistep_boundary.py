"""Public, dependency-light semantic validation for NVIDIA/Isaac-GR00T PR #786.

This file lives only on the validation branch. It imports the production
MultiStepWrapper from the exact PR head ancestry and checks Gymnasium episode
boundary semantics without loading the repository's global pytest configuration.
"""

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np

from gr00t.eval._horizon_contract import PolicyHorizonSpec
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper


ACTION = {"action": np.zeros((3, 1), dtype=np.float32)}


class FlagEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
    observation_space = gym.spaces.Dict(
        {"state.position": gym.spaces.Box(-100.0, 100.0, (1,), dtype=np.float32)}
    )

    def __init__(self, flags):
        super().__init__()
        self.flags = tuple(flags)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return {"state.position": np.array([0.0], dtype=np.float32)}, {}

    def step(self, action):
        terminated, truncated = self.flags[self.steps]
        self.steps += 1
        observation = {"state.position": np.array([float(self.steps)], dtype=np.float32)}
        return observation, 1.0, bool(terminated), bool(truncated), {}


class SuccessEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
    observation_space = gym.spaces.Dict(
        {"state.position": gym.spaces.Box(-100.0, 100.0, (1,), dtype=np.float32)}
    )

    def __init__(self):
        super().__init__()
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return {"state.position": np.array([0.0], dtype=np.float32)}, {"success": False}

    def step(self, action):
        self.steps += 1
        observation = {"state.position": np.array([float(self.steps)], dtype=np.float32)}
        return observation, 1.0, False, False, {"success": self.steps == 2}


def contract():
    return PolicyHorizonSpec(
        n_action_steps=3,
        action_horizon=3,
        video_delta_indices=(0,),
        state_delta_indices=(0,),
    )


def boundary_observation(observation):
    return float(observation["state.position"][0, 0])


def validate_inner_truncation():
    env = FlagEnv([(False, True), (False, False), (False, False)])
    wrapper = MultiStepWrapper(env=env, contract=contract(), max_episode_steps=100)
    wrapper.reset()
    observation, _, terminated, truncated, info = wrapper.step(ACTION)
    assert env.steps == 1
    assert terminated is False
    assert truncated is True
    assert boundary_observation(observation) == 1.0
    assert int(info["n_env_steps"]) == 1
    assert info["dones"].tolist() == [True]


def validate_joint_termination_and_truncation():
    env = FlagEnv([(True, True), (False, False), (False, False)])
    wrapper = MultiStepWrapper(env=env, contract=contract(), max_episode_steps=100)
    wrapper.reset()
    observation, _, terminated, truncated, info = wrapper.step(ACTION)
    assert env.steps == 1
    assert terminated is True
    assert truncated is True
    assert boundary_observation(observation) == 1.0
    assert int(info["n_env_steps"]) == 1
    assert info["dones"].tolist() == [True]


def validate_wrapper_time_limit():
    env = FlagEnv([(False, False), (False, False), (False, False)])
    wrapper = MultiStepWrapper(env=env, contract=contract(), max_episode_steps=2)
    wrapper.reset()
    observation, _, terminated, truncated, info = wrapper.step(ACTION)
    assert env.steps == 2
    assert terminated is False
    assert truncated is True
    assert boundary_observation(observation) == 2.0
    assert int(info["n_env_steps"]) == 2
    assert info["dones"].tolist() == [False, True]


def validate_success_boundary_bookkeeping():
    env = SuccessEnv()
    wrapper = MultiStepWrapper(
        env=env,
        contract=contract(),
        max_episode_steps=100,
        terminate_on_success=True,
    )
    wrapper.reset()
    observation, _, terminated, truncated, info = wrapper.step(ACTION)
    # Success is evaluated after the macro-step by current design.
    assert env.steps == 3
    assert terminated is True
    assert truncated is False
    assert boundary_observation(observation) == 3.0
    assert int(info["n_env_steps"]) == 3
    assert info["dones"].tolist() == [False, False, True]


def main():
    checks = {
        "inner_truncation": validate_inner_truncation,
        "joint_termination_and_truncation": validate_joint_termination_and_truncation,
        "wrapper_time_limit": validate_wrapper_time_limit,
        "success_boundary_bookkeeping": validate_success_boundary_bookkeeping,
    }
    for check in checks.values():
        check()
    print(
        json.dumps(
            {
                "status": "pass",
                "checks": list(checks),
                "production_wrapper": "gr00t.eval.sim.wrapper.multistep_wrapper.MultiStepWrapper",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
