"""Public VQ-BeT PushT support-response gate.

Mirrors the existing Diffusion PushT assay's physical state and intervention
protocol while changing only the frozen policy family.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from casj import EuclideanActionChart, collect_two_scale_probes, recover_two_scale_casj
from casj.pusht_state import PushTSnapshot, capture_snapshot, restore_snapshot
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.vqbet import VQBeTPolicy, make_vqbet_pre_post_processors
from lerobot.utils.constants import OBS_IMAGE, OBS_IMAGES, OBS_STATE

MODEL_ID = "lerobot/vqbet_pusht"
DATASET_ID = "lerobot/pusht"


def render_observation(env, snapshot: PushTSnapshot) -> dict[str, np.ndarray]:
    restore_snapshot(env, snapshot)
    obs = env.unwrapped.get_obs()
    if not isinstance(obs, dict) or "pixels" not in obs or "agent_pos" not in obs:
        raise RuntimeError("expected PushT pixels_agent_pos observation")
    return {
        "pixels": np.asarray(obs["pixels"]).copy(),
        "agent_pos": np.asarray(obs["agent_pos"], dtype=np.float32).copy(),
    }


def raw_frame(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        OBS_IMAGE: torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float() / 255.0,
        OBS_STATE: torch.from_numpy(obs["agent_pos"]).float(),
    }


def prepare_history(preprocessor, raw_history):
    processed = [preprocessor(raw_frame(obs)) for obs in raw_history]
    return {
        OBS_STATE: torch.stack([x[OBS_STATE][0] for x in processed], dim=0).unsqueeze(0),
        OBS_IMAGE: torch.stack([x[OBS_IMAGE][0] for x in processed], dim=0).unsqueeze(0),
    }


@torch.inference_mode()
def predict_chunk_with_common_randomness(policy, postprocessor, batch, *, sampling_seed: int):
    model_batch = dict(batch)
    model_batch[OBS_IMAGES] = torch.stack(
        [model_batch[key] for key in policy.config.image_features], dim=-4
    )

    devices = []
    parameter = next(policy.parameters())
    if parameter.is_cuda:
        devices = [parameter.device]

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(sampling_seed)
        if devices:
            torch.cuda.manual_seed_all(sampling_seed)
        normalized = policy.vqbet(model_batch, rollout=True)[:, : policy.config.action_chunk_size]

    action_chunk = postprocessor(normalized)
    return action_chunk.detach().cpu().numpy()[0]


@dataclass(frozen=True)
class VQRandomness:
    sampling_seed: int


@dataclass
class VQPolicyQuery:
    env: object
    policy: VQBeTPolicy
    preprocessor: object
    postprocessor: object

    def query(self, observation, *, randomness: VQRandomness) -> np.ndarray:
        raw = [render_observation(self.env, state) for state in observation]
        batch = prepare_history(self.preprocessor, raw)
        return predict_chunk_with_common_randomness(
            self.policy,
            self.postprocessor,
            batch,
            sampling_seed=randomness.sampling_seed,
        )


class PushTActionChart(EuclideanActionChart):
    pass


@dataclass(frozen=True)
class PushTBlockInterventionChart:
    num_supports: int = 1
    support_dim: int = 3

    def intervene(self, observation, *, direction: int, coefficients, magnitude: float):
        coeff = np.asarray(coefficients, dtype=np.float64)
        if coeff.shape != (1,):
            raise ValueError("PushT has one candidate support")
        if direction not in (0, 1, 2):
            raise ValueError("directions are block x, y, theta")

        signed = float(coeff[0]) * float(magnitude)
        delta_xy = np.zeros(2, dtype=np.float64)
        delta_angle = 0.0
        if direction < 2:
            delta_xy[direction] = signed
        else:
            delta_angle = signed
        return [
            state.shifted_block(delta_xy, delta_angle)
            for state in observation
        ]


def rollout_history(env, policy, preprocessor, postprocessor, *, seed: int, warmup: int):
    env.reset(seed=seed)
    history = deque(maxlen=policy.config.n_obs_steps)
    initial = capture_snapshot(env)
    for _ in range(policy.config.n_obs_steps):
        history.append(initial)

    for t in range(warmup):
        raw = [render_observation(env, state) for state in history]
        batch = prepare_history(preprocessor, raw)
        chunk = predict_chunk_with_common_randomness(
            policy,
            postprocessor,
            batch,
            sampling_seed=seed + 10_000 + t,
        )
        restore_snapshot(env, history[-1])
        _, _, terminated, truncated, _ = env.step(np.asarray(chunk[0], dtype=np.float32))
        history.append(capture_snapshot(env))
        if terminated or truncated:
            break
    return list(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup-steps", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--angle-epsilon", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("vqbet_pusht_support_gate.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    policy = VQBeTPolicy.from_pretrained(MODEL_ID)
    policy.config.device = str(device)
    policy.to(device)
    policy.eval()

    metadata = LeRobotDatasetMetadata(DATASET_ID)
    preprocessor, postprocessor = make_vqbet_pre_post_processors(
        policy.config,
        dataset_stats=metadata.stats,
    )

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )

    results = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "policy_family": "vqbet",
        "paired_randomness": "torch_rng_seed_replay_for_multinomial",
        "seed": args.seed,
        "epsilon_pixels": args.epsilon,
        "angle_epsilon_rad": args.angle_epsilon,
        "snapshots": {},
    }

    for warmup in args.warmup_steps:
        history = rollout_history(
            env,
            policy,
            preprocessor,
            postprocessor,
            seed=args.seed,
            warmup=warmup,
        )
        if len(history) != policy.config.n_obs_steps:
            raise RuntimeError("incomplete observation history")

        query = VQPolicyQuery(env, policy, preprocessor, postprocessor)
        randomness = VQRandomness(args.seed + 20_000 + warmup)

        baseline_a = query.query(history, randomness=randomness)
        baseline_b = query.query(history, randomness=randomness)
        replay_error = float(np.max(np.abs(baseline_a - baseline_b)))
        if replay_error > 1e-7:
            raise RuntimeError(
                f"paired VQ-BeT randomness did not replay exactly: {replay_error}"
            )

        changed_history = [
            state.shifted_block(np.asarray([args.epsilon, 0.0], dtype=np.float64), 0.0)
            for state in history
        ]
        base_pixels = [render_observation(env, state)["pixels"] for state in history]
        changed_pixels = [
            render_observation(env, state)["pixels"] for state in changed_history
        ]
        changed = any(
            not np.array_equal(a, b)
            for a, b in zip(base_pixels, changed_pixels, strict=True)
        )
        if not changed:
            raise RuntimeError(
                "physical support intervention did not change policy pixels"
            )

        probes = collect_two_scale_probes(
            policy=query,
            action_chart=PushTActionChart(),
            support_chart=PushTBlockInterventionChart(),
            observation=history,
            randomness=randomness,
            codes=np.ones((1, 1), dtype=np.float64),
            fine_epsilon=np.asarray(
                [args.epsilon, args.epsilon, args.angle_epsilon],
                dtype=np.float64,
            ),
            coarse_epsilon=np.asarray(
                [2.0 * args.epsilon, 2.0 * args.epsilon, 2.0 * args.angle_epsilon],
                dtype=np.float64,
            ),
            max_pairing_error=1e-7,
        )
        estimates = recover_two_scale_casj(probes, sparsity=1)
        jacobian_fine = estimates.fine.jacobian[:, 0]
        jacobian_coarse = estimates.coarse.jacobian[:, 0]
        denominator = np.maximum(
            np.maximum(
                np.linalg.norm(jacobian_fine, axis=(1, 2)),
                np.linalg.norm(jacobian_coarse, axis=(1, 2)),
            ),
            1e-12,
        )
        scale_drift = (
            np.linalg.norm(jacobian_fine - jacobian_coarse, axis=(1, 2))
            / denominator
        )

        results["snapshots"][f"warmup={warmup}"] = {
            "paired_randomness_replay_error": replay_error,
            "input_intervention_changed_pixels": changed,
            "query_count": probes.query_count,
            "fine_jacobian": jacobian_fine.tolist(),
            "coarse_jacobian": jacobian_coarse.tolist(),
            "scale_drift": scale_drift.tolist(),
            "median_scale_drift": float(np.median(scale_drift)),
            "max_scale_drift": float(np.max(scale_drift)),
            "action_chunk_shape": list(baseline_a.shape),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
