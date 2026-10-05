"""Public-checkpoint CASJ assay on lerobot/diffusion_pusht + gym-pusht.

This experiment is intentionally black-box with respect to policy weights:

1. load the public frozen LeRobot PushT diffusion checkpoint;
2. capture two exact PushT simulator states for the policy observation history;
3. perturb only the T-block pose while leaving the agent/history otherwise fixed;
4. reuse the same initial DDPM noise and the same scheduler generator seed for
   every baseline/counterfactual query;
5. estimate the Action-Support Jacobian by central finite differences;
6. test CASJ on a new held-out block displacement by predicting the fresh
   policy's action-chunk response.

The script uses DiffusionModel.conditional_sample directly because current
LeRobot public DiffusionPolicy inference does not expose the generator already
supported by that low-level sampler. This keeps the full DDPM randomness paired
without depending on exploration-branch code.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import gym_pusht
import numpy as np
import torch

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE

from casj import (
    RepairMode,
    certify_directional_remainder,
    infer_planar_rigid_anchor,
    certify_runtime,
    directional_second_derivative,
)


MODEL_ID = "lerobot/diffusion_pusht"
DATASET_ID = "lerobot/pusht"


@dataclass
class PushTSnapshot:
    agent_position: np.ndarray
    agent_velocity: np.ndarray
    block_position: np.ndarray
    block_angle: float
    block_velocity: np.ndarray
    block_angular_velocity: float

    def shifted_block(self, delta_xy: np.ndarray, delta_angle: float = 0.0) -> "PushTSnapshot":
        return PushTSnapshot(
            agent_position=self.agent_position.copy(),
            agent_velocity=self.agent_velocity.copy(),
            block_position=self.block_position + np.asarray(delta_xy, dtype=float),
            block_angle=float(self.block_angle + delta_angle),
            block_velocity=self.block_velocity.copy(),
            block_angular_velocity=float(self.block_angular_velocity),
        )


def _vec2(body_value: Any) -> np.ndarray:
    return np.asarray([body_value[0], body_value[1]], dtype=np.float64)


def capture_snapshot(env) -> PushTSnapshot:
    u = env.unwrapped
    return PushTSnapshot(
        agent_position=_vec2(u.agent.position),
        agent_velocity=_vec2(u.agent.velocity),
        block_position=_vec2(u.block.position),
        block_angle=float(u.block.angle),
        block_velocity=_vec2(u.block.velocity),
        block_angular_velocity=float(u.block.angular_velocity),
    )


def restore_snapshot(env, snapshot: PushTSnapshot) -> None:
    u = env.unwrapped
    u.agent.position = snapshot.agent_position.tolist()
    u.agent.velocity = snapshot.agent_velocity.tolist()
    u.block.position = snapshot.block_position.tolist()
    u.block.angle = snapshot.block_angle
    u.block.velocity = snapshot.block_velocity.tolist()
    u.block.angular_velocity = snapshot.block_angular_velocity


def render_observation(env, snapshot: PushTSnapshot) -> dict[str, np.ndarray]:
    restore_snapshot(env, snapshot)
    obs = env.unwrapped.get_obs()
    if not isinstance(obs, dict) or "pixels" not in obs or "agent_pos" not in obs:
        raise RuntimeError(
            "Expected gym-pusht obs_type='pixels_agent_pos' with pixels and agent_pos"
        )
    return {
        "pixels": np.asarray(obs["pixels"]).copy(),
        "agent_pos": np.asarray(obs["agent_pos"], dtype=np.float32).copy(),
    }


def _raw_policy_frame(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    image = torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float() / 255.0
    state = torch.from_numpy(obs["agent_pos"]).float()
    return {OBS_STATE: state, OBS_IMAGE: image}


def prepare_history(preprocessor, raw_history: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    processed = [preprocessor(_raw_policy_frame(obs)) for obs in raw_history]
    return {
        OBS_STATE: torch.stack([item[OBS_STATE][0] for item in processed], dim=0).unsqueeze(0),
        OBS_IMAGE: torch.stack([item[OBS_IMAGE][0] for item in processed], dim=0).unsqueeze(0),
    }


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    # torch.Generator accepts device types such as "cpu" and "cuda".
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


@torch.inference_mode()
def predict_chunk_with_common_randomness(
    policy: DiffusionPolicy,
    postprocessor,
    batch: dict[str, torch.Tensor],
    *,
    initial_noise: torch.Tensor,
    scheduler_seed: int,
) -> np.ndarray:
    """Mirror generate_actions while preserving complete DDPM randomness."""
    model = policy.diffusion
    batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
    if n_obs_steps != policy.config.n_obs_steps:
        raise ValueError(
            f"history has {n_obs_steps} steps, checkpoint expects {policy.config.n_obs_steps}"
        )

    global_cond = model._prepare_global_conditioning(batch)
    generator = make_generator(initial_noise.device, scheduler_seed)
    full = model.conditional_sample(
        batch_size,
        global_cond=global_cond,
        generator=generator,
        noise=initial_noise.clone(),
    )

    start = n_obs_steps - 1
    stop = start + policy.config.n_action_steps
    normalized_chunk = full[:, start:stop]
    action_chunk = postprocessor(normalized_chunk)
    return action_chunk.detach().cpu().numpy()[0]


def perturb_history(
    env,
    history: list[PushTSnapshot],
    *,
    delta_xy: np.ndarray,
    delta_angle: float = 0.0,
) -> list[dict[str, np.ndarray]]:
    return [
        render_observation(env, state.shifted_block(delta_xy, delta_angle))
        for state in history
    ]


def estimate_translation_casj(
    *,
    env,
    history: list[PushTSnapshot],
    policy,
    preprocessor,
    postprocessor,
    initial_noise,
    scheduler_seed: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate block-translation CASJ and its diagonal curvature witness.

    Returns
    -------
    baseline:
        Frozen-policy action chunk [H, 2].
    jacobian:
        Central finite-difference derivative [H, 2, 2].
    curvature:
        Per-support-axis second derivative [H, 2, 2]. The last dimension
        indexes the perturbed block x/y coordinate.
    """
    base_raw = [render_observation(env, state) for state in history]
    base_batch = prepare_history(preprocessor, base_raw)
    baseline = predict_chunk_with_common_randomness(
        policy,
        postprocessor,
        base_batch,
        initial_noise=initial_noise,
        scheduler_seed=scheduler_seed,
    )

    jacobian = np.zeros((baseline.shape[0], 2, 2), dtype=np.float64)
    curvature = np.zeros_like(jacobian)
    for support_axis in range(2):
        delta = np.zeros(2, dtype=np.float64)
        delta[support_axis] = epsilon

        plus_batch = prepare_history(
            preprocessor,
            perturb_history(env, history, delta_xy=delta),
        )
        minus_batch = prepare_history(
            preprocessor,
            perturb_history(env, history, delta_xy=-delta),
        )
        plus = predict_chunk_with_common_randomness(
            policy,
            postprocessor,
            plus_batch,
            initial_noise=initial_noise,
            scheduler_seed=scheduler_seed,
        )
        minus = predict_chunk_with_common_randomness(
            policy,
            postprocessor,
            minus_batch,
            initial_noise=initial_noise,
            scheduler_seed=scheduler_seed,
        )
        jacobian[:, :, support_axis] = (plus - minus) / (2.0 * epsilon)
        curvature[:, :, support_axis] = (
            plus + minus - 2.0 * baseline
        ) / (epsilon * epsilon)

    return baseline, jacobian, curvature




def estimate_angle_casj(
    *,
    env,
    history,
    policy,
    preprocessor,
    postprocessor,
    baseline,
    initial_noise,
    scheduler_seed: int,
    epsilon_angle: float,
) -> np.ndarray:
    """Estimate d(action_xy) / d(block_angle) around the current support center."""
    if epsilon_angle <= 0:
        raise ValueError("epsilon_angle must be positive")

    plus_batch = prepare_history(
        preprocessor,
        perturb_history(
            env,
            history,
            delta_xy=np.zeros(2, dtype=np.float64),
            delta_angle=epsilon_angle,
        ),
    )
    minus_batch = prepare_history(
        preprocessor,
        perturb_history(
            env,
            history,
            delta_xy=np.zeros(2, dtype=np.float64),
            delta_angle=-epsilon_angle,
        ),
    )
    plus = predict_chunk_with_common_randomness(
        policy,
        postprocessor,
        plus_batch,
        initial_noise=initial_noise,
        scheduler_seed=scheduler_seed,
    )
    minus = predict_chunk_with_common_randomness(
        policy,
        postprocessor,
        minus_batch,
        initial_noise=initial_noise,
        scheduler_seed=scheduler_seed,
    )
    return (plus - minus) / (2.0 * epsilon_angle)


def estimate_directional_curvature(
    *,
    env,
    history,
    policy,
    preprocessor,
    postprocessor,
    baseline,
    initial_noise,
    scheduler_seed: int,
    direction: np.ndarray,
    step: float,
) -> np.ndarray:
    """Estimate the second derivative along one physical block-motion direction."""
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        return np.zeros_like(baseline, dtype=np.float64)
    unit = direction / norm
    delta = step * unit

    plus_batch = prepare_history(
        preprocessor,
        perturb_history(env, history, delta_xy=delta),
    )
    minus_batch = prepare_history(
        preprocessor,
        perturb_history(env, history, delta_xy=-delta),
    )
    plus = predict_chunk_with_common_randomness(
        policy,
        postprocessor,
        plus_batch,
        initial_noise=initial_noise,
        scheduler_seed=scheduler_seed,
    )
    minus = predict_chunk_with_common_randomness(
        policy,
        postprocessor,
        minus_batch,
        initial_noise=initial_noise,
        scheduler_seed=scheduler_seed,
    )
    return directional_second_derivative(
        plus,
        minus,
        baseline,
        step=step,
    )


def rollout_to_snapshot(
    *,
    env,
    policy,
    preprocessor,
    postprocessor,
    steps: int,
    seed: int,
    device: torch.device,
) -> list[PushTSnapshot]:
    obs, _ = env.reset(seed=seed)
    history: deque[PushTSnapshot] = deque(maxlen=policy.config.n_obs_steps)
    initial = capture_snapshot(env)
    for _ in range(policy.config.n_obs_steps):
        history.append(initial)

    # Use deterministic public-policy actions to reach nontrivial phases.
    for t in range(steps):
        raw_history = [render_observation(env, state) for state in history]
        batch = prepare_history(preprocessor, raw_history)
        noise_gen = make_generator(device, seed + 1000 + t)
        initial_noise = torch.randn(
            (1, policy.config.horizon, policy.config.output_features["action"].shape[0]),
            generator=noise_gen,
            device=device,
            dtype=next(policy.parameters()).dtype,
        )
        action_chunk = predict_chunk_with_common_randomness(
            policy,
            postprocessor,
            batch,
            initial_noise=initial_noise,
            scheduler_seed=seed + 2000 + t,
        )
        restore_snapshot(env, history[-1])
        _, _, terminated, truncated, _ = env.step(action_chunk[0])
        history.append(capture_snapshot(env))
        if terminated or truncated:
            break

    return list(history)



def execute_chunk_branch(
    env,
    *,
    start_state: PushTSnapshot,
    chunk: np.ndarray,
) -> dict[str, Any]:
    """Execute one candidate chunk from an identical simulator state."""
    restore_snapshot(env, start_state)
    rewards = []
    last_info: dict[str, Any] = {}
    terminated = False
    truncated = False

    for index, action in enumerate(np.asarray(chunk, dtype=np.float64)):
        action32 = np.asarray(action, dtype=np.float32)
        if not env.action_space.contains(action32):
            return {
                "valid": False,
                "invalid_action_index": index,
                "invalid_action": action.tolist(),
                "steps": index,
                "success": False,
            }
        _, reward, terminated, truncated, info = env.step(action32)
        rewards.append(float(reward))
        last_info = dict(info)
        if terminated or truncated:
            break

    final_state = capture_snapshot(env)
    return {
        "valid": True,
        "steps": len(rewards),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(last_info.get("is_success", False)),
        "final_reward": rewards[-1] if rewards else None,
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "final_coverage": float(last_info["coverage"]) if "coverage" in last_info else None,
        "final_agent_position": final_state.agent_position.tolist(),
        "final_block_position": final_state.block_position.tolist(),
        "final_block_angle": final_state.block_angle,
    }


def evaluate_snapshot(
    *,
    env,
    history,
    policy,
    preprocessor,
    postprocessor,
    device,
    seed,
    epsilon,
    epsilon_angle,
    heldout_delta,
):
    noise_gen = make_generator(device, seed + 3000)
    initial_noise = torch.randn(
        (1, policy.config.horizon, policy.config.output_features["action"].shape[0]),
        generator=noise_gen,
        device=device,
        dtype=next(policy.parameters()).dtype,
    )

    baseline, casj_fine, curvature = estimate_translation_casj(
        env=env,
        history=history,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        initial_noise=initial_noise,
        scheduler_seed=seed + 4000,
        epsilon=epsilon,
    )
    baseline_coarse, casj_coarse, _ = estimate_translation_casj(
        env=env,
        history=history,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        initial_noise=initial_noise,
        scheduler_seed=seed + 4000,
        epsilon=2.0 * epsilon,
    )
    np.testing.assert_allclose(baseline_coarse, baseline, rtol=1e-6, atol=1e-6)

    angle_column = estimate_angle_casj(
        env=env,
        history=history,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        baseline=baseline,
        initial_noise=initial_noise,
        scheduler_seed=seed + 4000,
        epsilon_angle=epsilon_angle,
    )
    planar_casj = np.concatenate(
        [casj_fine, angle_column[:, :, None]],
        axis=2,
    )

    current_support = history[-1]
    anchor_fingerprint_objects = [
        infer_planar_rigid_anchor(
            planar_casj[t],
            support_position=current_support.block_position,
            support_angle=current_support.block_angle,
            action_point=baseline[t],
        )
        for t in range(len(baseline))
    ]

    fresh_batch = prepare_history(
        preprocessor,
        perturb_history(env, history, delta_xy=heldout_delta),
    )
    fresh = predict_chunk_with_common_randomness(
        policy,
        postprocessor,
        fresh_batch,
        initial_noise=initial_noise,
        scheduler_seed=seed + 4000,
    )

    casj_delta = np.einsum("tad,d->ta", casj_fine, heldout_delta)
    casj_repaired = baseline + casj_delta
    global_comp = baseline + heldout_delta[None, :]

    directional_fine = estimate_directional_curvature(
        env=env,
        history=history,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        baseline=baseline,
        initial_noise=initial_noise,
        scheduler_seed=seed + 4000,
        direction=heldout_delta,
        step=epsilon,
    )
    directional_coarse = estimate_directional_curvature(
        env=env,
        history=history,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        baseline=baseline,
        initial_noise=initial_noise,
        scheduler_seed=seed + 4000,
        direction=heldout_delta,
        step=2.0 * epsilon,
    )

    stale_mse = float(np.mean((baseline - fresh) ** 2))
    global_mse = float(np.mean((global_comp - fresh) ** 2))
    casj_mse = float(np.mean((casj_repaired - fresh) ** 2))

    identity = np.eye(2)
    jacobian_norm = np.linalg.norm(casj_fine, axis=(1, 2))
    identity_residual = np.linalg.norm(
        casj_fine - identity[None, :, :], axis=(1, 2)
    )
    scale_drift = np.linalg.norm(
        casj_fine - casj_coarse, axis=(1, 2)
    ) / np.maximum(
        np.maximum(
            np.linalg.norm(casj_fine, axis=(1, 2)),
            np.linalg.norm(casj_coarse, axis=(1, 2)),
        ),
        1e-12,
    )

    # Diagonal second-order effect at the held-out displacement. Mixed Hessian
    # terms are intentionally not inferred here, so this is a witness rather
    # than a complete Hessian certificate.
    second_order = 0.5 * np.einsum(
        "tad,d->ta",
        curvature,
        heldout_delta * heldout_delta,
    )
    second_order_norm = np.linalg.norm(second_order, axis=1)
    # Dimensionless witness: compare the second-order action effect with the
    # physical support displacement scale. For a rigid-follow relation the
    # first-order action change is naturally on this same scale.
    support_move_scale = max(float(np.linalg.norm(heldout_delta)), 1e-12)
    curvature_effect = second_order_norm / support_move_scale
    first_effect = np.linalg.norm(casj_delta, axis=1)
    curvature_ratio = second_order_norm / np.maximum(first_effect, 1e-12)

    # PushT has one candidate physical support (the T block), so the 1x1 code
    # design is exactly identifiable for sparsity one.
    codes = np.ones((1, 1), dtype=np.float64)
    certificates = []
    for t in range(len(baseline)):
        cert = certify_runtime(
            fine_blocks=casj_fine[t][None, :, :],
            coarse_blocks=casj_coarse[t][None, :, :],
            codes=codes,
            sparsity=1,
            fit_residual=0.0,
            active_code_condition=1.0,
            curvature_effect=float(curvature_effect[t]),
            curvature_ratio=float(curvature_ratio[t]),
        )
        directional = certify_directional_remainder(
            fine_second=directional_fine[t],
            coarse_second=directional_coarse[t],
            displacement_norm=float(np.linalg.norm(heldout_delta)),
            first_order_effect=casj_delta[t],
            reference_action_scale=max(float(np.linalg.norm(heldout_delta)), 1e-12),
        )
        accepted = cert.accepted and directional.accepted
        mode = cert.repair_mode if accepted else RepairMode.REPLAN
        reason = cert.reason if not cert.accepted else directional.reason

        anchor_fingerprint = anchor_fingerprint_objects[t]
        if (
            accepted
            and mode is RepairMode.EXACT_TRANSPORT
            and not anchor_fingerprint.accepted
        ):
            accepted = False
            mode = RepairMode.REPLAN
            reason = (
                "translation response looked rigid but angle intervention does not "
                f"support one consistent rigid anchor: {anchor_fingerprint.reason}"
            )
        certificates.append(
            {
                "accepted": accepted,
                "mode": mode.value,
                "relation": cert.relation.relation_type.value,
                "active_supports": list(cert.relation.active_supports),
                "reason": reason,
                "directional_remainder": {
                    "accepted": directional.accepted,
                    "curvature_scale_stability": directional.curvature_scale_stability,
                    "predicted_remainder_norm": directional.predicted_remainder_norm,
                    "remainder_to_first_order": directional.remainder_to_first_order,
                    "remainder_to_reference_scale": directional.remainder_to_reference_scale,
                    "reason": directional.reason,
                },
            }
        )

    # Materialize the runtime chunk only after the whole predicted suffix is
    # certified. PushT actions are absolute [x,y] target positions in the same
    # pixel coordinates as the block translation.
    any_refusal = any(not item["accepted"] for item in certificates)
    certified_chunk = baseline.copy()
    if any_refusal:
        certified_chunk = fresh.copy()
        certified_fallback = "fresh_requery"
    else:
        certified_fallback = None
        for t, item in enumerate(certificates):
            mode = RepairMode(item["mode"])
            if mode is RepairMode.KEEP:
                continue
            if mode is RepairMode.EXACT_TRANSPORT:
                certified_chunk[t] = baseline[t] + heldout_delta
            elif mode is RepairMode.LOCAL_LINEAR_REPAIR:
                certified_chunk[t] = casj_repaired[t]
            else:
                raise RuntimeError(f"unexpected certified mode {mode}")

    disturbed_state = history[-1].shifted_block(heldout_delta)
    rollout_branches = {
        "stale": execute_chunk_branch(
            env, start_state=disturbed_state, chunk=baseline
        ),
        "global": execute_chunk_branch(
            env, start_state=disturbed_state, chunk=global_comp
        ),
        "casj_raw_linear": execute_chunk_branch(
            env, start_state=disturbed_state, chunk=casj_repaired
        ),
        "casj_certified": execute_chunk_branch(
            env, start_state=disturbed_state, chunk=certified_chunk
        ),
        "fresh": execute_chunk_branch(
            env, start_state=disturbed_state, chunk=fresh
        ),
    }

    fresh_final = rollout_branches["fresh"].get("final_block_position")
    if fresh_final is not None:
        fresh_final = np.asarray(fresh_final, dtype=np.float64)
        for branch_result in rollout_branches.values():
            position = branch_result.get("final_block_position")
            if position is not None:
                branch_result["final_block_distance_to_fresh"] = float(
                    np.linalg.norm(np.asarray(position, dtype=np.float64) - fresh_final)
                )


    planar_anchor_fingerprints = [
        {
            "accepted": fingerprint.accepted,
            "anchor_world": fingerprint.anchor_world.tolist(),
            "anchor_local": fingerprint.anchor_local.tolist(),
            "translation_residual": fingerprint.translation_residual,
            "anchor_consistency_residual": fingerprint.anchor_consistency_residual,
            "jacobian_residual": fingerprint.jacobian_residual,
            "reason": fingerprint.reason,
        }
        for fingerprint in anchor_fingerprint_objects
    ]

    return {
        "stale_mse": stale_mse,
        "global_compensation_mse": global_mse,
        "casj_repair_mse": casj_mse,
        "casj_to_stale_ratio": casj_mse / max(stale_mse, 1e-12),
        "global_to_stale_ratio": global_mse / max(stale_mse, 1e-12),
        "jacobian_frobenius_norm": jacobian_norm.tolist(),
        "rigid_identity_residual": identity_residual.tolist(),
        "scale_drift": scale_drift.tolist(),
        "curvature_effect": curvature_effect.tolist(),
        "curvature_ratio": curvature_ratio.tolist(),
        "certificates": certificates,
        "certified_fallback": certified_fallback,
        "rollout_branches": rollout_branches,
        "jacobian_fine": casj_fine.tolist(),
        "block_angle_jacobian": angle_column.tolist(),
        "planar_casj_xytheta": planar_casj.tolist(),
        "planar_anchor_fingerprints": planar_anchor_fingerprints,
        "jacobian_coarse": casj_coarse.tolist(),
        "curvature_diagonal": curvature.tolist(),
        "directional_curvature_fine": directional_fine.tolist(),
        "directional_curvature_coarse": directional_coarse.tolist(),
        "baseline_chunk": baseline.tolist(),
        "fresh_chunk": fresh.tolist(),
        "casj_repaired_chunk": casj_repaired.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override the checkpoint diffusion inference steps. Leave unset for the checkpoint default.",
    )
    parser.add_argument("--warmup-steps", type=int, nargs="+", default=[0, 4, 8, 12])
    parser.add_argument("--epsilon", type=float, default=4.0, help="finite-difference block shift in pixels")
    parser.add_argument("--angle-epsilon", type=float, default=0.02, help="finite-difference block angle in radians")
    parser.add_argument("--heldout-dx", type=float, default=10.0)
    parser.add_argument("--heldout-dy", type=float, default=-6.0)
    parser.add_argument("--output", type=Path, default=Path("casj_pusht_public.json"))
    args = parser.parse_args()

    device = torch.device(args.device)

    policy = DiffusionPolicy.from_pretrained(args.model_id)
    policy.config.device = str(device)
    policy.to(device)
    policy.eval()
    checkpoint_inference_steps = int(policy.diffusion.num_inference_steps)
    if args.num_inference_steps is not None:
        if args.num_inference_steps <= 0:
            raise ValueError("--num-inference-steps must be positive")
        policy.diffusion.num_inference_steps = args.num_inference_steps
    effective_inference_steps = int(policy.diffusion.num_inference_steps)

    metadata = LeRobotDatasetMetadata(args.dataset_id)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        args.model_id,
        dataset_stats=metadata.stats,
    )

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )

    results = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "seed": args.seed,
        "checkpoint_num_inference_steps": checkpoint_inference_steps,
        "effective_num_inference_steps": effective_inference_steps,
        "uses_checkpoint_default_inference_steps": (
            effective_inference_steps == checkpoint_inference_steps
        ),
        "epsilon_pixels": args.epsilon,
        "heldout_delta_pixels": [args.heldout_dx, args.heldout_dy],
        "angle_epsilon_rad": args.angle_epsilon,
        "snapshots": {},
    }

    for warmup in args.warmup_steps:
        history = rollout_to_snapshot(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            steps=warmup,
            seed=args.seed,
            device=device,
        )
        if len(history) != policy.config.n_obs_steps:
            raise RuntimeError(
                f"expected {policy.config.n_obs_steps} history states, got {len(history)}"
            )
        result = evaluate_snapshot(
            env=env,
            history=history,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            seed=args.seed + warmup * 100,
            epsilon=args.epsilon,
            epsilon_angle=args.angle_epsilon,
            heldout_delta=np.asarray([args.heldout_dx, args.heldout_dy], dtype=np.float64),
        )
        results["snapshots"][str(warmup)] = result

    env.close()
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
