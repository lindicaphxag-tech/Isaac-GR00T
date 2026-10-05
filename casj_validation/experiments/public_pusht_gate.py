"""Public-checkpoint CASJ assay on lerobot/diffusion_pusht + gym-pusht.

This experiment is intentionally black-box with respect to policy weights:

1. load the public frozen LeRobot PushT diffusion checkpoint;
2. capture two exact PushT simulator states for the policy observation history;
3. perturb only the T-block pose (translation and rotation) while leaving the
   agent/history otherwise fixed;
4. reuse the same initial DDPM noise and the same scheduler generator seed for
   every baseline/counterfactual query;
5. estimate the Action-Support Jacobian by central finite differences;
6. test CASJ on a new held-out block pose by predicting the fresh policy's
   action-chunk response and applying the recovered latent anchor for finite
   rotation transport.

The script uses DiffusionModel.conditional_sample directly because current
LeRobot public DiffusionPolicy inference does not expose the generator already
supported by that low-level sampler. This keeps the full DDPM randomness paired
without depending on exploration-branch code.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import gym_pusht  # noqa: F401  # registers the PushT Gymnasium environment
import gymnasium as gym
import numpy as np
import torch
from casj import (
    EuclideanActionChart,
    RepairMode,
    certify_directional_remainder,
    certify_directional_trust_region,
    certify_runtime,
    collect_two_scale_probes,
    infer_planar_rigid_anchor,
    recover_two_scale_casj,
    transport_planar_anchor,
)
from casj.pusht_assay import estimate_directional_curvature_via_query
from casj.pusht_state import PushTSnapshot, capture_snapshot, restore_snapshot
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.policies.diffusion.processor_diffusion import (
    make_diffusion_pre_post_processors,
)
from lerobot.utils.constants import OBS_IMAGE, OBS_IMAGES, OBS_STATE

MODEL_ID = "lerobot/diffusion_pusht"
DATASET_ID = "lerobot/pusht"


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


def prepare_history(
    preprocessor, raw_history: list[dict[str, np.ndarray]]
) -> dict[str, torch.Tensor]:
    processed = [preprocessor(_raw_policy_frame(obs)) for obs in raw_history]
    return {
        OBS_STATE: torch.stack(
            [item[OBS_STATE][0] for item in processed], dim=0
        ).unsqueeze(0),
        OBS_IMAGE: torch.stack(
            [item[OBS_IMAGE][0] for item in processed], dim=0
        ).unsqueeze(0),
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

    # Mirror DiffusionPolicy.predict_action_chunk: the public processor keeps
    # camera tensors under their configured feature keys, while DiffusionModel
    # consumes a stacked internal OBS_IMAGES tensor.
    model_batch = dict(batch)
    if policy.config.image_features:
        model_batch[OBS_IMAGES] = torch.stack(
            [model_batch[key] for key in policy.config.image_features],
            dim=-4,
        )

    global_cond = model._prepare_global_conditioning(model_batch)
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


@dataclass(frozen=True)
class PushTRandomness:
    """Replayable DDPM randomness for one paired CASJ assay."""

    initial_noise: torch.Tensor
    scheduler_seed: int


@dataclass
class PushTPolicyQuery:
    env: Any
    policy: DiffusionPolicy
    preprocessor: Any
    postprocessor: Any

    def query(
        self,
        observation: list[PushTSnapshot],
        *,
        randomness: PushTRandomness,
    ) -> np.ndarray:
        raw_history = [render_observation(self.env, state) for state in observation]
        batch = prepare_history(self.preprocessor, raw_history)
        return predict_chunk_with_common_randomness(
            self.policy,
            self.postprocessor,
            batch,
            initial_noise=randomness.initial_noise,
            scheduler_seed=randomness.scheduler_seed,
        )


class PushTActionChart(EuclideanActionChart):
    def decode(self, raw_chunk: np.ndarray, *, observation) -> np.ndarray:
        # postprocessor has already mapped the normalized policy action to the
        # physical PushT absolute target coordinates used by the environment.
        return super().decode(raw_chunk, observation=observation)


@dataclass(frozen=True)
class PushTBlockInterventionChart:
    num_supports: int = 1
    support_dim: int = 3

    def intervene(
        self,
        observation: list[PushTSnapshot],
        *,
        direction: int,
        coefficients: np.ndarray,
        magnitude: float,
    ) -> list[PushTSnapshot]:
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape != (1,):
            raise ValueError("PushT has exactly one candidate support")
        if direction not in (0, 1, 2):
            raise ValueError("PushT support directions are block x, y, theta")

        signed = float(coefficients[0]) * float(magnitude)
        delta_xy = np.zeros(2, dtype=np.float64)
        delta_angle = 0.0
        if direction < 2:
            delta_xy[direction] = signed
        else:
            delta_angle = signed

        return [state.shifted_block(delta_xy, delta_angle) for state in observation]


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
    _obs, _ = env.reset(seed=seed)
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
            (
                1,
                policy.config.horizon,
                policy.config.output_features["action"].shape[0],
            ),
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
        "final_coverage": float(last_info["coverage"])
        if "coverage" in last_info
        else None,
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
    heldout_delta_xy,
    heldout_delta_angle,
):
    noise_gen = make_generator(device, seed + 3000)
    initial_noise = torch.randn(
        (1, policy.config.horizon, policy.config.output_features["action"].shape[0]),
        generator=noise_gen,
        device=device,
        dtype=next(policy.parameters()).dtype,
    )

    randomness = PushTRandomness(
        initial_noise=initial_noise,
        scheduler_seed=seed + 4000,
    )
    policy_query = PushTPolicyQuery(
        env=env,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    baseline_images = [render_observation(env, state) for state in history]
    intervention = [
        state.shifted_block(np.asarray([epsilon, 0.0], dtype=np.float64))
        for state in history
    ]
    intervention_images = [
        render_observation(env, state)
        for state in intervention
    ]
    if not any(
        not np.array_equal(base["pixels"], changed["pixels"])
        for base, changed in zip(
            baseline_images, intervention_images, strict=True
        )
    ):
        raise RuntimeError(
            "INVALID_INPUT_PERTURBATION: block translation did not change any "
            "rendered policy observation; refusing to score CASJ"
        )
    action_chart = PushTActionChart()
    support_chart = PushTBlockInterventionChart()
    probes = collect_two_scale_probes(
        policy=policy_query,
        action_chart=action_chart,
        support_chart=support_chart,
        observation=history,
        randomness=randomness,
        codes=np.ones((1, 1), dtype=np.float64),
        fine_epsilon=np.asarray([epsilon, epsilon, epsilon_angle], dtype=np.float64),
        coarse_epsilon=np.asarray(
            [2.0 * epsilon, 2.0 * epsilon, 2.0 * epsilon_angle],
            dtype=np.float64,
        ),
        max_pairing_error=1e-6,
    )
    estimates = recover_two_scale_casj(probes, sparsity=1)

    baseline = probes.baseline
    planar_casj = estimates.fine.jacobian[:, 0]
    planar_casj_coarse = estimates.coarse.jacobian[:, 0]
    casj_fine = planar_casj[:, :, :2]
    casj_coarse = planar_casj_coarse[:, :, :2]
    angle_column = planar_casj[:, :, 2]
    fine_curvature = np.transpose(
        probes.fine_second[:, 0],
        (1, 2, 0),
    )
    curvature = fine_curvature

    current_support = history[-1]
    heldout_delta = np.asarray(
        [heldout_delta_xy[0], heldout_delta_xy[1], heldout_delta_angle],
        dtype=np.float64,
    )
    anchor_fingerprint_objects = [
        infer_planar_rigid_anchor(
            planar_casj[t],
            support_position=current_support.block_position,
            support_angle=current_support.block_angle,
            action_point=baseline[t],
        )
        for t in range(len(baseline))
    ]

    fresh_history = [
        state.shifted_block(heldout_delta[:2], heldout_delta[2]) for state in history
    ]
    fresh = policy_query.query(
        fresh_history,
        randomness=randomness,
    )

    casj_delta = np.einsum("tad,d->ta", planar_casj, heldout_delta)
    casj_repaired = baseline + casj_delta
    # Baseline global compensation knows the object's translation but has no
    # estimate of the policy's support-relative action anchor.
    global_comp = baseline + heldout_delta[:2][None, :]

    fine_fraction = (
        max(
            abs(float(heldout_delta[0])) / epsilon,
            abs(float(heldout_delta[1])) / epsilon,
            abs(float(heldout_delta[2])) / epsilon_angle,
            1.0,
        )
        ** -1
    )
    coarse_fraction = 2.0 * fine_fraction

    directional_fine = estimate_directional_curvature_via_query(
        policy_query=policy_query,
        history=history,
        baseline=baseline,
        randomness=randomness,
        support_delta=heldout_delta,
        step_fraction=fine_fraction,
    )
    directional_coarse = estimate_directional_curvature_via_query(
        policy_query=policy_query,
        history=history,
        baseline=baseline,
        randomness=randomness,
        support_delta=heldout_delta,
        step_fraction=coarse_fraction,
    )

    stale_mse = float(np.mean((baseline - fresh) ** 2))
    global_mse = float(np.mean((global_comp - fresh) ** 2))
    casj_mse = float(np.mean((casj_repaired - fresh) ** 2))
    anchor_transport_chunk = np.stack(
        [
            transport_planar_anchor(
                anchor_local=fingerprint.anchor_local,
                new_support_position=current_support.block_position + heldout_delta[:2],
                new_support_angle=current_support.block_angle + heldout_delta[2],
            )
            for fingerprint in anchor_fingerprint_objects
        ],
        axis=0,
    )
    anchor_transport_available = all(
        fingerprint.accepted for fingerprint in anchor_fingerprint_objects
    )
    anchor_transport_mse = (
        float(np.mean((anchor_transport_chunk - fresh) ** 2))
        if anchor_transport_available
        else None
    )

    identity = np.eye(2)
    jacobian_norm = np.linalg.norm(casj_fine, axis=(1, 2))
    identity_residual = np.linalg.norm(casj_fine - identity[None, :, :], axis=(1, 2))
    scale_drift = np.linalg.norm(casj_fine - casj_coarse, axis=(1, 2)) / np.maximum(
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
    reference_action_scale = np.maximum(
        np.linalg.norm(casj_delta, axis=1),
        1.0,
    )
    curvature_effect = second_order_norm / reference_action_scale
    first_effect = np.linalg.norm(casj_delta, axis=1)
    curvature_ratio = second_order_norm / np.maximum(first_effect, 1e-12)

    # PushT has one candidate physical support (the T block), so the 1x1 code
    # design is exactly identifiable for sparsity one.
    codes = np.ones((1, 1), dtype=np.float64)
    trust_regions = [
        certify_directional_trust_region(
            fine_second_per_unit2=directional_fine[t],
            coarse_second_per_unit2=directional_coarse[t],
            first_order_per_unit=casj_delta[t],
            reference_action_scale=max(float(np.linalg.norm(casj_delta[t])), 1.0),
        )
        for t in range(len(baseline))
    ]

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
            displacement_norm=1.0,
            first_order_effect=casj_delta[t],
            reference_action_scale=max(float(np.linalg.norm(casj_delta[t])), 1.0),
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
                fingerprint = anchor_fingerprint_objects[t]
                certified_chunk[t] = transport_planar_anchor(
                    anchor_local=fingerprint.anchor_local,
                    new_support_position=(
                        current_support.block_position + heldout_delta[:2]
                    ),
                    new_support_angle=(current_support.block_angle + heldout_delta[2]),
                )
            elif mode is RepairMode.LOCAL_LINEAR_REPAIR:
                certified_chunk[t] = casj_repaired[t]
            else:
                raise RuntimeError(f"unexpected certified mode {mode}")

    disturbed_state = history[-1].shifted_block(heldout_delta[:2], heldout_delta[2])
    rollout_branches = {
        "stale": execute_chunk_branch(env, start_state=disturbed_state, chunk=baseline),
        "global": execute_chunk_branch(
            env, start_state=disturbed_state, chunk=global_comp
        ),
        "casj_raw_linear": execute_chunk_branch(
            env, start_state=disturbed_state, chunk=casj_repaired
        ),
        "recovered_anchor_transport": (
            execute_chunk_branch(
                env,
                start_state=disturbed_state,
                chunk=anchor_transport_chunk,
            )
            if anchor_transport_available
            else {
                "valid": False,
                "reason": "one or more per-position anchor fingerprints were rejected",
            }
        ),
        "casj_certified": execute_chunk_branch(
            env, start_state=disturbed_state, chunk=certified_chunk
        ),
        "fresh": execute_chunk_branch(env, start_state=disturbed_state, chunk=fresh),
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
        "assay_status": "VALID_INPUT_PERTURBATION",
        "paired_randomness_replay_error": probes.pairing_error,
        "abi_probe_query_count": probes.query_count,
        "abi_fine_epsilon": probes.fine_epsilon.tolist(),
        "abi_coarse_epsilon": probes.coarse_epsilon.tolist(),
        "stale_mse": stale_mse,
        "global_compensation_mse": global_mse,
        "casj_repair_mse": casj_mse,
        "recovered_anchor_transport_mse": anchor_transport_mse,
        "casj_to_stale_ratio": casj_mse / max(stale_mse, 1e-12),
        "global_to_stale_ratio": global_mse / max(stale_mse, 1e-12),
        "jacobian_frobenius_norm": jacobian_norm.tolist(),
        "rigid_identity_residual": identity_residual.tolist(),
        "scale_drift": scale_drift.tolist(),
        "curvature_effect": curvature_effect.tolist(),
        "curvature_ratio": curvature_ratio.tolist(),
        "certificates": certificates,
        "directional_trust_regions": [
            {
                "accepted": trust.accepted,
                "radius_fraction_of_requested_motion": trust.radius,
                "curvature_scale_stability": trust.curvature_scale_stability,
                "first_order_norm_per_unit": trust.first_order_norm_per_unit,
                "curvature_norm_per_unit2": trust.curvature_norm_per_unit2,
                "radius_from_first_order": trust.radius_from_first_order,
                "radius_from_reference_scale": trust.radius_from_reference_scale,
                "limiting_constraint": trust.limiting_constraint,
                "reason": trust.reason,
            }
            for trust in trust_regions
        ],
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
        "recovered_anchor_transport_chunk": (
            anchor_transport_chunk.tolist() if anchor_transport_available else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override the checkpoint diffusion inference steps. Leave unset for the checkpoint default.",
    )
    parser.add_argument("--warmup-steps", type=int, nargs="+", default=[0, 4, 8, 12])
    parser.add_argument(
        "--epsilon",
        type=float,
        default=4.0,
        help="finite-difference block shift in pixels",
    )
    parser.add_argument(
        "--angle-epsilon",
        type=float,
        default=0.02,
        help="finite-difference block angle in radians",
    )
    parser.add_argument("--heldout-dx", type=float, default=10.0)
    parser.add_argument("--heldout-dy", type=float, default=-6.0)
    parser.add_argument(
        "--heldout-dtheta",
        type=float,
        default=0.35,
        help="held-out block rotation in radians; exact CASJ transport uses the recovered anchor",
    )
    parser.add_argument(
        "--heldout-pose",
        type=float,
        nargs=3,
        action="append",
        metavar=("DX", "DY", "DTHETA"),
        help=(
            "additional held-out block pose; repeat this option to evaluate a pose grid. "
            "If omitted, --heldout-dx/--heldout-dy/--heldout-dtheta define one pose."
        ),
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=5,
        help="independent environment reset seeds per warmup phase",
    )
    parser.add_argument("--output", type=Path, default=Path("casj_pusht_public.json"))
    args = parser.parse_args()

    if args.replicates <= 0:
        raise ValueError("--replicates must be positive")

    heldout_poses = args.heldout_pose or [
        [args.heldout_dx, args.heldout_dy, args.heldout_dtheta]
    ]

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
    # The official PushT checkpoint predates serialized processor JSON.
    # Rebuild the current DiffusionPolicy normalization pipeline from the
    # public lerobot/pusht dataset statistics without mutating the checkpoint.
    preprocessor, postprocessor = make_diffusion_pre_post_processors(
        policy.config,
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
        "environment_id": "gym-pusht/PushT-v0",
        "environment_package_version": package_version("gym-pusht"),
        "gymnasium_version": package_version("gymnasium"),
        "pymunk_version": package_version("pymunk"),
        "state_restore_protocol": "reset-fresh-space-exact-readback-v1",
        "seed": args.seed,
        "replicates_per_warmup": args.replicates,
        "checkpoint_num_inference_steps": checkpoint_inference_steps,
        "effective_num_inference_steps": effective_inference_steps,
        "uses_checkpoint_default_inference_steps": (
            effective_inference_steps == checkpoint_inference_steps
        ),
        "epsilon_pixels": args.epsilon,
        "heldout_delta_pixels": list(map(float, heldout_poses[0][:2])),
        "heldout_delta_angle_rad": float(heldout_poses[0][2]),
        "heldout_delta_pose": list(map(float, heldout_poses[0])),
        "angle_epsilon_rad": args.angle_epsilon,
        "heldout_poses": [list(map(float, pose)) for pose in heldout_poses],
        "snapshots": {},
        "summary": {},
    }

    for warmup in args.warmup_steps:
        for replicate in range(args.replicates):
            episode_seed = args.seed + replicate * 1_000_003
            history = rollout_to_snapshot(
                env=env,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                steps=warmup,
                seed=episode_seed,
                device=device,
            )
            if len(history) != policy.config.n_obs_steps:
                raise RuntimeError(
                    f"expected {policy.config.n_obs_steps} history states, got {len(history)}"
                )

            for pose_index, pose in enumerate(heldout_poses):
                delta = np.asarray(pose, dtype=np.float64)
                result = evaluate_snapshot(
                    env=env,
                    history=history,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    device=device,
                    seed=episode_seed + warmup * 100,
                    epsilon=args.epsilon,
                    epsilon_angle=args.angle_epsilon,
                    heldout_delta_xy=delta[:2],
                    heldout_delta_angle=float(delta[2]),
                )
                key = f"warmup={warmup}/replicate={replicate}/pose={pose_index}"
                results["snapshots"][key] = result
                summary_key = f"warmup={warmup}/pose={pose_index}"
                group = results["summary"].setdefault(
                    summary_key,
                    {
                        "warmup_steps": warmup,
                        "pose_index": pose_index,
                        "heldout_pose": delta.tolist(),
                        "replicates": [],
                    },
                )
                group["replicates"].append(
                    {
                        "replicate": replicate,
                        "seed": episode_seed,
                        "stale_mse": result["stale_mse"],
                        "global_compensation_mse": result["global_compensation_mse"],
                        "casj_repair_mse": result["casj_repair_mse"],
                        "recovered_anchor_transport_mse": result[
                            "recovered_anchor_transport_mse"
                        ],
                        "rollout_branches": {
                            name: {
                                "success": branch.get("success"),
                                "final_reward": branch.get("final_reward"),
                                "final_coverage": branch.get("final_coverage"),
                                "final_block_distance_to_fresh": branch.get(
                                    "final_block_distance_to_fresh"
                                ),
                            }
                            for name, branch in result["rollout_branches"].items()
                        },
                    }
                )

        for summary in results["summary"].values():
            if summary["warmup_steps"] != warmup:
                continue
            rows = summary["replicates"]
            scalar_fields = (
                "stale_mse",
                "global_compensation_mse",
                "casj_repair_mse",
                "recovered_anchor_transport_mse",
            )
            summary["n"] = len(rows)
            summary["metrics"] = {}
            for field in scalar_fields:
                values = np.asarray(
                    [row[field] for row in rows if row[field] is not None],
                    dtype=np.float64,
                )
                summary["metrics"][field] = {
                    "n": int(values.size),
                    "mean": float(np.mean(values)) if values.size else None,
                    "std": (float(np.std(values, ddof=1)) if values.size > 1 else None),
                    "median": float(np.median(values)) if values.size else None,
                }
            paired_gaps = [
                row["casj_repair_mse"] - row["global_compensation_mse"]
                for row in rows
                if row["casj_repair_mse"] is not None
                and row["global_compensation_mse"] is not None
            ]
            summary["paired_casj_minus_global_mse"] = {
                "n": len(paired_gaps),
                "mean": float(np.mean(paired_gaps)) if paired_gaps else None,
                "std": (
                    float(np.std(paired_gaps, ddof=1)) if len(paired_gaps) > 1 else None
                ),
                "values": [float(value) for value in paired_gaps],
            }
            branch_names = sorted(
                {name for row in rows for name in row["rollout_branches"]}
            )
            summary["rollout_metrics"] = {}
            for branch_name in branch_names:
                branch_rows = [
                    row["rollout_branches"][branch_name]
                    for row in rows
                    if branch_name in row["rollout_branches"]
                ]
                numeric = {}
                for field in (
                    "final_reward",
                    "final_coverage",
                    "final_block_distance_to_fresh",
                ):
                    values = np.asarray(
                        [row[field] for row in branch_rows if row[field] is not None],
                        dtype=np.float64,
                    )
                    numeric[field] = {
                        "n": int(values.size),
                        "mean": float(np.mean(values)) if values.size else None,
                        "std": (
                            float(np.std(values, ddof=1)) if values.size > 1 else None
                        ),
                        "median": float(np.median(values)) if values.size else None,
                    }
                successful = [
                    row["success"] for row in branch_rows if row["success"] is not None
                ]
                summary["rollout_metrics"][branch_name] = {
                    "n": len(branch_rows),
                    "success_rate": (
                        float(np.mean(np.asarray(successful, dtype=np.float64)))
                        if successful
                        else None
                    ),
                    "numeric": numeric,
                }

    env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()