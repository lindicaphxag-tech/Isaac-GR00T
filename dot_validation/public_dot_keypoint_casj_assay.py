"""Public frozen DOT PushT-keypoints CASJ assay.

Requires the DOT-compatible LeRobot branch documented in
PUBLIC_DOT_KEYPOINT_CASJ_PROTOCOL_V0_1.md and access to the public checkpoint.

This script does not train the base policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ENV_KEY = "observation.environment_state"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def translate_keypoints(environment_state: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    """Rigidly translate all eight PushT block keypoints."""
    state = np.asarray(environment_state, dtype=np.float32)
    delta = np.asarray(delta_xy, dtype=np.float32)
    if state.shape[-1] != 16:
        raise ValueError("PushT environment_state must end in 16 values")
    if delta.shape != (2,):
        raise ValueError("delta_xy must have shape [2]")
    out = state.copy()
    points = out.reshape(*out.shape[:-1], 8, 2)
    points += delta
    return out


def rotate_keypoints(environment_state: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rigidly rotate all eight T-block keypoints about their centroid."""
    state = np.asarray(environment_state, dtype=np.float32)
    if state.shape[-1] != 16:
        raise ValueError("PushT environment_state must end in 16 values")
    out = state.copy()
    points = out.reshape(*out.shape[:-1], 8, 2)
    center = points.mean(axis=-2, keepdims=True)
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float32,
    )
    points[...] = (points - center) @ rotation.T + center
    return out


def transform_keypoints_se2(
    environment_state: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """Apply [dx, dy, dtheta] rigid motion about the block centroid."""
    delta = np.asarray(delta, dtype=np.float32)
    if delta.shape != (3,):
        raise ValueError("SE(2) support delta must have shape [3]")
    rotated = rotate_keypoints(environment_state, float(delta[2]))
    return translate_keypoints(rotated, delta[:2])


def block_centroid(environment_state: np.ndarray) -> np.ndarray:
    state = np.asarray(environment_state, dtype=np.float32)
    if state.shape[-1] != 16:
        raise ValueError("PushT environment_state must end in 16 values")
    return state.reshape(*state.shape[:-1], 8, 2).mean(axis=-2)


def intervene_block_pose_history(
    environment_history: np.ndarray,
    delta: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Apply an SE(2) block-pose intervention with explicit history semantics."""
    history = np.asarray(environment_history, dtype=np.float32)
    if history.ndim != 2 or history.shape[1] != 16:
        raise ValueError("environment_history must have shape [T,16]")
    if mode == "current_only":
        out = history.copy()
        out[-1] = transform_keypoints_se2(out[-1], delta)
        return out
    if mode == "all_history":
        return transform_keypoints_se2(history, delta)
    raise ValueError(f"unknown history intervention mode {mode!r}")


def transform_action_points_se2(
    actions: np.ndarray,
    *,
    center: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """Finite rigid-point compensation baseline for 2D action targets."""
    actions = np.asarray(actions, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("actions must have shape [H,2]")
    if center.shape != (2,) or delta.shape != (3,):
        raise ValueError("center must be [2] and delta must be [3]")
    cosine = np.cos(delta[2])
    sine = np.sin(delta[2])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return (actions - center) @ rotation.T + center + delta[:2]


def rigid_point_se2_jacobian(actions: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Ideal 2x3 infinitesimal Jacobian for a point rigidly following the block."""
    actions = np.asarray(actions, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    relative = actions - center
    out = np.zeros((len(actions), 2, 3), dtype=np.float64)
    out[:, :, :2] = np.eye(2)[None]
    out[:, 0, 2] = -relative[:, 1]
    out[:, 1, 2] = relative[:, 0]
    return out


def intervene_block_history(
    environment_history: np.ndarray,
    delta_xy: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Apply a support intervention without silently rewriting history.

    current_only is the runtime CASJ estimand: past observations remain factual
    and only the current support pose changes.

    all_history is retained as a separate trajectory-level counterfactual
    control in which the block is shifted in every conditioning frame.
    """
    history = np.asarray(environment_history, dtype=np.float32)
    if history.ndim != 2 or history.shape[1] != 16:
        raise ValueError("environment_history must have shape [T,16]")
    if mode == "current_only":
        out = history.copy()
        out[-1] = translate_keypoints(out[-1], delta_xy)
        return out
    if mode == "all_history":
        return translate_keypoints(history, delta_xy)
    raise ValueError(f"unknown history intervention mode {mode!r}")


def intervene_agent_history(
    agent_history: np.ndarray,
    delta_xy: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    history = np.asarray(agent_history, dtype=np.float32)
    if history.ndim != 2 or history.shape[1] != 2:
        raise ValueError("agent_history must have shape [T,2]")
    if mode == "current_only":
        out = history.copy()
        out[-1] = shift_agent(out[-1], delta_xy)
        return out
    if mode == "all_history":
        return shift_agent(history, delta_xy)
    raise ValueError(f"unknown history intervention mode {mode!r}")


def intervene_deformation_history(
    environment_history: np.ndarray,
    delta_xy: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    history = np.asarray(environment_history, dtype=np.float32)
    if history.ndim != 2 or history.shape[1] != 16:
        raise ValueError("environment_history must have shape [T,16]")
    if mode == "current_only":
        out = history.copy()
        out[-1] = deform_keypoints(out[-1], delta_xy)
        return out
    if mode == "all_history":
        return deform_keypoints(history, delta_xy)
    raise ValueError(f"unknown history intervention mode {mode!r}")


def shift_agent(agent_state: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    """Translate only the robot/agent state, leaving the T-block unchanged."""
    state = np.asarray(agent_state, dtype=np.float32)
    delta = np.asarray(delta_xy, dtype=np.float32)
    if state.shape[-1] != 2 or delta.shape != (2,):
        raise ValueError("agent state and delta must end in shape [2]")
    return state + delta


def deform_keypoints(
    environment_state: np.ndarray,
    delta_xy: np.ndarray,
) -> np.ndarray:
    """Apply a zero-mean non-rigid keypoint deformation as a negative control.

    Alternating keypoints move in opposite directions, so the T-block centroid
    is unchanged. This is deliberately not a valid rigid support transform.
    """
    state = np.asarray(environment_state, dtype=np.float32)
    delta = np.asarray(delta_xy, dtype=np.float32)
    if state.shape[-1] != 16 or delta.shape != (2,):
        raise ValueError("environment state must end in 16 values and delta in 2")
    out = state.copy()
    points = out.reshape(*out.shape[:-1], 8, 2)
    signs = np.where(np.arange(8) % 2 == 0, 1.0, -1.0).astype(np.float32)
    points += signs.reshape(*([1] * (points.ndim - 2)), 8, 1) * delta
    return out


def load_model_history(args, n_obs: int):
    """Load publication-grade model history or build an explicit smoke input."""
    if args.history_npz is not None:
        payload = np.load(args.history_npz)
        environment_history = np.asarray(
            payload["environment_history"],
            dtype=np.float32,
        )
        agent_history = np.asarray(
            payload["agent_history"],
            dtype=np.float32,
        )
        if "history_offsets" not in payload:
            raise ValueError(
                "publication-grade DOT history must record history_offsets"
            )
        history_offsets = tuple(
            int(v) for v in np.asarray(payload["history_offsets"]).reshape(-1)
        )
        expected_offsets = (-10, -1, 0)
        if history_offsets != expected_offsets:
            raise ValueError(
                f"DOT runtime history must use offsets {expected_offsets}, "
                f"got {history_offsets}"
            )
        if environment_history.shape != (n_obs, 16):
            raise ValueError(
                "history_npz environment_history must have shape "
                f"{(n_obs, 16)}, got {environment_history.shape}"
            )
        if agent_history.shape != (n_obs, 2):
            raise ValueError(
                "history_npz agent_history must have shape "
                f"{(n_obs, 2)}, got {agent_history.shape}"
            )
        return environment_history, agent_history, "natural_history"

    if args.environment_state is None or args.agent_state is None:
        raise ValueError(
            "provide --history-npz for publication-grade input, or both "
            "--environment-state and --agent-state for repeated-snapshot smoke mode"
        )

    environment_history = np.repeat(
        np.asarray(args.environment_state, dtype=np.float32)[None],
        n_obs,
        axis=0,
    )
    agent_history = np.repeat(
        np.asarray(args.agent_state, dtype=np.float32)[None],
        n_obs,
        axis=0,
    )
    return environment_history, agent_history, "repeated_snapshot_smoke"


@torch.no_grad()
def query_raw_dot_chunk(policy, environment_history, agent_history, device):
    """Query the raw frozen DOT chunk without rolling action ensembling."""
    environment_history = np.asarray(environment_history, dtype=np.float32)
    agent_history = np.asarray(agent_history, dtype=np.float32)

    if environment_history.ndim != 2 or environment_history.shape[1] != 16:
        raise ValueError("environment_history must have shape [T,16]")
    if agent_history.shape != (len(environment_history), 2):
        raise ValueError("agent_history must have shape [T,2]")

    expected = int(policy.config.n_obs_steps)
    if len(environment_history) != expected:
        raise ValueError(
            f"checkpoint expects n_obs_steps={expected}, got {len(environment_history)}"
        )

    batch = {
        ENV_KEY: torch.as_tensor(environment_history[None], device=device),
        STATE_KEY: torch.as_tensor(agent_history[None], device=device),
    }
    batch = policy.normalize_inputs(batch)
    prediction = policy.model(batch)[:, -policy.config.inference_horizon :]
    prediction = policy.unnormalize_outputs({ACTION_KEY: prediction})[ACTION_KEY]
    return prediction[0].detach().cpu().numpy()


def central_se2_casj(
    policy,
    environment_history,
    agent_history,
    *,
    translation_epsilon,
    rotation_epsilon_rad,
    device,
    history_intervention,
):
    baseline = query_raw_dot_chunk(
        policy,
        environment_history,
        agent_history,
        device,
    )
    horizon = len(baseline)
    jacobian = np.zeros((horizon, 2, 3), dtype=np.float64)
    curvature = np.zeros((horizon, 2, 3), dtype=np.float64)

    epsilons = (
        float(translation_epsilon),
        float(translation_epsilon),
        float(rotation_epsilon_rad),
    )
    for q, epsilon_q in enumerate(epsilons):
        if epsilon_q <= 0:
            raise ValueError("all SE(2) intervention radii must be positive")
        direction = np.zeros(3, dtype=np.float32)
        direction[q] = 1.0
        plus = query_raw_dot_chunk(
            policy,
            intervene_block_pose_history(
                environment_history,
                epsilon_q * direction,
                mode=history_intervention,
            ),
            agent_history,
            device,
        )
        minus = query_raw_dot_chunk(
            policy,
            intervene_block_pose_history(
                environment_history,
                -epsilon_q * direction,
                mode=history_intervention,
            ),
            agent_history,
            device,
        )
        jacobian[:, :, q] = (plus - minus) / (2.0 * epsilon_q)
        curvature[:, :, q] = (
            plus + minus - 2.0 * baseline
        ) / (epsilon_q * epsilon_q)

    return baseline, jacobian, curvature


def central_control_jacobian(
    policy,
    environment_history,
    agent_history,
    *,
    epsilon,
    device,
    intervention,
    history_intervention,
):
    """Central derivative for a named non-support control intervention."""
    baseline = query_raw_dot_chunk(
        policy,
        environment_history,
        agent_history,
        device,
    )
    horizon = len(baseline)
    jacobian = np.zeros((horizon, 2, 2), dtype=np.float64)

    for q in range(2):
        direction = np.zeros(2, dtype=np.float32)
        direction[q] = 1.0

        if intervention == "agent":
            plus_env = environment_history
            minus_env = environment_history
            plus_agent = intervene_agent_history(
                agent_history,
                epsilon * direction,
                mode=history_intervention,
            )
            minus_agent = intervene_agent_history(
                agent_history,
                -epsilon * direction,
                mode=history_intervention,
            )
        elif intervention == "deformation":
            plus_env = intervene_deformation_history(
                environment_history,
                epsilon * direction,
                mode=history_intervention,
            )
            minus_env = intervene_deformation_history(
                environment_history,
                -epsilon * direction,
                mode=history_intervention,
            )
            plus_agent = minus_agent = agent_history
        else:
            raise ValueError(f"unknown intervention {intervention!r}")

        plus = query_raw_dot_chunk(
            policy, plus_env, plus_agent, device
        )
        minus = query_raw_dot_chunk(
            policy, minus_env, minus_agent, device
        )
        jacobian[:, :, q] = (plus - minus) / (2.0 * epsilon)

    return jacobian


def summarize(
    baseline,
    fine,
    coarse,
    curvature,
    fresh,
    delta_test,
    *,
    block_center,
    agent_jacobian,
    deformation_jacobian,
):
    global_comp = transform_action_points_se2(
        baseline,
        center=block_center,
        delta=delta_test,
    )
    casj_repair = baseline + np.einsum("hdo,o->hd", fine, delta_test)

    stale_mse = float(np.mean((baseline - fresh) ** 2))
    global_mse = float(np.mean((global_comp - fresh) ** 2))
    casj_mse = float(np.mean((casj_repair - fresh) ** 2))

    expected_rigid = rigid_point_se2_jacobian(baseline, block_center)
    norm = np.linalg.norm(fine, axis=(1, 2))
    rigid_residual = np.linalg.norm(fine - expected_rigid, axis=(1, 2))
    scale_drift = np.linalg.norm(fine - coarse, axis=(1, 2)) / np.maximum(
        np.linalg.norm(fine, axis=(1, 2)),
        1e-12,
    )
    curvature_norm = np.linalg.norm(curvature, axis=(1, 2))

    return {
        "stale_vs_fresh_mse": stale_mse,
        "global_vs_fresh_mse": global_mse,
        "casj_vs_fresh_mse": casj_mse,
        "casj_to_stale_error_ratio": casj_mse / max(stale_mse, 1e-12),
        "casj_to_global_error_ratio": casj_mse / max(global_mse, 1e-12),
        "per_step_casj_norm": norm.tolist(),
        "per_step_identity_residual": rigid_residual.tolist(),
        "per_step_scale_drift": scale_drift.tolist(),
        "per_step_curvature_norm": curvature_norm.tolist(),
        "per_step_agent_state_jacobian_norm": np.linalg.norm(
            agent_jacobian, axis=(1, 2)
        ).tolist(),
        "per_step_nonrigid_deformation_jacobian_norm": np.linalg.norm(
            deformation_jacobian, axis=(1, 2)
        ).tolist(),
    }


def history_sha256(environment_history: np.ndarray, agent_history: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(environment_history).tobytes())
    digest.update(np.ascontiguousarray(agent_history).tobytes())
    return digest.hexdigest()


def save_evidence_bundle(
    output_dir: str,
    *,
    summary: dict,
    environment_history: np.ndarray,
    agent_history: np.ndarray,
    baseline: np.ndarray,
    fine: np.ndarray,
    coarse: np.ndarray,
    curvature: np.ndarray,
    fresh: np.ndarray,
    delta_test: np.ndarray,
    block_center: np.ndarray,
    agent_jacobian: np.ndarray,
    deformation_jacobian: np.ndarray,
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    global_comp = transform_action_points_se2(
        baseline,
        center=block_center,
        delta=delta_test,
    )
    casj_repair = baseline + np.einsum("hdo,o->hd", fine, delta_test)

    np.savez_compressed(
        directory / "traces.npz",
        environment_history=environment_history,
        agent_history=agent_history,
        baseline=baseline,
        fine_casj=fine,
        coarse_casj=coarse,
        curvature=curvature,
        heldout_fresh=fresh,
        heldout_delta_se2=delta_test,
        rigid_global_compensation=global_comp,
        casj_repair=casj_repair,
        block_center=block_center,
        agent_jacobian=agent_jacobian,
        nonrigid_deformation_jacobian=deformation_jacobian,
    )
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="IliaLarchenko/dot_pusht_keypoints_best",
    )
    parser.add_argument(
        "--checkpoint-revision",
        default=None,
        help=(
            "Exact Hugging Face commit/tag passed to from_pretrained. "
            "Required for publication-grade evidence."
        ),
    )
    parser.add_argument(
        "--dot-source-revision",
        default="42cca283322bbb68b6d3f1b4a436ada5b5d3393a",
        help="Expected IliaLarchenko/lerobot source revision for the DOT adapter.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for summary.json and raw traces.npz.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--translation-epsilon",
        type=float,
        default=4.0,
        help="Central-difference radius in PushT position units.",
    )
    parser.add_argument(
        "--rotation-epsilon-deg",
        type=float,
        default=2.0,
        help="Central-difference radius for T-block orientation.",
    )
    parser.add_argument("--heldout-dx", type=float, default=10.0)
    parser.add_argument("--heldout-dy", type=float, default=-6.0)
    parser.add_argument("--heldout-dtheta-deg", type=float, default=5.0)
    parser.add_argument(
        "--history-intervention",
        choices=["current_only", "all_history"],
        default="current_only",
        help=(
            "current_only is the runtime CASJ estimand and freezes past "
            "observations; all_history is a trajectory-level control."
        ),
    )
    parser.add_argument(
        "--history-npz",
        type=str,
        default=None,
        help=(
            "Publication-grade input containing environment_history [T,16] "
            "and agent_history [T,2] in the exact DOT model-history order."
        ),
    )
    parser.add_argument(
        "--environment-state",
        type=float,
        nargs=16,
        default=None,
        help="Smoke mode only: one T-block state repeated across history.",
    )
    parser.add_argument(
        "--agent-state",
        type=float,
        nargs=2,
        default=None,
        help="Smoke mode only: one agent state repeated across history.",
    )
    args = parser.parse_args()

    from lerobot.common.policies.dot.modeling_dot import DOTPolicy

    device = torch.device(args.device)
    policy = DOTPolicy.from_pretrained(
        args.checkpoint,
        revision=args.checkpoint_revision,
        map_location=args.device,
    ).to(device).eval()

    n_obs = int(policy.config.n_obs_steps)
    environment_history, agent_history, history_mode = load_model_history(
        args,
        n_obs,
    )

    baseline, fine, curvature = central_se2_casj(
        policy,
        environment_history,
        agent_history,
        translation_epsilon=args.translation_epsilon,
        rotation_epsilon_rad=np.deg2rad(args.rotation_epsilon_deg),
        device=device,
        history_intervention=args.history_intervention,
    )
    _, coarse, _ = central_se2_casj(
        policy,
        environment_history,
        agent_history,
        translation_epsilon=2.0 * args.translation_epsilon,
        rotation_epsilon_rad=2.0 * np.deg2rad(args.rotation_epsilon_deg),
        device=device,
        history_intervention=args.history_intervention,
    )

    agent_jacobian = central_control_jacobian(
        policy,
        environment_history,
        agent_history,
        epsilon=args.translation_epsilon,
        device=device,
        intervention="agent",
        history_intervention=args.history_intervention,
    )
    deformation_jacobian = central_control_jacobian(
        policy,
        environment_history,
        agent_history,
        epsilon=args.translation_epsilon,
        device=device,
        intervention="deformation",
        history_intervention=args.history_intervention,
    )

    delta_test = np.asarray(
        [
            args.heldout_dx,
            args.heldout_dy,
            np.deg2rad(args.heldout_dtheta_deg),
        ],
        dtype=np.float32,
    )
    fresh = query_raw_dot_chunk(
        policy,
        intervene_block_pose_history(
            environment_history,
            delta_test,
            mode=args.history_intervention,
        ),
        agent_history,
        device,
    )

    center = block_centroid(environment_history[-1])
    history_hash = history_sha256(environment_history, agent_history)
    publication_grade = (
        history_mode == "natural_history"
        and args.history_intervention == "current_only"
        and args.checkpoint_revision is not None
    )

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.checkpoint_revision,
        "dot_source_revision": args.dot_source_revision,
        "history_sha256": history_hash,
        "n_obs_steps": n_obs,
        "history_mode": history_mode,
        "history_intervention": args.history_intervention,
        "runtime_estimand": args.history_intervention == "current_only",
        "publication_grade_history": history_mode == "natural_history",
        "publication_grade_configuration": publication_grade,
        "translation_epsilon": args.translation_epsilon,
        "rotation_epsilon_deg": args.rotation_epsilon_deg,
        "heldout_delta_se2": delta_test.tolist(),
        **summarize(
            baseline,
            fine,
            coarse,
            curvature,
            fresh,
            delta_test,
            block_center=center,
            agent_jacobian=agent_jacobian,
            deformation_jacobian=deformation_jacobian,
        ),
    }
    if args.output_dir is not None:
        save_evidence_bundle(
            args.output_dir,
            summary=result,
            environment_history=environment_history,
            agent_history=agent_history,
            baseline=baseline,
            fine=fine,
            coarse=coarse,
            curvature=curvature,
            fresh=fresh,
            delta_test=delta_test,
            block_center=center,
            agent_jacobian=agent_jacobian,
            deformation_jacobian=deformation_jacobian,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
