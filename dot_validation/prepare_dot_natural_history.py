from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset


ENV_KEY = "observation.environment_state"
STATE_KEY = "observation.state"
EPISODE_KEY = "episode_index"
FRAME_KEY = "frame_index"


def array_sha256(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def select_history(table, *, episode_index: int, target_frame: int, offsets: tuple[int, ...]):
    rows = {}
    wanted = {target_frame + offset for offset in offsets}
    for row in table:
        if int(row[EPISODE_KEY]) != episode_index:
            continue
        frame = int(row[FRAME_KEY])
        if frame in wanted:
            rows[frame] = row
        if len(rows) == len(wanted):
            break

    missing = sorted(wanted - set(rows))
    if missing:
        raise ValueError(
            f"episode {episode_index} is missing required frames {missing} "
            f"for target {target_frame} and offsets {offsets}"
        )

    ordered = [rows[target_frame + offset] for offset in offsets]
    environment_history = np.asarray([r[ENV_KEY] for r in ordered], dtype=np.float32)
    agent_history = np.asarray([r[STATE_KEY] for r in ordered], dtype=np.float32)
    return environment_history, agent_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--target-frame", type=int, default=20)
    args = parser.parse_args()

    freeze = json.loads(args.revision_freeze.read_text())
    dataset_id = freeze["dataset_id"]
    dataset_revision = freeze["dataset_revision"]

    # Pinned DOT source 42cca... uses an 11-frame inference queue and returns
    # the oldest buffered observation plus the two newest observations when
    # n_obs_steps=3: [t-10, t-1, t].
    offsets = (-10, -1, 0)

    ds = load_dataset(dataset_id, split="train", revision=dataset_revision)
    environment_history, agent_history = select_history(
        ds,
        episode_index=args.episode,
        target_frame=args.target_frame,
        offsets=offsets,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz = args.output_dir / "history.npz"
    np.savez_compressed(
        npz,
        environment_history=environment_history,
        agent_history=agent_history,
        history_offsets=np.asarray(offsets, dtype=np.int64),
        episode_index=np.asarray(args.episode, dtype=np.int64),
        target_frame=np.asarray(args.target_frame, dtype=np.int64),
    )

    meta = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "episode_index": args.episode,
        "target_frame": args.target_frame,
        "history_offsets": list(offsets),
        "history_sha256": array_sha256(environment_history, agent_history),
        "environment_shape": list(environment_history.shape),
        "agent_shape": list(agent_history.shape),
        "semantics": "pinned DOT inference history: [t-10, t-1, t]",
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
