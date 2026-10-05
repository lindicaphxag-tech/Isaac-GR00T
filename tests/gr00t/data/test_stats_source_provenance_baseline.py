"""Baseline witnesses for stale stats cache reuse after source replacement.

These tests intentionally run against the unmodified upstream base.  They define
the desired cache-freshness behavior without importing any helper introduced by
the candidate fix.
"""

import json

import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import generate_rel_stats, generate_stats


EMBODIMENT = EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT


def _write_info(dataset):
    meta = dataset / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [17]},
                    "action": {"dtype": "float32", "shape": [17]},
                    "timestamp": {"dtype": "float32", "shape": [1]},
                }
            }
        )
    )


def _write_source(dataset, payload: bytes):
    shard = dataset / "data" / "chunk-000" / "episode_000000.parquet"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(payload)
    return shard


def _normal_stats():
    return {
        "mean": [0.0] * 17,
        "std": [1.0] * 17,
        "min": [-1.0] * 17,
        "max": [1.0] * 17,
        "q01": [-0.99] * 17,
        "q99": [0.99] * 17,
    }


def _relative_stats():
    return {
        "max": np.ones(9, dtype=np.float32),
        "min": -np.ones(9, dtype=np.float32),
        "q01": -np.ones(9, dtype=np.float32) * 0.99,
        "q99": np.ones(9, dtype=np.float32) * 0.99,
        "mean": np.zeros(9, dtype=np.float32),
        "std": np.ones(9, dtype=np.float32),
    }


def test_normal_stats_cache_invalidates_when_parquet_source_changes(tmp_path, monkeypatch):
    _write_info(tmp_path)
    shard = _write_source(tmp_path, b"source-v1")
    calls = []

    def fake(parquet_paths, features=None):
        calls.append(list(features or []))
        return {feature: _normal_stats() for feature in (features or [])}

    monkeypatch.setattr("gr00t.data.stats.calculate_dataset_statistics", fake)

    generate_stats(tmp_path)
    calls.clear()
    shard.write_bytes(b"source-version-two")

    generate_stats(tmp_path)

    assert calls, "changed parquet source must invalidate cached normal statistics"


def test_relative_stats_cache_invalidates_when_parquet_source_changes(tmp_path, monkeypatch):
    (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
    shard = _write_source(tmp_path, b"source-v1")
    calls = []

    def fake(dataset_path, embodiment_tag, group_key, max_episodes=-1):
        calls.append(group_key)
        return _relative_stats()

    monkeypatch.setattr("gr00t.data.stats.calculate_stats_for_key", fake)

    generate_rel_stats(tmp_path, EMBODIMENT)
    calls.clear()
    shard.write_bytes(b"source-version-two")

    generate_rel_stats(tmp_path, EMBODIMENT)

    assert calls, "changed parquet source must invalidate cached relative statistics"