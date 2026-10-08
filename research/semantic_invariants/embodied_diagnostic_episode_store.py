"""Crash-stable, single-dispatch reservations for semantic diagnostic probes.

SQLite BEGIN IMMEDIATE serializes competing workers. The reservation is
durably committed *before* the caller attempts a physical probe. A crash,
timeout, or lost observation leaves the episode in RESERVED (ambiguous), not
automatically eligible for re-dispatch. A trusted controller may mark an
ambiguous episode ABORTED but must never reset its cursor to retry the effect.

The database and episode initializer are inside the trusted deployment
boundary. Sensor observations are NOT authenticated by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sqlite3
from secrets import token_hex
from typing import Callable

from .embodied_diagnostic_execution import (
    DiagnosticExecutionRejected,
    DiagnosticObservedStep,
    DiagnosticProbeAuthority,
    DiagnosticResolution,
    authorize_next_diagnostic_probe,
    diagnostic_tree_commitment,
)
from .embodied_semantic_experiment_design import SemanticExperimentPlan


@dataclass(frozen=True)
class DiagnosticEpisodeSnapshot:
    episode_id: str
    status: str
    step_index: int
    risk_charged: float
    pending_experiment: str | None


class DiagnosticEpisodeStore:
    """An atomic, persistent dispatch cursor, never an actuator interface."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("a persistent SQLite path is required")
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS diagnostic_episode (
                    episode_id TEXT PRIMARY KEY,
                    problem_digest TEXT NOT NULL,
                    tree_commitment TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('READY', 'RESERVED', 'DONE', 'ABORTED')
                    ),
                    trace_json TEXT NOT NULL,
                    pending_experiment TEXT,
                    pending_token TEXT,
                    risk_charged REAL NOT NULL,
                    next_index INTEGER NOT NULL
                )"""
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _trace(row: sqlite3.Row) -> tuple[DiagnosticObservedStep, ...]:
        return tuple(
            DiagnosticObservedStep(
                item["experiment_name"],
                tuple(item["observation_signature"]),
                item["evidence_id"],
            )
            for item in json.loads(row["trace_json"])
        )

    @staticmethod
    def _encode_trace(trace: tuple[DiagnosticObservedStep, ...]) -> str:
        return json.dumps(
            [
                {
                    "experiment_name": item.experiment_name,
                    "observation_signature": item.observation_signature,
                    "evidence_id": item.evidence_id,
                }
                for item in trace
            ],
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )

    @staticmethod
    def _check_plan(row: sqlite3.Row, plan: SemanticExperimentPlan) -> None:
        if (
            row["problem_digest"] != plan.digest
            or row["tree_commitment"] != diagnostic_tree_commitment(plan)
        ):
            raise DiagnosticExecutionRejected("trusted episode plan identity changed")

    @staticmethod
    def _row(conn: sqlite3.Connection, episode_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM diagnostic_episode WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            raise DiagnosticExecutionRejected("unknown diagnostic episode")
        return row

    def create_episode(
        self,
        *, episode_id: str, plan: SemanticExperimentPlan,
        trusted_problem_digest: str, trusted_tree_commitment: str,
    ) -> None:
        if not episode_id:
            raise ValueError("episode_id must be nonempty")
        decision = authorize_next_diagnostic_probe(
            plan=plan,
            trusted_problem_digest=trusted_problem_digest,
            trusted_tree_commitment=trusted_tree_commitment,
        )
        if not isinstance(decision, DiagnosticProbeAuthority):
            raise DiagnosticExecutionRejected("episode must start with a valid probe")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """INSERT INTO diagnostic_episode VALUES
                    (?, ?, ?, 'READY', '[]', NULL, NULL, 0.0, 0)""",
                    (episode_id, plan.digest, trusted_tree_commitment),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise DiagnosticExecutionRejected(
                    "duplicate episode identifier; no reset or replay"
                ) from exc

    def reserve_next(
        self, *, episode_id: str, plan: SemanticExperimentPlan,
    ) -> DiagnosticProbeAuthority:
        """Commit exclusive probe reservation before the physical side effect."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._row(conn, episode_id)
                self._check_plan(row, plan)
                if row["status"] != "READY":
                    raise DiagnosticExecutionRejected(
                        "episode has no unconsumed probe authorization"
                    )
                trace = self._trace(row)
                decision = authorize_next_diagnostic_probe(
                    plan=plan,
                    trusted_problem_digest=row["problem_digest"],
                    trusted_tree_commitment=row["tree_commitment"],
                    trace=trace,
                )
                if not isinstance(decision, DiagnosticProbeAuthority):
                    raise DiagnosticExecutionRejected("episode is already resolved")
                if decision.step_index != row["next_index"]:
                    raise DiagnosticExecutionRejected("diagnostic cursor has drifted")
                if abs(decision.risk_consumed - row["risk_charged"]) > 1e-9:
                    raise DiagnosticExecutionRejected("persistent risk charge drift")
                reservation_token = token_hex(32)
                conn.execute(
                    """UPDATE diagnostic_episode SET
                    status='RESERVED', pending_experiment=?, pending_token=?,
                    risk_charged=?, next_index=?
                    WHERE episode_id=? AND status='READY'""",
                    (
                        decision.experiment.name,
                        reservation_token,
                        decision.risk_after_probe,
                        decision.step_index + 1,
                        episode_id,
                    ),
                )
                conn.commit()
                return replace(decision, reservation_token=reservation_token)
            except Exception:
                conn.rollback()
                raise

    def commit_observation(
        self, *, episode_id: str, plan: SemanticExperimentPlan,
        observation: DiagnosticObservedStep,
        reservation_token: str,
        evidence_verifier: Callable[[str, int, str, DiagnosticObservedStep], bool],
    ) -> DiagnosticEpisodeSnapshot:
        """Require trusted evidence bound to the episode, step and nonce.

        The caller-supplied verifier must be a trusted independently configured
        sensor/effect authority, not a self-attestation function from the model
        proposing a repair. It runs under a write transaction to keep the
        claimed observation coupled to this exact reservation.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._row(conn, episode_id)
                self._check_plan(row, plan)
                if row["status"] != "RESERVED":
                    raise DiagnosticExecutionRejected("no outstanding probe reservation")
                if observation.experiment_name != row["pending_experiment"]:
                    raise DiagnosticExecutionRejected("observation does not match reservation")
                if (
                    not reservation_token
                    or reservation_token != row["pending_token"]
                ):
                    raise DiagnosticExecutionRejected(
                        "missing or stale physical dispatch reservation token"
                    )
                if not callable(evidence_verifier):
                    raise DiagnosticExecutionRejected(
                        "trusted evidence verification is mandatory"
                    )
                try:
                    is_trusted = evidence_verifier(
                        episode_id,
                        row["next_index"] - 1,
                        reservation_token,
                        observation,
                    )
                except Exception as exc:
                    raise DiagnosticExecutionRejected(
                        "trusted evidence verifier failed"
                    ) from exc
                if is_trusted is not True:
                    raise DiagnosticExecutionRejected(
                        "physical observation was not authenticated by the trusted verifier"
                    )
                trace = self._trace(row)
                full_trace = trace + (observation,)
                decision = authorize_next_diagnostic_probe(
                    plan=plan,
                    trusted_problem_digest=row["problem_digest"],
                    trusted_tree_commitment=row["tree_commitment"],
                    trace=full_trace,
                )
                next_status = "DONE" if isinstance(decision, DiagnosticResolution) else "READY"
                if len(full_trace) != row["next_index"]:
                    raise DiagnosticExecutionRejected("observation cursor drift")
                conn.execute(
                    """UPDATE diagnostic_episode SET status=?, trace_json=?,
                    pending_experiment=NULL, pending_token=NULL
                    WHERE episode_id=? AND status='RESERVED'""",
                    (next_status, self._encode_trace(full_trace), episode_id),
                )
                conn.commit()
                return DiagnosticEpisodeSnapshot(
                    episode_id, next_status, row["next_index"],
                    row["risk_charged"], None,
                )
            except Exception:
                conn.rollback()
                raise

    def abort(self, *, episode_id: str) -> None:
        """Preserve risk and prevent retry even if physical outcome is unknown."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, episode_id)
            if row["status"] not in ("READY", "RESERVED"):
                raise DiagnosticExecutionRejected("episode cannot be re-opened")
            conn.execute(
                "UPDATE diagnostic_episode SET status='ABORTED' WHERE episode_id=?",
                (episode_id,),
            )
            conn.commit()

    def snapshot(self, *, episode_id: str) -> DiagnosticEpisodeSnapshot:
        with self._connect() as conn:
            row = self._row(conn, episode_id)
            return DiagnosticEpisodeSnapshot(
                episode_id, row["status"], row["next_index"],
                row["risk_charged"], row["pending_experiment"],
            )
