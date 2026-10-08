"""Crash-stable, single-dispatch reservations for semantic diagnostic probes.

SQLite BEGIN IMMEDIATE serializes competing workers. The reservation is
durably committed *before* the caller attempts a physical probe. A crash,
timeout, or lost observation leaves the episode in RESERVED (ambiguous), not
automatically eligible for re-dispatch. A trusted controller may mark an
ambiguous episode ABORTED but must never reset its cursor to retry the effect.

The database and episode initializer are inside the trusted deployment
boundary. Sensor receipts are HMAC-authenticated under a trust-root secret
provisioned separately from the candidate repair generator. HMAC does not
prove physical ground truth, only possession of that sensor-side secret.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import hmac
from hashlib import sha256
from pathlib import Path
import sqlite3
from secrets import token_hex
from typing import Iterator

from .embodied_diagnostic_execution import (
    DiagnosticExecutionRejected,
    DiagnosticObservedStep,
    DiagnosticProbeAuthority,
    DiagnosticResolution,
    authorize_next_diagnostic_probe,
    diagnostic_tree_commitment,
)
from .embodied_semantic_experiment_design import SemanticExperimentPlan
from .embodied_robust_diagnosis import (
    RobustDiagnosisPlan,
    RobustDiagnosisRejected,
    RobustObservation,
    RobustProbe,
    RobustResolution,
    advance_robust_diagnosis,
)


SupportedDiagnosticPlan = SemanticExperimentPlan | RobustDiagnosisPlan


def _plan_commitment(plan: SupportedDiagnosticPlan) -> str:
    if isinstance(plan, RobustDiagnosisPlan):
        # Robust v2 hashes the complete source-bound prediction table and
        # all outcome branches as one frozen plan identity.
        return plan.digest
    return diagnostic_tree_commitment(plan)


def _next_decision(
    *, plan: SupportedDiagnosticPlan,
    trusted_problem_digest: str,
    trusted_tree_commitment: str,
    trace: tuple[DiagnosticObservedStep, ...] = (),
) -> DiagnosticProbeAuthority | DiagnosticResolution:
    if isinstance(plan, RobustDiagnosisPlan):
        if trusted_problem_digest != plan.digest or trusted_tree_commitment != plan.digest:
            raise DiagnosticExecutionRejected("robust plan does not match trusted frozen identity")
        try:
            decision = advance_robust_diagnosis(
                plan=plan,
                trusted_plan_digest=trusted_tree_commitment,
                observations=tuple(
                    RobustObservation(item.experiment_name, item.observation_signature)
                    for item in trace
                ),
            )
        except RobustDiagnosisRejected as exc:
            raise DiagnosticExecutionRejected(
                f"robust sensor-evidence/authority rejection: {exc}"
            ) from exc
        if isinstance(decision, RobustResolution):
            return DiagnosticResolution(
                hypotheses=decision.consistent_hypotheses,
                identified=len(decision.consistent_hypotheses) == 1,
                risk_consumed=decision.spent_risk,
                observations_used=len(trace),
            )
        return DiagnosticProbeAuthority(
            experiment=decision.experiment,
            plan_digest=plan.digest,
            tree_commitment=plan.digest,
            step_index=len(trace),
            risk_consumed=decision.spent_risk,
            risk_after_probe=decision.spent_risk + decision.experiment.risk,
            remaining_risk_after_probe=max(
                0.0, plan.risk_budget-decision.spent_risk-decision.experiment.risk
            ),
            hypotheses=decision.remaining_hypotheses,
        )
    return authorize_next_diagnostic_probe(
        plan=plan,
        trusted_problem_digest=trusted_problem_digest,
        trusted_tree_commitment=trusted_tree_commitment,
        trace=trace,
    )


def diagnostic_observation_mac(
    secret_key: bytes,
    *,
    episode_id: str,
    step_index: int,
    reservation_token: str,
    observation: DiagnosticObservedStep,
) -> str:
    """Authenticate a sensor receipt under a separately managed secret key.

    The key must be provisioned by the trusted execution/sensor control plane;
    the candidate repair generator must never obtain it.
    """
    if not isinstance(secret_key, bytes) or len(secret_key) < 32:
        raise ValueError("trusted diagnostic evidence key must be >=32 bytes")
    if not episode_id or step_index < 0 or not reservation_token:
        raise ValueError("diagnostic evidence identity is invalid")
    encoded = json.dumps(
        {
            "schema": "semrepair-diagnostic-observation-hmac-v1",
            "episode_id": episode_id,
            "step_index": step_index,
            "reservation_token": reservation_token,
            "experiment_name": observation.experiment_name,
            "signature": list(observation.observation_signature),
            "evidence_id": observation.evidence_id,
        },
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hmac.new(secret_key, encoded, sha256).hexdigest()


@dataclass(frozen=True)
class DiagnosticEpisodeSnapshot:
    episode_id: str
    status: str
    step_index: int
    risk_charged: float
    pending_experiment: str | None


class DiagnosticEpisodeStore:
    """An atomic, persistent dispatch cursor, never an actuator interface."""

    def __init__(self, path: str | Path, *, trusted_evidence_key: bytes):
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("a persistent SQLite path is required")
        if (
            not isinstance(trusted_evidence_key, bytes)
            or len(trusted_evidence_key) < 32
        ):
            raise ValueError("trusted evidence key must contain at least 32 bytes")
        self._trusted_evidence_key = trusted_evidence_key
        self._key_commitment = sha256(trusted_evidence_key).hexdigest()
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS diagnostic_episode (
                    episode_id TEXT PRIMARY KEY,
                    problem_digest TEXT NOT NULL,
                    tree_commitment TEXT NOT NULL,
                    key_commitment TEXT NOT NULL,
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

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # sqlite.Connection's native context manager commits/rolls back but
        # does not close the underlying descriptor. The robot host can keep
        # episodes for hours, so every operation must close deterministically.
        conn = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

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

    def _check_plan(self, row: sqlite3.Row, plan: SupportedDiagnosticPlan) -> None:
        if row["key_commitment"] != self._key_commitment:
            raise DiagnosticExecutionRejected(
                "sensor verification key changed since episode creation"
            )
        if (
            row["problem_digest"] != plan.digest
            or row["tree_commitment"] != _plan_commitment(plan)
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
        *, episode_id: str, plan: SupportedDiagnosticPlan,
        trusted_problem_digest: str, trusted_tree_commitment: str,
    ) -> None:
        if not episode_id:
            raise ValueError("episode_id must be nonempty")
        decision = _next_decision(
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
                    (?, ?, ?, ?, 'READY', '[]', NULL, NULL, 0.0, 0)""",
                    (
                        episode_id, plan.digest, trusted_tree_commitment,
                        self._key_commitment,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise DiagnosticExecutionRejected(
                    "duplicate episode identifier; no reset or replay"
                ) from exc

    def reserve_next(
        self, *, episode_id: str, plan: SupportedDiagnosticPlan,
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
                decision = _next_decision(
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
        self, *, episode_id: str, plan: SupportedDiagnosticPlan,
        observation: DiagnosticObservedStep,
        reservation_token: str,
        evidence_mac: str,
    ) -> DiagnosticEpisodeSnapshot:
        """Require an authenticated sensor receipt with frozen verification key.

        The caller cannot swap in a permissive verifier on a per-call basis.
        Trusted hardware/telemetry software must supply the HMAC separately.
        This proves origin under that key, not ground-truth physical effect.
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
                try:
                    expected_mac = diagnostic_observation_mac(
                        self._trusted_evidence_key,
                        episode_id=episode_id,
                        step_index=row["next_index"] - 1,
                        reservation_token=reservation_token,
                        observation=observation,
                    )
                except (TypeError, ValueError, OverflowError) as exc:
                    raise DiagnosticExecutionRejected(
                        "malformed physical observation receipt"
                    ) from exc
                if (
                    not isinstance(evidence_mac, str)
                    or not hmac.compare_digest(expected_mac, evidence_mac)
                ):
                    raise DiagnosticExecutionRejected(
                        "physical observation HMAC authentication failed"
                    )
                trace = self._trace(row)
                full_trace = trace + (observation,)
                decision = _next_decision(
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
            if row["key_commitment"] != self._key_commitment:
                raise DiagnosticExecutionRejected(
                    "sensor verification key changed since episode creation"
                )
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
