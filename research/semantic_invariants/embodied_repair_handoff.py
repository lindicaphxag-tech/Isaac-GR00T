"""Source- and evidence-bound *one-time reservation* for a repair payload.

Security boundary:
  - This module never actuates a robot. It returns one opaque handoff token
    after committing a non-reissuable reservation in trusted SQLite.
  - A loss of connection after commit is ambiguous: NO automatic retry.
  - The payload must match the exact sha256 digest named by the robust
    diagnosis's already frozen concrete repair authority.
  - Every observation in the durable diagnosis is independently reverified
    against its original sensor HMAC before a handoff is recorded.
  - Guarantees require a trusted non-rollback SQLite database, trusted plan
    installation, uncompromised sensor HMAC key and exclusive trusted dispatch
    process. A receiver that duplicates a token is outside this model.
  - A byte hash says which bytes were approved, NOT that code is safe or
    that they were executed successfully. No execution-result claim exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import sqlite3
from secrets import token_hex

from .embodied_diagnostic_episode_store import (
    DiagnosticEpisodeStore, _next_decision,
)
from .embodied_diagnostic_execution import (
    DiagnosticExecutionRejected, DiagnosticResolution,
)
from .embodied_robust_diagnosis import RobustDiagnosisPlan


@dataclass(frozen=True)
class RepairHandoffReservation:
    """Opaque at-most-once issuance, never confirmation of physical execution."""

    episode_id: str
    plan_digest: str
    concrete_authority_id: str
    payload_sha256: str
    one_time_token: str
    status: str = "RESERVED_UNCONFIRMED"


@dataclass(frozen=True)
class RepairHandoffSnapshot:
    episode_id: str
    status: str
    plan_digest: str
    concrete_authority_id: str
    payload_sha256: str
    token_sha256: str


class RepairDispatchStore(DiagnosticEpisodeStore):
    """Atomically verify sensor-proven repair and reserve a single handoff.

    This is a *simulation-independent software authorization boundary*.
    The caller must still implement a trusted physical actuator consumer
    and an independently verifiable observed outcome. Reservation alone
    is not an actuator-side exactly-once guarantee.
    """

    def __init__(self, path, *, trusted_evidence_key: bytes):
        super().__init__(path, trusted_evidence_key=trusted_evidence_key)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS repair_handoff (
                    episode_id TEXT PRIMARY KEY,
                    plan_digest TEXT NOT NULL,
                    authority_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    token_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL
                      CHECK(status = 'RESERVED_UNCONFIRMED')
                )"""
            )
            conn.commit()

    def reserve_repair_once(
        self, *, episode_id: str, plan: RobustDiagnosisPlan,
        payload: bytes,
    ) -> RepairHandoffReservation:
        """Reserve a source-bound repair payload once, never retry after crash.

        A caller cannot choose an arbitrary authority identifier: the stored
        source-pinned plan must already resolve to the *actual* SHA256 digest
        of precisely these bytes. Empty, oversized or non-byte payloads
        and descriptive pseudo-sha labels fail closed.
        """
        if not isinstance(plan, RobustDiagnosisPlan):
            raise DiagnosticExecutionRejected(
                "repair handoff requires a robust source-bound plan"
            )
        if not isinstance(payload, bytes) or not 0 < len(payload) <= 16 * 1024 * 1024:
            raise DiagnosticExecutionRejected(
                "repair payload must be 1..16777216 bytes"
            )
        payload_digest = sha256(payload).hexdigest()
        authority_digest = f"sha256:{payload_digest}"

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._row(conn, episode_id)
                self._check_plan(row, plan)
                if (
                    row["status"] != "DONE"
                    or row["pending_token"] is not None
                    or row["pending_experiment"] is not None
                ):
                    raise DiagnosticExecutionRejected(
                        "repair handoff requires a finished signed diagnosis"
                    )
                # This check is deliberately *inside* the transaction: the
                # validated receipt trace and inserted handoff must have the
                # same durable database snapshot.
                trace = self._authenticated_trace(row=row, episode_id=episode_id)
                decision = _next_decision(
                    plan=plan,
                    trusted_problem_digest=row["problem_digest"],
                    trusted_tree_commitment=row["tree_commitment"],
                    trace=trace,
                )
                if (
                    not isinstance(decision, DiagnosticResolution)
                    or decision.authority_id != row["resolved_authority_id"]
                    or decision.observations_used != row["next_index"]
                    or abs(decision.risk_consumed - row["risk_charged"]) > 1e-9
                ):
                    raise DiagnosticExecutionRejected(
                        "repair handoff failed authoritative sensor trace replay"
                    )
                if not hmac.compare_digest(
                    str(decision.authority_id), authority_digest
                ):
                    raise DiagnosticExecutionRejected(
                        "repair payload bytes do not match authorized source artifact"
                    )
                token = token_hex(32)
                # Store a non-recoverable commitment, not the capability
                # itself; even a read-only database dump is not a dispatch
                # token. Repeat reservations are rejected by PRIMARY KEY.
                conn.execute(
                    """INSERT INTO repair_handoff (
                        episode_id, plan_digest, authority_id, payload_sha256,
                        token_sha256, status
                    ) VALUES (?, ?, ?, ?, ?, 'RESERVED_UNCONFIRMED')""",
                    (
                        episode_id, plan.digest, authority_digest,
                        payload_digest, sha256(token.encode("ascii")).hexdigest(),
                    ),
                )
                conn.commit()
                return RepairHandoffReservation(
                    episode_id=episode_id,
                    plan_digest=plan.digest,
                    concrete_authority_id=authority_digest,
                    payload_sha256=payload_digest,
                    one_time_token=token,
                )
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise DiagnosticExecutionRejected(
                    "repair handoff already reserved: never redispatch"
                ) from exc
            except Exception:
                conn.rollback()
                raise

    def handoff_snapshot(self, *, episode_id: str) -> RepairHandoffSnapshot:
        """Read-only status. Does NOT return or mint the one-time token."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM repair_handoff WHERE episode_id=?",
                (episode_id,),
            ).fetchone()
            if row is None:
                raise DiagnosticExecutionRejected(
                    "no repair handoff reservation exists"
                )
            return RepairHandoffSnapshot(
                episode_id=row["episode_id"],
                status=row["status"],
                plan_digest=row["plan_digest"],
                concrete_authority_id=row["authority_id"],
                payload_sha256=row["payload_sha256"],
                token_sha256=row["token_sha256"],
            )
