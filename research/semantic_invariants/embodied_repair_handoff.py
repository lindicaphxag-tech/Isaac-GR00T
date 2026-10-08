"""Source- and evidence-bound *one-time reservation* for a repair payload.

Security boundary:
  - This module never actuates a robot. It returns one opaque handoff token
    after committing a non-reissuable reservation in trusted SQLite.
  - A loss of connection after commit is ambiguous: NO automatic retry.
  - The payload must match the exact sha256 digest named by the robust
    diagnosis's already frozen concrete repair authority.
  - Every observation in the durable diagnosis is independently reverified
    against its original sensor HMAC before a handoff is recorded.
  - SQLite-only writers cannot mint tokens or reset claimed status: the
    handoff row (INCLUDING its state) is MAC-sealed under a SEPARATE
    deployment-provisioned 32+ byte dispatch secret, not the sensor key.
  - Guarantees still require a non-rollback SQLite database, trusted plan
    bootstrap and secret provisioning. Replacing the ENTIRE database with
    an older signed snapshot is outside this local verifier's ability.
  - A receiver that duplicates an actual external effect is out of scope.
  - A byte hash says which bytes were approved, NOT that code is safe or
    that they were executed successfully. No execution-result claim exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
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


@dataclass(frozen=True)
class RepairExecutionClaim:
    """One trusted consumer claimed the pending repair; NOT a robot result."""

    episode_id: str
    plan_digest: str
    concrete_authority_id: str
    payload_sha256: str
    status: str = "CLAIMED_UNCONFIRMED"


class RepairDispatchStore(DiagnosticEpisodeStore):
    """Atomically verify sensor-proven repair and reserve a single handoff.

    This is a *simulation-independent software authorization boundary*.
    The caller must still implement a trusted physical actuator consumer
    and an independently verifiable observed outcome. Reservation alone
    is not an actuator-side exactly-once guarantee.
    """

    def __init__(
        self, path, *, trusted_evidence_key: bytes,
        trusted_dispatch_key: bytes,
    ):
        if (
            not isinstance(trusted_dispatch_key, bytes)
            or len(trusted_dispatch_key) < 32
            or trusted_dispatch_key == trusted_evidence_key
        ):
            raise ValueError(
                "independently provisioned dispatch MAC key must be "
                "32+ bytes and distinct from sensor verification key"
            )
        self._trusted_dispatch_key = trusted_dispatch_key
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
                      CHECK(status IN (
                        'RESERVED_UNCONFIRMED', 'CLAIMED_UNCONFIRMED'
                      )),
                    handoff_receipt_mac TEXT
                )"""
            )
            # Migrate old unsigned handoff rows without granting them a
            # retrospectively fabricated MAC. All old rows fail closed.
            names = {
                row["name"] for row in conn.execute(
                    "PRAGMA table_info(repair_handoff)"
                )
            }
            if "handoff_receipt_mac" not in names:
                conn.execute(
                    "ALTER TABLE repair_handoff ADD COLUMN handoff_receipt_mac TEXT"
                )
            conn.commit()

    def _handoff_mac(
        self, *, episode_id: str, plan_digest: str,
        authority_id: str, payload_sha256: str, token_sha256: str,
        status: str,
    ) -> str:
        encoded = json.dumps(
            {
                "schema": "semrepair-source-pinned-repair-handoff-hmac-v1",
                "episode_id": episode_id,
                "plan_digest": plan_digest,
                "authority_id": authority_id,
                "payload_sha256": payload_sha256,
                "token_sha256": token_sha256,
                "status": status,
            },
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return hmac.new(
            self._trusted_dispatch_key, encoded, sha256
        ).hexdigest()

    def _verify_handoff_row(self, row: sqlite3.Row) -> None:
        signature = row["handoff_receipt_mac"]
        if not isinstance(signature, str) or len(signature) != 64:
            raise DiagnosticExecutionRejected(
                "missing dispatch integrity MAC; legacy handoff is untrusted"
            )
        expected = self._handoff_mac(
            episode_id=row["episode_id"],
            plan_digest=row["plan_digest"],
            authority_id=row["authority_id"],
            payload_sha256=row["payload_sha256"],
            token_sha256=row["token_sha256"],
            status=row["status"],
        )
        if not hmac.compare_digest(signature, expected):
            raise DiagnosticExecutionRejected(
                "repair handoff integrity MAC mismatch: tampered database row"
            )

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
                token_commitment = sha256(
                    token.encode("ascii")
                ).hexdigest()
                status = "RESERVED_UNCONFIRMED"
                signed_row = self._handoff_mac(
                    episode_id=episode_id,
                    plan_digest=plan.digest,
                    authority_id=authority_digest,
                    payload_sha256=payload_digest,
                    token_sha256=token_commitment,
                    status=status,
                )
                conn.execute(
                    """INSERT INTO repair_handoff (
                        episode_id, plan_digest, authority_id, payload_sha256,
                        token_sha256, status, handoff_receipt_mac
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        episode_id, plan.digest, authority_digest,
                        payload_digest, token_commitment, status, signed_row,
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

    def consume_repair_token_once(
        self, *, episode_id: str, plan: RobustDiagnosisPlan,
        payload: bytes, one_time_token: str,
    ) -> RepairExecutionClaim:
        """Atomically consume the handoff token *before* an external effect.

        Trusted executors must invoke this operation immediately before
        attempting their own physical action. Once claimed, success and
        failure are BOTH ambiguous here and never authorize retry.
        Without actuator-side durable fencing, this is only one trusted
        claim, NOT exactly-once physical effect.
        """
        if not isinstance(plan, RobustDiagnosisPlan):
            raise DiagnosticExecutionRejected(
                "only robust diagnosis authorizes a concrete repair claim"
            )
        if not isinstance(payload, bytes) or not 0 < len(payload) <= 16 * 1024 * 1024:
            raise DiagnosticExecutionRejected(
                "repair claim requires the exact nonempty payload bytes"
            )
        if (
            not isinstance(one_time_token, str)
            or len(one_time_token) != 64
            or any(ch not in "0123456789abcdef" for ch in one_time_token)
        ):
            raise DiagnosticExecutionRejected("invalid one-time repair token")
        digest = sha256(payload).hexdigest()
        authority = f"sha256:{digest}"
        token_digest = sha256(one_time_token.encode("ascii")).hexdigest()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                handoff = conn.execute(
                    "SELECT * FROM repair_handoff WHERE episode_id=?",
                    (episode_id,),
                ).fetchone()
                if handoff is None:
                    raise DiagnosticExecutionRejected(
                        "repair token is absent, previously claimed or revoked"
                    )
                self._verify_handoff_row(handoff)
                if handoff["status"] != "RESERVED_UNCONFIRMED":
                    raise DiagnosticExecutionRejected(
                        "repair token is absent, previously claimed or revoked"
                    )
                if (
                    not hmac.compare_digest(handoff["token_sha256"], token_digest)
                    or handoff["plan_digest"] != plan.digest
                    or handoff["payload_sha256"] != digest
                    or handoff["authority_id"] != authority
                ):
                    raise DiagnosticExecutionRejected(
                        "repair token / frozen source artifact mismatch"
                    )
                episode = self._row(conn, episode_id)
                self._check_plan(episode, plan)
                if (
                    episode["status"] != "DONE"
                    or episode["pending_token"] is not None
                    or episode["pending_experiment"] is not None
                ):
                    raise DiagnosticExecutionRejected(
                        "repair claim not supported by a completed diagnosis"
                    )
                trace = self._authenticated_trace(
                    row=episode, episode_id=episode_id
                )
                decision = _next_decision(
                    plan=plan,
                    trusted_problem_digest=episode["problem_digest"],
                    trusted_tree_commitment=episode["tree_commitment"],
                    trace=trace,
                )
                if (
                    not isinstance(decision, DiagnosticResolution)
                    or decision.authority_id != authority
                    or episode["resolved_authority_id"] != authority
                    or decision.observations_used != episode["next_index"]
                    or abs(decision.risk_consumed - episode["risk_charged"]) > 1e-9
                ):
                    raise DiagnosticExecutionRejected(
                        "repair token cannot authorize altered sensor evidence"
                    )
                claimed_status = "CLAIMED_UNCONFIRMED"
                next_mac = self._handoff_mac(
                    episode_id=episode_id, plan_digest=plan.digest,
                    authority_id=authority,
                    payload_sha256=digest, token_sha256=token_digest,
                    status=claimed_status,
                )
                updated = conn.execute(
                    """UPDATE repair_handoff
                       SET status=?, handoff_receipt_mac=?
                       WHERE episode_id=? AND status='RESERVED_UNCONFIRMED'""",
                    (claimed_status, next_mac, episode_id),
                )
                if updated.rowcount != 1:
                    raise DiagnosticExecutionRejected(
                        "concurrent repair token consumer already claimed"
                    )
                conn.commit()
                return RepairExecutionClaim(
                    episode_id=episode_id, plan_digest=plan.digest,
                    concrete_authority_id=authority, payload_sha256=digest,
                )
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
            self._verify_handoff_row(row)
            return RepairHandoffSnapshot(
                episode_id=row["episode_id"],
                status=row["status"],
                plan_digest=row["plan_digest"],
                concrete_authority_id=row["authority_id"],
                payload_sha256=row["payload_sha256"],
                token_sha256=row["token_sha256"],
            )
