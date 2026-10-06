"""Runtime refinement evidence for non-forgeable embodied semantics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

from .embodied_semantic_types import SemanticTensorType, semantic_mismatches


class InvalidSemanticEvidence(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticEvidence:
    kind: str
    subject_id: str
    claims: Mapping[str, Any]
    issuer: str

    @property
    def digest(self) -> str:
        payload = {
            "kind": self.kind,
            "subject_id": self.subject_id,
            "claims": dict(self.claims),
            "issuer": self.issuer,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
        return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RefinementResult:
    before: SemanticTensorType
    after: SemanticTensorType
    evidence_digest: str
    effects: tuple[str, ...]
    proof_context: tuple[tuple[str, object], ...] = ()


def refine_executed_action(
    action_type: SemanticTensorType,
    evidence: SemanticEvidence,
) -> RefinementResult:
    """Refine requested -> executed only from an execution receipt."""

    if action_type.role != "action":
        raise InvalidSemanticEvidence("execution evidence can refine only action values")
    if action_type.provenance != "requested":
        raise InvalidSemanticEvidence("source action must have requested provenance")
    if evidence.kind != "execution_receipt":
        raise InvalidSemanticEvidence("requested->executed requires execution_receipt")
    if evidence.claims.get("executed") is not True:
        raise InvalidSemanticEvidence("receipt does not assert actual execution")
    if not evidence.claims.get("controller_id"):
        raise InvalidSemanticEvidence("receipt lacks controller_id")
    return RefinementResult(
        before=action_type,
        after=action_type.updated(provenance="executed"),
        evidence_digest=evidence.digest,
        effects=("execute",),
    )


def refine_sensor_freshness(
    observation_type: SemanticTensorType,
    evidence: SemanticEvidence,
    *,
    due_time: float,
) -> RefinementResult:
    """Refine unknown/stale -> fresh only from a sufficiently recent sample."""

    if observation_type.role not in {"observation", "state"}:
        raise InvalidSemanticEvidence("sensor evidence refines observation/state values")
    if evidence.kind != "sensor_sample":
        raise InvalidSemanticEvidence("freshness requires sensor_sample evidence")
    sample_time = evidence.claims.get("sample_time")
    if not isinstance(sample_time, (int, float)):
        raise InvalidSemanticEvidence("sensor_sample lacks numeric sample_time")
    if float(sample_time) < float(due_time):
        raise InvalidSemanticEvidence("sensor sample predates the required freshness boundary")
    return RefinementResult(
        before=observation_type,
        after=observation_type.updated(freshness="fresh"),
        evidence_digest=evidence.digest,
        effects=("observe",),
        proof_context=(("due_time", float(due_time)),),
    )


def refine_episode_time(
    value_type: SemanticTensorType,
    evidence: SemanticEvidence,
) -> RefinementResult:
    """Refine dataset-global time into episode-local time with an origin witness."""

    if evidence.kind != "episode_origin":
        raise InvalidSemanticEvidence("episode retiming requires episode_origin evidence")
    if "episode_id" not in evidence.claims or "origin_time" not in evidence.claims:
        raise InvalidSemanticEvidence("episode_origin evidence is incomplete")
    if value_type.clock != "dataset_global":
        raise InvalidSemanticEvidence("source clock must be dataset_global")
    return RefinementResult(
        before=value_type,
        after=value_type.updated(clock="episode_local", scope="episode"),
        evidence_digest=evidence.digest,
        effects=("retime",),
    )


def refinement_satisfies(
    result: RefinementResult,
    target: SemanticTensorType,
) -> bool:
    return not semantic_mismatches(result.after, target)