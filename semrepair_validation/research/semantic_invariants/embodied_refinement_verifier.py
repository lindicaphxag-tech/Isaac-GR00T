"""Independent verification of event-owned semantic refinements.

The certificate binds refinement records, but the checker still requires the
actual owner-boundary evidence objects. It recomputes each refinement rule from
that evidence and rejects forged, stale, missing, or context-mismatched claims.

Trust boundary: evidence content is checked, while authenticity of the issuer
identity remains an external system assumption unless receipts are separately
signed/attested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .embodied_compilation_certificate import SemanticCompilationCertificate
from .embodied_refinement import (
    InvalidSemanticEvidence,
    SemanticEvidence,
    refine_episode_time,
    refine_executed_action,
    refine_sensor_freshness,
)
from .embodied_semantic_types import SemanticTensorType


@dataclass(frozen=True)
class RefinementProofCheck:
    valid: bool
    reasons: tuple[str, ...]
    verified_refinements: int


def _type_from_record(record: Mapping[str, object]) -> SemanticTensorType:
    return SemanticTensorType(**dict(record))  # type: ignore[arg-type]


def _proof_context(record: Mapping[str, object]) -> dict[str, object]:
    raw = record.get("proof_context", ())
    out: dict[str, object] = {}
    for pair in raw:  # type: ignore[assignment]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"invalid proof-context entry: {pair!r}")
        out[str(pair[0])] = pair[1]
    return out


def verify_refinement_records(
    certificate: SemanticCompilationCertificate,
    *,
    evidence_by_digest: Mapping[str, SemanticEvidence],
) -> RefinementProofCheck:
    """Recompute certificate-bound event refinement rules from raw evidence."""

    reasons: list[str] = []
    verified = 0

    try:
        current = _type_from_record(certificate.inferred_source)
        expected_final = _type_from_record(certificate.refined_source)
    except (TypeError, ValueError) as exc:
        return RefinementProofCheck(
            valid=False,
            reasons=(f"invalid semantic type record: {exc}",),
            verified_refinements=0,
        )

    for index, record in enumerate(certificate.refinement_records):
        try:
            before = _type_from_record(record["before"])  # type: ignore[arg-type]
            after = _type_from_record(record["after"])  # type: ignore[arg-type]
            digest = str(record["evidence_digest"])
            effects = tuple(str(item) for item in record.get("effects", ()))
            context = _proof_context(record)
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(f"refinement[{index}] record malformed: {exc}")
            break

        if before != current:
            reasons.append(
                f"refinement[{index}] before-type does not match current chain state"
            )
            break

        evidence = evidence_by_digest.get(digest)
        if evidence is None:
            reasons.append(
                f"refinement[{index}] missing raw evidence for digest {digest!r}"
            )
            break
        if evidence.digest != digest:
            reasons.append(
                f"refinement[{index}] raw evidence digest mismatch"
            )
            break

        try:
            if evidence.kind == "execution_receipt":
                recomputed = refine_executed_action(before, evidence)
            elif evidence.kind == "sensor_sample":
                due_time = context.get("due_time")
                if not isinstance(due_time, (int, float)):
                    raise InvalidSemanticEvidence(
                        "sensor refinement proof context lacks numeric due_time"
                    )
                recomputed = refine_sensor_freshness(
                    before,
                    evidence,
                    due_time=float(due_time),
                )
            elif evidence.kind == "episode_origin":
                recomputed = refine_episode_time(before, evidence)
            else:
                raise InvalidSemanticEvidence(
                    f"unsupported refinement evidence kind {evidence.kind!r}"
                )
        except InvalidSemanticEvidence as exc:
            reasons.append(f"refinement[{index}] rule rejected evidence: {exc}")
            break

        if recomputed.before != before:
            reasons.append(f"refinement[{index}] recomputed before-type mismatch")
            break
        if recomputed.after != after:
            reasons.append(f"refinement[{index}] recomputed after-type mismatch")
            break
        if recomputed.effects != effects:
            reasons.append(f"refinement[{index}] effect mismatch")
            break
        if dict(recomputed.proof_context) != context:
            reasons.append(f"refinement[{index}] proof-context mismatch")
            break
        if recomputed.evidence_digest != digest:
            reasons.append(f"refinement[{index}] evidence digest mismatch after recomputation")
            break

        current = recomputed.after
        verified += 1

    if not reasons and current != expected_final:
        reasons.append(
            "verified refinement chain does not equal certificate refined_source"
        )

    return RefinementProofCheck(
        valid=not reasons,
        reasons=tuple(reasons),
        verified_refinements=verified,
    )
