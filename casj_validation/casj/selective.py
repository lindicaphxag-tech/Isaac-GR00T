from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SelectiveRepairRecord:
    """One frozen repair/refusal decision with externally measured losses."""

    accepted: bool
    stale_loss: float
    raw_repair_loss: float
    fallback_loss: float
    fallback_exact: bool = True
    label: str = ""


def _wilson_upper(
    successes: int, total: int, z: float = 1.959963984540054
) -> float | None:
    """Return the two-sided 95% Wilson interval upper endpoint."""
    if total <= 0:
        return None
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    radius = (
        z
        * sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total))
        / denom
    )
    return min(1.0, center + radius)


def evaluate_selective_repair(
    records: Iterable[SelectiveRepairRecord],
    *,
    requery_cost_ratios: Sequence[float] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0),
    harm_margin: float = 0.0,
    eps: float = 1e-12,
) -> dict:
    """Evaluate a frozen repair certificate as a selective decision rule.

    Losses are normalized by each state's stale loss. A rejected decision pays
    its measured fallback loss plus a dimensionless re-query cost ratio. This
    evaluates the certificate as frozen; it does not tune any acceptance
    threshold from the observed outcomes.
    """
    rows = list(records)
    if not rows:
        raise ValueError("at least one selective-repair record is required")
    if harm_margin < 0:
        raise ValueError("harm_margin must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    costs = tuple(float(value) for value in requery_cost_ratios)
    if any(value < 0 for value in costs):
        raise ValueError("requery_cost_ratios must be non-negative")

    normalized = []
    for row in rows:
        losses = (row.stale_loss, row.raw_repair_loss, row.fallback_loss)
        if not all(np.isfinite(value) and value >= 0 for value in losses):
            raise ValueError(f"losses must be finite and non-negative: {row!r}")
        denom = max(float(row.stale_loss), eps)
        raw_ratio = float(row.raw_repair_loss) / denom
        fallback_ratio = float(row.fallback_loss) / denom
        harmful = (
            float(row.raw_repair_loss)
            > float(row.stale_loss) * (1.0 + harm_margin)
        )
        beneficial = (
            float(row.raw_repair_loss)
            < float(row.stale_loss) * (1.0 - harm_margin)
        )
        normalized.append(
            {
                "label": row.label,
                "accepted": bool(row.accepted),
                "raw_to_stale_ratio": raw_ratio,
                "fallback_to_stale_ratio": fallback_ratio,
                "raw_harmful": harmful,
                "raw_beneficial": beneficial,
                "fallback_exact": bool(row.fallback_exact),
            }
        )

    n = len(normalized)
    accepted = [row for row in normalized if row["accepted"]]
    rejected = [row for row in normalized if not row["accepted"]]
    harmful = [row for row in normalized if row["raw_harmful"]]
    beneficial = [row for row in normalized if row["raw_beneficial"]]
    false_accepts = [row for row in accepted if row["raw_harmful"]]
    rejected_harm = [row for row in rejected if row["raw_harmful"]]
    rejected_benefit = [row for row in rejected if row["raw_beneficial"]]

    def mean_or_none(values):
        return float(np.mean(values)) if values else None

    raw_ratios = [row["raw_to_stale_ratio"] for row in normalized]
    fallback_ratios = [row["fallback_to_stale_ratio"] for row in normalized]
    curve = []
    for cost in costs:
        selective = [
            row["raw_to_stale_ratio"]
            if row["accepted"]
            else row["fallback_to_stale_ratio"] + cost
            for row in normalized
        ]
        always_fallback = [value + cost for value in fallback_ratios]
        curve.append(
            {
                "requery_cost_ratio": cost,
                "selective_normalized_risk": float(np.mean(selective)),
                "always_raw_normalized_risk": float(np.mean(raw_ratios)),
                "always_fallback_normalized_risk": float(np.mean(always_fallback)),
                "stale_normalized_risk": 1.0,
                "query_savings_vs_always_fallback": len(accepted) / n,
            }
        )

    return {
        "n": n,
        "coverage": len(accepted) / n,
        "requery_rate": len(rejected) / n,
        "raw_harm_rate": len(harmful) / n,
        "false_accept_count": len(false_accepts),
        "false_accept_rate_overall": len(false_accepts) / n,
        "false_accept_rate_given_acceptance": (
            len(false_accepts) / len(accepted) if accepted else None
        ),
        "false_accept_wilson95_upper_given_acceptance": _wilson_upper(
            len(false_accepts), len(accepted)
        ),
        "harmful_repair_rejection_rate": (
            len(rejected_harm) / len(harmful) if harmful else None
        ),
        "beneficial_repair_rejection_rate": (
            len(rejected_benefit) / len(beneficial) if beneficial else None
        ),
        "fallback_integrity_rate_on_rejection": (
            sum(row["fallback_exact"] for row in rejected) / len(rejected)
            if rejected
            else None
        ),
        "accepted_mean_raw_to_stale_ratio": mean_or_none(
            [row["raw_to_stale_ratio"] for row in accepted]
        ),
        "rejected_mean_raw_to_stale_ratio": mean_or_none(
            [row["raw_to_stale_ratio"] for row in rejected]
        ),
        "cost_aware_risk_curve": curve,
        "rows": normalized,
    }
