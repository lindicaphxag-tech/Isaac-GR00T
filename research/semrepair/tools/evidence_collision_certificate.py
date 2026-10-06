#!/usr/bin/env python3
"""Detect sampled evidence collisions for semantic-repair authorization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def certify(payload: dict) -> dict:
    variants = payload["variants"]
    internal = payload["internal_evidence"]
    external = payload["external_semantic_oracle"]
    tol = float(internal["equivalence_tolerance"])
    separation = float(external["material_separation"])
    anchor_verified = bool(payload.get("external_anchor_verified", False))

    collisions = []
    for i, left in enumerate(variants):
        for right in variants[i + 1 :]:
            internally_equivalent = all(
                abs(
                    float(left["internal_scores"][metric])
                    - float(right["internal_scores"][metric])
                )
                <= tol
                for metric in internal["metrics"]
            )
            semantic_distance = abs(
                float(left["external_semantic_score"])
                - float(right["external_semantic_score"])
            )
            if internally_equivalent and semantic_distance >= separation:
                collisions.append(
                    {
                        "left": left["id"],
                        "right": right["id"],
                        "external_semantic_distance": semantic_distance,
                        "internal_scores_left": left["internal_scores"],
                        "internal_scores_right": right["internal_scores"],
                    }
                )

    if collisions:
        identifiability = "non_identifying"
        authority = "reject_until_external_anchor"
    elif anchor_verified:
        identifiability = "anchored_for_sampled_hypotheses"
        authority = "advance_to_next_gate"
    else:
        identifiability = "undetermined"
        authority = "reject_until_external_anchor"

    return {
        "schema_version": 1,
        "sampled_hypothesis_count": len(variants),
        "collision_count": len(collisions),
        "collisions": collisions,
        "identifiability": identifiability,
        "authority": authority,
        "external_anchor_verified": anchor_verified,
        "claim_boundary": (
            "A detected collision proves the supplied internal evidence is "
            "non-identifying on the sampled hypotheses. No observed collision "
            "does not prove global identifiability."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = certify(json.loads(args.input.read_text()))
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
