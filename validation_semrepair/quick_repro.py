"""Two-minute CPU-only falsification surface for Semantic ABI v0.1.

Prints one JSON object.  Any assertion failure is a failed reproduction.
No simulator, checkpoint, GPU, or network access is required.
"""

from __future__ import annotations

import json

from validation_semrepair.embodied_semantic_experiment_design import (
    SemanticExperiment,
    solve_optimal_semantic_diagnosis,
)
from validation_semrepair.embodied_semantic_repair_aware_design import (
    BoundRepairAuthority,
    RepairAuthority,
    bind_repair_authority,
    solve_repair_aware_semantic_diagnosis,
    verify_repair_aware_semantic_diagnosis,
)


def experiment(name, outcomes, *, cost=1.0, risk=0.0, atol=0.0):
    return SemanticExperiment(
        name=name,
        outcomes=tuple(outcomes.items()),
        cost=cost,
        risk=risk,
        observation_atol=atol,
    )


def main() -> None:
    hypotheses = ("a0", "a1", "b", "c")
    experiments = (
        experiment(
            "repair-class-test",
            {"a0": "A", "a1": "A", "b": "B", "c": "C"},
            cost=1.0,
        ),
        experiment(
            "identity-refiner",
            {"a0": 0, "a1": 1, "b": 0, "c": 0},
            cost=5.0,
        ),
    )
    authorities = tuple(
        RepairAuthority(hypothesis=h, authority_id=a)
        for h, a in {
            "a0": "bundle:A",
            "a1": "bundle:A",
            "b": "bundle:B",
            "c": "reject",
        }.items()
    )

    full = solve_optimal_semantic_diagnosis(
        hypotheses, experiments, objective="uniform_expected"
    )
    repair = solve_repair_aware_semantic_diagnosis(
        hypotheses,
        experiments,
        authorities,
        objective="uniform_expected",
    )
    verification = verify_repair_aware_semantic_diagnosis(repair, experiments)

    assert full.complete and repair.complete and verification.valid
    assert abs(full.optimal_cost - 3.5) <= 1e-12
    assert abs(repair.optimal_cost - 1.0) <= 1e-12

    aliases = ("correct", "masked-double-swap")
    alias_authorities = (
        RepairAuthority("correct", "noop"),
        RepairAuthority(
            "masked-double-swap",
            "bundle:producer+controller",
        ),
    )
    external = (
        experiment(
            "external-output",
            {
                "correct": (1.0, 0.0),
                "masked-double-swap": (1.0, 0.0),
            },
        ),
    )
    blocked = solve_repair_aware_semantic_diagnosis(
        aliases, external, alias_authorities
    )
    assert blocked.status == "unsafe_unidentifiable"

    with_tap = external + (
        experiment(
            "tap:producer",
            {
                "correct": (1.0, 0.0),
                "masked-double-swap": (0.0, 1.0),
            },
            cost=2.0,
        ),
    )
    resolved = solve_repair_aware_semantic_diagnosis(
        aliases, with_tap, alias_authorities
    )
    assert resolved.complete
    assert abs(resolved.optimal_cost - 2.0) <= 1e-12
    assert verify_repair_aware_semantic_diagnosis(resolved, with_tap).valid

    binding = BoundRepairAuthority(
        repair_bundle_id="scale-mm-to-m",
        implementation_ids=("impl:scale-v3",),
        evidence_digest="evidence:v1",
        dependency_digest="deps:v1",
    )
    drifted = BoundRepairAuthority(
        repair_bundle_id="scale-mm-to-m",
        implementation_ids=("impl:scale-v3",),
        evidence_digest="evidence:v1",
        dependency_digest="deps:v2",
    )
    assert binding.authority_id != drifted.authority_id
    same_authority = (
        bind_repair_authority("cause-a", binding),
        bind_repair_authority("cause-b", binding),
    )
    zero_cost = solve_repair_aware_semantic_diagnosis(
        ("cause-a", "cause-b"), (), same_authority
    )
    assert zero_cost.complete and zero_cost.optimal_cost == 0.0

    print(
        json.dumps(
            {
                "schema": "semrepair-semantic-abi-quick-repro-v1",
                "full_abi_expected_cost": full.optimal_cost,
                "repair_authority_expected_cost": repair.optimal_cost,
                "authority_saving": full.optimal_cost - repair.optimal_cost,
                "external_alias_status": blocked.status,
                "tap_resolved_status": resolved.status,
                "tap_resolved_cost": resolved.optimal_cost,
                "dependency_drift_splits_authority": (
                    binding.authority_id != drifted.authority_id
                ),
                "same_authority_zero_probe_cost": zero_cost.optimal_cost,
                "independent_verifier_valid": verification.valid,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
