#!/usr/bin/env python3
"""Small public reproduction capsule for joint semantic experiment design."""

from __future__ import annotations

import json

from research.semantic_invariants.embodied_semantic_autorepair_bridge import (
    compile_diagnosis_to_repair_plan,
)
from research.semantic_invariants.embodied_semantic_experiment_design import (
    SemanticExperiment,
    synthesize_optimal_experiment_plan,
)
from research.semantic_invariants.embodied_semantic_observability import (
    SemanticDiagnosisHypothesis,
)
from research.semantic_invariants.embodied_semantic_transport import (
    MonomialSemanticTransport,
    SemanticTransportFactor,
)


def factor(name, transport, evidence):
    return SemanticTransportFactor(name, transport, evidence)


def hypothesis(name, factors):
    return SemanticDiagnosisHypothesis(name=name, factors=tuple(factors))


def main() -> None:
    identity = MonomialSemanticTransport.identity(2)
    swap = MonomialSemanticTransport((1, 0), (1.0, 1.0))

    correct = hypothesis(
        "correct",
        (
            factor("producer", identity, "correct/p"),
            factor("controller", identity, "correct/c"),
        ),
    )
    visible = hypothesis(
        "visible-swap",
        (
            factor("producer", swap, "visible/p"),
            factor("controller", identity, "visible/c"),
        ),
    )
    masked = hypothesis(
        "masked-double-swap",
        (
            factor("producer", swap, "masked/p"),
            factor("controller", swap, "masked/c"),
        ),
    )

    experiments = (
        SemanticExperiment(
            name="output/e0",
            probe=(1.0, 0.0),
            tap_after_factor=None,
            cost=1.0,
            risk=0.0,
        ),
        SemanticExperiment(
            name="producer/e0",
            probe=(1.0, 0.0),
            tap_after_factor="producer",
            cost=3.0,
            risk=0.0,
        ),
    )

    plan = synthesize_optimal_experiment_plan(
        (correct, visible, masked),
        experiments,
        priors={"correct": 1.0, "visible-swap": 1.0, "masked-double-swap": 1.0},
        objective="expected",
    )
    masked_pair = synthesize_optimal_experiment_plan(
        (correct, masked),
        experiments,
        objective="worst_case",
    )
    repair = compile_diagnosis_to_repair_plan((masked,))

    payload = {
        "schema_version": 1,
        "experiment_plan": {
            "complete": plan.complete,
            "root_experiment": (
                None if plan.root.experiment is None else plan.root.experiment.name
            ),
            "expected_total_cost": plan.expected_total_cost,
            "worst_case_total_cost": plan.worst_case_total_cost,
            "digest": plan.digest,
        },
        "masked_pair": {
            "complete": masked_pair.complete,
            "root_experiment": (
                None
                if masked_pair.root.experiment is None
                else masked_pair.root.experiment.name
            ),
            "expected_total_cost": masked_pair.expected_total_cost,
            "worst_case_total_cost": masked_pair.worst_case_total_cost,
            "digest": masked_pair.digest,
        },
        "diagnose_before_repair": {
            "hypothesis": repair.hypothesis_name,
            "nonidentity_boundaries": list(repair.nonidentity_boundaries),
            "requires_interaction_assay": repair.requires_interaction_assay,
            "eligible_for_independent_verification": (
                repair.eligible_for_independent_verification
            ),
            "obligations": [
                {
                    "boundary": item.boundary,
                    "status": item.status,
                    "program_name": item.program_name,
                    "implementation_ids": list(item.program_implementation_ids),
                }
                for item in repair.obligations
            ],
            "digest": repair.digest,
        },
        "claim_boundary": (
            "Self-contained deterministic reproduction only; not external "
            "adoption, prospective I2, or robot-safety evidence."
        ),
    }

    assert payload["experiment_plan"]["root_experiment"] == "output/e0"
    assert payload["masked_pair"]["root_experiment"] == "producer/e0"
    assert payload["diagnose_before_repair"]["requires_interaction_assay"] is True
    assert (
        payload["diagnose_before_repair"][
            "eligible_for_independent_verification"
        ]
        is False
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
