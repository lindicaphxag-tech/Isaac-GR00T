# SemRepair public validation branch

This branch is an isolated validation surface for the SemRepair embodied
semantic compiler. It is intentionally separate from the upstream-facing GR00T
fix branches and does not modify NVIDIA production code.

Source snapshot:
- private research repository: lindicaphxag-tech/lindicaphxag-tech
- branch: research/invariantbench-l8
- source head: 6f2c8549a8c323ee55fcbb4683b533894b44587c
- validation target: semantic inference -> type/effect compilation -> repair
  synthesis -> independent verification -> certified runtime mediation ->
  formal executable oracles -> deterministic closed-loop consequences.

Run locally from this repository root:

    PYTHONPATH=semrepair_validation python -m pytest -q semrepair_validation/tests

Behavioral evidence:

    PYTHONPATH=semrepair_validation python -m research.semantic_invariants.embodied_behavioral_evidence --json

Claim boundary:
- this is public executable validation of the current SemRepair core;
- it is not NVIDIA/Isaac-GR00T adoption;
- it is not a prospective discovery;
- it is not, by itself, L8 or L9 evidence.
