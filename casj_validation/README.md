# CASJ public-checkpoint validation

This directory is a fork-only execution surface for the CASJ paper core.

It runs the frozen CASJ assay against the public LeRobot PushT diffusion policy,
with exact PushT state reconstruction and paired diffusion randomness. Results
from this directory are method-validation evidence only; they are not
NVIDIA/Isaac-GR00T adoption and are never counted toward the SemRepair external
adoption gate.

The first public workflow run intentionally overrides the checkpoint diffusion
steps for a CPU feasibility smoke. A later result may be promoted to scientific
public-checkpoint evidence only if it uses the checkpoint-default inference
configuration, multiple independent reset seeds, and the predeclared multi-phase
protocol.
