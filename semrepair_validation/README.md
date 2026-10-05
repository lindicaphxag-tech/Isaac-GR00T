# SemRepair 0.3.0-rc1

SemRepair is a research prototype for **semantic type-and-effect compilation**
across embodied-AI and robot-learning software boundaries.

It targets failures where ordinary tensor shape and dtype remain valid while
physical meaning changes silently: rotation representation, joint ordering,
clock/scope, action provenance, sensor freshness, and related cross-layer
semantics.

## Method path

```text
source / config / runtime evidence
        |
        v
hidden semantic inference
        |
        v
type + effect compilation
        |
        +--> certified explicit adapter
        +--> bounded repair candidate
        +--> proof obligation
        +--> owner-boundary runtime refinement
        +--> fail closed
        |
        v
independent verification certificate
        |
        v
certified runtime mediation
```

## Install this public release candidate

From the repository root:

```bash
python -m pip install -e semrepair_validation
```

The distribution name is `semrepair`. A stable convenience import is also
provided:

```python
import semrepair
print(semrepair.__version__)
```

## One-command evidence

Behavioral assays:

```bash
semrepair-behavior --json
```

Bounded core theorem checks:

```bash
semrepair-core-theorems --json
```

Portable source-to-compiler example:

```bash
semrepair-source \
  --manifest semrepair_validation/examples/maniskill_style/semrepair.json \
  --json
```

The example infers an axis-angle producer and XYZ-Euler consumer from Python
source and synthesizes the explicit two-hop representation repair.

## Formal core

The release candidate also carries a Lean 4 core under
`semrepair_validation/formal`. Public CI rejects `sorry` / `admit` and builds
the pinned formal project.

The formal claims are intentionally narrow: the current core mechanizes selected
semantic-preservation / non-forgeability properties. It does not prove the
entire Python compiler or general robot safety.

## Public evidence

The deterministic behavioral surface contains three distinct intervention
classes:

1. **joint ordering** — compile and install an explicit reorder adapter;
2. **rotation representation + sign** — synthesize, independently verify, and
   certificate-gate a composed repair;
3. **sensor freshness** — refuse a fake stale->fresh cast and require an
   owner-boundary resample/refinement.

## Source lineage

This branch is a public release-candidate staging surface derived from the
private research repository:

- source repository: `lindicaphxag-tech/lindicaphxag-tech`
- source branch: `research/invariantbench-l8`
- current source lineage head used for this release work:
  `fdc527015491cfb2c790371bb953202a8dfc1c94`

It is deliberately isolated from the NVIDIA upstream-facing fix branches and
does not modify NVIDIA production code.

## Citation and license

- citation metadata: `semrepair_validation/CITATION.cff`
- license: Apache-2.0 in `semrepair_validation/LICENSE`
- changelog: `semrepair_validation/CHANGELOG.md`
- machine-readable status: `semrepair_validation/RELEASE_STATUS.json`

## Claim boundary

This is a **public release candidate**, not a maintained NVIDIA/Isaac-GR00T
integration.

It does **not** count as:

- NVIDIA adoption;
- a prospective I2 mechanism match;
- project-native real-stack closed-loop validation;
- L8 achieved;
- L9 achieved;
- a GitHub tagged Release.

Those gates stay false until external evidence exists.
