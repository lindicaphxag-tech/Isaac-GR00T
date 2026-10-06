# SemRepair 0.3.0-rc2

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
interaction-aware repair-set authorization
        |
        v
proof-carrying semantic runtime
        |
        +--> stable physical-effect identity
        +--> independent observation-plane evidence
        +--> ambiguity-preserving commit / reconcile / block
        |
        v
certified runtime mediation
```

## 0.3.0-rc2 hardening

This candidate adds three authority-boundary properties that were not strong
enough in rc1:

1. **Executable identity binding.** Repair certificates bind stable primitive
   identity plus inspectable callable/source-file digests, defaults and closure
   captures. A changed implementation invalidates an old certificate even when
   its symbolic repair name is unchanged.
2. **No self-declared verification authority.** Runtime installation requires a
   matching `RepairVerificationCertificate`; a caller-provided
   `verification_status="verified"` string has no authority.
3. **Repair-set atomicity.** A complete factorial interaction certificate can
   identify compensating semantic defects. Non-empty proper subsets of a
   strict compensating bundle fail closed even when each local repair is
   individually correct.

A public ManiSkill `PegInsertionSide-v1` four-way replay assay (workflow
`37388740482`) measured main=8/10, converter-only=1/10,
controller-only=0/10, and the composed pair=8/10; main and composed both
retained 1238 saved steps. This is self-authored real-stack replay evidence,
not learned-policy success, maintainer adoption, or a prospective I2 result.

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

Production compiler vs independent exhaustive oracle:

```bash
semrepair-conformance --json
```

Proof-carrying runtime release gate:

```bash
semrepair-runtime-gate --depth 4 --json
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

The rc1 predecessor is already published as `semrepair-v0.3.0rc1` and passed clean-install validation. The rc2 branch must obtain its own green validation before publication.

Historical rc1 validation:

- head: `304f19661b21bf2a510f736b63f78d3bbe6d3aa9`;
- workflow run: **37316616190**;
- Python 3.10 / 3.12 / 3.13 package jobs: success;
- wheel + sdist build: success;
- public Git URL external-consumer install: success;
- Lean core with explicit `sorry/admit` rejection: success;
- production-vs-exhaustive conformance: **2017 cases**, valid;
- bounded replay model in runtime gate: **4662 traces**;
- proof-carrying runtime gate: **7/7 checks**;
- pinned real-source GR00T and ManiSkill semantic-source jobs: success.

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
- rc1 source lineage: published immutable snapshot `8cd7e7ad01e50aa18f42d333765f4fd242228d66`;
- rc2 method source: `lindicaphxag-tech/lindicaphxag-tech@research/semrepair-rc1`;
- rc2 public staging branch: `lindicaphxag-tech/Isaac-GR00T@semrepair-0.3.0rc2`

It is deliberately isolated from the NVIDIA upstream-facing fix branches and
does not modify NVIDIA production code.

## Citation and license

- citation metadata: `semrepair_validation/CITATION.cff`
- license: Apache-2.0 in `semrepair_validation/LICENSE`
- changelog: `semrepair_validation/CHANGELOG.md`
- machine-readable status: `semrepair_validation/RELEASE_STATUS.json`

## Claim boundary

This is a **public release-candidate update derived from the published 0.3.0rc1 snapshot**, not a maintained NVIDIA/Isaac-GR00T integration.

It does **not** count as:

- NVIDIA adoption;
- a prospective I2 mechanism match;
- project-native real-stack closed-loop validation;
- L8 achieved;
- L9 achieved;
- a new rc2 GitHub tagged Release before rc2 publication.

Those gates stay false until external evidence exists.