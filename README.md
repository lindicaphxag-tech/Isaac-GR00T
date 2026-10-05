# SemRepair 0.4.0a0 development

> **Standalone public tree.** This branch contains only SemRepair. The surrounding
> GitHub repository name is inherited from the public fork used to host the
> original prerelease; SemRepair does not depend on GR00T at runtime. A future
> dedicated repository can import this branch without restructuring the tree.

SemRepair is a research prototype for **proof-carrying semantic repair and
runtime execution across embodied-AI software boundaries**.

It targets failures where tensors remain shape/dtype-valid while physical
meaning changes silently: rotation representation, joint ordering, clock/scope,
action provenance, sensor freshness, dependency provenance, and physical-effect
completion under crash/replay.

## Development line: semantic observability and atomic repair

The standalone `main` development line extends the published 0.3.0rc1 kernel
with a narrowly scoped interaction model for **compensating semantic faults**.

The claim is **not** that fault masking, higher-order faults, or diagnosability
are new. The development contribution is the composition of:

1. algebraic semantic transports for ordering/sign/per-axis scale boundaries;
2. constructive observability analysis that either synthesizes an external
   basis witness or proves that an exactly canceling chain is endpoint-
   indistinguishable and therefore requires an internal evidence tap;
3. complete factorial repair-lattice analysis with non-monotone-edge and
   interaction detection;
4. implementation-bound atomic repair authorization that rejects a partial
   hotfix when the measured interaction evidence requires a repair bundle; and
5. a Lean theorem for the exact-cancellation core: if two non-identity
   transports cancel end-to-end, repairing either boundary alone unmasks the
   remaining fault, while repairing both restores the identity specification.

The real ManiSkill #1472/#1495 2x2 witness is a **partial-masking empirical
case**, not an exact instance of the Lean theorem. The theorem and empirical
case are deliberately kept separate.

## Published prerelease

SemRepair **0.3.0rc1 is already published** as a GitHub prerelease:

- tag: `semrepair-v0.3.0rc1`
- immutable tag target: `8cd7e7ad01e50aa18f42d333765f4fd242228d66`
- release id: `403794471`
- wheel SHA-256:
  `fa2a16877175043e5f8a727babfbedda945ccfac5170c64201dea898025e8b2c`
- sdist SHA-256:
  `996439a19a93f802005e57afe2e836b747345aecccd68491a42eb75466d2a1ec`

Install the immutable published tag:

```bash
python -m pip install \
  "git+https://github.com/lindicaphxag-tech/Isaac-GR00T.git@semrepair-v0.3.0rc1#subdirectory=semrepair_validation"
```

Install the frozen standalone rc1 tree directly from repository root:

```bash
python -m pip install \
  "git+https://github.com/lindicaphxag-tech/Isaac-GR00T.git@semrepair-standalone-v0.3.0rc1"
```

Install the post-rc1 standalone development line:

```bash
python -m pip install \
  "git+https://github.com/lindicaphxag-tech/Isaac-GR00T.git@semrepair-standalone-main"
```

The development line reports package version `0.4.0a0`; it is **not** a
published release and must not be cited as 0.3.0rc1 evidence.

Or install the published wheel from the GitHub prerelease.

The branch `semrepair-v0.3-rc1` is the **post-release development surface**.
It contains additional validation and adoption work that is not retroactively
claimed as part of the immutable rc1 snapshot.

## Core execution path

```text
source / config / runtime evidence
        ↓
semantic type + effect inference
        ↓
typed repair synthesis
        ↓
independent repair verification
        ↓
semantic compilation certificate
        ↓
proof-carrying runtime authorization
        ↓
physical-effect dispatch
        ↓
independent observation-plane evidence
        ↓
COMMITTED / ABORTED / AMBIGUOUS
```

The runtime fails closed when semantics cannot be justified. In particular,
requested→executed and stale→fresh are event-owned transitions and cannot be
forged by a pure adapter.

## Repair-interaction CLI

A host project can analyze a complete factorial repair experiment without
integrating the full runtime:

```bash
semrepair-repair-lattice \
  --input examples/maniskill_pr1472_pr1495_factorial.json \
  --json
```

The included ManiSkill example is bound to public workflow run
`37370267001`. It classifies the #1472/#1495 pair as **partial masking**:
each singleton repair makes the old end-to-end orientation metric much worse,
while the pair closes the representation/sign chain to approximately
`4.83e-06°`.

This command classifies supplied evidence; it does not upgrade self-authored
measurements into maintainer adoption or prospective evidence.

## One-command evidence

```bash
semrepair-behavior --json
semrepair-core-theorems --json
semrepair-conformance --json
semrepair-runtime-gate --depth 4 --json
semrepair-effect-fault-assay --json
```

## Post-rc1 public validation

Latest fully green development-surface validation:

- development head: `9e60f6404e21c08aad5935387065d6e172ea5e7d`;
- workflow run: **37327372332**;
- conclusion: success.

The immediately preceding code-validation run
**37326497957** at `b24d33ce816e2967b3ec8701d30a2213ecf72991`
adds an exact native ManiSkill pre→patch→post gate and is fully green.

Validated surfaces include:

- Python 3.10 / 3.12 / 3.13;
- wheel + sdist;
- public Git consumer install;
- reusable composite Action consumer smoke;
- Lean build with explicit `sorry` / `admit` rejection;
- production compiler vs independent exhaustive oracle: **2017 cases**;
- bounded replay model: **4662 traces**;
- proof-carrying runtime release gate: **7/7**;
- real-source GR00T / LeRobot / ManiSkill audits;
- native MuJoCo closed-loop semantic-corruption / repair;
- native ManiSkill production converter→controller exact-patch red→green gate.

### Native ManiSkill #1138 boundary

For target XYZ Euler `(0.5, 0.5, 0.5)` rad, the real frozen production
converter/controller boundary measures:

- before exact patch: **12.9285035°** SO(3) error;
- after exact patch: **0.0092307°**;
- SAPIEN pose-storage floor in the same assay: **0.0092307°**.

Artifact:
`semrepair-maniskill-1138-native-boundary`,
id **11351603467**,
SHA-256
`48120ac10ccf5fa53c6170d36044013751e82653aeeb2477247a05158912488f`.

The separate normalized-rotation sign issue (#1469/#1472) is intentionally
outside this claim.

### Native MuJoCo closed loop

A real `MjModel` / `MjData` loop executes `mujoco.mj_step`, injects a
hidden joint-order corruption, and applies the explicit compiled permutation
repair.

Artifact:
`semrepair-mujoco-native-closed-loop`,
id **11352960143**,
SHA-256
`be76e8cc87af4acf1973e6fc0b5b0af8606a35656f4b84d6f72bd8b0a9b9492c`.

## External adoption

SemRepair exposes:

- package / CLI;
- source-boundary manifests;
- `semrepair-adopt-init` native-regression scaffold;
- adoption-manifest validator;
- composite GitHub Action.

These lower adoption friction but **do not self-award adoption credit**.
L8/L9 counters change only when an independently maintained repository retains
the regression/spec/tooling.

## Claim boundary

The versioned public prerelease and post-release validation establish
reproducibility and method maturity. They do **not** by themselves establish:

- maintained external adoption;
- a prospective I2 mechanism match;
- L8;
- L9.
