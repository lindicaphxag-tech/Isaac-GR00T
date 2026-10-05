# SemRepair 0.3.0rc3

SemRepair is a research prototype for **proof-carrying semantic repair and
runtime execution across embodied-AI software boundaries**.

It targets failures where tensors remain shape/dtype-valid while physical
meaning changes silently: rotation representation, joint ordering, clock/scope,
action provenance, sensor freshness, dependency provenance, and physical-effect
completion under crash/replay.

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

## Install

Development release-candidate branch:

```bash
python -m pip install \
  "git+https://github.com/lindicaphxag-tech/Isaac-GR00T.git@semrepair-v0.3-rc3#subdirectory=semrepair_validation"
```

A fixed semver-style branch `semrepair-v0.3.0rc3` should be frozen only after the
corresponding release-candidate commit passes the complete public workflow.
Until then, use an exact reviewed commit SHA for reproducible rc3 evaluation.

The distribution name is `semrepair`:

```python
import semrepair
print(semrepair.__version__)
```

## One-command evidence

```bash
semrepair-behavior --json
semrepair-core-theorems --json
semrepair-conformance --json
semrepair-runtime-gate --depth 4 --json
semrepair-effect-fault-assay --json
```

Source-boundary example:

```bash
semrepair-source \
  --manifest semrepair_validation/examples/maniskill_style/semrepair.json \
  --json
```

## Language-neutral proof bundle

0.3.0rc3 keeps the v0.1 compatibility surface and adds **v0.2 exact-decimal wire authorization** for non-Python consumers.

The reference verifier independently checks:

- bundle / compilation / effect certificate digests;
- adapter-registry and contextual-evidence identity;
- selected semantic adapter-path replay;
- bounded unique minimum-cost repair;
- non-forgeable provenance/freshness constraints;
- effect-intent identity;
- exact compilation -> physical-effect dependency binding.

CLI:

```bash
semrepair-proof-bundle --input proof-bundle.json --json
```

Schema:

`semrepair_validation/schemas/semrepair-execution-proof-bundle-v0.2.schema.json`
(v0.1 remains available as the compatibility surface).

The digests are integrity/dependency receipts, not digital signatures.

## Public validation

Latest fully green inherited baseline validation (0.3.0rc2):

- validated code head: `094e5d96c74440410f9bfeff1ec6413985483c2b`;
- workflow run: **37329404920**;
- Python 3.10 / 3.12 / 3.13: success;
- wheel + sdist: success;
- install from public Git URL in a consumer-style job: success;
- reusable composite Action consumer smoke: success;
- packaged proof-bundle JSON Schema: success;
- installed proof-bundle generation + independent CLI verification: success;
- Lean core build with explicit `sorry` / `admit` rejection: success;
- production compiler vs independent exhaustive oracle: **2017 cases**, valid;
- bounded replay model: **4662 traces**;
- proof-carrying runtime release gate: **7/7 checks**;
- pinned real-source GR00T / LeRobot / ManiSkill audits: success;
- native MuJoCo closed-loop semantic-corruption / repair assay: success;
- native ManiSkill production converter→controller **red→green patch gate**:
  success.

### Native ManiSkill #1138 representation boundary

The public job executes the real frozen ManiSkill production functions:

```text
mani_skill.trajectory.utils.actions.conversion.delta_pose_to_pd_ee_delta
→
mani_skill.agents.controllers.PDEEPoseController.compute_target_pose
```

For target XYZ Euler `(0.5, 0.5, 0.5)` rad:

- current frozen source: **12.9285035°** SO(3) error;
- after applying the exact minimal converter patch: **0.0092307°**;
- SAPIEN pose round-trip numerical floor in the same assay: **0.0092307°**.

The before/after JSON evidence is retained in workflow artifact
`semrepair-maniskill-1138-native-boundary` (artifact id **11352879806**,
SHA-256 `8b00a6061e1f6dc4569db4e40e8b7c61df0aec07bf596de6f7c04bb32d795654`).

This validates the representation boundary only. The separate normalized
rotation-sign issue (#1469/#1472) is intentionally outside this claim.

### Native MuJoCo closed loop

The public workflow constructs real `MjModel` / `MjData`, executes
`mujoco.mj_step`, injects a hidden joint-order semantic corruption, and checks
the explicit compiled permutation repair.

Artifact:
`semrepair-mujoco-native-closed-loop` (id **11354160530**, SHA-256
`0918ecc438a845ab1c265534426f49faa8f583f0634867de4dbd98751e8dcb07`).

## Formal claim boundary

The release candidate includes a Lean 4 formal core and bounded executable
theorem/conformance checks. It does **not** claim a proof of the entire Python
implementation or general robot safety.

## External adoption

A repository can consume SemRepair through either:

- the package / CLI;
- a source-boundary manifest;
- a repository-native regression generated by `semrepair-adopt-init`;
- the composite GitHub Action in `semrepair_validation/action.yml`.

Self-authored consumer smoke tests remain **zero external-adoption credit**.
L8/L9 adoption changes only when an independently maintained repository retains
the regression/spec/tooling.

## Citation and license

- `semrepair_validation/CITATION.cff`
- `semrepair_validation/LICENSE`
- `semrepair_validation/CHANGELOG.md`
- `semrepair_validation/RELEASE_STATUS.json`

## Current claim boundary

`0.3.0rc1` is the currently published immutable prerelease.

`0.3.0rc2` is the latest **green public release candidate** baseline.

`0.3.0rc3` is the current moving release candidate. It adds exact-decimal v0.2
wire authorization and remains pending until the complete rc3 public workflow
is green and an immutable branch/tag is frozen.

None of these release states are NVIDIA adoption, a prospective I2 mechanism
match, or L8/L9 by themselves.