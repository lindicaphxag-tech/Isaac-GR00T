# Reviewer Start Here — SemRepair

This page is a compact, falsifiable entry point for reviewers, maintainers, and
researchers who want to assess SemRepair without reading the full development
history.

SemRepair studies a narrow class of embodied-software failures:

> **Values can satisfy ordinary API, tensor-shape, dtype, and range checks while
> carrying the wrong physical / temporal / execution meaning.**

The project asks when those semantic boundary faults can be detected,
constructively repaired, independently certified, and safely dispatched.

## 60-second evidence map

### 1. Public standalone implementation

Branch: `semrepair-standalone-main`

Published prerelease: `semrepair-v0.3.0rc1`

The standalone branch does not require GR00T at runtime. The surrounding
repository name is inherited from the public fork used to publish the first
artifact.

### 2. Full public development CI

Workflow run: **37388039152**

Head: `d6c89553c9778d139e2cd9b79eb2fad84f57d608`

The same run contains:

- Python 3.10 / 3.12 / 3.13 jobs;
- standalone package installation;
- full standalone tests;
- evidence CLIs;
- wheel + sdist build;
- Lean proof build with `sorry` / `admit` rejection;
- native MuJoCo exact-masking assay.

This is self-authored public evidence, not external adoption.

### 3. Counterintuitive physical witness: two wrong boundaries can look correct

File: `examples/native_mujoco_complete_masking.py`

Two independent joint-order faults are injected around a real MuJoCo execution
path:

```text
producer order fault -> dispatch order fault -> MuJoCo
```

Because both faults are the same swap, they cancel before dispatch.

Public run **37388039152** measures:

| configuration | trajectory L∞ error vs correct |
|---|---:|
| both semantic faults present | **0.0** |
| producer repaired only | **5.969498771552663** |
| dispatch repaired only | **5.969498771552663** |
| both repaired | **0.0** |

The faulty baseline and fully correct trajectory are byte-for-byte equal under
the deterministic assay.

Therefore endpoint task behavior alone cannot certify that each internal
semantic boundary is correct.

### 4. Formal exact-cancellation result

File: `formal/SemRepairFormal/Masking.lean`

The Lean surface proves, for the scoped exact-cancellation model:

- canceling faults are black-box indistinguishable at the endpoints;
- repairing only the consumer unmasks a non-identity producer;
- repairing only the producer unmasks the remaining consumer;
- repairing both restores the identity specification;
- an exact canceling pair therefore requires an atomic pair repair under the
  declared endpoint specification.

The theorem deliberately does **not** claim that arbitrary robot dynamics are
formally verified.

### 5. Real upstream-derived partial-masking witness

ManiSkill PRs:

- `mani-skill/ManiSkill#1472` — controller rotation-sign semantics;
- `mani-skill/ManiSkill#1495` — converter rotation-representation semantics.

Public 2×2 causal run: **37370267001**

Measured SO(3) error:

| converter | controller | error |
|---|---|---:|
| old | old | 5.076696° |
| old | #1472 | 64.747338° |
| #1495 | old | 66.128005° |
| #1495 | #1472 | **4.83e-06°** |

This is a partial-masking empirical case: locally correcting one semantic fault
makes the old end-to-end metric dramatically worse; repairing both closes the
chain.

Neither open upstream PR is counted as maintained adoption until a maintainer
retains it.

### 6. Independent upstream boundary candidate

NVIDIA Isaac-GR00T PR **#786** is a one-commit / two-file fix for episode-boundary
semantics in `MultiStepWrapper`.

The external project issue was independently reported; SemRepair does not claim
ownership of that report. The value to this research program is that the same
semantic-boundary abstraction can describe episode termination/truncation,
rotation representation, ordering, temporal anchors, and provenance without
changing tensor shape/type.

Again: an open PR is submitted evidence, not adoption.

## Core research claim

The project does **not** claim that runtime VLA checking, fault masking,
proof-carrying actions, typed robot contracts, or runtime monitors are
individually new.

The current research wedge is the composition:

```text
semantic type/effect boundary
    -> constructive bounded repair
    -> independent implementation-bound certificate
    -> prepare/dispatch/commit reauthorization
    -> independent post-effect evidence closure
```

The new interaction axis adds:

```text
semantic transport composition
    -> observability / internal-witness synthesis
    -> complete repair lattice
    -> implementation-bound atomic repair deployment
```

## 10-minute falsification checklist

A skeptical reviewer can try to falsify the project by checking whether:

1. the full public CI run actually executed all three Python versions;
2. the MuJoCo baseline with both faults is exactly equal to the correct
   trajectory;
3. either singleton repair really makes the physical trajectory worse;
4. `Masking.lean` builds under the pinned Lean toolchain without
   `sorry` / `admit`;
5. the ManiSkill factorial evidence is bound to the stated PR revisions;
6. the runtime rejects a partial implementation set for a certified
   compensating bundle;
7. claim-boundary files still report maintained external adoption and resolved
   prospective-I2 counts as zero unless independent evidence changes them.

A failure of any item should reduce the corresponding claim rather than be
explained away.

## Maturity boundary

As of this public research record:

- public reproducible artifact: **yes**;
- native simulator execution: **yes**;
- formal scoped core: **yes**;
- real upstream-derived causal cases: **yes**;
- maintained independent SemRepair adoption: **0**;
- resolved prospective I2 mechanism confirmations: **0**;
- L8 claim: **false**;
- L9 claim: **false**.

The last three lines are intentional. The project treats external retention and
prospective confirmation as evidence that cannot be self-awarded.

## Suggested reading order

1. this file;
2. `README.md`;
3. `RESEARCH_RECORD_0_4_ALPHA.json`;
4. `formal/SemRepairFormal/Masking.lean`;
5. `examples/native_mujoco_complete_masking.py`;
6. `ADOPTION.md` for the external-retention policy.
