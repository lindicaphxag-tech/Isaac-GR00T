# SemRepair v0.3 — Public Research Record

**Status:** public artifact note for an L8-candidate embodied-AI research method.  
**Date:** 2026-10-05  
**Validation repository:** `lindicaphxag-tech/Isaac-GR00T`  
**Validation branch:** `validation/semrepair-core-v0.3`

This note is intentionally conservative. It records the method, frozen
prospective denominator, public executable evidence, negative results, upstream
status, and threats to validity. It does **not** claim NVIDIA adoption, a
standalone release, L8, or L9.

## 1. Method

SemRepair is a semantic type-and-effect compilation method for embodied-learning
pipelines.

The constructive path is:

```text
source/config/runtime evidence
-> hidden semantic-type inference
-> type/effect boundary compilation
-> explicit adapter synthesis or bounded repair synthesis
-> runtime proof obligation for context-dependent semantics
-> owner-boundary refinement for non-forgeable semantics
-> held-out verification certificate
-> certificate-gated runtime mediation
```

Tracked semantic axes include representation, frame, unit, clock/scope,
freshness, provenance, ordering, embodiment, mode, and convention.

The central distinction is between:

- **forgeable representation semantics**, where a certified conversion can be
  synthesized; and
- **non-forgeable event semantics**, such as requested→executed or stale→fresh,
  which require evidence from the owner boundary rather than a cast.

## 2. Frozen protocol

Primary prospective evidence uses:

- contract specification: **v0.2**;
- prospective protocol: **v2**;
- watchset generation: **1**;
- effective exposure cutoff: **2026-10-04T20:17:41Z**.

The v2 information barrier permits the issue body/reproducer, frozen source, and
pre-existing documentation before prediction freeze. It forbids later
root-cause comments, candidate-fix diffs, later fixes, or external discussion
that reveals the repair.

Primary evidence uses **I2** cases only: symptom/reproducer visible, concrete
mechanism and repair boundary not exposed.

## 3. Complete prospective denominator at this record

No eligible I2 case has been admitted yet.

The complete post-cutoff screened denominator is:

| Repository | Issue | Created | Classification | Primary eligible | Reason |
|---|---:|---|---|---|---|
| google-deepmind/mujoco | #3654 | 2026-10-05T03:31:49Z | I0 | no | Original issue body already gives the concrete multicontact/polygonClip mechanism and proposed repair. |
| isaac-sim/IsaacLab | #8294 | 2026-10-05T04:10:39Z | I0 | no | Original issue body already identifies the sensor-vs-asset body-id indexing mismatch and acceptance criterion. |

Current counters:

```text
screened_post_cutoff = 2
I0_excluded          = 2
admitted              = 0
I2_predictions        = 0
resolved_I2           = 0
mechanism_matches     = 0
mismatches            = 0
contaminated          = 0
```

The zero denominator is a negative result, not missing data to be backfilled.
Older retrospective bugs are not promoted into prospective evidence.

## 4. Public executable artifact evidence

The public validation surface contains the SemRepair dependency-light core plus
tests and deterministic behavioral assays.

### Python implementation

Public code-validation snapshot:

- public head: `939f0453a65f53e6aa3ab7481dde8eebf9f4124e`;
- workflow run: **37312334137**;
- Python: **3.10 / 3.12 / 3.13**;
- tests: **102 passed per matrix job**;
- machine-readable behavioral evidence: success.

The same core continued to pass after the formal-validation files were added at
head `c32f7438c6faf04853a9a03ba3d7cddff5b01843`.

Deterministic behavioral assays report:

| Semantic failure | Broken metric | Repaired / guarded metric |
|---|---:|---:|
| joint ordering | 8.48528137423857 | 0.0 |
| rotation representation + sign | 76.01498504259729 degrees | 0.0 |
| sensor freshness | 1.0 | 0.0 |

These assays establish causal executable witnesses only. They are **not** a
robotics benchmark and do not substitute for project-native closed-loop task
evaluation.

### Real GR00T source audit

A public test reads the actual repository source:

- `gr00t/data/types.py`;
- `gr00t/data/state_action/state_action_processor.py`;
- production embodiment configuration;
- SO100 example configuration.

The frozen audit infers:

- `RELATIVE`: explicitly handled;
- `ABSOLUTE`: intentional baseline only after two independent evidence planes;
- `DELTA`: unresolved.

SemRepair does **not** map “unresolved” to “bug” automatically.

## 5. Mechanized formal core

A small Lean 4 core is publicly compiled using Lean **4.34.1**.

Public formal run:

- validation head: `c32f7438c6faf04853a9a03ba3d7cddff5b01843`;
- workflow: **SemRepair formal core**;
- workflow run: **37313104123**;
- `lake update`: success;
- explicit `sorry/admit` rejection: success;
- `lake build`: success.

Mechanized theorem surface currently includes:

- pure-step canonical-meaning preservation;
- pure-step provenance preservation;
- pure-step freshness preservation;
- successful event refinement requires a receipt;
- successful event refinement records event lineage;
- total boundary progress classification;
- accepted-boundary and unique-repair specializations.

This formalization covers only the small core calculus. It does not verify the
full Python compiler or prove task correctness.

## 6. Exact upstream evidence

### NVIDIA Isaac-GR00T #786

A real upstream PR exists for multi-step episode-boundary semantics.

At this record:

- upstream PR: `NVIDIA/Isaac-GR00T#786`;
- state: open;
- mergeable: true at last audit;
- maintainer review: none at last audit;
- merged: false;
- upstream Actions: created but blocked by `action_required`;
- maintained-adoption credit: **0**.

Independent public validation for the submitted regression exists:

- run **37292195890**: public Python 3.10/3.12/3.13 boundary-semantics matrix,
  success;
- run **37292636798**: focused episode-boundary validation, success.

This is a publicly validated upstream submission, **not** maintainer
confirmation and **not** maintained adoption.

### ManiSkill #1138

ManiSkill #1138 remains a retrospective E4 representation-consistency seed and
an upstream-ready repair packet. It does not count as prospective evidence or
maintained adoption. A separate controller-sign issue (#1469) demonstrates why
representation-only repair must not be described as complete end-to-end
restoration.

## 7. Negative results and unresolved work

The following remain explicitly unresolved:

1. **Prospective I2 evidence:** 0 admitted / 0 resolved.
2. **Maintained external adoption:** 0 repositories.
3. **Maintainer-retained SemRepair CI:** 0.
4. **External-authored reuse:** 0.
5. **Standalone versioned public release:** not yet created.
6. **Citation surface:** not yet created.
7. **Native ManiSkill closed-loop execution:** not yet established.
8. **Full formal verification of the Python compiler:** not claimed.

## 8. Threats to validity

### Retrospective-seed bias

Many motivating mechanisms were known before the v2 freeze. They are useful for
method construction but cannot establish prospective discovery ability.

### Synthetic closed-loop gap

The three deterministic assays show semantic errors can change closed-loop
behavior, but they are mathematical/proxy environments rather than full native
robot-learning task runs.

### Source-rule coverage

Static semantic inference currently depends on registered source rules and may
miss implicit semantics encoded through dynamic dispatch, generated code, or
external libraries.

### Repair grammar incompleteness

The bounded repair DSL contains interpretable physical transformations but is
not complete. “No repair found” means no repair in the registered grammar, not
that no physically valid repair exists.

### Certificate scope

A repair-verification certificate binds a specific contract/program/evidence
bank. Passing that bank does not prove correctness outside its certified domain.

### External-validation gap

Public CI is independently inspectable but still runs on a repository owned by
the original author. It establishes reproducibility, not independent adoption.

### Formalization scope

The Lean core proves properties of a small model. The relationship between that
model and the entire Python implementation is supported by differential tests,
not a full refinement proof.

## 9. Current portfolio-gate interpretation

This public record satisfies the **public research record** requirement only.

It does **not** satisfy:

- prospective-case gates;
- external-maintained-adoption gate;
- two independent upstream confirmations/merges;
- any L9 adoption criterion.

SemRepair therefore remains **L8-candidate**.

## 10. Reproduction pointers

Public validation PR:

`lindicaphxag-tech/Isaac-GR00T#3`

Python core:

```text
https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37312334137
```

Lean formal core:

```text
https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37313104123
```

The exact provenance snapshot is:

`semrepair_validation/SOURCE_SNAPSHOT.json`
