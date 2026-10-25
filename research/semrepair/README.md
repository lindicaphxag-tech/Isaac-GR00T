# SemRepair — Public Research Evidence

SemRepair studies **authorization of executable semantic repairs in embodied-AI software**.

The project asks a narrower question than generic automated program repair:

> When locally plausible semantic fixes interact with latent defects, hidden control protocols, or non-identifying evidence, what proof is required before the executable behavior is allowed to change?

This directory is a public evidence surface for the self-authored SemRepair research line. It is **not** NVIDIA adoption of SemRepair.

## Release

Public prerelease:

- [SemRepair 0.3.0rc1](https://github.com/lindicaphxag-tech/Isaac-GR00T/releases/tag/semrepair-v0.3.0rc1)
- wheel SHA-256: `fa2a16877175043e5f8a727babfbedda945ccfac5170c64201dea898025e8b2c`
- sdist SHA-256: `996439a19a93f802005e57afe2e836b747345aecccd68491a42eb75466d2a1ec`

## Case A — ManiSkill compensating repairs

Public validation surface:

- https://github.com/lindicaphxag-tech/ManiSkill/pull/1
- paired semantic run: https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37401619096

On two frozen request corpora, all four repair cells are evaluated on identical requests with episode-level inference.

The paired result is:

```
converter-only repair  -> semantic error increases
controller-only repair -> semantic error increases
composed repair        -> returns to / improves on main semantic neighborhood
```

This is a real repair-interaction / non-monotonicity witness.

A separate execution replay showed that semantic restoration does not automatically imply task-level non-regression. Later same-base runs also exposed simulator-level replay variability, so **single-run 8/10 vs 9/10 counts are no longer treated as deterministic authority evidence**.

Current execution work therefore includes a repeatability sensitivity probe before any stronger deployment claim.

## Case B — LeRobot evidence collision

Pinned upstream:

- `huggingface/lerobot@8c920c4270460851cedd2737657584586d3dc66f`
- issue: https://github.com/huggingface/lerobot/issues/3863
- community fix PR: https://github.com/huggingface/lerobot/pull/4111

Public source-bound witness:

- https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37406064561

The current relative-action forward and inverse helpers share the same positional-prefix state anchor. On a non-prefix state/action layout:

| metric | result |
| --- | ---: |
| one-way semantic L∞ error | **59.7** |
| forward→inverse roundtrip L∞ error | **0.0** |
| deterministic sweep cases | **32** |
| sweep cases with exact roundtrip | **32/32** |
| sweep cases with wrong one-way semantics | **32/32** |

This is an **evidence collision**: two semantic hypotheses can be indistinguishable under internal roundtrip evidence while materially different under an external semantic oracle.

The underlying LeRobot defect/fix belongs to the LeRobot community. SemRepair claims only the source-bound identifiability witness and its authorization implication.

## Authorization gates

The current SemRepair evidence model is:

```
implementation identity
        ↓
external semantic identifiability
        ↓
value semantics
        ↓
protocol-effect semantics
        ↓
repair-set interaction
        ↓
execution-domain stability / non-regression
        ↓
policy-level effect, when required
        ↓
authorization
```

A local unit test, cycle-consistency check, or status string is not deployment authority.

## Executable identifiability gate

`tools/evidence_collision_certificate.py` detects sampled **evidence collisions**.

A collision exists when two semantic hypotheses are equivalent under supplied internal evidence but materially separated under an external semantic oracle.

The checker is deliberately fail-closed:

- collision found → `non_identifying / reject_until_external_anchor`;
- no sampled collision + no verified external anchor → `undetermined / reject_until_external_anchor`;
- verified anchor → may advance to the next gate.

Absence of a sampled collision is **not** promoted to a theorem of global identifiability.

The included LeRobot fixture is validated by public CI.

## Reproduction / falsification

External disagreement is welcome.

Useful evidence includes:

- an independent reproduction of the ManiSkill paired semantic interaction;
- a reproduction or falsification of the LeRobot evidence collision;
- another real embodied-software case where an internally strong invariant is non-identifying;
- a counterexample showing an authorization gate is unnecessarily conservative.

Please retain exact SHAs, source hashes, environment details, commands, raw results, and protocol deviations.

Because fork Issues are disabled, this public draft PR is the discussion surface: external reproductions can be posted directly as PR comments.

## External-recognition boundary

Current counters are intentionally conservative:

- independent third-party SemRepair reproductions: **0**
- upstream maintainer reviews of the SemRepair mechanism: **0**
- upstream-retained SemRepair-derived regressions: **0**
- upstream-retained SemRepair-derived production fixes: **0**
- external citations of the SemRepair mechanism: **0**

Public self-authored evidence does not increment those counters.

See [EXTERNAL_EVIDENCE.md](./EXTERNAL_EVIDENCE.md).

## Novelty boundary

Cycle-consistency ambiguity, mathematical identifiability, proof-carrying code, program contracts, metamorphic testing, multiple-fault masking, and semantic program repair are established topics.

SemRepair does not claim those concepts as new.

The narrower contribution under test is:

> make semantic identifiability, protocol effects, repair interaction, and execution stability explicit authorization obligations for executable repairs in embodied-AI software, and bind those obligations to real cross-stack failures.
