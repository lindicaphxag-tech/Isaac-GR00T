# Release handoff — SemRepair 0.3.0rc1

The technical release-candidate surface is validated.

Latest fully green code validation:

- code head: `b24d33ce816e2967b3ec8701d30a2213ecf72991`
- workflow: `37326497957`
- conclusion: success

Recommended release sequence:

1. let the documentation-only release-freeze commit complete CI;
2. freeze branch `semrepair-v0.3.0rc1` at that green commit;
3. create tag `semrepair-v0.3.0rc1` from the fixed branch;
4. create GitHub prerelease titled
   `SemRepair 0.3.0rc1 — Proof-Carrying Semantic Runtime`;
5. use `semrepair_validation/RELEASE_NOTES_0_3_0_RC1.md`.

Verified surfaces include:

- LICENSE and CITATION.cff;
- package version 0.3.0rc1;
- Python 3.10 / 3.12 / 3.13;
- wheel + sdist;
- public Git consumer install;
- composite Action consumer smoke;
- Lean build with incomplete-proof rejection;
- production/compiler conformance;
- proof-carrying runtime release gate;
- physical-effect fault assay;
- real-source GR00T / LeRobot / ManiSkill;
- native MuJoCo closed loop;
- native ManiSkill exact-patch red→green gate.

Native ManiSkill measurement:

```text
before patch: 12.9285035° SO(3) error
after patch:   0.0092307°
SAPIEN floor:  0.0092307°
```

The connected GitHub tool exposes release reads but not tag/release creation, so
no GitHub tag/release is claimed by this handoff.
