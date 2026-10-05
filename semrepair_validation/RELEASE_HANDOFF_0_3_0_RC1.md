# Release handoff — 0.3.0rc1

All technical release prerequisites are present on branch
`semrepair-v0.3-rc1`.

Recommended GitHub release fields:

- tag: `semrepair-v0.3.0rc1`
- target: `e015dff78a3177f61b500947c9a46bdb57fe9beb`
- title: `SemRepair 0.3.0rc1 — Proof-Carrying Semantic Runtime`
- prerelease: yes
- notes: use `semrepair_validation/RELEASE_NOTES_0_3_0_RC1.md`

Verified before release:

- LICENSE present;
- CITATION.cff present;
- package version 0.3.0rc1;
- wheel/sdist build succeeds;
- public Git install succeeds;
- Python 3.10/3.12/3.13 validation succeeds;
- Lean formal build succeeds with incomplete-proof rejection;
- native ManiSkill boundary succeeds;
- native MuJoCo closed-loop assay succeeds.

The connected GitHub tool available in this session does not expose tag/release
creation, so no release/tag is claimed here.
