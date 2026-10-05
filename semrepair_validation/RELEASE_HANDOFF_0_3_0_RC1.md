# SemRepair 0.3.0rc1 — published release record

SemRepair 0.3.0rc1 has already been published as a GitHub prerelease.

Published release:

- tag: `semrepair-v0.3.0rc1`
- release id: `403794471`
- immutable target:
  `8cd7e7ad01e50aa18f42d333765f4fd242228d66`
- wheel SHA-256:
  `fa2a16877175043e5f8a727babfbedda945ccfac5170c64201dea898025e8b2c`
- sdist SHA-256:
  `996439a19a93f802005e57afe2e836b747345aecccd68491a42eb75466d2a1ec`

Published wheel and immutable-tag Git installation were independently exercised
by public workflow run **37323937866**.

## Post-release development evidence

The moving branch `semrepair-v0.3-rc1` has continued to strengthen the
artifact after publication.

Latest fully green development-surface run:

```text
head 9e60f6404e21c08aad5935387065d6e172ea5e7d
workflow 37327372332
success
```

The exact native ManiSkill patch gate was validated in run **37326497957**:

```text
before: 12.9285035° SO(3) error
after:   0.0092307°
floor:   0.0092307°
```

These post-release improvements are **not retroactively claimed as part of the
immutable rc1 snapshot**.

## Remaining L8 gates

The release/public-artifact gate is complete.

The remaining level-changing gates are external:

1. clean prospective I2 mechanism confirmation;
2. maintained upstream adoption / retained regression.

A future rc2 is optional packaging work, not itself an L8 requirement.
