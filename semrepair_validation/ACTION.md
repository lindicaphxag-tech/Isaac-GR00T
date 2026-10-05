# SemRepair GitHub Action

SemRepair ships a composite GitHub Action so another robotics repository can run
the proof-carrying runtime gate without vendoring the research implementation.

Prefer the immutable published rc1 tag:

```yaml
- uses: lindicaphxag-tech/Isaac-GR00T/semrepair_validation@semrepair-v0.3.0rc1
  with:
    manifest: semrepair-adoption.json
    repository-root: .
    model-depth: "3"
```

The tag `semrepair-v0.3.0rc1` is published and points to the immutable rc1
snapshot. For maximum provenance control, a consumer may instead pin the exact
tag commit `8cd7e7ad01e50aa18f42d333765f4fd242228d66`.

Do not pin the moving `semrepair-v0.3-rc1` development branch for a maintained
integration unless you intentionally want post-release development changes.

The Action:

1. installs SemRepair from its action checkout;
2. runs the proof-carrying semantic-runtime gate;
3. optionally validates `semrepair-adoption.json`;
4. verifies that the declared maintained path exists in the consumer repository.

A self-authored Action smoke test remains zero external-adoption credit.
