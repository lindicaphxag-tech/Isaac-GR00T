# Reusable proof-bundle verification Action

Downstream repositories can verify a
`semrepair-execution-proof-bundle/v0.2` without running the SemRepair compiler.

Example:

```yaml
- uses: lindicaphxag-tech/Isaac-GR00T/semrepair_validation/proof_action@<pinned-ref>
  with:
    bundle: artifacts/authorization.json
```

The Action fails closed unless the same bundle passes both:

1. the independent Python wire verifier; and
2. the zero-third-party-dependency Node.js reference verifier.

The caller should pin an immutable SemRepair release/tag or commit. Using a
moving branch for production authorization is discouraged.

This is a reusable integration surface, not evidence of external adoption by
itself.
