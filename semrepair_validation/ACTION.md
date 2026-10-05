# SemRepair GitHub Action

SemRepair ships a composite GitHub Action so another robotics repository can run
the proof-carrying runtime gate without vendoring the research implementation.

After the fixed release branch is frozen, prefer:

```yaml
- uses: lindicaphxag-tech/Isaac-GR00T/semrepair_validation@semrepair-v0.3.0rc1
  with:
    manifest: semrepair-adoption.json
    repository-root: .
    model-depth: "3"
```

Until that branch exists, the moving development RC is
`semrepair-v0.3-rc1`.

For maximum reproducibility, pin the Action to the reviewed commit SHA used by
your integration.

The Action:

1. installs SemRepair from its own action checkout;
2. runs `semrepair-runtime-gate`;
3. optionally validates a consumer-side `semrepair-adoption.json`;
4. verifies that the declared maintained path exists in the consumer repository.

Generate an adoption skeleton with:

```bash
semrepair-adopt-init \
  --repository owner/repo \
  --contract-id embodied/representation/controller-roundtrip@0.2 \
  --semantic-layer controller/action-representation \
  --integration-type native-regression \
  --maintained-path tests/test_semantic_contract.py
```

The generated regression intentionally fails until replaced with real
repository-native semantic evidence.

A green self-test or self-authored consumer fixture is **not** external-adoption
credit. External adoption requires the independent repository to retain the
native regression/spec/tooling.
