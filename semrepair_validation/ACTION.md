# SemRepair GitHub Action

This directory is also a composite GitHub Action.

A robotics repository can add SemRepair to CI without vendoring the research
runtime:

```yaml
- uses: lindicaphxag-tech/Isaac-GR00T/semrepair_validation@semrepair-v0.3-rc1
  with:
    manifest: semrepair-adoption.json
    repository-root: .
    model-depth: 3
```

The action performs two independent checks:

1. installs the SemRepair package from the action checkout and runs the
   proof-carrying semantic-runtime release gate;
2. if `manifest` is supplied, validates the consumer repository's adoption
   manifest and verifies that its declared maintained path exists inside the
   checked-out repository.

A green action run is **not** external-adoption credit by itself. The independent
repository must intentionally retain its native regression/spec/tooling. The
SemRepair ledger verifies that external fact separately.

For a minimal candidate manifest, first run:

```bash
semrepair-adopt-init \
  --repository owner/repo \
  --contract-id embodied/representation/controller-roundtrip@0.2 \
  --semantic-layer controller/action-representation \
  --integration-type native-regression \
  --maintained-path tests/test_semantic_contract.py
```

Then replace the generated failing placeholder with a real repository-native
semantic regression.
