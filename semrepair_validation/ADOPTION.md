# Adopting SemRepair in another robotics repository

SemRepair is designed so an upstream project can adopt a semantic contract
without adopting a new robotics framework.

## Install the immutable rc1 prerelease

```bash
python -m pip install \
  "git+https://github.com/lindicaphxag-tech/Isaac-GR00T.git@semrepair-v0.3.0rc1#subdirectory=semrepair_validation"
```

Published tag target:

```text
8cd7e7ad01e50aa18f42d333765f4fd242228d66
```

The moving branch `semrepair-v0.3-rc1` contains post-release development and
should not be mistaken for the immutable released snapshot.

## Minimal source-boundary integration

Create a manifest naming producer/consumer source locations, partial semantic
types, inference rules, and allowed certified adapters, then run:

```bash
semrepair-source --manifest path/to/semrepair.json --json
```

The compiler fails closed on conflicting evidence, unknown repairs, ambiguity,
or missing proof evidence.

## Runtime event semantics

Do not use pure adapters to relabel requested→executed or stale→fresh. Those
transitions require owner-boundary runtime evidence.

## Repository-native adoption

For upstream projects, a native regression may be preferable to importing the
full runtime. Generate a conservative skeleton with:

```bash
semrepair-adopt-init \
  --repository my-org/my-robot-stack \
  --contract-id embodied/provenance/executed-action@0.2 \
  --semantic-layer dataset/safety-provenance \
  --integration-type native-regression \
  --maintained-path tests/test_executed_action_provenance.py
```

The generated placeholder intentionally fails until replaced with project-native
semantic evidence.

## Composite Action

```yaml
- uses: actions/checkout@v4

- uses: lindicaphxag-tech/Isaac-GR00T/semrepair_validation@semrepair-v0.3.0rc1
  with:
    manifest: semrepair-adoption.json
    repository-root: .
    model-depth: "3"
```

## Evidence rule

A fork, copied snippet, issue comment, reviewer request, self-authored fixture,
or self-test counts as **zero maintained external adoption**. Only independent
retention/reuse in another maintained repository changes the L8/L9 adoption
ledger.
