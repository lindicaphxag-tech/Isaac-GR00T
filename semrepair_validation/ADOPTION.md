# Adopting SemRepair in another robotics repository

SemRepair is designed so an upstream project does **not** need to adopt a new
robotics framework.

## Package install

Moving release-candidate branch:

```bash
python -m pip install \
  "git+https://github.com/lindicaphxag-tech/Isaac-GR00T.git@semrepair-v0.3-rc1#subdirectory=semrepair_validation"
```

Once frozen, prefer the fixed branch `semrepair-v0.3.0rc1` or an exact reviewed
commit SHA.

## Minimal source-boundary integration

Create a manifest that names producer/consumer source locations, partial semantic
types, inference rules, and allowed certified adapters. Then run:

```bash
semrepair-source --manifest path/to/semrepair.json --json
```

The compiler fails closed on conflicting evidence, unknown repairs, ambiguity,
or missing proof evidence.

## Runtime event semantics

Do not encode pure adapters that relabel:

- requested → executed;
- stale → fresh.

These transitions require owner-boundary evidence such as an execution receipt
or a fresh sensor sample.

## Repository-native adoption

For upstream projects, a native regression is often preferable to importing the
full runtime. Examples include:

- converter output reconstructed by the controller decoder preserves rotation;
- logged executed action equals the action actually sent after safety mediation;
- episode-relative lookup is invariant to storage repartition;
- environment permutation only relabels independent trajectories within the
  declared numerical floor.

Generate a skeleton:

```bash
semrepair-adopt-init \
  --repository my-org/my-robot-stack \
  --contract-id embodied/provenance/executed-action@0.2 \
  --semantic-layer dataset/safety-provenance \
  --integration-type native-regression \
  --maintained-path tests/test_executed_action_provenance.py
```

The generated placeholder deliberately raises until project-native evidence is
implemented.

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

A fork, copied snippet, issue comment, or self-authored consumer fixture counts
as **zero maintained external adoption**. SemRepair's L8/L9 ledger changes only
when an independently maintained repository intentionally retains the
regression/spec/tooling in its own source or CI.
