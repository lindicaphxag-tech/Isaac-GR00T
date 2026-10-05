# Adopting SemRepair in another robotics repository

SemRepair is designed so an upstream project does **not** need to adopt a new
robotics framework.

The smallest useful integration is one source-boundary manifest plus one CI
command.

## 1. Install the current public release candidate

```bash
python -m pip install   "git+https://github.com/lindicaphxag-tech/Isaac-GR00T.git@semrepair-v0.3-rc1#subdirectory=semrepair_validation"
```

This branch is a staging surface, not NVIDIA adoption.

## 2. Add a source-boundary manifest

Copy the structure from:

`examples/maniskill_style/semrepair.json`

The manifest names:

- producer source path and partial semantic type;
- consumer source path and partial semantic type;
- source semantic inference rules;
- certified explicit adapters.

## 3. Run fail-closed source compilation

```bash
semrepair-source --manifest path/to/semrepair.json --json
```

Exit behavior:

- success: semantic boundary accepted or repaired using explicit certified
  adapters;
- exit 2: conflicting source evidence, ambiguity, unknown repair, missing proof
  evidence, or other fail-closed condition.

## 4. Keep runtime event semantics separate

Do not create adapters that relabel:

- requested -> executed;
- stale -> fresh.

Those semantics require owner-boundary runtime evidence. A project should wire
the controller execution receipt or sensor sample into SemRepair refinement
rather than convert the label.

## 5. Upstream-friendly integration

For upstream adoption, prefer a repository-native regression over importing the
full research artifact when that is simpler.

A useful regression should preserve the semantic contract itself, for example:

- converter output reconstructed by the controller decoder preserves rotation;
- action logged as executed equals the command actually sent after safety
  mediation;
- episode-relative frame lookup is invariant to storage repartition;
- environment permutation only permutes trajectories within a justified
  numerical floor.

## Evidence needed to count as external adoption

SemRepair's own L8/L9 ledger counts an external project only when the project
maintains the regression/spec/tooling in its own repository or CI. A fork,
comment, copied snippet, or self-authored demo is intentionally insufficient.

## 6. Generate an upstream-native adoption skeleton

To lower integration friction, the release candidate includes a scaffold
generator. It deliberately creates a **failing placeholder regression** rather
than a test that can pass without project-native semantic evidence.

Example:

```bash
semrepair-adopt-init \
  --repository my-org/my-robot-stack \
  --contract-id embodied/provenance/executed-action@0.2 \
  --semantic-layer dataset/safety-provenance \
  --integration-type native-regression \
  --maintained-path tests/test_executed_action_provenance.py
```

The generated directory contains:

- `semrepair-adoption.json`: machine-readable candidate integration metadata;
- `test_semantic_contract.py`: a repository-native regression skeleton that
  raises `NotImplementedError` until replaced with real project code.

For a source-boundary CI integration, use:

```bash
semrepair-adopt-init \
  --repository my-org/my-robot-stack \
  --contract-id embodied/representation/controller-roundtrip@0.2 \
  --semantic-layer controller/action-representation \
  --integration-type source-boundary-ci
```

The adoption manifest schema is:

`schemas/adoption-manifest-v1.schema.json`

A generated manifest has `status=candidate`. SemRepair's own evidence ledger
must not promote it to maintained adoption merely because this command was run.
Promotion requires the named external repository to actually retain the
regression/spec/tooling.

## 7. Reusable GitHub Action

For CI adoption, SemRepair also exposes a composite Action:

```yaml
- uses: actions/checkout@v4

- uses: lindicaphxag-tech/Isaac-GR00T/semrepair_validation@semrepair-v0.3-rc1
  with:
    manifest: semrepair-adoption.json
    repository-root: .
    model-depth: "3"
```

The Action installs the package from its own checked-out action directory,
executes the proof-carrying semantic runtime gate, and optionally validates the
repository's adoption manifest.

For a durable integration, pin the Action to a reviewed commit SHA rather than
a moving branch.

Running this Action in a self-authored fixture still counts as **zero external
adoption**. The L8/L9 ledger changes only when an independently maintained
repository retains the integration.
