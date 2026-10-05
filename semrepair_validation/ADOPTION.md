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
