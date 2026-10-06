# Start Here — Semantic ABI v0.1

This artifact is designed to be **falsified quickly**, not trusted from the PR description.

## 30-second question

> How much evidence must an embodied system collect before it may authorize one concrete repair?

The artifact distinguishes:

- **full hidden-ABI identification** — learn every hidden semantic cause;
- **repair-authority identification** — stop once every surviving cause authorizes the same exact implementation/evidence-bound repair;
- **unsafe aliasing** — reject when observationally indistinguishable causes require different repairs.

## One command

From the repository root on branch `artifact/semantic-abi-v0.1`:

```bash
python -m validation_semrepair.quick_repro
```

No simulator, GPU, checkpoint, or network access is required.

Expected headline facts in the JSON output:

```text
full_abi_expected_cost          3.5
repair_authority_expected_cost  1.0
external_alias_status           unsafe_unidentifiable
tap_resolved_status             optimal
tap_resolved_cost               2.0
dependency_drift_splits_authority true
same_authority_zero_probe_cost  0.0
independent_verifier_valid      true
```

A disagreement, exception, or different numeric result is a useful falsification.

## What the first result means

Two hidden semantic causes share one exact repair authority.

Full identity insists on distinguishing them and pays an expected cost of 3.5.
Authority-optimal diagnosis stops earlier because both surviving worlds authorize
the same concrete repair; expected cost is 1.0.

The claim is **not** that the Bellman mathematics is novel.  Equivalence-class
decision problems are prior art.  The embodied research object is the authority
class: it binds factorized repair, executable implementation identity, evidence,
and dependencies.

## What the second result means

Two worlds are externally identical:

```text
correct             -> noop
masked-double-swap  -> producer+controller repair bundle
```

External behavior alone cannot authorize either action.  Adding an internal
producer tap separates the worlds at cost 2.0.  The system refuses to collapse a
compensated double fault into a net-identity/no-op decision.

## Real-stack evidence

The minimal CPU capsule is paired with real software and dynamics anchors:

- ManiSkill #429: production action-chart ambiguity, base-red / patch-green;
- ManiSkill #1495 + #1472: compensating representation/controller-sign faults;
- NVIDIA Isaac-GR00T #786: episode-boundary semantics;
- LeRobot: exact forward/inverse self-consistency despite wrong one-way semantics;
- native MuJoCo: nuisance-confounded worlds become identifiable only after an
  internal pre-transmission tap.

See `SEMANTIC_ABI_CAPSULE.md` and `INDEPENDENT_REPLICATION.md`.

## Strong independent account-level evidence

Separately from SemRepair, Braindecode maintainers independently replicated the
author's NeuroRVQ port before merging braindecode/braindecode#1218:

- 5 seeds x 10 subject-independent folds;
- 0.837 +/- 0.003 balanced accuracy;
- paper value 0.869 +/- 0.026;
- 3.7% relative gap, inside their preregistered 5% gate;
- maintainer APPROVED and merged.

This does **not** count as SemRepair adoption.  It is listed only to distinguish
self-authored artifact validation from a genuine third-party validation event.

## Claim boundary

Current SemRepair counters remain:

```text
resolved prospective primary I2       0
maintained external SemRepair reuse    0
independent third-party reproduction   0
```

The artifact should not be cited as NVIDIA, ManiSkill, or LeRobot adoption.


## Stable checkout

A versioned public branch is frozen at the validated artifact snapshot:

```bash
git clone https://github.com/lindicaphxag-tech/Isaac-GR00T.git
cd Isaac-GR00T
git checkout semantic-abi-v0.1
python -m validation_semrepair.quick_repro
```

For strict reproducibility, record the commit printed by:

```bash
git rev-parse HEAD
```

The validated release snapshot is expected to resolve to the commit recorded in
PR #20.  If the branch and PR disagree, treat that as an artifact integrity
failure and report it rather than silently proceeding.

## Citation

Machine-readable citation metadata is in `CITATION.cff`.  This is a software
artifact citation only; it is not a claim of peer-reviewed publication.
