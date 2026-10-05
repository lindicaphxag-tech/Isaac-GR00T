# Contract Compatibility Policy v0.1

SemRepair canonical contract IDs use:

```text
embodied/<family>/<name>@<major>.<minor>
```

This policy defines the minimum compatibility discipline required for
cross-repository use.

Within the same major version, a revision may:

- add compatibility aliases;
- move a contract from specified to executable;
- change implementation adapters without changing the advertised relation;
- add new contracts to a registry.

Within the same major version, a revision may **not**:

- change the canonical semantic relation;
- remove a compatibility alias;
- downgrade an executable contract to specified;
- move the version backwards;
- remove a contract from a registry without an explicit successor policy.

A semantic change requires a major-version bump.

The public checker is:

```bash
semrepair-contract-compat --old old-registry.json --new new-registry.json --json
```

This policy protects public semantic identity. It does not prove that arbitrary
adapter implementations are behaviorally equivalent; executable adapters remain
subject to their own verification/certificate rules.
