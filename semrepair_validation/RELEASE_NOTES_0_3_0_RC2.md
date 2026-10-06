# SemRepair 0.3.0rc2

Status: **staging for public validation**.

SemRepair 0.3.0rc2 is an authority-boundary hardening release derived from the
published and clean-install validated 0.3.0rc1 snapshot.

## New in rc2

### Executable-bound repair certificates

Repair verification no longer treats a symbolic primitive name as executable
identity. Certificates bind a stable implementation identifier together with
inspectable callable/source-file digests, defaults, and closure captures.

Changing the executable repair implementation invalidates the old certificate.

### Certificate-only runtime authority

A caller-provided `verification_status="verified"` value cannot install a
repair. Runtime mediation requires a real `RepairVerificationCertificate`
matching the contract and current executable repair.

### Interaction-aware repair authorization

rc2 adds factorial repair-lattice analysis and repair-set atomicity.

When end-to-end evidence identifies a strict compensating repair bundle,
non-empty proper subsets fail closed even if their individual local repairs are
correct. The complete bundle may proceed only when the measured end-to-end
policy allows it.

## Public real-stack evidence

Two self-authored public ManiSkill executions replay the same first 10 official
`PegInsertionSide-v1` motion-planning demonstrations across four cells.

Parallel execution (`37388740482`) produced 8/10 main, 1/10 converter-only,
0/10 controller-only, and 8/10 composed.

Serial exact-head execution (`37394500249`) produced 9/10 main, 1/10
converter-only, 0/10 controller-only, and 8/10 composed.

The singleton regressions and composed recovery are repeatable, but the
main-vs-composed relation is not. rc2 therefore does not treat task-success
evidence as sufficient authority for the composed repair. Replicated
authorization fails closed until a direct semantic-fidelity metric resolves
the ambiguity.

This is real-stack interaction evidence, not learned-policy improvement,
real-robot safety, maintainer-retained adoption, or prospective I2 confirmation.

## External evidence boundary

At rc2 staging time:

- maintained external adoption: **0**;
- prospective resolved I2 cases: **0**;
- rc2 tagged release: **not yet**;
- rc2 public validation: **pending**.

The release must remain a candidate until those publication-specific checks are
completed.
