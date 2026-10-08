# SemRepair v0.2.4: source-bound, one-use concrete repair handoff

**Public, CPU-only software proof — not robot actuation, not independent adoption.**

Frozen source: `semantic-abi-noisy-v0.2.4` (SHA recorded after branch freeze). Related code:

- `embodied_repair_handoff.py` — independent sensor + dispatch trust roots, SQLite reservation, one-consumer atomic claim, and replay of source-bound signed observations
- `tests/test_embodied_repair_handoff.py` — concurrency, crash/restart, wrong payload, key separation, SQLite-only row manipulation and source-version tests
- `repair_handoff_quick_repro.py` — single-command pure-stdlib positive/negative capability handoff

### Reproduce

```bash
git clone https://github.com/lindicaphxag-tech/Isaac-GR00T.git
cd Isaac-GR00T
git checkout semantic-abi-noisy-v0.2.4
python -m research.semantic_invariants.repair_handoff_quick_repro
python -m pytest -q research/semantic_invariants/tests/test_embodied_repair_handoff.py
```

The first command uses **inert literal byte strings** as the repaired adapter payload and public test-only HMAC keys, with no GPU. The byte digest of the exact payload must match the signed diagnosis' implementation/evidence-bound `sha256:<actual bytes>` identity. A generic descriptive label that happens to start with `sha256:` is not accepted as a deployable repair.

### Contract and trust boundaries

The API consists of two distinct atomic stages:

1. **`reserve_repair_once(...)`** accepts only a robust finished episode with unchanged source/plan digest and correct full signed sensor chain. It computes the true payload digest, verifies the selected frozen repair identity and inserts an irreversible (under no rollback) `RESERVED_UNCONFIRMED` record, returning one secret token **once**. The database retains a token hash but not the token.
2. **`consume_repair_token_once(...)`** verifies the live token and exact payload, independently verifies the sensor observations and repair identity again, verifies an HMAC over all handoff fields, and *before the external effect* atomically marks that row `CLAIMED_UNCONFIRMED`. Any concurrent/later consumer is denied. The status explicitly means **claim issued, external result unknown**.

A **separate, explicitly provisioned dispatch HMAC key** signs the handoff row, including `status`, `episode_id`, `plan_digest`, `authority_id`, `payload_sha256` and `token_sha256`. The sensor key authenticates external observations; the dispatch key authenticates locally issued handoff credentials. Neither key is derivable from the candidate repair bytes. The two keys are required to differ. Legacy unsigned handoff records are not backfilled.

### Real discovered failures and red → green evidence

After the initial consumer interface passed 3-version CI, adversarial tests found two actual credential-store bugs with a **SQLite-only attacker without dispatch key**:

- Replace `token_sha256` with SHA-256 of the attacker's self-generated token. Initial consumer code accepted that forged token.
- After legitimate claim, manually set `status='RESERVED_UNCONFIRMED'`. Initial consumer code accepted the same token again.

The public *red* [run 37749620816](https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37749620816) showed **2 security counterexample failures** in Python 3.10/3.12/3.13. The fix cryptographically binds both mutable fields to a separate dispatch secret, and verifies that row before any claim or read-only exposure.

Public *green* [diagnostic CI 37749889350](https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37749889350) ran **69 tests** per Python version (3.10, 3.12, 3.13) and includes 24-worker parallel claim tests. The companion [authority CI 37749889476](https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37749889476) passed in all three versions. These are author-supplied tests; an independent maintainer or researcher has **not** yet reproduced them.

### Unresolved external effect problem — strict nonclaims

This software hands out at most one claim in a trusted, non-rollback SQLite transaction. It does **not** prove that an actual ROS/SAPIEN/physical controller executes an action once or even executes at all.

A crash **after the claim and before executing** the action can result in **zero** external executions. If an untrusted actuator duplicates/ignores the claim, multiple effects are possible. A lost acknowledgement leaves physical completion uncertain. The only correct terminal description of this software receipt is `CLAIMED_UNCONFIRMED`; there is deliberately no `executed_successfully=True` field.

This distinction reflects established outbox/inbox and idempotent side-effect designs, not newly discovered distributed-systems theory. See [Azure's transactional outbox documentation](https://learn.microsoft.com/en-us/azure/architecture/databases/guide/transactional-out-box-cosmos) and [Pobiega, token-based side effects](https://exactly-once.github.io/posts/side-effects/). Exactly-once physical action requires additional independently controlled actuator-side fencing, stable per-effect identity, controller logs, and closed-loop observation.

**Excluded adversaries:** dispatch/sensor secret compromise, malicious code inside the trusted claimant, incorrect sensory ground truth, complete signed database snapshot rollback or deletion, and replay of an already claimed external command at a separate consumer. Cryptographically sealing a local row does not provide an external monotonic counter.

### What would make this an L8/L9 embodied-intelligence contribution?

HMAC, byte hashes, SQLite uniqueness, nonce tokens and fail-closed migrations are established engineering, not a new scientific algorithm. The high-value hypothesis is whether a *risk-bounded, actively selected semantic observation* can disambiguate an embodied control-contract error, synthesize the right concrete repair and improve **measured held-out task recovery**, while propagating evidence/source identity through the eventual physical/robot controller effect.

A credible comparison requires fixed failure injection, the same source-policy checkpoints, official ManiSkill/LeRobot baseline, a diagnosis-only/no-certificate ablation, fixed-rule or reject-only baseline, a generic active diagnosis baseline, multiple seeds, on-robot or simulator observations, recovery success and latency, false repair authorization, and independent third-party replication. None of those claims follow from the present CPU-only handoff tests.

### Reviewer challenge

Find a minimal transaction/evidence/source trace under the declared trust boundary that can (a) issue two valid claims to the same episode, (b) authorize a byte payload other than the robust plan's concrete digest, (c) accept an attacker-controlled token/status edit without the dispatch secret, or (d) accept a forged sensor observation without the independent sensor key. Include Python version, exact frozen SHA, SQLite trace and the observed CI test result. Equally useful is demonstrating that a stated trust assumption cannot be maintained for an actual robotic deployment.
