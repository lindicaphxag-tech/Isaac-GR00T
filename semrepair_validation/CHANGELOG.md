# Changelog

## 0.3.0-rc4

This release candidate hardens repair authority after public real-stack
interaction testing exposed a non-monotone semantic-repair failure mode.

### Executable repair identity

- binds verification certificates to a stable implementation ID **and** the
  actual callable source, containing source-file digest, defaults, keyword
  defaults, and closure captures;
- invalidates an old certificate when the executable `apply_fn` changes even
  if the symbolic primitive name and declared implementation ID stay the same;
- rejects custom primitives without an inspectable/stable executable identity.

### Runtime authority

- a caller can no longer install a repair by self-declaring
  `verification_status="verified"`;
- runtime installation requires a real program-bound
  `RepairVerificationCertificate`.

### Interaction-aware multi-plane authorization

- adds evidence-bound repair interaction certificates;
- fails closed on incomplete strict compensating repair bundles;
- separates local semantic fidelity from execution-domain non-regression;
- final physical authority requires **every declared evidence plane** to pass.

The motivating public ManiSkill assay deliberately preserves the negative
result: on the same first 10 official PegInsertionSide demonstrations,
current-main replay saved 9/10 episodes, converter-only saved 1/10,
controller-only 0/10, and the composed repair 8/10.  A separate direct
converter->controller SO(3) assay shows the composed repair is semantically
accurate while the singleton repairs are catastrophically wrong.  Therefore
semantic correctness alone does not authorize execution.

### Public validation

- trust-boundary matrix: workflow `37400490466`, Python 3.10/3.12/3.13,
  focused trust tests + full public core suite: success;
- complete release-candidate workflow `37400490470`: success, including Lean,
  native ManiSkill, native MuJoCo, real-source GR00T/LeRobot, external-consumer
  and reusable-action-consumer jobs.

### Claim boundary

This remains self-authored public software validation. It contributes zero
maintained external-adoption credit and zero prospective-I2 credit.

## 0.3.0-rc3

This release candidate hardens the language-neutral proof bundle against
cross-language numeric ambiguity.

### Exact-decimal wire authorization

- adds `semrepair-execution-proof-bundle/v0.2`;
- removes binary floating-point values from the hashed wire surface;
- encodes repair costs as canonical exponent-free decimal strings;
- uses exact `Decimal` arithmetic for independent path replay and minimum-cost
  uniqueness;
- normalizes Unicode text to NFC before hashing;
- separates producer-internal certificate digests from reproducible wire
  digests;
- adds a regression where `0.1 + 0.2` and `0.3` must remain an equal-cost
  ambiguity rather than becoming a false unique repair;
- keeps v0.1 available as a compatibility surface.

### Claim boundary

This is interoperability/self-validation work. It does not increment maintained
external adoption or prospective I2 counts.

# Changelog

## 0.3.0-rc2

This release candidate adds a language-neutral proof-carrying execution bundle
without changing the frozen semantic contract generation.

### Interoperability surface

- canonical JSON proof bundle `semrepair-execution-proof-bundle/v0.1`;
- JSON Schema for non-Python consumers;
- independent wire verifier that does not call compiler/search code;
- adapter-registry and contextual-evidence digest verification;
- independent selected-path replay and bounded unique minimum-cost verification;
- effect-certificate / logical-intent / compilation-dependency binding;
- fail-closed rejection of non-forgeable adapter transitions and unverified
  repair candidates;
- installed `semrepair-proof-bundle` verification CLI.

### Claim boundary

SHA-256 receipts provide deterministic integrity and dependency binding. They
are not issuer authentication or digital signatures, and the bundle does not by
itself prove the physical postcondition.


## 0.3.0-rc1

This release candidate isolates the constructive SemRepair method from the
larger research workspace.

### Method surface

- multi-axis embodied semantic tensor types;
- active semantic-hypothesis inference with probe cost/risk;
- source-code semantic fact extraction and source-to-compiler bridge;
- fail-closed type/effect boundary compilation;
- unique minimum-cost explicit adapter synthesis;
- bounded CEGIS repair synthesis for missing numerical adapters;
- non-forgeable runtime refinement for provenance and freshness;
- independent held-out repair verification certificates;
- certificate-gated runtime mediation;
- executable small-step calculus and bounded theorem checker;
- Lean 4 mechanization of the current formal core.

### Behavioral evidence

Deterministic assays cover:

1. hidden joint-order mismatch;
2. composed rotation representation + sign mismatch;
3. stale-sensor freshness requiring owner-boundary resampling.

These assays are causal prototype evidence, not real-robot benchmark results.

### External evidence boundary

This release candidate does not claim NVIDIA adoption, maintained upstream
reuse, L8/L9 achievement, or a prospective I2 mechanism match.