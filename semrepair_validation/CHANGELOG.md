# Changelog

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