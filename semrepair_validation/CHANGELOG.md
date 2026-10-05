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