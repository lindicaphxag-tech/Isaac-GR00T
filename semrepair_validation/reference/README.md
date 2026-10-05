# Independent reference verifiers

SemRepair's language-neutral proof claim is not validated only by the Python
implementation.

## Node.js v0.2 verifier

`reference/verify_bundle_v2.mjs` is a zero-third-party-dependency verifier for
`semrepair-execution-proof-bundle/v0.2`.

It independently implements:

- the `semrepair-wire-c14n/v0.1` canonicalization profile;
- UTF-8 SHA-256 wire digests;
- NFC string normalization;
- rejection of fractional/exponent JSON number tokens;
- exact decimal addition/comparison with `BigInt`;
- semantic adapter replay;
- non-forgeable provenance/freshness checks;
- bounded exhaustive simple-path enumeration;
- unique minimum-cost authorization;
- effect/intent/compilation dependency binding.

It does **not** import the Python compiler or search code.

This is portability evidence, not external adoption. L8/L9 adoption counters
remain zero until an independently maintained repository retains the verifier,
contract, regression, or repair pattern.
