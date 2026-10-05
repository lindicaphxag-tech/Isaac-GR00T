# SemRepair execution proof bundle v0.1

Status: **experimental interoperability surface for the next SemRepair release**.

## Goal

A host project should be able to verify the authorization chain for an embodied
semantic repair without importing the SemRepair compiler or trusting its search
algorithm.

The v0.1 bundle is canonical JSON and binds:

- semantic-contract identity;
- semantic compilation certificate;
- exact adapter registry;
- contextual evidence identities;
- independently replayable selected adapter path;
- minimum-cost uniqueness profile;
- physical-effect certificate;
- logical physical-effect identity;
- compilation -> effect dependency provenance.

## Verification

A conforming verifier must independently:

1. recompute the outer bundle SHA-256 over canonical JSON;
2. recompute the semantic-compilation certificate digest;
3. recompute the adapter-registry digest;
4. recompute the contextual-evidence digest;
5. replay the selected adapter path from refined source semantics to target;
6. verify selected path cost;
7. exhaustively check that the selected path is the unique minimum-cost solution
   within the declared bounded verification profile;
8. recompute the physical-effect certificate digest;
9. verify effect certificate and logical intent identities agree;
10. verify semantic-contract identity;
11. verify both effect certificate and intent are bound to the exact compilation
    decision digest.

The reference verifier in `execution_bundle_wire.py` performs these operations
over plain mappings and does not call the production compiler/search routines.

## Canonical JSON

All digests use UTF-8 JSON with:

- object keys sorted;
- separators `,` and `:` with no added whitespace;
- Unicode preserved rather than ASCII-escaped.

Digest:

```text
sha256(canonical_json(payload_without_digest))
```

## Security / claim boundary

The SHA-256 fields are deterministic integrity and dependency-binding receipts.
They are **not digital signatures** and do not authenticate who issued a
certificate.

The v0.1 wire verifier also does not prove the physical postcondition itself.
Physical commit remains governed by the semantic contract and independent
observation-plane policy.

Therefore the bundle establishes:

```text
declared semantic environment
-> independently verifiable repair authorization
-> exact physical-effect dependency binding
```

not:

```text
trusted identity of issuer
or
universal proof that the physical world reached the desired state.
```

## Interoperability target

A C++, Rust, ROS, or other runtime can implement this specification directly.
It need not run the SemRepair Python compiler.

The reference JSON Schema is:

`semrepair-execution-proof-bundle-v0.1.schema.json`.

Breaking changes require a new bundle schema identifier.