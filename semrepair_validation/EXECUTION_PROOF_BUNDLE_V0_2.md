# SemRepair execution proof bundle v0.2

Status: **release-candidate interoperability surface**.

## Why v0.2 exists

v0.1 used ordinary JSON numbers in the hashed wire representation. That is not
strong enough for a language-neutral authorization format: different runtimes
can serialize the same binary floating-point value differently, and binary
arithmetic can change minimum-cost ordering.

The canonical example is:

```text
0.1 + 0.2 = 0.30000000000000004   # common binary float result
0.3       = 0.3
```

A repair verifier must not turn two mathematically equal decimal-cost paths into
a false unique minimum because of host-language floating-point rounding.

## v0.2 wire rules

v0.2 therefore separates producer-internal certificate identities from the
language-neutral wire proof.

The hashed wire surface has **no binary floating-point values**:

- adapter costs are canonical exponent-free decimal strings;
- arbitrary float evidence is converted to a reserved tagged-decimal object;
- minimum-cost verification uses exact decimal arithmetic;
- Unicode strings are normalized to NFC;
- object keys are sorted;
- JSON is UTF-8 with no insignificant whitespace;
- integers remain JSON integers.

Canonicalization profile:

```text
semrepair-wire-c14n/v0.1
```

Bundle schema:

```text
semrepair-execution-proof-bundle/v0.2
```

## Independent verification

A conforming v0.2 verifier independently checks:

1. outer bundle wire digest;
2. adapter-registry wire digest;
3. contextual-evidence wire digest;
4. compilation-record wire digest;
5. effect-record wire digest;
6. selected adapter-path replay;
7. exact-decimal unique minimum-cost authorization;
8. non-forgeable provenance/freshness constraints;
9. semantic-contract identity;
10. effect/intent agreement;
11. compilation -> physical-effect dependency identity.

The producer's Python-side `decision_digest` remains embedded as an **opaque
dependency identity**. v0.2 does not pretend a foreign runtime can reproduce a
producer-specific historical serialization rule. Instead, the normalized wire
record receives its own independently reproducible digest.

## Claim boundary

Wire digests are deterministic integrity and dependency-binding receipts. They
are not digital signatures, issuer authentication, or a proof that the physical
postcondition happened.

Physical commit still requires the runtime's independent observation-plane
evidence policy.

## Interoperability target

C++, Rust, ROS 2, JVM, or other consumers can implement the v0.2 verifier
without importing SemRepair's Python compiler or search implementation.
