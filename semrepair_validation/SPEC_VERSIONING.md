# SemRepair specification versioning

SemRepair separates software versioning from semantic-specification versioning.

## Software versions

The Python package follows semantic versioning once a stable 1.0 surface is
declared. Until then, 0.x versions may change APIs, but every release candidate
must state breaking changes explicitly.

Current software version: **0.3.0-rc3**.

## Manifest schema

The portable source-boundary manifest currently uses:

```json
{"schema_version": 1}
```

A parser must reject a schema version it does not understand. Unknown semantic
fields are not ignored silently.

Compatible v1 changes may:

- add optional documentation;
- add optional metadata outside the validated semantic core only when the schema
  explicitly permits it;
- add new rule/adaptor instances that use existing fields.

A new schema version is required when:

- the meaning of an existing field changes;
- a required field is added or removed;
- assignability semantics change;
- a non-forgeable field becomes forgeable or vice versa;
- effect/refinement authorization changes.

## Execution proof bundle schema

The language-neutral proof-carrying execution bundle currently uses:

```text
semrepair-execution-proof-bundle/v0.1
```

Its canonical JSON, digest rules, non-forgeable adapter constraints, and
minimum-cost verification profile are versioned independently of the Python API.
A breaking change to those semantics requires a new bundle schema identifier.

The v0.1 digests are integrity/dependency receipts, not digital signatures.

## Semantic type compatibility

Consumer requirements are covariant only over **unspecified** axes: a target
field set to null/unspecified imposes no requirement. A concrete target field
must match exactly or be reached through an explicit certified repair/refinement
path.

Unknown producer semantics never satisfy a concrete consumer requirement.

## Contract identity

A contract or repair certificate must bind to:

- semantic contract id;
- software/spec generation;
- repair program identity when applicable;
- evidence fingerprint where verification is claimed.

Certificates are not transferable across incompatible contract generations.

## Change process

Proposed semantic changes should include:

1. motivating real boundary or counterexample;
2. old and proposed semantics;
3. compatibility impact;
4. false-positive/false-repair risk;
5. tests covering pass, fail, and ambiguity cases;
6. whether a schema/spec version bump is required.

Changes that alter prospective adjudication are applied only to a future frozen
protocol generation. Historical predictions are never re-scored under a newer
rule.