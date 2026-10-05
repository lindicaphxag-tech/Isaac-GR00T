# Contributing to SemRepair

Contributions are welcome, especially real embodied-system semantic failures
that can be reduced to a reproducible producer/consumer boundary.

## Required discipline

A semantic rule or repair should include:

- the physical meaning being preserved;
- source and target semantic types;
- why the transformation is forgeable or non-forgeable;
- a minimal counterexample;
- ambiguity and negative controls;
- tests that fail before the change and pass after it.

Do not add a generic converter merely because two tensor shapes match.

## Prospective cases

Prospective discovery evidence is stricter than ordinary bug reports. The
mechanism prediction must be frozen before reading a later maintainer diagnosis
or candidate fix. Negative, unresolved, and mismatched cases stay in the
denominator.

## Repair synthesis

Synthesized repair programs are candidates only. Runtime installation requires
independent verification evidence bound to the contract and program identity.

## Backwards compatibility

Semantic changes follow SPEC_VERSIONING.md. Historical evidence and frozen
prospective cases are never rewritten to fit a new method version.
